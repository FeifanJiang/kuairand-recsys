"""Performance benchmark of the service (serving/README.md, completion criterion 4). Run from the repository root.

Starts uvicorn (one worker) with the bundle, waits for /health, warms up, then measures /recommend latency (p50/p95)
and throughput at several concurrency levels for the exact and the HNSW index, with random known users and the
default as_of. Records peak RSS of the server process, load and index-build times, stage timings reported by the
service, hardware, threads and cache conditions. Writes reports/expansion/serving/benchmark.json.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # torch and faiss-cpu each ship an OpenMP runtime on macOS; without this the process aborts (OMP Error #15)
import asyncio
import json
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
import httpx
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'reports/expansion/serving'; BUNDLE = ROOT/'artifacts/serving_bundle/v1'
CONFIGS = [('flat', 128), ('hnsw', 128)]; CONCURRENCY = [1, 4, 16]; REQUESTS = {1: 200, 4: 400, 16: 800}; THREADS = 8

def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p

async def run_load(url, users, n, conc, index, ef):
    lat, stage_r, stage_f = [], [], []; sem = asyncio.Semaphore(conc); rng = np.random.default_rng(3)
    async with httpx.AsyncClient(timeout=120) as c:
        async def one(u):
            async with sem:
                t0 = time.perf_counter(); r = await c.get(f'{url}/recommend', params={'user_id': int(u), 'index': index, 'ef_search': ef}); dt = time.perf_counter()-t0
                r.raise_for_status(); j = r.json(); lat.append(dt*1000); stage_r.append(j['timing_ms']['retrieval']); stage_f.append(j['timing_ms']['features_and_scoring'])
        t0 = time.perf_counter(); await asyncio.gather(*[one(u) for u in rng.choice(users, n)]); wall = time.perf_counter()-t0
    q = lambda a: {'p50_ms': float(np.percentile(a, 50)), 'p95_ms': float(np.percentile(a, 95)), 'mean_ms': float(np.mean(a))}
    return {'requests': n, 'concurrency': conc, 'wall_seconds': wall, 'throughput_rps': n/wall, 'latency': q(lat), 'stage_retrieval': q(stage_r), 'stage_features_scoring': q(stage_f)}

def main():
    port = free_port(); url = f'http://127.0.0.1:{port}'; env = dict(**__import__('os').environ, KUAIRAND_BUNDLE=str(BUNDLE), KUAIRAND_THREADS=str(THREADS))
    t0 = time.time(); proc = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'serving.app:app', '--host', '127.0.0.1', '--port', str(port), '--workers', '1', '--log-level', 'warning'], cwd=ROOT, env=env)
    p = psutil.Process(proc.pid); peak = {'rss': 0}; stop = threading.Event()
    def watch():
        while not stop.is_set():
            try: peak['rss'] = max(peak['rss'], p.memory_info().rss)
            except psutil.Error: break
            time.sleep(.2)
    threading.Thread(target=watch, daemon=True).start()
    health = None
    for _ in range(600):
        try: health = httpx.get(f'{url}/health', timeout=5).json(); break
        except Exception: time.sleep(1)
    assert health, 'server did not start'; startup_wall = time.time()-t0
    try:
        ials = np.load(BUNDLE/'ials.npz'); users = ials['users'][ials['user_valid']]; manifest = json.loads((BUNDLE/'manifest.json').read_text())
        asyncio.run(run_load(url, users, 50, 4, 'flat', 128)); asyncio.run(run_load(url, users, 50, 4, 'hnsw', 128))        # warm-up
        results = []
        for index, ef in CONFIGS:
            for conc in CONCURRENCY:
                r = asyncio.run(run_load(url, users, REQUESTS[conc], conc, index, ef)); r.update({'index': index, 'ef_search': ef if index == 'hnsw' else None}); results.append(r); print(json.dumps(r), flush=True)
        stop.set()
        out = {'hardware': {'machine': platform.machine(), 'cpu': subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip() or platform.processor(),
                            'cores': psutil.cpu_count(logical=True), 'memory_gb': psutil.virtual_memory().total/1e9, 'os': platform.platform()},
               'server': {'workers': 1, 'faiss_threads': THREADS, 'python': platform.python_version(), 'bundle_version': manifest['bundle_version'], 'startup_wall_seconds': startup_wall, 'load_seconds': health['load_seconds'], 'peak_rss_gb': peak['rss']/1e9},
               'index_build_seconds': {'flat': manifest['build_info']['ials']['seconds_flat_build'], 'hnsw': manifest['build_info']['ials']['seconds_hnsw_build'], 'hnsw_M': manifest['build_info']['ials']['hnsw_M'], 'hnsw_efConstruction': manifest['build_info']['ials']['hnsw_efConstruction']},
               'bundle_build_seconds_total': manifest['build_seconds_total'],
               'conditions': 'warm process (50 warm-up requests per index); random known users with valid factors; default as_of; k=20, 200 candidates; the service caches nothing per request; client and server on the same machine; latency measured at the client',
               'results': results}
        OUT.mkdir(parents=True, exist_ok=True); (OUT/'benchmark.json').write_text(json.dumps(out, indent=2, default=float)); print(json.dumps({k: v for k, v in out.items() if k != 'results'}, indent=2, default=float))
    finally:
        proc.terminate(); proc.wait(timeout=30)

if __name__ == '__main__':
    main()
