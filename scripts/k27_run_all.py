"""Serial driver for the 27K tuning phase with resource monitoring (protocol_confirmed.json 'resources').

Each job runs as its own process under /usr/bin/time -l; swap-out pages (vm_stat) before and after, wall
time, peak memory and the macOS memory-pressure level (sampled every 10 s) are appended to
run_dev/resource_log.jsonl. Rule revision 2026-10-21 (user-approved, see resource_incidents_zh.md):
swap-outs are recorded but never pause by themselves; the driver pauses when a job fails (including the
non-finite checks inside each stage), when memory pressure stays critical for 2 minutes (the job is
terminated), or when a job takes more than twice the baseline of the same workload (iALS: 640 s, the
uncontended grid runs; a Tree seed rerun: the wall time of the same configuration at seed 0). Finished
jobs (result file present) are skipped, so a rerun resumes. The driver holds no data; no job reads
report-group scores.
"""
import argparse
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'reports/expansion/run_dev'; PY = str(ROOT/'.venv/bin/python')
PAGE = 16384; IALS_BASELINE = 640; CRITICAL = 4; CRITICAL_POLLS = 12
MACOS = sys.platform == 'darwin'; TIME = ['/usr/bin/time', '-l'] if MACOS else ['/usr/bin/time', '-v']   # swap/pressure monitoring is macOS-only

def swapouts():
    if not MACOS: return 0
    try: return int(re.search(r'Swapouts:\s+(\d+)', subprocess.run(['vm_stat'], capture_output=True, text=True).stdout).group(1))
    except Exception: return 0

def pressure():
    if not MACOS: return 0
    try: return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], capture_output=True, text=True).stdout.strip())
    except Exception: return 0

def baseline(tag, stage, expected):
    if stage == 'ials': return IALS_BASELINE
    if stage in ('tree', 'deep') and not tag.endswith('_s0'):
        same = tag.rsplit('_s', 1)[0]+'_s0'
        for line in open(OUT/'resource_log.jsonl'):
            r = json.loads(line)
            if r['job'] == same and r['exit'] == 0: wall = r['wall_seconds']
        return wall
    return expected

def tuning_jobs():
    jobs = [('eval', [], 'eval', 600), ('lr_L1', ['--level', 'L1'], 'lr_L1', 900), ('lr_L2', ['--level', 'L2'], 'lr_L2', 900)]
    for level in ['L1', 'L2']:
        for c in range(8): jobs.append((f'tree_{level}_c{c}_s0', ['--level', level, '--config', str(c), '--seed', '0'], 'tree', 1200))
    for tau, alpha, l2 in itertools.product([.5, 2], [10, 30, 100], [30, 100, 300]):
        jobs.append((f'ials_tau{tau:g}_alpha{alpha:g}_l2{l2:g}_s0', ['--tau', str(tau), '--alpha', str(alpha), '--l2', str(l2), '--seed', '0'], 'ials', 900))
    return jobs

def run(jobs):
    OUT.mkdir(parents=True, exist_ok=True); log = OUT/'resource_log.jsonl'
    for tag, args, stage, expected in jobs:
        if (OUT/f'{tag}.json').exists(): continue
        script, stage_name = {'deep': ('k27_deep.py', 'train'), 'lr_fusion': ('k27_deep.py', 'lr-fusion')}.get(stage, ('k27_experiment.py', stage if stage in ('eval', 'tree', 'ials') else 'lr'))
        extra = ['--out', str(OUT/f'{tag}.json')] if script == 'k27_deep.py' else []
        base = baseline(tag, stage, expected); before = swapouts(); t0 = time.time(); levels = []; streak = 0; killed = False
        proc = subprocess.Popen([*TIME, PY, str(ROOT/'scripts'/script), stage_name, *args, *extra], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT/'scripts', start_new_session=True)
        while True:
            try:
                out, err = proc.communicate(timeout=10); break
            except subprocess.TimeoutExpired:
                lv = pressure(); levels.append(lv); streak = streak+1 if lv >= CRITICAL else 0
                if streak >= CRITICAL_POLLS and not killed: os.killpg(proc.pid, signal.SIGTERM); killed = True   # whole job tree, incl. the iALS child
        wall = time.time()-t0; delta = (swapouts()-before)*PAGE
        peak = re.search(r'(\d+)\s+peak memory footprint', err) if MACOS else re.search(r'Maximum resident set size \(kbytes\): (\d+)', err)
        rec = {'job': tag, 'exit': proc.returncode, 'wall_seconds': wall, 'baseline_seconds': base, 'peak_bytes': (int(peak.group(1))*(1 if MACOS else 1024)) if peak else None, 'swapout_bytes': delta,
               'pressure_samples': {str(k): levels.count(k) for k in sorted(set(levels))}, 'terminated_for_pressure': killed}
        with open(log, 'a') as fh: fh.write(json.dumps(rec)+'\n')
        (OUT/'logs').mkdir(exist_ok=True); (OUT/'logs'/f'{tag}.log').write_text(out+'\n'+err)
        print(json.dumps(rec), flush=True)
        if killed: sys.exit(f'paused: {tag} terminated after memory pressure stayed critical for 2 minutes')
        if proc.returncode != 0: sys.exit(f'paused: {tag} failed')
        if wall > 2*base: sys.exit(f'paused: {tag} took {wall:.0f} s (> 2x baseline {base:.0f} s)')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--jobs-file'); a = p.parse_args()
    run([tuple(j) for j in json.loads(Path(a.jobs_file).read_text())] if a.jobs_file else tuning_jobs())
