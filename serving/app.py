"""FastAPI service over the two-stage recommender. Run from the repository root:

    .venv/bin/uvicorn serving.app:app --host 127.0.0.1 --port 8000 --workers 1

Environment: KUAIRAND_BUNDLE (default artifacts/serving_bundle/v1), KUAIRAND_THREADS (default 8).
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # torch and faiss-cpu each ship an OpenMP runtime on macOS; without this the process aborts (OMP Error #15)
import os
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'serving'))
from model import Recommender  # noqa: E402

BUNDLE = Path(os.environ.get('KUAIRAND_BUNDLE', ROOT/'artifacts/serving_bundle/v1')); THREADS = int(os.environ.get('KUAIRAND_THREADS', '8'))
app = FastAPI(title='kuairand-recsys serving demo', version='1.0'); rec = Recommender(BUNDLE, threads=THREADS)

class ScoreRequest(BaseModel):
    user_id: int
    video_ids: List[int]
    as_of_ms: Optional[int] = None

@app.get('/health')
def health():
    return {'status': 'ok', 'bundle_version': rec.manifest['bundle_version'], 'load_seconds': rec.load_seconds, 'threads': rec.threads, 'items': rec.n_items}

@app.get('/manifest')
def manifest():
    return rec.manifest

@app.get('/recommend')
def recommend(user_id: int, as_of_ms: Optional[int] = None, k: int = Query(20, ge=1, le=200), n_candidates: int = Query(200, ge=1, le=2000),
              index: str = Query('flat', pattern='^(flat|hnsw)$'), ef_search: int = Query(128, ge=1, le=4096)):
    try: return rec.recommend(user_id, as_of_ms, k, n_candidates, index, ef_search)
    except ValueError as e: raise HTTPException(400, str(e))

@app.post('/score')
def score(req: ScoreRequest):
    if not req.video_ids: raise HTTPException(400, 'video_ids is empty')
    try: scores, feats = rec.score(req.user_id, np.asarray(req.video_ids, np.int64), req.as_of_ms)
    except ValueError as e: raise HTTPException(400, str(e))
    except KeyError as e: raise HTTPException(404, str(e))
    return {'user_id': req.user_id, 'as_of_ms': rec.check_as_of(req.as_of_ms), 'scores': [float(s) for s in scores], 'has_factors': [bool(f) for f in feats['has_factors']],
            'user_known': rec.user_index(req.user_id) is not None, 'bundle_version': rec.manifest['bundle_version']}
