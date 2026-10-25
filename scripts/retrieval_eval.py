"""Shared frozen validation protocol for full-catalog retrieval models.

Matches the popularity baseline: standard validation long-view targets, deduplicated
user-video pairs, training-known users with at least one in-catalog target, and
seen items allowed. Test and random-exposure files are never read.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
KS = [20, 50, 100]

def load_targets(users, catalog, max_train, data=DATA, split='validation',
                 baseline_path=ROOT/'reports/popularity/per_user_metrics.csv'):
    targets=[]; validation_min=10**18
    for chunk in pd.read_csv(data/f'{split}.csv.gz',usecols=['user_id','video_id','long_view','time_ms'],chunksize=250000):
        validation_min=min(validation_min,int(chunk.time_ms.min()))
        targets.append(chunk.loc[chunk.long_view.eq(1),['user_id','video_id']])
    assert max_train<validation_min
    target=pd.concat(targets,ignore_index=True).drop_duplicates()
    known=target.loc[target.user_id.isin(users)]
    eligible=known.loc[known.video_id.isin(catalog)]
    groups=eligible.groupby('user_id').video_id.agg(set)
    baseline=pd.read_csv(baseline_path).set_index('user_id')
    assert set(groups.index)==set(baseline.index)
    assert all(len(v)==int(baseline.loc[k,'eligible_target_count']) for k,v in groups.items())
    return groups, known.groupby('user_id').size(), len(eligible), baseline

def exact_top100(score_block, n_users, n_items, log, block=8192):
    """score_block(start, stop) returns an (n_users, stop-start) score tensor."""
    best_scores=torch.full((n_users,100),float('-inf'))
    best_ids=torch.full((n_users,100),-1,dtype=torch.int64)
    with torch.no_grad():
        for start in range(0,n_items,block):
            stop=min(start+block,n_items)
            top,loc=score_block(start,stop).topk(min(100,stop-start),dim=1)
            combined=torch.cat([best_scores,top],dim=1)
            combined_ids=torch.cat([best_ids,loc+start],dim=1)
            best_scores,keep=combined.topk(100,dim=1)
            best_ids=torch.gather(combined_ids,1,keep)
            if start%(block*50)==0: log(f'Exact retrieval: {stop:,}/{n_items:,} items')
    assert np.isfinite(best_scores.numpy()).all()
    return best_ids.numpy(), best_scores.numpy()

def evaluate(eval_users, recommendations, groups, all_counts, eligible_pairs, baseline, seed=2026):
    assert all(len(set(row))==100 for row in recommendations)
    rows=[]
    for user_id,recs in zip(eval_users,recommendations):
        truth=groups.loc[user_id]
        row={'user_id':int(user_id),'eligible_target_count':len(truth),'all_target_count':int(all_counts.loc[user_id])}
        for k in KS:
            hits=len(truth.intersection(recs[:k]));row[f'hits_at_{k}']=hits;row[f'recall_at_{k}']=hits/len(truth)
        rows.append(row)
    per_user=pd.DataFrame(rows).set_index('user_id')
    metrics=[];boot_rng=np.random.default_rng(seed)
    for k in KS:
        differences=(per_user[f'recall_at_{k}']-baseline.loc[per_user.index,f'recall_at_{k}']).to_numpy()
        draws=boot_rng.choice(differences,size=(2000,len(differences)),replace=True).mean(axis=1)
        hits=int(per_user[f'hits_at_{k}'].sum())
        metrics.append({'k':k,'eligible_users':len(per_user),'eligible_target_pairs':eligible_pairs,'hits':hits,
            'macro_recall':float(per_user[f'recall_at_{k}'].mean()),'micro_recall':hits/eligible_pairs,
            'user_hit_rate':float(per_user[f'hits_at_{k}'].gt(0).mean()),
            'popularity_macro_recall':float(baseline[f'recall_at_{k}'].mean()),
            'paired_macro_delta':float(differences.mean()),'delta_ci95_low':float(np.quantile(draws,.025)),
            'delta_ci95_high':float(np.quantile(draws,.975)),
            'known_user_all_target_macro_recall':float((per_user[f'hits_at_{k}'].reindex(all_counts.index,fill_value=0)/all_counts).mean())})
    return per_user, pd.DataFrame(metrics)
