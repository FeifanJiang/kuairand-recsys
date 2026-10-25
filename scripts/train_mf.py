"""Train BPR-MF or implicit ALS on training long-view edges and evaluate the frozen validation protocol.

Same edges (B_uv = 1 for any training long view), catalog, cohort, and retrieval as the LSM.
BPR (Rendle et al. 2009): score b_v + p_u^T q_v, loss -log sigmoid(s_pos - s_neg) on uniform
negatives that reject training edges; l2 per occurrence (as published) or a fixed Gaussian
prior (touches divided by expected touches, as in train_lsm.py).
iALS (Hu, Koren, Volinsky 2008): every cell enters the loss; edges have preference 1 and
confidence 1 + alpha, all other cells preference 0 and confidence 1; l2 lambda on all factors.
Solved exactly with the Gram-matrix identity, so no negative sampling.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from retrieval_eval import load_targets, exact_top100, evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'

def load_edges():
    catalog = np.sort(pd.read_csv(DATA/'candidate_videos.csv').video_id.to_numpy(np.int64))
    users = np.sort(pd.read_csv(DATA/'training_users.csv').user_id.to_numpy(np.int64))
    positives=[]; max_train=0
    for chunk in pd.read_csv(DATA/'train.csv.gz',usecols=['user_id','video_id','long_view','time_ms'],chunksize=250000):
        max_train=max(max_train,int(chunk.time_ms.max()))
        positives.append(chunk.loc[chunk.long_view.eq(1),['user_id','video_id']].to_numpy(np.int64))
    positive=np.concatenate(positives)
    pair_keys=np.unique(np.searchsorted(users,positive[:,0])*len(catalog)+np.searchsorted(catalog,positive[:,1]))
    return users, catalog, pair_keys, max_train

def train_bpr(args, users, catalog, pair_keys, rng, log):
    n_items=len(catalog); u=pair_keys//n_items; i=pair_keys%n_items; m=args.negatives
    p=nn.Embedding(len(users),args.dim); q=nn.Embedding(n_items,args.dim,sparse=True); b=nn.Embedding(n_items,1,sparse=True)
    nn.init.normal_(p.weight,std=.1); nn.init.normal_(q.weight,std=.1); nn.init.zeros_(b.weight)
    degree_u=np.bincount(u,minlength=len(users)); degree_v=np.bincount(i,minlength=n_items)
    if args.penalty=='gaussian':
        user_weight=torch.from_numpy(1/np.maximum(degree_u,1)).float()
        item_weight=torch.from_numpy(1/(degree_v+m*len(pair_keys)/n_items)).float()
    else:
        user_weight=torch.ones(len(users)); item_weight=torch.ones(n_items)
    sparse_opt=torch.optim.SparseAdam([q.weight,b.weight],lr=args.item_lr); dense_opt=torch.optim.Adam([p.weight],lr=args.user_lr)
    losses=[]; started=time.perf_counter()
    for epoch in range(args.epochs):
        order=rng.permutation(len(u)); loss_sum=0.
        for start in range(0,len(order),args.batch_size):
            idx=order[start:start+args.batch_size]; ub=u[idx]; ib=i[idx]
            neg=rng.integers(0,n_items,size=(len(idx),m),dtype=np.int64)
            for attempt in range(100):
                keys=ub[:,None]*n_items+neg; loc=np.searchsorted(pair_keys,keys)
                bad=(loc<len(pair_keys)) & (pair_keys[np.minimum(loc,len(pair_keys)-1)]==keys)
                if not bad.any(): break
                neg[bad]=rng.integers(0,n_items,size=int(bad.sum()))
            else: raise RuntimeError('Negative rejection did not converge')
            ut=torch.from_numpy(ub); items=torch.from_numpy(np.concatenate([ib[:,None],neg],axis=1))
            pu=p(ut); qv=q(items); bv=b(items).squeeze(-1)
            score=bv+(pu[:,None,:]*qv).sum(-1)
            nll=F.softplus(-(score[:,:1]-score[:,1:])).sum()/len(idx)
            penalty=(user_weight[ut]*pu.pow(2).sum(-1)).sum()+(item_weight[items]*(qv.pow(2).sum(-1)+bv.pow(2))).sum()
            loss=nll+args.l2*penalty/len(idx)
            assert torch.isfinite(loss)
            sparse_opt.zero_grad(set_to_none=True); dense_opt.zero_grad(set_to_none=True)
            loss.backward(); sparse_opt.step(); dense_opt.step()
            loss_sum+=nll.item()*len(idx)
        losses.append({'epoch':epoch+1,'loss_per_edge':loss_sum/len(u),'elapsed_seconds':time.perf_counter()-started})
        log(f'Epoch {epoch+1} BPR loss per edge: {losses[-1]["loss_per_edge"]:.6f}')
    def score_block(eval_idx,start,stop):
        items=torch.arange(start,stop)
        return b(items).squeeze(-1)[None,:]+p(eval_idx)@q(items).T
    return score_block, losses, b.weight.detach().numpy().ravel()

def train_ials(args, users, catalog, pair_keys, log):
    n_items=len(catalog); k=args.dim; u=torch.from_numpy(pair_keys//n_items); i=torch.from_numpy(pair_keys%n_items)
    X=torch.randn(len(users),k)*.01; Y=torch.zeros(n_items,k)
    active=torch.unique(i); Y[active]=torch.randn(len(active),k)*.01
    eye=torch.eye(k)*args.l2; a=args.alpha
    by_item=torch.argsort(i,stable=True); i_sorted=i[by_item]; u_by_item=u[by_item]
    bounds=torch.searchsorted(i_sorted,active); bounds=torch.cat([bounds,torch.tensor([len(i_sorted)])])
    def solve_rows(F_other, rows, cols, n_rows, row_ids):
        """Least squares for rows that have edges: (G + a*sum f f^T + lambda I) x = (1+a) sum f."""
        G=F_other.T@F_other
        A=G.expand(len(row_ids),k,k).clone(); rhs=torch.zeros(len(row_ids),k)
        local=torch.searchsorted(row_ids,rows)
        for s in range(0,len(rows),20000):
            f=F_other[cols[s:s+20000]]; r=local[s:s+20000]
            A.index_add_(0,r,a*f[:,:,None]*f[:,None,:]); rhs.index_add_(0,r,(1+a)*f)
        return torch.linalg.solve(A+eye,rhs)
    losses=[]; started=time.perf_counter()
    for it in range(args.iterations):
        X=solve_rows(Y,u,i,len(users),torch.arange(len(users)))
        for c in range(0,len(active),20000):
            ids=active[c:c+20000]; lo,hi=int(bounds[c]),int(bounds[min(c+20000,len(active))])
            Y[ids]=solve_rows(X,i_sorted[lo:hi],u_by_item[lo:hi],len(ids),ids)
        # Full-matrix objective via Gram identity: sum_all (x^T y)^2 + edge corrections + penalty.
        pred=(X[u]*Y[i]).sum(-1)
        total=float(((X.T@X)*(Y.T@Y)).sum()-pred.pow(2).sum()+((1+a)*(1-pred).pow(2)).sum()
                    +args.l2*(X.pow(2).sum()+Y.pow(2).sum()))
        losses.append({'iteration':it+1,'objective':total,'elapsed_seconds':time.perf_counter()-started})
        log(f'Iteration {it+1} iALS objective: {total:.2f}')
    def score_block(eval_idx,start,stop):
        return X[eval_idx]@Y[start:stop].T
    return score_block, losses, None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['bpr','ials'],required=True)
    parser.add_argument('--dim',type=int,default=32)
    parser.add_argument('--l2',type=float,default=1.)
    parser.add_argument('--penalty',choices=['occurrence','gaussian'],default='occurrence',help='BPR only')
    parser.add_argument('--alpha',type=float,default=10.,help='iALS confidence for edges')
    parser.add_argument('--epochs',type=int,default=10); parser.add_argument('--iterations',type=int,default=15)
    parser.add_argument('--batch-size',type=int,default=2048); parser.add_argument('--negatives',type=int,default=4)
    parser.add_argument('--item-lr',type=float,default=.02); parser.add_argument('--user-lr',type=float,default=.01)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(4); rng=np.random.default_rng(args.seed)
    tag=f'_k{args.dim}_l2-{args.l2:g}'+(f'_{args.penalty}' if args.model=='bpr' else f'_a{args.alpha:g}')+(f'_s{args.seed}' if args.seed!=42 else '')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_{args.model}'+tag
    out=ROOT/'reports/mf'/run_id; art=ROOT/'artifacts/mf'/run_id; out.mkdir(parents=True); art.mkdir(parents=True)
    def log(msg):
        print(msg,flush=True)
        with (out/'run.log').open('a') as f: f.write(msg+'\n')
    log(f'Run: {run_id}; device=cpu; threads=4')
    users,catalog,pair_keys,max_train=load_edges()
    config={'run_id':run_id,'model':args.model,'seed':args.seed,'latent_dimension':args.dim,'l2':args.l2,
        'edge_definition':'B_uv = 1 if any standard training long_view=1 event; repeats collapsed','training_edges':len(pair_keys),
        'training_users':len(users),'catalog_size':len(catalog),'checkpoint_selection':'fixed final epoch or iteration',
        'retrieval':'exact blocked full-catalog scoring; no ANN','seen_item_policy':'allowed','test_accessed':False}
    if args.model=='bpr':
        config.update({'score':'b_v + p_u^T q_v','loss':'-log sigmoid(s_pos - s_neg), mean over edges of sum over negatives',
            'penalty':args.penalty,'negative_samples':args.negatives,'epochs':args.epochs,'batch_size':args.batch_size,
            'item_learning_rate':args.item_lr,'user_learning_rate':args.user_lr})
    else:
        config.update({'score':'x_u^T y_v','loss':'sum over all cells c_uv (p_uv - x_u^T y_v)^2 + l2 (|X|^2 + |Y|^2)',
            'confidence':f'1 + alpha on edges, 1 elsewhere; alpha={args.alpha:g}','iterations':args.iterations,
            'zero_degree_items':'closed-form solution is 0 and is not recomputed'})
    (out/'config.json').write_text(json.dumps(config,indent=2))
    if args.model=='bpr': score_block,losses,item_bias=train_bpr(args,users,catalog,pair_keys,rng,log)
    else: score_block,losses,item_bias=train_ials(args,users,catalog,pair_keys,log)
    pd.DataFrame(losses).to_csv(out/'training_loss.csv',index=False)
    groups,all_counts,eligible_pairs,baseline=load_targets(users,catalog,max_train)
    eval_users=groups.index.to_numpy(np.int64); eval_idx=torch.from_numpy(np.searchsorted(users,eval_users))
    started=time.perf_counter()
    with torch.no_grad():
        best_ids,best_scores=exact_top100(lambda a,b: score_block(eval_idx,a,b),len(eval_users),len(catalog),log)
    recommendations=catalog[best_ids]
    np.savez_compressed(art/'validation_top100.npz',user_ids=eval_users,video_ids=recommendations,scores=best_scores)
    per_user,metrics=evaluate(eval_users,recommendations,groups,all_counts,eligible_pairs,baseline)
    per_user.to_csv(out/'per_user_metrics.csv'); metrics.to_csv(out/'metrics.csv',index=False)
    degree_v=np.bincount(pair_keys%len(catalog),minlength=len(catalog)); slot_degree=degree_v[best_ids.ravel()]
    diagnostics={'distinct_recommended_videos':int(len(np.unique(best_ids))),'recommendation_slots':int(best_ids.size),
        'recommended_slot_training_degree_median':float(np.median(slot_degree)),
        'recommended_slot_share_degree_at_most_2':float((slot_degree<=2).mean()),'exact_retrieval_seconds':time.perf_counter()-started}
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    log(json.dumps(diagnostics)); log(metrics.to_string(index=False)); log('Finished.')

if __name__=='__main__': main()
