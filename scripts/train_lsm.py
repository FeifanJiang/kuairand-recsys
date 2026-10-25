"""Fit a bipartite inner-product latent space model for retrieval.

logit P(B_uv = 1) = z_u^T w_v + alpha_u + alpha_v, where B_uv = 1 if user u had any
training long view of video v. Zeros are case-control sampled uniformly per user from
non-edges; the logit offset -log r_u keeps alpha and z consistent for the full adjacency.
Variant 'degree' drops z and w; its ranking by alpha_v should recover popularity.
--item-features adds a feature part to each video: alpha_v = a_v + h_alpha(x_v) and
w_v = e_v + h_w(x_v), with h linear (summed field embeddings) or a small MLP. x_v uses static
metadata and training-period logs only: author, tags, upload type, music type, video type,
duration bucket, and video age at the training cutoff.
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
META = ROOT/'data/raw/KuaiRand-1K/data/video_features_basic_1k.csv'
FIELDS = ['upload_type','music_type','video_type','duration_bucket','age_bucket']

def item_feature_table(catalog, data, cutoff):
    """Integer codes per catalog video. Tags are multi-valued; index 0 pads."""
    meta = pd.read_csv(META, usecols=['video_id','author_id','upload_dt','upload_type','music_type','video_type','tag'])
    meta = meta.loc[meta.video_id.isin(catalog)].set_index('video_id').reindex(catalog)
    durations = []
    for chunk in pd.read_csv(data/'train.csv.gz', usecols=['video_id','duration_ms'], chunksize=1000000):
        durations.append(chunk)
    duration = pd.concat(durations).groupby('video_id').duration_ms.median().reindex(catalog)
    age = (cutoff-pd.to_datetime(meta.upload_dt, errors='coerce')).dt.days
    codes = {'author': pd.factorize(meta.author_id.fillna(-1))[0],
        'upload_type': pd.factorize(meta.upload_type.fillna('missing'))[0],
        'music_type': pd.factorize(meta.music_type.fillna(-1))[0],
        'video_type': pd.factorize(meta.video_type.fillna('missing'))[0],
        'duration_bucket': np.digitize(duration.fillna(-1).to_numpy(), [0, 1, 7000, 18001, 60000, 180000]),
        'age_bucket': np.where(age.isna(), 0, 1+np.digitize(age.fillna(0).to_numpy(), [2, 4, 8, 31, 91, 366]))}
    tags = meta.tag.fillna('none').astype(str).str.split(',')
    vocab = {t: n+1 for n, t in enumerate(sorted({t for row in tags for t in row}))}
    width = int(tags.str.len().max())
    tag_codes = np.zeros((len(catalog), width), np.int64)
    for r, row in enumerate(tags):
        tag_codes[r, :len(row)] = [vocab[t] for t in row]
    return {k: torch.from_numpy(np.asarray(v, np.int64)) for k, v in codes.items()}, torch.from_numpy(tag_codes)

class BipartiteLSM(nn.Module):
    def __init__(self, user_alpha, n_items, dim):
        super().__init__()
        self.dim = dim
        self.user_alpha = nn.Parameter(torch.from_numpy(user_alpha).float())
        self.item_alpha = nn.Embedding(n_items, 1, sparse=True)
        nn.init.zeros_(self.item_alpha.weight)
        if dim:
            self.z = nn.Embedding(len(user_alpha), dim)
            self.w = nn.Embedding(n_items, dim, sparse=True)
            nn.init.normal_(self.z.weight, std=.1)
            nn.init.normal_(self.w.weight, std=.1)
    def item_params(self, items):
        """Returns (alpha_v, w_v) used for scoring, the video's own (a_v, e_v), and any feature penalty."""
        alpha_v = self.item_alpha(items).squeeze(-1)
        wv = self.w(items) if self.dim else None
        return alpha_v, wv, alpha_v, wv, 0.
    def logits(self, users, items, user_weight, item_weight):
        """users: (B,), items: (B, m). Returns theta (B, m) and the weighted squared norm of touched rows."""
        alpha_v, wv, own_alpha, own_w, feature_penalty = self.item_params(items)
        theta = self.user_alpha[users][:,None] + alpha_v
        penalty = (item_weight[items]*own_alpha.pow(2)).sum() + feature_penalty
        if self.dim:
            zu = self.z(users)
            theta = theta + (zu[:,None,:]*wv).sum(-1)
            penalty = penalty + (user_weight[users]*zu.pow(2).sum(-1)).sum() + (item_weight[items]*own_w.pow(2).sum(-1)).sum()
        return theta, penalty
    def score_block(self, users, start, stop):
        """Retrieval score alpha_v + z_u^T w_v; alpha_u is constant within a user."""
        items = torch.arange(start, stop)
        alpha_v, wv, _, _, _ = self.item_params(items)
        score = alpha_v[None,:].expand(len(users), -1)
        if self.dim:
            score = score + self.z(users) @ wv.T
        return score

class FeatureLSM(BipartiteLSM):
    """alpha_v = a_v + h_alpha(x_v), w_v = e_v + h_w(x_v). Feature parts start at zero output."""
    def __init__(self, user_alpha, n_items, dim, codes, tags, mode, author_weight, field_dim=16, hidden=64):
        super().__init__(user_alpha, n_items, dim)
        self.mode, self.codes, self.tags, self.author_weight = mode, codes, tags, author_weight
        width = dim+1 if mode=='linear' else field_dim
        sizes = {f: int(codes[f].max())+1 for f in FIELDS}
        self.author = nn.Embedding(int(codes['author'].max())+1, width, sparse=True)
        self.tag = nn.Embedding(int(tags.max())+1, width, padding_idx=0)
        self.fields = nn.ModuleDict({f: nn.Embedding(n, width) for f, n in sizes.items()})
        if mode=='linear':
            for e in [self.author, self.tag, *self.fields.values()]: nn.init.zeros_(e.weight)
        else:
            for e in [self.author, self.tag, *self.fields.values()]: nn.init.normal_(e.weight, std=.1)
            with torch.no_grad(): self.tag.weight[0] = 0
            self.mlp = nn.Sequential(nn.Linear(width*(2+len(FIELDS)), hidden), nn.ReLU(), nn.Linear(hidden, dim+1))
            nn.init.zeros_(self.mlp[2].weight); nn.init.zeros_(self.mlp[2].bias)
    def item_params(self, items):
        own_alpha = self.item_alpha(items).squeeze(-1); own_w = self.w(items)
        author = self.codes['author'][items]; tag_codes = self.tags[items]
        a = self.author(author)
        t = self.tag(tag_codes).sum(-2)/tag_codes.gt(0).sum(-1, keepdim=True).clamp(min=1)
        parts = [a, t]+[self.fields[f](self.codes[f][items]) for f in FIELDS]
        h = sum(parts) if self.mode=='linear' else self.mlp(torch.cat(parts, -1))
        penalty = (self.author_weight[author]*a.pow(2).sum(-1)).sum()
        return own_alpha+h[...,0], own_w+h[...,1:], own_alpha, own_w, penalty

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['degree','full'],required=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=2048)
    parser.add_argument('--dim',type=int,default=32)
    parser.add_argument('--negatives',type=int,default=4)
    parser.add_argument('--l2',type=float,default=1e-4)
    # occurrence: l2 on every touched row, so prior precision grows with usage.
    # gaussian: each touch is divided by the row's expected touches per epoch, so one
    # epoch accumulates l2 * sum of squared rows in expectation: a fixed Gaussian prior.
    parser.add_argument('--penalty',choices=['occurrence','gaussian'],default='occurrence')
    parser.add_argument('--seed',type=int,default=42)
    # test: fit on train + validation (prepare_test_phase.py) and evaluate once on standard test.
    parser.add_argument('--phase',choices=['validation','test'],default='validation')
    parser.add_argument('--skip-model',action='store_true',help='do not save model.pt; runs are deterministic given the seed')
    parser.add_argument('--skip-latest',action='store_true',help='do not update latest_<variant>.txt (grid members)')
    # Diagnostic learning curves: evaluate validation recall after every epoch. Evaluation
    # draws no random numbers, so the training trajectory matches a run without it.
    parser.add_argument('--eval-every-epoch',action='store_true')
    parser.add_argument('--item-features',choices=['none','linear','mlp'],default='none')
    parser.add_argument('--item-lr',type=float,default=.02)
    parser.add_argument('--user-lr',type=float,default=.01)
    args = parser.parse_args()
    dim = args.dim if args.variant=='full' else 0
    seed = args.seed
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_{args.variant}'
    if args.penalty=='gaussian': run_id+=f'_k{dim}_l2-{args.l2:g}_gaussian'
    if args.item_features!='none': run_id+=f'_feat-{args.item_features}'
    if args.negatives!=4: run_id+=f'_m{args.negatives}'
    if args.epochs!=10: run_id+=f'_e{args.epochs}'
    if seed!=42: run_id+=f'_s{seed}'
    test = args.phase=='test'
    data = ROOT/'data/processed/test_phase' if test else DATA
    reports = ROOT/('reports/test_phase/lsm' if test else 'reports/lsm')
    popularity = ROOT/('artifacts/test_phase/popularity' if test else 'artifacts/popularity')
    out = reports/run_id
    art = ROOT/('artifacts/test_phase/lsm' if test else 'artifacts/lsm')/run_id
    out.mkdir(parents=True); art.mkdir(parents=True)
    def log(msg):
        print(msg,flush=True)
        with (out/'run.log').open('a') as f:
            f.write(msg+'\n')
    log(f'Run: {run_id}; device=cpu; threads=4')
    catalog = np.sort(pd.read_csv(data/'candidate_videos.csv').video_id.to_numpy(np.int64))
    users = np.sort(pd.read_csv(data/'training_users.csv').user_id.to_numpy(np.int64))
    positives=[]; max_train=0; training_rows=0
    for chunk in pd.read_csv(data/'train.csv.gz',usecols=['user_id','video_id','long_view','time_ms'],chunksize=250000):
        training_rows+=len(chunk); max_train=max(max_train,int(chunk.time_ms.max()))
        positives.append(chunk.loc[chunk.long_view.eq(1),['user_id','video_id']].to_numpy(np.int64))
    positive=np.concatenate(positives); del positives
    n_items=len(catalog)
    # B is binary: repeated long views of the same pair form one edge.
    pair_keys=np.unique(np.searchsorted(users,positive[:,0])*n_items+np.searchsorted(catalog,positive[:,1]))
    u=pair_keys//n_items; i=pair_keys%n_items; del positive
    degree_u=np.bincount(u,minlength=len(users)); degree_v=np.bincount(i,minlength=n_items)
    # Each edge draws m zeros per epoch, so user u's zeros are sampled at rate r_u.
    sample_rate=np.where(degree_u>0,args.negatives*degree_u/np.maximum(n_items-degree_u,1),1.)
    offset=torch.from_numpy(-np.log(sample_rate)).float()
    # Null-model MLE with alpha_v = 0: alpha_u = logit(degree_u / n_items).
    user_alpha=np.log(np.maximum(degree_u,.5)/(n_items-degree_u))
    if args.penalty=='gaussian':
        user_weight=torch.from_numpy(1/np.maximum(degree_u,1)).float()
        item_weight=torch.from_numpy(1/(degree_v+args.negatives*len(pair_keys)/n_items)).float()
    else:
        user_weight=torch.ones(len(users)); item_weight=torch.ones(n_items)
    config={'run_id':run_id,'variant':args.variant,
        'model':'logit P(B_uv=1) = alpha_u + alpha_v' + (' + z_u^T w_v' if dim else ''),
        'seed':seed,'device':'cpu','threads':4,'epochs':args.epochs,'batch_size':args.batch_size,
        'latent_dimension':dim,'negative_samples':args.negatives,'l2':args.l2,
        'item_learning_rate':args.item_lr,'user_learning_rate':args.user_lr,
        'edge_definition':'B_uv = 1 if any standard training long_view=1 event; repeats collapsed',
        'penalty':args.penalty,
        'loss':'Bernoulli log-likelihood on edges and case-control sampled zeros, offset -log r_u, plus '+(
            'l2 on touched rows per occurrence' if args.penalty=='occurrence' else
            'l2 on touched rows divided by expected touches per epoch (deg_u for users; deg_v + m*E/N for items), i.e. a fixed Gaussian prior on alpha_v, w_v, z_u in expectation; alpha_u unpenalized'),
        'negative_distribution':'uniform over the training catalog per edge; reject all training edges',
        'initialization':'alpha_u = logit(degree_u / catalog size), alpha_v = 0, z and w ~ N(0, 0.1^2)',
        'identifiability':'not constrained; retrieval scores alpha_v + z_u^T w_v are invariant to the remaining transformations',
        'training_rows':training_rows,'training_edges':len(u),'training_users':len(users),
        'catalog_size':n_items,'items_with_zero_degree':int((degree_v==0).sum()),
        'checkpoint_selection':'fixed final epoch, no validation-based selection',
        'retrieval':'exact blocked full-catalog scoring of alpha_v + z_u^T w_v; no ANN approximation',
        'seen_item_policy':'allowed','phase':args.phase,'test_accessed':test,'random_exposure_accessed':False,
        'limitations':['No content, covariate, or temporal history features','No cold-start support',
            'Zero-degree items are identified only through the l2 penalty',
            'Per-occurrence l2 is a frequency-weighted prior, not a fixed Gaussian prior' if args.penalty=='occurrence' else
            'Gaussian prior holds in expectation over sampled batches; alpha_v and w_v share one precision',
            'Single seed; validation results do not establish online gains']}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    if args.item_features=='none':
        model=BipartiteLSM(user_alpha,n_items,dim)
        sparse_params=[model.item_alpha.weight]+([model.w.weight] if dim else [])
        dense_params=[model.user_alpha]+([model.z.weight] if dim else [])
    else:
        assert dim, 'item features require the full variant'
        cutoff=pd.Timestamp(max_train,unit='ms',tz='UTC').tz_convert('Asia/Shanghai').tz_localize(None).normalize()
        codes,tags=item_feature_table(catalog,data,cutoff)
        # Author rows get the same fixed Gaussian prior: touches divided by expected author touches.
        expected=(degree_v+args.negatives*len(pair_keys)/n_items) if args.penalty=='gaussian' else np.ones(n_items)
        author_touches=np.bincount(codes['author'].numpy(),weights=expected)
        model=FeatureLSM(user_alpha,n_items,dim,codes,tags,args.item_features,torch.from_numpy(1/author_touches).float())
        sparse_params=[model.item_alpha.weight,model.w.weight,model.author.weight]
        dense_params=[model.user_alpha,model.z.weight]+[p for n,p in model.named_parameters()
            if n.startswith(('tag.','fields.','mlp.'))]
        config.update({'item_features':args.item_features,'feature_fields':['author','tags (mean of tag embeddings)']+FIELDS,
            'feature_cutoff_date':str(cutoff.date()),
            'feature_model':'summed field embeddings of width k+1' if args.item_features=='linear' else
                'field embeddings of width 16, concatenated, MLP 112-64-(k+1) with ReLU; last layer zero-initialized',
            'feature_prior':'author embeddings: fixed Gaussian via touches divided by expected author touches; other feature parameters unpenalized'})
        (out/'config.json').write_text(json.dumps(config,indent=2))
    sparse_opt=torch.optim.SparseAdam(sparse_params,lr=args.item_lr)
    dense_opt=torch.optim.Adam(dense_params,lr=args.user_lr)
    if args.eval_every_epoch:
        assert not test, 'learning curves are validation diagnostics'
        groups,all_counts,eligible_pairs,baseline=load_targets(users,catalog,max_train)
        eval_users=groups.index.to_numpy(np.int64); eval_idx=torch.from_numpy(np.searchsorted(users,eval_users))
        curve=[]
    m=args.negatives; losses=[]; started=time.perf_counter()
    for epoch in range(args.epochs):
        order=rng.permutation(len(u)); loss_sum=0.; examples=0
        for step,start in enumerate(range(0,len(order),args.batch_size)):
            idx=order[start:start+args.batch_size]; ub=u[idx]; ib=i[idx]
            neg=rng.integers(0,n_items,size=(len(idx),m),dtype=np.int64)
            for attempt in range(100):
                keys=ub[:,None]*n_items+neg
                loc=np.searchsorted(pair_keys,keys)
                bad=(loc<len(pair_keys)) & (pair_keys[np.minimum(loc,len(pair_keys)-1)]==keys)
                if not bad.any(): break
                neg[bad]=rng.integers(0,n_items,size=int(bad.sum()))
            else: raise RuntimeError('Negative rejection did not converge')
            ut=torch.from_numpy(ub)
            items=torch.from_numpy(np.concatenate([ib[:,None],neg],axis=1))
            theta,penalty=model.logits(ut,items,user_weight,item_weight)
            labels=torch.zeros_like(theta); labels[:,0]=1
            nll=F.binary_cross_entropy_with_logits(theta+offset[ut][:,None],labels,reduction='sum')/len(idx)
            loss=nll+args.l2*penalty/len(idx)
            assert torch.isfinite(loss)
            sparse_opt.zero_grad(set_to_none=True); dense_opt.zero_grad(set_to_none=True)
            loss.backward(); sparse_opt.step(); dense_opt.step()
            loss_sum+=nll.item()*len(idx); examples+=len(idx)
            if step%200==0:
                log(f'Epoch {epoch+1} step {step}: nll={nll.item():.6f}; elapsed={time.perf_counter()-started:.1f}s')
        losses.append({'epoch':epoch+1,'nll_per_edge':loss_sum/examples,'edges':examples,
                       'elapsed_seconds':time.perf_counter()-started})
        log(f'Epoch {epoch+1} mean nll per edge: {losses[-1]["nll_per_edge"]:.6f}')
        pd.DataFrame(losses).to_csv(out/'training_loss.csv',index=False)
        if args.eval_every_epoch:
            with torch.no_grad():
                ids,_=exact_top100(lambda a,b: model.score_block(eval_idx,a,b),len(eval_users),n_items,lambda msg: None)
            _,epoch_metrics=evaluate(eval_users,catalog[ids],groups,all_counts,eligible_pairs,baseline)
            slot=degree_v[ids.ravel()]
            curve.append({'epoch':epoch+1,'nll_per_edge':losses[-1]['nll_per_edge'],
                **{f'recall_at_{k}':float(epoch_metrics.set_index('k').loc[k,'macro_recall']) for k in [20,50,100]},
                'slot_share_degree_at_most_2':float((slot<=2).mean()),'distinct_recommended_videos':int(len(np.unique(ids)))})
            pd.DataFrame(curve).to_csv(out/'epoch_metrics.csv',index=False)
            log(f'Epoch {epoch+1} validation Recall@100: {curve[-1]["recall_at_100"]:.4%}')
    if not args.skip_model:
        torch.save({'state_dict':model.state_dict(),'config':config,'user_ids':torch.from_numpy(users),
                    'video_ids':torch.from_numpy(catalog)},art/'model.pt')
    log(f'Training finished; loading standard {"test" if test else "validation"} targets.')
    groups,all_counts,eligible_pairs,baseline=load_targets(users,catalog,max_train,data=data,
        split='test' if test else 'validation',baseline_path=ROOT/('reports/test_phase/popularity' if test else 'reports/popularity')/'per_user_metrics.csv')
    eval_users=groups.index.to_numpy(np.int64)
    eval_idx=torch.from_numpy(np.searchsorted(users,eval_users))
    model.eval(); retrieval_start=time.perf_counter()
    best_ids,best_scores=exact_top100(lambda a,b: model.score_block(eval_idx,a,b),len(eval_users),n_items,log)
    retrieval_seconds=time.perf_counter()-retrieval_start
    recommendations=catalog[best_ids]
    np.savez_compressed(art/'validation_top100.npz',user_ids=eval_users,video_ids=recommendations,scores=best_scores)
    per_user,metrics=evaluate(eval_users,recommendations,groups,all_counts,eligible_pairs,baseline)
    per_user.to_csv(out/'per_user_metrics.csv'); metrics.to_csv(out/'metrics.csv',index=False)
    # Diagnostics: does alpha_v track training degree, and how concentrated are recommendations?
    alpha_v=model.item_alpha.weight.detach().numpy().ravel()
    popularity_top=pd.read_csv(popularity/'top_100.csv').video_id.to_numpy()
    alpha_top=catalog[np.argsort(-alpha_v,kind='stable')[:100]]
    slot_degree=degree_v[best_ids.ravel()]
    diagnostics={'alpha_v_degree_spearman':float(pd.Series(alpha_v).rank().corr(pd.Series(degree_v).rank())),
        'alpha_v_top100_overlap_with_popularity_top100':int(len(set(alpha_top)&set(popularity_top))),
        'distinct_recommended_videos':int(len(np.unique(best_ids))),'recommendation_slots':int(best_ids.size),
        'recommended_slot_training_degree_median':float(np.median(slot_degree)),
        'recommended_slot_training_degree_mean':float(slot_degree.mean()),
        'recommended_slot_share_degree_at_most_2':float((slot_degree<=2).mean())}
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    checks={'same_users_and_target_counts_as_baseline':True,'strict_training_evaluation_time_order':True,
        'finite_training_losses':True,'unique_full_catalog_top100':True,'test_accessed':test,
        'bootstrap':'2000 paired user resamples; seed 2026; conditional on this cohort, not training-seed uncertainty',
        'exact_retrieval_seconds':retrieval_seconds}
    (out/'validation_checks.json').write_text(json.dumps(checks,indent=2))
    if not args.skip_latest: (reports/f'latest_{args.variant}.txt').write_text(run_id+'\n')
    log(json.dumps(diagnostics)); log(metrics.to_string(index=False)); log('Finished.')

if __name__=='__main__': main()
