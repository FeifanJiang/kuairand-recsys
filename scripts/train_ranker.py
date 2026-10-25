"""Fit an exposure-level ranking model for long_view with an edge-covariate latent space model.

logit P(long_view=1 | exposure) = b0 + alpha_u + alpha_v + z_u^T w_v + X_uv^T beta,
fit on standard training exposures, whose zeros are observed non-long views. Covariates
are known before exposure. alpha_u, alpha_v, z_u, w_v get a fixed Gaussian prior: each
touch is divided by the row's training exposure count, so one epoch accumulates
l2 * sum of squared rows. Unknown users and videos fall back to the prior mean 0.
Evaluation reads standard and random-exposure validation only.
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
META = ROOT/'data/raw/KuaiRand-1K/data/video_features_basic_1k.csv'
NUMERIC = ['log_duration','short_video','zero_duration','log_video_age','missing_video_age','is_ad']
VARIANTS = {'degree':(True,False,False),'covariates':(False,True,False),
            'degree_covariates':(True,True,False),'full':(True,True,True)}

def features(frame, meta):
    """Pre-exposure covariates; long_view needs a full view when duration <= 18 s."""
    frame = frame.merge(meta, on='video_id', how='left')
    local = pd.to_datetime(frame.time_ms, unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
    age = (local.dt.tz_localize(None).dt.normalize()-frame.upload).dt.days
    out = pd.DataFrame({'log_duration':np.log1p(frame.duration_ms/1000),
        'short_video':frame.duration_ms.le(18000).astype(float),'zero_duration':frame.duration_ms.eq(0).astype(float),
        'log_video_age':np.log1p(age.clip(lower=0).fillna(0)),'missing_video_age':age.isna().astype(float),
        'is_ad':frame.video_type.eq('AD').astype(float)})
    return out, frame.tab.to_numpy(np.int64), local.dt.hour.to_numpy(np.int64)

def load(name, meta):
    frame = pd.read_csv(DATA/f'{name}.csv.gz', usecols=['user_id','video_id','time_ms','tab','duration_ms','long_view'])
    numeric, tab, hour = features(frame, meta)
    return frame, numeric, tab, hour

class Ranker(nn.Module):
    def __init__(self, n_users, n_items, dim, use_degree, use_covariates):
        super().__init__()
        self.dim, self.use_degree, self.use_covariates = dim, use_degree, use_covariates
        self.b0 = nn.Parameter(torch.zeros(1))
        self.user_alpha = nn.Embedding(n_users+1, 1)
        self.item_alpha = nn.Embedding(n_items+1, 1, sparse=True)
        self.beta = nn.Linear(len(NUMERIC), 1, bias=False)
        self.tab = nn.Embedding(15, 1); self.hour = nn.Embedding(24, 1)
        self.z = nn.Embedding(n_users+1, max(dim,1)); self.w = nn.Embedding(n_items+1, max(dim,1), sparse=True)
        for p in [self.user_alpha.weight,self.item_alpha.weight,self.beta.weight,self.tab.weight,self.hour.weight]:
            nn.init.zeros_(p)
        nn.init.normal_(self.z.weight, std=.1); nn.init.normal_(self.w.weight, std=.1)
        # Last row is the unknown user or video and stays at the prior mean 0.
        with torch.no_grad():
            self.z.weight[-1] = 0; self.w.weight[-1] = 0
    def forward(self, u, v, x, tab, hour, user_weight=None, item_weight=None):
        au = self.user_alpha(u).squeeze(-1)
        logit = self.b0 + au
        penalty = (user_weight[u]*au.pow(2)).sum() if user_weight is not None else 0.
        if self.use_degree:
            av = self.item_alpha(v).squeeze(-1); logit = logit + av
            if item_weight is not None: penalty = penalty + (item_weight[v]*av.pow(2)).sum()
        if self.use_covariates:
            logit = logit + self.beta(x).squeeze(-1) + self.tab(tab).squeeze(-1) + self.hour(hour).squeeze(-1)
        if self.dim:
            zu = self.z(u); wv = self.w(v); logit = logit + (zu*wv).sum(-1)
            if user_weight is not None:
                penalty = penalty + (user_weight[u]*zu.pow(2).sum(-1)).sum() + (item_weight[v]*wv.pow(2).sum(-1)).sum()
        return logit, penalty

def auc(y, s):
    r = pd.Series(s).rank().to_numpy(); n1 = y.sum(); n0 = len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def per_user_auc(users, y, s):
    frame = pd.DataFrame({'user_id':users,'y':y,'s':s})
    frame['r'] = frame.groupby('user_id').s.rank()
    g = frame.assign(rp=frame.r*frame.y).groupby('user_id').agg(n=('y','size'),n1=('y','sum'),rank_pos=('rp','sum'))
    g = g.loc[g.n1.gt(0) & g.n1.lt(g.n)]
    return ((g.rank_pos-g.n1*(g.n1+1)/2)/(g.n1*(g.n-g.n1))).rename('auc')

def metrics(y, logit, users, in_catalog):
    p = 1/(1+np.exp(-logit))
    rows = []
    for name, mask in [('all',np.ones(len(y),bool)),('in_train_catalog',in_catalog==1),('outside_train_catalog',in_catalog==0)]:
        if mask.sum()==0 or y[mask].min()==y[mask].max(): continue
        yy, pp = y[mask], np.clip(p[mask],1e-7,1-1e-7)
        pua = per_user_auc(users[mask], yy, logit[mask])
        rows.append({'subset':name,'exposures':int(mask.sum()),'long_view_rate':float(yy.mean()),
            'mean_prediction':float(pp.mean()),'log_loss':float(-(yy*np.log(pp)+(1-yy)*np.log(1-pp)).mean()),
            'auc':auc(yy, logit[mask]),'gauc_users':len(pua),'gauc':float(pua.mean())})
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',choices=list(VARIANTS),required=True)
    parser.add_argument('--dim',type=int,default=16)
    parser.add_argument('--l2',type=float,default=10.)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=4096)
    parser.add_argument('--item-lr',type=float,default=.02)
    parser.add_argument('--dense-lr',type=float,default=.01)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    use_degree, use_covariates, use_latent = VARIANTS[args.variant]
    dim = args.dim if use_latent else 0
    torch.manual_seed(args.seed); torch.set_num_threads(4); rng = np.random.default_rng(args.seed)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_{args.variant}'+(f'_s{args.seed}' if args.seed!=42 else '')
    out = ROOT/'reports/ranking'/run_id; art = ROOT/'artifacts/ranking'/run_id
    out.mkdir(parents=True); art.mkdir(parents=True)
    def log(msg):
        print(msg,flush=True)
        with (out/'run.log').open('a') as f: f.write(msg+'\n')
    log(f'Run: {run_id}; device=cpu; threads=4')
    meta = pd.read_csv(META, usecols=['video_id','video_type','upload_dt'])
    meta['upload'] = pd.to_datetime(meta.upload_dt, errors='coerce'); meta = meta.drop(columns='upload_dt')
    train, x_train, tab_train, hour_train = load('train', meta)
    users = np.sort(train.user_id.unique()); catalog = np.sort(pd.read_csv(DATA/'candidate_videos.csv').video_id.to_numpy(np.int64))
    def index(ids, known):
        loc = np.searchsorted(known, ids).clip(max=len(known)-1)
        return np.where(known[loc]==ids, loc, len(known))
    mean, sd = x_train.mean(), x_train.std().replace(0,1)
    def tensors(frame, x, tab, hour):
        return (torch.from_numpy(index(frame.user_id.to_numpy(), users)), torch.from_numpy(index(frame.video_id.to_numpy(), catalog)),
                torch.from_numpy(((x-mean)/sd).to_numpy(np.float32)), torch.from_numpy(tab), torch.from_numpy(hour))
    u, v, x, tab, hour = tensors(train, x_train, tab_train, hour_train)
    y = torch.from_numpy(train.long_view.to_numpy(np.float32)); max_train = int(train.time_ms.max())
    user_count = np.bincount(u.numpy(), minlength=len(users)+1); item_count = np.bincount(v.numpy(), minlength=len(catalog)+1)
    user_weight = torch.from_numpy(1/np.maximum(user_count,1)).float(); item_weight = torch.from_numpy(1/np.maximum(item_count,1)).float()
    config = {'run_id':run_id,'variant':args.variant,
        'model':'logit P(long_view=1 | exposure) = b0 + alpha_u'+(' + alpha_v' if use_degree else '')+
            (' + z_u^T w_v' if dim else '')+(' + X_uv^T beta' if use_covariates else ''),
        'seed':args.seed,'device':'cpu','threads':4,'epochs':args.epochs,'batch_size':args.batch_size,
        'latent_dimension':dim,'l2':args.l2,'item_learning_rate':args.item_lr,'dense_learning_rate':args.dense_lr,
        'covariates':NUMERIC+['tab (15 levels)','hour of day in Asia/Shanghai (24 levels)'] if use_covariates else [],
        'covariate_standardization':'training mean and standard deviation',
        'prior':'fixed Gaussian on alpha_u, alpha_v, z_u, w_v via per-touch l2 divided by training exposure count; b0, beta, tab, hour unpenalized',
        'hyperparameters':'k and l2 carried over from retrieval (k=16, l2=10); not tuned for ranking',
        'unknown_ids':'validation users and videos outside training map to a zero row (prior mean)',
        'training_exposures':len(y),'training_long_view_rate':float(y.mean()),'training_users':len(users),'catalog_size':len(catalog),
        'checkpoint_selection':'fixed final epoch, no validation-based selection',
        'test_accessed':False,
        'limitations':['No user history features beyond IDs','Validation history frozen at training cutoff',
            'Single seed','Hyperparameters not tuned for ranking','beta reported as point estimates without inference']}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    model = Ranker(len(users), len(catalog), dim, use_degree, use_covariates)
    with torch.no_grad(): rate = float(y.mean()); model.b0.fill_(float(np.log(rate/(1-rate))))
    sparse_params = [model.item_alpha.weight, model.w.weight]
    dense_params = [p for n,p in model.named_parameters() if n not in ('item_alpha.weight','w.weight')]
    sparse_opt = torch.optim.SparseAdam(sparse_params, lr=args.item_lr); dense_opt = torch.optim.Adam(dense_params, lr=args.dense_lr)
    losses = []; started = time.perf_counter()
    for epoch in range(args.epochs):
        order = torch.from_numpy(rng.permutation(len(y))); loss_sum = 0.
        for step, start in enumerate(range(0, len(order), args.batch_size)):
            idx = order[start:start+args.batch_size]
            logit, penalty = model(u[idx], v[idx], x[idx], tab[idx], hour[idx], user_weight, item_weight)
            nll = F.binary_cross_entropy_with_logits(logit, y[idx], reduction='sum')/len(idx)
            loss = nll+args.l2*penalty/len(idx)
            assert torch.isfinite(loss)
            sparse_opt.zero_grad(set_to_none=True); dense_opt.zero_grad(set_to_none=True)
            loss.backward(); sparse_opt.step(); dense_opt.step()
            with torch.no_grad():
                model.z.weight[-1] = 0; model.w.weight[-1] = 0; model.user_alpha.weight[-1] = 0; model.item_alpha.weight[-1] = 0
            loss_sum += nll.item()*len(idx)
            if step%400==0: log(f'Epoch {epoch+1} step {step}: nll={nll.item():.6f}; elapsed={time.perf_counter()-started:.1f}s')
        losses.append({'epoch':epoch+1,'train_log_loss':loss_sum/len(y),'elapsed_seconds':time.perf_counter()-started})
        log(f'Epoch {epoch+1} train log loss: {losses[-1]["train_log_loss"]:.6f}')
        pd.DataFrame(losses).to_csv(out/'training_loss.csv', index=False)
    torch.save({'state_dict':model.state_dict(),'config':config,'user_ids':torch.from_numpy(users),'video_ids':torch.from_numpy(catalog),
                'covariate_mean':mean.to_dict(),'covariate_sd':sd.to_dict()}, art/'model.pt')
    model.eval(); results = []; predictions = {}
    for name in ['validation','random_validation']:
        frame, xe, tabe, houre = load(name, meta)
        assert int(frame.time_ms.min())>max_train
        tensors_e = tensors(frame, xe, tabe, houre)
        with torch.no_grad():
            logit = torch.cat([model(*(t[i:i+262144] for t in tensors_e))[0] for i in range(0, len(frame), 262144)]).numpy()
        ye = frame.long_view.to_numpy(); in_cat = np.isin(frame.video_id.to_numpy(), catalog).astype(int)
        results.append(metrics(ye, logit, frame.user_id.to_numpy(), in_cat).assign(split=name))
        per_user_auc(frame.user_id.to_numpy(), ye, logit).to_csv(out/f'per_user_auc_{name}.csv')
        predictions[name] = logit.astype(np.float32)
    np.savez_compressed(art/'validation_logits.npz', **predictions)
    results = pd.concat(results, ignore_index=True)[['split','subset','exposures','long_view_rate','mean_prediction','log_loss','auc','gauc_users','gauc']]
    results.to_csv(out/'metrics.csv', index=False)
    if use_covariates:
        names = NUMERIC+[f'tab_{i}' for i in range(15)]+[f'hour_{i}' for i in range(24)]
        values = np.concatenate([model.beta.weight.detach().numpy().ravel(), model.tab.weight.detach().numpy().ravel(),
                                 model.hour.weight.detach().numpy().ravel()])
        pd.DataFrame({'covariate':names,'coefficient':values}).to_csv(out/'coefficients.csv', index=False)
    (out/'validation_checks.json').write_text(json.dumps({'strict_time_order':True,'finite_training_losses':True,
        'test_accessed':False,'random_exposure_used_for_evaluation_only':True},indent=2))
    (ROOT/f'reports/ranking/latest_{args.variant}.txt').write_text(run_id+'\n')
    log(results.to_string(index=False)); log('Finished.')

if __name__=='__main__': main()
