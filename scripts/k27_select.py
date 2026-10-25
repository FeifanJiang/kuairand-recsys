"""Selection from seed-0 tune-group AUCs only; never reads report scores.

  python k27_select.py initial    LR / Tree / iALS on the 18-setting grid; writes the iALS edge-extension jobs (protocol_confirmed.json)
  python k27_select.py final      re-select iALS over grid + extension, choose the main L2 method, write the seed jobs
  python k27_select.py deep-grid  job list for the deep-ranker grid, baselines from the smoke run in deep_checks.json (protocol_deep.json)
  python k27_select.py deep       select Deep-L2 / Deep-L2+iALS / LR controls, fix the best new method, write seed and control jobs
Ties within 0.0005 go to the simpler setting by each protocol's order.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; RUN = ROOT/'reports/expansion/run_dev'; TIE = 5e-4
REFERENCE = (0.5, 30.0, 100.0)

def results(prefix):
    return [json.loads(p.read_text())['result'] for p in sorted(RUN.glob(f'{prefix}*_s0.json'))] if prefix != 'lr' else \
           [r for lvl in ['L1', 'L2'] for r in json.loads((RUN/f'lr_{lvl}.json').read_text())['result']]

def pick(rows, simpler):
    best = max(r['tune_auc'] for r in rows); near = [r for r in rows if r['tune_auc'] >= best-TIE]
    return sorted(near, key=simpler)[0]

def lr(level): return pick([r for r in results('lr') if r['name'].startswith(f'LR-{level}_')], lambda r: -r['l2'])
def tree(level): return pick(results(f'tree_{level}_'), lambda r: (r['config']['max_leaf_nodes'], -r['config']['min_samples_leaf'], -r['config']['l2_regularization']))
def ials(rows): return pick(rows, lambda r: (-r['l2'], r['alpha'], r['tau'] != 0.5))

def job(tau, alpha, l2, seed, expected=900):
    return [f'ials_tau{tau:g}_alpha{alpha:g}_l2{l2:g}_s{seed}', ['--tau', str(tau), '--alpha', str(alpha), '--l2', str(l2), '--seed', str(seed)], 'ials', expected]

def initial():
    grid = results('ials_'); assert len(grid) == 18, len(grid)
    sel = {'LR-L1': lr('L1'), 'LR-L2': lr('L2'), 'Tree-L1': tree('L1'), 'Tree-L2': tree('L2'), 'iALS_grid': ials(grid)}
    s = sel['iALS_grid']; jobs = []
    new_alpha = {100: 300, 10: 3}.get(s['alpha']); new_l2 = {300: 1000, 30: 10}.get(s['l2'])
    if new_alpha: jobs += [job(s['tau'], new_alpha, l, 0) for l in [30, 100, 300]]
    if new_l2: jobs += [job(s['tau'], a, new_l2, 0) for a in [10, 30, 100]]
    sel['iALS_extension_jobs'] = [j[0] for j in jobs]
    (RUN/'selection_initial.json').write_text(json.dumps(sel, indent=2)); (RUN/'jobs_ials_extension.json').write_text(json.dumps(jobs, indent=2))
    print(json.dumps(sel, indent=2))

def final():
    first = json.loads((RUN/'selection_initial.json').read_text())
    pool = results('ials_')
    assert len(pool) == 18+len(first['iALS_extension_jobs']), 'extension jobs missing'
    sel = {k: first[k] for k in ['LR-L1', 'LR-L2', 'Tree-L1', 'Tree-L2']}; sel['iALS_tuned'] = ials(pool)
    sel['iALS_reference'] = next(r for r in pool if (r['tau'], r['alpha'], r['l2']) == REFERENCE)
    a, b = sel['LR-L2']['tune_auc'], sel['Tree-L2']['tune_auc']
    sel['main_L2_method'] = 'Tree-L2' if b > a+TIE else 'LR-L2'
    sel['main_L2_rule'] = 'seed-0 tune AUC; Tree-L2 only if it exceeds LR-L2 by more than 0.0005'
    jobs = []
    for lvl in ['L1', 'L2']:
        c = sel[f'Tree-{lvl}']; cfg = int(c['name'].split('_c')[1].split('_')[0])
        jobs += [[f'tree_{lvl}_c{cfg}_s{s}', ['--level', lvl, '--config', str(cfg), '--seed', str(s)], 'tree', 1200] for s in [1, 2]]
    settings = {(sel['iALS_tuned']['tau'], sel['iALS_tuned']['alpha'], sel['iALS_tuned']['l2']), REFERENCE}
    jobs += [job(*st, s) for st in sorted(settings) for s in [1, 2, 3, 4]]
    sel['seed_jobs'] = [j[0] for j in jobs]
    (RUN/'selection_final.json').write_text(json.dumps(sel, indent=2)); (RUN/'jobs_seeds.json').write_text(json.dumps(jobs, indent=2))
    print(json.dumps(sel, indent=2))


# ---- deep-ranker stage (protocol_deep.json) ----
def pick_deep(rows, simpler):
    best = max(r['tune_auc'] for r in rows); near = [r for r in rows if r['tune_auc'] >= best-TIE]
    return sorted(near, key=simpler)[0]

def wall(tag):
    w = None
    for line in open(RUN/'resource_log.jsonl'):
        r = json.loads(line)
        if r['job'] == tag and r['exit'] == 0: w = r['wall_seconds']
    assert w is not None, f'no finished run for {tag}'; return w

def deep():
    sel = {}
    for m in ['deep', 'deep_ials']:
        rows = [json.loads((RUN/f'{m}_c{c}_s0.json').read_text())['result'] for c in range(8)]
        sel[m] = pick_deep(rows, lambda r: (-r['config']['emb_l2'], -r['config']['dropout'], r['config']['lr']))
        sel[f'{m}_grid'] = [{'config_index': r['config_index'], **r['config'], 'tune_auc': r['tune_auc'], 'best_epoch': r['best_epoch'], 'epochs_run': r['epochs_run']} for r in rows]
    sel['LR-L2all'] = pick_deep(json.loads((RUN/'lr_L2all.json').read_text())['result'], lambda r: -r['l2'])
    sel['LR-L2+iALS_late'] = pick_deep(json.loads((RUN/'lr_fusion.json').read_text())['result'], lambda r: -r['l2'])
    hyb = json.loads((RUN/'hybrid.json').read_text())
    cands = {'hybrid': hyb['tune_curve_seed0'][str(hyb['selected_w'])], 'LR-L2+iALS': sel['LR-L2+iALS_late']['tune_auc'],
             'Deep-L2': sel['deep']['tune_auc'], 'Deep-L2+iALS': sel['deep_ials']['tune_auc']}
    best = max(cands.values()); order = ['hybrid', 'LR-L2+iALS', 'Deep-L2', 'Deep-L2+iALS']                     # simpler first
    sel['best_new_method'] = sorted([k for k, v in cands.items() if v >= best-TIE], key=order.index)[0]
    sel['best_new_method_candidates_tune_auc_seed0'] = cands; sel['tie_rule'] = 'within 0.0005 -> simpler'
    cd, ci = sel['deep']['config_index'], sel['deep_ials']['config_index']
    jobs = [[f'deep_c{cd}_s{s}', ['--model', 'deep', '--config', str(cd), '--seed', str(s)], 'deep', 0] for s in [1, 2]]
    jobs += [[f'deep_ials_c{ci}_s{s}', ['--model', 'deep_ials', '--config', str(ci), '--seed', str(s)], 'deep', 0] for s in [1, 2]]
    jobs += [[f'deep_late_c{ci}_s{s}', ['--model', 'deep_late', '--config', str(ci), '--seed', str(s)], 'deep', round(wall(f'deep_ials_c{ci}_s0'))] for s in [0, 1, 2]]
    sel['seed_and_control_jobs'] = [j[0] for j in jobs]
    (RUN/'selection_deep.json').write_text(json.dumps(sel, indent=2)); (RUN/'jobs_deep_seeds.json').write_text(json.dumps(jobs, indent=2))
    print(json.dumps({k: v for k, v in sel.items() if not k.endswith('_grid')}, indent=2))


def deep_grid():
    import k27_deep as kd
    chk = json.loads((ROOT/'reports/expansion/deep_checks.json').read_text()); meta = json.loads((kd.CACHE/'meta.json').read_text())
    per_row = chk['5_smoke_epoch_seconds_200k_rows']/(200_000*(1-kd.HOLDOUT))
    base = {m: round(per_row*rows*(1-kd.HOLDOUT)*kd.MAX_EPOCHS+120) for m, rows in [('deep', meta['rows']), ('deep_ials', meta['late_rows'])]}
    jobs = [['lr_L2all', ['--level', 'L2', '--feat', 'all'], 'lr', 1200], ['lr_fusion', [], 'lr_fusion', 1200]]
    for m in ['deep', 'deep_ials']:
        jobs += [[f'{m}_c{c}_s0', ['--model', m, '--config', str(c), '--seed', '0'], 'deep', base[m]] for c in range(8)]
    (RUN/'jobs_deep_grid.json').write_text(json.dumps(jobs, indent=2)); print(json.dumps({'baselines_seconds': base, 'jobs': [j[0] for j in jobs]}, indent=2))


if __name__ == '__main__':
    {'initial': initial, 'final': final, 'deep': deep, 'deep-grid': deep_grid}[sys.argv[1]]()
