"""Pillar 1 of docs/02 section 6: the split-brain benchmark as one callable, so that any pair of models can be
scored on the same five numbers.

    from benchmark import score
    score(model, world_fn, n=1000) -> dict

The five numbers (each defined in docs/02 section 4.4):
    T2_committee / T2_shared   cross-midline same/different, read out by summing each side's log-odds (committee)
                               or from both states together (shared). Integration vs unity at the exit.
    T5_entry_ratio             prediction error when an object crosses the midline, over the ordinary error.
    T7_recovery                steps for the prediction error to come back within 20% of normal after one
                               hemisphere is zeroed for 4 steps, and the same with the link cut.
    T8_wake / T8_dream         fraction of crossing objects the receiver draws with the same identity, with
                               observations on and with observations cut after step 10.
    channel_receiving / channel_correlated
                               overlap of the silencing-derived and the correlation-derived direction with the true
                               channel (only available when the ground truth is computable, i.e. a linked model).

The model interface is the one of toy_split_brain.Pair: .L, .R (each with .enc, .gru, .dec), .msg(i, h), .d, and
hidden size H. Anything with that interface can be scored.

usage: python benchmark.py RESULTS_DIR [out.json]
"""
import sys, os, json, numpy as np, torch
here = os.path.dirname(os.path.abspath(__file__))
R_ = sys.argv[1] if len(sys.argv) > 1 else 'prototype/results'
OUTJ = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R_, 'benchmark.json')
sys.argv = [sys.argv[0], R_, '1']
exec(open(os.path.join(here, 'toy_split_brain.py')).read().split("if __name__")[0])
_g = open(os.path.join(here, 'ground_truth_channel.py')).read()
exec(_g[_g.index('def orth('):_g.index("if __name__ == '__main__'")], globals())
_d = open(os.path.join(here, 'dream_consistency.py')).read()
exec(_d[_d.index('def run('):_d.index("if __name__ == '__main__'")], globals())
_r = open(os.path.join(here, 'recovery.py')).read()
exec(_r[_r.index('T0, SIL ='):_r.index("if __name__ == '__main__'")].replace('def run(', 'def run_sil(', 1), globals())


def score(m, n=4000, seed=7, verbose=False):
    """n: trials for the probes. The T2 shared readout needs a few thousand (with n = 1000 it falls from 0.68 to
    0.53 on the same model, because only trials with exactly one object per half can be used)."""
    out = {}
    ev, (hL, hR, ids, P) = evaluate(m, n=n)
    pr = probes(hL, hR, ids, P)
    out['T2_committee'] = pr['T2']['committee']; out['T2_shared'] = pr['T2']['shared']
    out['T1_committee'] = pr['T1']['committee']; out['T1_shared'] = pr['T1']['shared']
    out['T5_entry_ratio'] = ev['entry_ratio']
    X, _, _ = world(n, seed=seed + 24)
    out['T8_wake'] = consistency(*run(m, X), 10)[0]
    out['T8_dream'] = consistency(*run(m, X, dream_from=10), 10)[0]
    Xr, _, _ = world(min(n, 800), seed=seed + 48)
    e0, _, _ = run_sil(m, Xr); es, _, _ = run_sil(m, Xr, silence=0); ec, _, _ = run_sil(m, Xr, silence=0, cut_from=T0)
    out['T7_recovery'] = recovery_step(es, e0); out['T7_recovery_link_cut'] = recovery_step(ec, e0)
    if getattr(m, 'd', 0):
        Xg, _, _ = world(300, seed=seed + 14); k = min(m.d, 4); gt = {}
        for src, nm in ((0, 'L_to_R'), (1, 'R_to_L')):
            Tm = orth(true_channel(m, Xg, src), k); est = estimates(m, Xg, src, k)
            gt[nm] = {kk: float(np.sum((Tm.T @ S) ** 2) / k) for kk, S in est.items()}
        out['channel_receiving'] = float(np.mean([gt[nm]['receiving'] for nm in gt]))
        out['channel_correlated'] = float(np.mean([gt[nm]['correlated'] for nm in gt]))
        out['channel_random'] = k / H
    if verbose:
        print(json.dumps(out, indent=1))
    return out


def report(rows):
    """one line per model, in the order of docs/02 section 4.4"""
    cols = ['T5_entry_ratio', 'T2_committee', 'T2_shared', 'T7_recovery', 'T7_recovery_link_cut',
            'T8_wake', 'T8_dream', 'channel_receiving', 'channel_correlated']
    w = max(len(k) for k in rows)
    print(' ' * w + ''.join(f'{c[:13]:>15s}' for c in cols))
    for k, v in rows.items():
        print(f'{k:{w}s}' + ''.join(f'{v.get(c, float("nan")):>15.3f}' if isinstance(v.get(c), float) else
                                    f'{str(v.get(c, "-")):>15s}' for c in cols))


if __name__ == '__main__':
    rows = {}
    for d in (0, 1, 2, 4, 16):
        f = os.path.join(R_, f'model_d{d}.pt')
        if not os.path.exists(f): continue
        m = Pair(d); m.load_state_dict(torch.load(f)); m.eval()
        rows[f'd{d}'] = score(m)
        print('scored', d, flush=True)
    report(rows)
    json.dump(rows, open(OUTJ, 'w'), indent=1)
    print('wrote', OUTJ)
