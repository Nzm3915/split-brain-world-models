"""T7 (recovery) on the toy models, the artificial counterpart of 01's silencing-and-recovery result.

One hemisphere's state is set to zero for S steps (the analogue of photoinhibition). Afterwards, how long does its
prediction take to return to normal, and does the recovery come through the link (the partner writes it back) or
from its own dynamics? Three runs on the same sequences:
  normal      nothing silenced
  silenced    hemisphere L zeroed for steps t0..t0+S
  silenced+cut  the same, but the link is cut from t0 on, so nothing can be written back
Recovery time = the first step after the silencing at which L's prediction error is back within 20% of normal.
The gap between 'silenced' and 'silenced+cut' is how much of the recovery the link contributes.
In 01, silencing one ALM hemisphere cost ~6 points of performance, recovery took ~0.5 s, and the silenced side
returned to its own pre-silencing state rather than being written back by the partner (B9 rejected).

usage: python recovery.py RESULTS_DIR
"""
import sys, os, json, numpy as np, torch
R_ = sys.argv[1] if len(sys.argv) > 1 else 'prototype/results'
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], R_, '1']
exec(open(os.path.join(here, 'toy_split_brain.py')).read().split("if __name__")[0])
T0, SIL = 14, 4


def run(m, X, silence=None, cut_from=None):
    n = X.shape[0]; hL = torch.zeros(n, H); hR = torch.zeros(n, H); eL, eR, states = [], [], []
    with torch.no_grad():
        for t in range(X.shape[1] - 1):
            cut = cut_from is not None and t >= cut_from
            mLR = torch.zeros(n, H) if cut else m.msg(0, hL); mRL = torch.zeros(n, H) if cut else m.msg(1, hR)
            hL = m.L.gru(torch.cat([m.L.enc(X[:, t, :W].reshape(n, -1)), mRL], 1), hL)
            hR = m.R.gru(torch.cat([m.R.enc(X[:, t, W:].reshape(n, -1)), mLR], 1), hR)
            if silence is not None and T0 <= t < T0 + SIL:
                if silence == 0: hL = torch.zeros_like(hL)
                else: hR = torch.zeros_like(hR)
            states.append((hL.clone(), hR.clone()))
            eL.append(((m.L.dec(hL) - X[:, t + 1, :W].reshape(n, -1)) ** 2).mean(1))
            eR.append(((m.R.dec(hR) - X[:, t + 1, W:].reshape(n, -1)) ** 2).mean(1))
    return torch.stack(eL, 1).numpy(), torch.stack(eR, 1).numpy(), states


def recovery_step(err, base, tol=0.2):
    for t in range(T0 + SIL, err.shape[1]):
        if err[:, t].mean() <= base[:, t].mean() * (1 + tol):
            return t - (T0 + SIL)
    return None


if __name__ == '__main__':
    X, _, _ = world(800, seed=55)
    out = {}
    for d in (0, 1, 2, 4, 16):
        f = os.path.join(R_, f'model_d{d}.pt')
        if not os.path.exists(f): continue
        m = Pair(d); m.load_state_dict(torch.load(f)); m.eval()
        e0L, e0R, s0 = run(m, X)
        esL, esR, ss = run(m, X, silence=0)
        ecL, ecR, sc = run(m, X, silence=0, cut_from=T0)
        rec = recovery_step(esL, e0L); rec_cut = recovery_step(ecL, e0L)
        # does the partner's state carry the silenced side's lost information? compare the state L returns to with
        # the state it would have had, and with the state it reaches when nothing can be written back
        t_after = T0 + SIL + 1
        a = torch.stack([s[0] for s in [s0[t_after]]])[0]; b = ss[t_after][0]; c = sc[t_after][0]
        cos = lambda u, v: float((u * v).sum() / (u.norm() * v.norm() + 1e-9))
        out[f'd{d}'] = dict(recovery_steps=rec, recovery_steps_link_cut=rec_cut,
                            err_ratio_during=float(esL[:, T0:T0 + SIL].mean() / e0L[:, T0:T0 + SIL].mean()),
                            err_ratio_after1=float(esL[:, t_after].mean() / e0L[:, t_after].mean()),
                            err_ratio_after1_cut=float(ecL[:, t_after].mean() / e0L[:, t_after].mean()),
                            cos_normal_vs_silenced=cos(a, b), cos_normal_vs_cut=cos(a, c),
                            partner_err_ratio=float(esR[:, T0:T0 + SIL].mean() / e0R[:, T0:T0 + SIL].mean()))
        print(d, json.dumps(out[f'd{d}']), flush=True)
    json.dump(out, open(os.path.join(R_, 'recovery.json'), 'w'), indent=1)
