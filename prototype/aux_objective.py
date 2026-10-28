"""H13: does a link trained only for prediction carry what cross-midline comparison (T2) needs? The toy said no (the
committee T2 stays at chance even with d = 16). Here each hemisphere is given an extra objective - describe the
OTHER half of the world: decode which object types are present in the opposite half at time t (4 binary labels),
from its own state only. The link is the only way to get that information. If the committee T2 now rises above
chance, what a link carries is set by the training objective, not by its width.

Conditions: d in {0, 4}, aux weight lam in {0, 1}. Same world, training length and probes as toy_split_brain.py.

usage: python aux_objective.py RESULTS_DIR [steps]
"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
OUTD = sys.argv[1]; STEPS_ = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], OUTD, str(STEPS_)]
exec(open(os.path.join(here, 'toy_split_brain.py')).read().split("if __name__")[0])


def other_types(X, t):
    """(n, C) presence of each type in the right half (for L) and left half (for R) at time t"""
    oR = (X[:, t, W:] > 0.5).any(1).float(); oL = (X[:, t, :W] > 0.5).any(1).float()
    return oR, oL


def train_aux(d, steps, lam, seed=1):
    torch.manual_seed(seed)
    m = Pair(d); aL, aR = nn.Linear(H, C), nn.Linear(H, C)
    opt = torch.optim.Adam(list(m.parameters()) + list(aL.parameters()) + list(aR.parameters()), 2e-3)
    bce = nn.BCEWithLogitsLoss()
    for s in range(steps):
        X, _, _ = world(32, seed=1000 + s)
        preds, states = m(X)
        loss = 0; aux = 0
        for t in range(T - 1):
            loss = loss + ((preds[t][0] - X[:, t + 1, :W].reshape(32, -1)) ** 2).mean() + \
                   ((preds[t][1] - X[:, t + 1, W:].reshape(32, -1)) ** 2).mean()
            if lam and t >= 3:
                yR, yL = other_types(X, t)
                aux = aux + bce(aL(states[t][0]), yR) + bce(aR(states[t][1]), yL)
        opt.zero_grad(); ((loss + lam * 0.02 * aux) / (T - 1)).backward(); opt.step()
    return m, (aL, aR)


if __name__ == '__main__':
    os.makedirs(OUTD, exist_ok=True); fn = os.path.join(OUTD, 'aux_objective.json')
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for d, lam in ((4, 1.0), (0, 1.0), (4, 0.0)):
        key = f'd{d}_lam{lam}'
        if key in res: continue
        t0 = time.time(); m, (aL, aR) = train_aux(d, STEPS_, lam); m.eval()
        ev, (hL, hR, ids, P) = evaluate(m)
        r = dict(entry_ratio=ev['entry_ratio'], ordinary_err=ev['ordinary_err'], probes=probes(hL, hR, ids, P),
                 sec=time.time() - t0)
        res[key] = r; json.dump(res, open(fn, 'w'), indent=1)
        print(key, json.dumps(r), flush=True)
