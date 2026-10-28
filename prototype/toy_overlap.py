"""H7 needs a world with common input. In toy_split_brain.py each hemisphere sees only its own half, so even awake,
the receiver learns of an incoming object only through the link and the wake/dream comparison is empty.
Here each hemisphere also sees the O pixels on the other side of the midline (an overlapping central strip, like
the bilateral representation of the vertical meridian), and predicts its own half plus that strip.
Awake, an incoming object is then seen by both sides before it crosses (common input). In the dream (open loop
from step 10) each hemisphere feeds back its own prediction of half + strip, so the common input is gone and only
the link can keep the two renderings consistent.

Prediction (H7):  Delta_dream = C_dream(link) - C_dream(no link)  >  Delta_wake = C_wake(link) - C_wake(no link)

usage: python toy_overlap.py RESULTS_DIR [steps] [O]
"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
OUTD = sys.argv[1]; STEPS_ = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; O = int(sys.argv[3]) if len(sys.argv) > 3 else 2
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], OUTD, str(STEPS_)]
exec(open(os.path.join(here, 'toy_split_brain.py')).read().split("if __name__")[0])
src = open(os.path.join(here, 'dream_consistency.py')).read()
exec(src[src.index('def decode('):src.index("if __name__ == '__main__'")], globals())


def views(X, t):
    n = X.shape[0]
    return X[:, t, :W + O].reshape(n, -1), X[:, t, W - O:].reshape(n, -1)       # L sees 0..W+O-1, R sees W-O..2W-1


class PairO(Pair):
    def __init__(self, d):
        super().__init__(d)
        for h in (self.L, self.R):
            h.enc = nn.Sequential(nn.Linear((W + O) * C, 64), nn.ReLU()); h.dec = nn.Linear(H, (W + O) * C)

    def step(self, hL, hR, xL, xR):
        mLR, mRL = self.msg(0, hL), self.msg(1, hR)
        return self.L.gru(torch.cat([self.L.enc(xL), mRL], 1), hL), self.R.gru(torch.cat([self.R.enc(xR), mLR], 1), hR)


def train_o(d, steps, seed=1):
    torch.manual_seed(seed); m = PairO(d); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for s in range(steps):
        X, _, _ = world(32, seed=1000 + s); n = 32
        hL = torch.zeros(n, H); hR = torch.zeros(n, H); loss = 0
        for t in range(T - 1):
            xL, xR = views(X, t); hL, hR = m.step(hL, hR, xL, xR); yL, yR = views(X, t + 1)
            loss = loss + ((m.L.dec(hL) - yL) ** 2).mean() + ((m.R.dec(hR) - yR) ** 2).mean()
        opt.zero_grad(); (loss / (T - 1)).backward(); opt.step()
    return m


def run_o(m, X, dream_from=None):
    """returns own-half frames and predictions (n, T-1, W, C), as dream_consistency.run"""
    n = X.shape[0]; hL = torch.zeros(n, H); hR = torch.zeros(n, H)
    fL, fR, pL_, pR_ = [], [], [], []
    xL, xR = views(X, 0)
    with torch.no_grad():
        for t in range(X.shape[1] - 1):
            if dream_from is None or t < dream_from:
                xL, xR = views(X, t)
            fL.append(xL.reshape(n, W + O, C)[:, :W]); fR.append(xR.reshape(n, W + O, C)[:, O:])
            hL, hR = m.step(hL, hR, xL, xR)
            pL, pR = m.L.dec(hL), m.R.dec(hR)
            pL_.append(pL.reshape(n, W + O, C)[:, :W]); pR_.append(pR.reshape(n, W + O, C)[:, O:])
            xL, xR = pL, pR
    f = lambda L: torch.stack(L, 1).numpy()
    return f(fL), f(fR), f(pL_), f(pR_)


if __name__ == '__main__':
    os.makedirs(OUTD, exist_ok=True); fn = os.path.join(OUTD, f'overlap_O{O}.json')
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    X, _, _ = world(1000, seed=31)
    for seed in (1, 2):
        for d in (0, 4):
            key = f'd{d}_s{seed}'
            if key in res: continue
            t0 = time.time(); m = train_o(d, STEPS_, seed); m.eval()
            cw, nw = consistency(*run_o(m, X), 10); cd, nd = consistency(*run_o(m, X, dream_from=10), 10)
            res[key] = dict(wake=cw, dream=cd, n_wake=nw, n_dream=nd, sec=time.time() - t0)
            json.dump(res, open(fn, 'w'), indent=1); print(key, res[key], flush=True)
    for seed in (1, 2):
        a, b = res.get(f'd4_s{seed}'), res.get(f'd0_s{seed}')
        if a and b:
            res[f'delta_s{seed}'] = dict(wake=a['wake'] - b['wake'], dream=a['dream'] - b['dream'])
    json.dump(res, open(fn, 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k.startswith('delta')}))
