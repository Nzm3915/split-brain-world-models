"""Follow-up to stitch_frozen.py: does the intervention-calibrated bridge still work when the two frozen modules
do not match - different architecture, different latent size, different training data emphasis?

That is the realistic case in engineering (a convolutional image encoder joined to a decoder that was trained
inside a different model), and the realistic case for a machine hemisphere, which will not share the biological
hemisphere's architecture.

A: convolutional encoder, latent ZA.      B: fully connected decoder, latent ZB != ZA.
The bridge is a ZA x ZB matrix of rank k, fitted three ways as in stitch_frozen.py (correlational, Procrustes on
the common subspace, interventional), with the recording optionally contaminated by shared nuisance variance.

usage: python stitch_mismatch.py OUTDIR [steps]
"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
OUT = sys.argv[1] if len(sys.argv) > 1 else 'prototype/results/stitch'
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
S, ZA, ZB = 16, 12, 20
NUIS = 0.0


def images(n, seed):
    r = np.random.default_rng(seed); x = np.zeros((n, S, S), np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    for i in range(n):
        for _ in range(2):
            a, b = r.integers(0, S - 4, 2); w, h = r.integers(2, 6, 2)
            x[i, a:a + h, b:b + w] += r.uniform(0.4, 1.0)
        cy, cx = r.integers(3, S - 3, 2); rad = r.uniform(1.5, 3.0)
        x[i][(yy - cy) ** 2 + (xx - cx) ** 2 < rad ** 2] += r.uniform(0.4, 1.0)
    return torch.tensor(np.clip(x, 0, 1).reshape(n, -1))


class ConvAE(nn.Module):
    """A: convolutional encoder"""
    def __init__(self, Z):
        super().__init__()
        self.c = nn.Sequential(nn.Unflatten(1, (1, S, S)), nn.Conv2d(1, 8, 3, 2, 1), nn.ReLU(),
                               nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(16 * 16, Z))
        self.dec = nn.Sequential(nn.Linear(Z, 128), nn.ReLU(), nn.Linear(128, S * S), nn.Sigmoid())

    def encode(self, x, u=None): return self.c(x)
    def forward(self, x, u=None): return self.dec(self.encode(x))


class MlpAE(nn.Module):
    """B: fully connected, wider latent"""
    def __init__(self, Z):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(S * S, 192), nn.ReLU(), nn.Linear(192, 96), nn.ReLU(), nn.Linear(96, Z))
        self.dec = nn.Sequential(nn.Linear(Z, 96), nn.ReLU(), nn.Linear(96, 192), nn.ReLU(),
                                 nn.Linear(192, S * S), nn.Sigmoid())

    def encode(self, x, u=None): return self.enc(x)
    def forward(self, x, u=None): return self.dec(self.encode(x))


def train(cls, Z, seed, steps):
    torch.manual_seed(seed); m = cls(Z); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for s in range(steps):
        x = images(64, 10000 * seed + s)
        loss = ((m(x) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, float(loss.detach())


def rank_k(M, k):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def contaminate(zA, zB, amp, seed=3):
    if not amp: return zA, zB
    g = np.random.default_rng(seed)
    dA = g.normal(size=(2, zA.shape[1])) / np.sqrt(zA.shape[1]); dB = g.normal(size=(2, zB.shape[1])) / np.sqrt(zB.shape[1])
    u = g.normal(size=(zA.shape[0], 2)); sc = amp * zA.std()
    return zA + sc * u @ dA, zB + sc * u @ dB


def jac(dec, z0, eps=0.05, reps=24, seed=0):
    g = torch.Generator().manual_seed(seed); A, B = [], []
    with torch.no_grad():
        base = dec(z0)
        for _ in range(reps):
            d = torch.randn(z0.shape, generator=g) * eps
            A.append(d.numpy()); B.append((dec(z0 + d) - base).numpy())
    return np.linalg.lstsq(np.concatenate(A), np.concatenate(B), rcond=None)[0]


def fits(mA, mB, x, k):
    zA = mA.encode(x).numpy(); zB = mB.encode(x).numpy()
    cA, cB = contaminate(zA, zB, NUIS)
    out = {}
    X = cA - cA.mean(0); Y = cB - cB.mean(0)
    G = X.T @ X + 1e-2 * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1])
    out['correlational'] = (rank_k(np.linalg.solve(G, X.T @ Y), k), cA.mean(0), cB.mean(0))
    U, _, Vt = np.linalg.svd(X.T @ Y)                      # rectangular Procrustes
    out['procrustes'] = (rank_k(U[:, :min(X.shape[1], Y.shape[1])] @ Vt[:min(X.shape[1], Y.shape[1])], k),
                         cA.mean(0), cB.mean(0))
    JA = jac(mA.dec, torch.tensor(zA), seed=1); JB = jac(mB.dec, torch.tensor(zB), seed=2)
    W = np.linalg.lstsq(JB.T, JA.T, rcond=None)[0].T
    out['interventional'] = (rank_k(W, k), zA.mean(0), zB.mean(0))
    return out


def mse(mA, mB, W, muA, muB, x):
    with torch.no_grad():
        z = (mA.encode(x).numpy() - muA) @ W + muB
        xh = mB.dec(torch.tensor(z, dtype=torch.float32))
    return float(((xh - x) ** 2).mean())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    mA, lA = train(ConvAE, ZA, 1, STEPS); mB, lB = train(MlpAE, ZB, 2, STEPS)
    xf = images(2000, 777); xt = images(2000, 888)
    with torch.no_grad():
        res = dict(ZA=ZA, ZB=ZB, own_A=float(((mA(xt) - xt) ** 2).mean()), own_B=float(((mB(xt) - xt) ** 2).mean()),
                   blank=float(((torch.full_like(xt, float(xt.mean())) - xt) ** 2).mean()))
    print(f'trained conv A {lA:.4f} (latent {ZA}), mlp B {lB:.4f} (latent {ZB})  [{round(time.time()-t0)} s]', flush=True)
    res['by_nuisance'] = {}
    for amp in (0.0, 1.0, 2.0, 4.0):
        globals()['NUIS'] = amp; row = {}
        for k in (6, 12):
            f = fits(mA, mB, xf, k)
            row[k] = {n: mse(mA, mB, W, muA, muB, xt) for n, (W, muA, muB) in f.items()}
        res['by_nuisance'][amp] = row
        print(f'  nuisance {amp:3.1f}: ' + ' | '.join(f'k={k} ' + ' '.join(f'{n[:4]} {v:.4f}' for n, v in row[k].items())
                                                      for k in (6, 12)), flush=True)
    print(f"own A {res['own_A']:.4f}  own B {res['own_B']:.4f}  blank {res['blank']:.4f}")
    json.dump(res, open(os.path.join(OUT, 'stitch_mismatch.json'), 'w'), indent=1)
