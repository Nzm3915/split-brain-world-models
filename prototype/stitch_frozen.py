"""Item 2 of the review log: take the intervention-calibrated bridge out of the toy brain and point it at two
frozen modules that were never connected.

Setting (model stitching, the engineering form of the same problem): two autoencoders are trained independently on
the same images, with different seeds. Then encoder A is joined to decoder B through a thin linear bridge, and
nothing but the bridge may be fitted - both modules stay frozen, and no gradient is taken through them.

    x --[encoder A]--> zA --[bridge W, rank k]--> zB' --[decoder B]--> x_hat

Three ways to fit W, all with the same budget:
  correlational    zB ~ zA by ridge regression over normal data, truncated to rank k (standard stitching)
  interventional   perturb A's latent and B's latent and match their effects on the image:
                     JA = d x_hat_A / d zA   (how A's own decoder renders a latent direction)
                     JB = d x_hat_B / d zB
                   choose W so that B renders what A meant:  JB W ~ JA, rank(W) <= k. Uses only perturbations of
                   the two modules, never their joint statistics.
  procrustes       orthogonal Procrustes on paired latents (a common alignment baseline)

The toy predicted that the correlational route fails when the recording contains variation the bridge cannot see.
That is tested here with NUISANCE > 0: the *recording used to fit the bridge* is contaminated by a shared nuisance,
zA_obs = zA + s uA, zB_obs = zB + s uB with the same u drawn per sample and fixed random directions - the engineer's
measurements of the two modules share variance that has nothing to do with the images. Evaluation always uses the
clean encoder. The correlational and Procrustes fits see the contaminated latents; the interventional fit does not,
because it works from differences, in which the nuisance cancels.

A first version instead gave each autoencoder an extra noisy input channel during training. That failed to create
the condition: the noise does not help reconstruction, so the encoders simply ignored it and the latents stayed
clean. Recorded below as a reminder that a nuisance has to enter the quantity being recorded.

usage: python stitch_frozen.py OUTDIR [steps] [--nuisance]
"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
OUT = sys.argv[1] if len(sys.argv) > 1 else 'prototype/results/stitch'
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3000
NUIS = 0.6 if '--nuisance' in sys.argv else 0.0
S, Z, K = 16, 16, 4            # image side, latent size, bridge rank


def images(n, seed):
    """n images of S x S: two rectangles and a disc, random position, size and brightness"""
    r = np.random.default_rng(seed); x = np.zeros((n, S, S), np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    for i in range(n):
        for _ in range(2):
            a, b = r.integers(0, S - 4, 2); w, h = r.integers(2, 6, 2)
            x[i, a:a + h, b:b + w] += r.uniform(0.4, 1.0)
        cy, cx = r.integers(3, S - 3, 2); rad = r.uniform(1.5, 3.0)
        x[i][(yy - cy) ** 2 + (xx - cx) ** 2 < rad ** 2] += r.uniform(0.4, 1.0)
    return torch.tensor(np.clip(x, 0, 1).reshape(n, -1))


class AE(nn.Module):
    def __init__(self, nuis=0):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(S * S + nuis, 128), nn.ReLU(), nn.Linear(128, Z))
        self.dec = nn.Sequential(nn.Linear(Z, 128), nn.ReLU(), nn.Linear(128, S * S), nn.Sigmoid())
        self.nuis = nuis

    def encode(self, x, u=None):
        return self.enc(torch.cat([x, u], 1) if self.nuis else x)

    def forward(self, x, u=None):
        return self.dec(self.encode(x, u))


def train(seed, steps, nuis):
    torch.manual_seed(seed); m = AE(nuis); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for s in range(steps):
        x = images(64, 10000 * seed + s)
        u = torch.randn(64, nuis) * NUIS if nuis else None
        loss = ((m(x, u) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, float(loss)


def rank_k(M, k):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def jac_decoder(dec, z0, eps=0.05, reps=24, seed=0):
    """how the rendered image moves per latent direction, averaged over the data: (Z, S*S)"""
    g = torch.Generator().manual_seed(seed); A, B = [], []
    with torch.no_grad():
        base = dec(z0)
        for _ in range(reps):
            d = torch.randn_like(z0) * eps
            A.append(d.numpy()); B.append((dec(z0 + d) - base).numpy())
    return np.linalg.lstsq(np.concatenate(A), np.concatenate(B), rcond=None)[0]


def fit_interventional(mA, mB, x, k=K, u=None):
    zA = mA.encode(x, u); zB = mB.encode(x, u)
    JA = jac_decoder(mA.dec, zA, seed=1); JB = jac_decoder(mB.dec, zB, seed=2)
    # want JB^T W^T = JA^T  (row convention: dzB = dzA @ W, dx = dzB @ JB)
    W = np.linalg.lstsq(JB.T, JA.T, rcond=None)[0].T              # (Z, Z): dzA -> dzB
    return rank_k(W, k)


def contaminate(zA, zB, amp, seed=3):
    if not amp:
        return zA, zB
    g = np.random.default_rng(seed)
    dA = g.normal(size=(2, zA.shape[1])) / np.sqrt(zA.shape[1]); dB = g.normal(size=(2, zB.shape[1])) / np.sqrt(zB.shape[1])
    u = g.normal(size=(zA.shape[0], 2))
    sc = amp * zA.std()
    return zA + sc * u @ dA, zB + sc * u @ dB


def fit_correlational(mA, mB, x, k=K, u=None, ridge=1e-2):
    zA, zB = contaminate(mA.encode(x, u).numpy(), mB.encode(x, u).numpy(), NUIS)
    X = zA - zA.mean(0); Y = zB - zB.mean(0)
    G = X.T @ X + ridge * np.trace(X.T @ X) / Z * np.eye(Z)
    return rank_k(np.linalg.solve(G, X.T @ Y), k), zA.mean(0), zB.mean(0)


def fit_procrustes(mA, mB, x, k=K, u=None):
    zA, zB = contaminate(mA.encode(x, u).numpy(), mB.encode(x, u).numpy(), NUIS)
    X = zA - zA.mean(0); Y = zB - zB.mean(0)
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return rank_k(U @ Vt, k), zA.mean(0), zB.mean(0)


def mse(mA, mB, W, x, muA=None, muB=None, u=None):
    with torch.no_grad():
        zA = mA.encode(x, u).numpy()
        zB = (zA - (muA if muA is not None else 0)) @ W + (muB if muB is not None else 0)
        xh = mB.dec(torch.tensor(zB, dtype=torch.float32))
    return float(((xh - x) ** 2).mean())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    nuis = 0                      # the nuisance now enters the recording, not the training
    mA, lA = train(1, STEPS, nuis); mB, lB = train(2, STEPS, nuis)
    print(f'trained: A {lA:.4f}  B {lB:.4f}  [{round(time.time()-t0)} s]', flush=True)
    xf = images(2000, 777); xt = images(2000, 888)
    uf = torch.randn(2000, nuis) * NUIS if nuis else None
    ut = torch.randn(2000, nuis) * NUIS if nuis else None
    with torch.no_grad():
        own_A = float(((mA(xt, ut) - xt) ** 2).mean()); own_B = float(((mB(xt, ut) - xt) ** 2).mean())
        blank = float(((torch.full_like(xt, float(xt.mean())) - xt) ** 2).mean())
    res = dict(nuisance=NUIS, own_A=own_A, own_B=own_B, blank=blank, rank=K, latent=Z)
    res['identity'] = mse(mA, mB, np.eye(Z), xt, u=ut)
    # how each fit degrades as the recording is contaminated by shared variance the bridge cannot see
    res['by_nuisance'] = {}
    for amp in (0.0, 0.5, 1.0, 2.0, 4.0):
        globals()['NUIS'] = amp
        row = {}
        for k in (8, 16):
            Wc, muA, muB = fit_correlational(mA, mB, xf, k=k, u=uf)
            Wi = fit_interventional(mA, mB, xf, k=k, u=uf)
            Wp, pA, pB = fit_procrustes(mA, mB, xf, k=k, u=uf)
            row[k] = dict(correlational=mse(mA, mB, Wc, xt, muA, muB, u=ut),
                          procrustes=mse(mA, mB, Wp, xt, pA, pB, u=ut),
                          interventional=mse(mA, mB, Wi, xt, muA, muB, u=ut))
        res['by_nuisance'][amp] = row
        print(f'  nuisance {amp:3.1f}: ' + ' | '.join(f'k={k} ' + ' '.join(f'{n[:4]} {v:.4f}' for n, v in row[k].items()) for k in (8, 16)), flush=True)
    globals()['NUIS'] = 0.0
    res['by_rank'] = {}
    for k in (2, 4, 8, 16):
        Wc, muA, muB = fit_correlational(mA, mB, xf, k=k, u=uf)
        Wp, pA, pB = fit_procrustes(mA, mB, xf, k=k, u=uf)
        Wi = fit_interventional(mA, mB, xf, k=k, u=uf)
        res['by_rank'][k] = dict(correlational=mse(mA, mB, Wc, xt, muA, muB, u=ut),
                                 procrustes=mse(mA, mB, Wp, xt, pA, pB, u=ut),
                                 interventional=mse(mA, mB, Wi, xt, muA, muB, u=ut))
        print(f'  rank {k:2d}: ' + '  '.join(f'{n} {v:.5f}' for n, v in res['by_rank'][k].items()), flush=True)
    for k, v in res.items():
        print(f'{k:16s} {v}' if not isinstance(v, float) else f'{k:16s} {v:.5f}', flush=True)
    json.dump(res, open(os.path.join(OUT, f'stitch{"_nuis" if NUIS else ""}.json'), 'w'), indent=1)
