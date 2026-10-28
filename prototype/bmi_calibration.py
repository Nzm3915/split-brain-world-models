"""Main question 3 in silico: replace a cut callosum with a linear BMI, calibrated either from correlations of normal
activity or from interventions. Which restores integration?

'Intact brain' = a trained linked model (rank-d link). The link is cut (messages set to 0) and replaced by a BMI that
adds a rank-k linear term to each receiver's state after its own update:
    h_B(t) <- GRU_B(...no message...) + M_{A->B} (h_A(t-1) - mean)
M is estimated in the intact system in two ways (the only data a BMI engineer would have):
  correlational   normal activity: h_B(t) = a + Mb h_B(t-1) + M h_A(t-1) + e  (ridge, then truncated to rank k);
                  the sender coefficient M is the BMI (Granger-style, conditions on the receiver's own past)
  cross-corr.     the same without the receiver's own past: h_B(t) = a + M h_A(t-1) + e
  interventional  small random perturbations of h_A(t-1) at random times in the intact system; the change of h_B(t)
                  relative to the unperturbed run, regressed on the perturbation (rank k): the causal effect
Evaluation on held-out sequences: T5 midline entry ratio, wake consistency and dream consistency, for intact, cut,
cut + correlational BMI, cut + interventional BMI.

With DRIVE_AMP > 0 (env), the fitting recording contains a hidden common drive (see rollout); evaluation never does.

usage: [DRIVE_AMP=0.3] python bmi_calibration.py RESULTS_DIR [d]
"""
import sys, os, json, numpy as np, torch
R = sys.argv[1] if len(sys.argv) > 1 else 'prototype/results'; D = int(sys.argv[2]) if len(sys.argv) > 2 else 16
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], R, '1']
exec(open(os.path.join(here, 'toy_split_brain.py')).read().split("if __name__")[0])
src = open(os.path.join(here, 'dream_consistency.py')).read()
exec(src[src.index('def decode('):src.index("if __name__ == '__main__'")], globals())
torch.manual_seed(0); rng = np.random.default_rng(0)
AMP = float(os.environ.get('DRIVE_AMP', '0'))


def step(m, hL, hR, xL, xR, bmi=None, cut=False, pert=None):
    mLR, mRL = (torch.zeros_like(hL), torch.zeros_like(hR)) if cut else (m.msg(0, hL), m.msg(1, hR))
    hA = hL + (pert[0] if pert is not None else 0); hB = hR + (pert[1] if pert is not None else 0)
    if pert is not None and not cut:
        mLR, mRL = m.msg(0, hA), m.msg(1, hB)
    nL = m.L.gru(torch.cat([m.L.enc(xL), mRL], 1), hL); nR = m.R.gru(torch.cat([m.R.enc(xR), mLR], 1), hR)
    if bmi is not None:
        (MRL, muR), (MLR, muL) = bmi                     # MRL: R->L map (applied to L), MLR: L->R map
        nL = nL + (hR - muR) @ MRL.T; nR = nR + (hL - muL) @ MLR.T
    return nL, nR


DRIVE = dict(amp=0.0)                                    # hidden common drive during the BMI-fitting recording


def rollout(m, X, bmi=None, cut=False, dream_from=None, drive=False):
    """drive=True adds an unobserved, slow (AR(1), tau = 5 steps) 2-d common input to both hemispheres' states,
    along fixed random directions - the analogue of brainstem/thalamic drive that a BMI engineer cannot record"""
    n = X.shape[0]; hL = torch.zeros(n, H); hR = torch.zeros(n, H)
    if drive and DRIVE['amp']:
        g = torch.Generator().manual_seed(5); DL = torch.randn(2, H, generator=g) / np.sqrt(H); DR = torch.randn(2, H, generator=g) / np.sqrt(H)
        z = torch.zeros(n, 2); rho = np.exp(-1 / 5)
    fL, fR, pL_, pR_, sL, sR = [], [], [], [], [], []
    xL = X[:, 0, :W].reshape(n, -1); xR = X[:, 0, W:].reshape(n, -1)
    with torch.no_grad():
        for t in range(X.shape[1] - 1):
            if dream_from is None or t < dream_from:
                xL = X[:, t, :W].reshape(n, -1); xR = X[:, t, W:].reshape(n, -1)
            fL.append(xL); fR.append(xR)
            hL, hR = step(m, hL, hR, xL, xR, bmi, cut)
            if drive and DRIVE['amp']:
                z = rho * z + np.sqrt(1 - rho ** 2) * torch.randn(n, 2)
                hL = hL + DRIVE['amp'] * np.sqrt(H) * z @ DL; hR = hR + DRIVE['amp'] * np.sqrt(H) * z @ DR
            pL, pR = m.L.dec(hL), m.R.dec(hR); pL_.append(pL); pR_.append(pR); sL.append(hL); sR.append(hR)
            xL, xR = pL, pR
    f = lambda L: torch.stack(L, 1).reshape(n, len(L), W, C).numpy()
    return f(fL), f(fR), f(pL_), f(pR_), torch.stack(sL, 1).numpy(), torch.stack(sR, 1).numpy()


def entry_ratio(X, pL, pR, P):
    n = X.shape[0]; Xn = X.numpy()
    err = np.stack([((pL - Xn[:, 1:, :W]) ** 2).mean((2, 3)), ((pR - Xn[:, 1:, W:]) ** 2).mean((2, 3))], -1)
    ent = []
    for t in range(X.shape[1] - 1):
        for k in range(K):
            a, b = P[:, t, k], P[:, t + 1, k]
            ent += list(err[(a == W - 1) & (b == W), t, 1]) + list(err[(a == W) & (b == W - 1), t, 0])
    return float(np.mean(ent) / err[:, 5:].mean()), float(err[:, 5:].mean())


def rank_k(M, k):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def fit_correlational(m, X, k, own_history=True):
    """own_history=True: lagged regression that also conditions on the receiver's own past (Granger-style);
    False: the receiver's state regressed on the sender's past alone (plain cross-correlation)"""
    _, _, _, _, sL, sR = rollout(m, X, drive=True)
    out = []
    for snd, rcv in ((sR, sL), (sL, sR)):                # R->L first, then L->R
        Y = rcv[:, 6:].reshape(-1, H)
        Xr = np.c_[rcv[:, 5:-1].reshape(-1, H), snd[:, 5:-1].reshape(-1, H)] if own_history else \
             np.c_[np.zeros((Y.shape[0], H)), snd[:, 5:-1].reshape(-1, H)]
        mu = Xr.mean(0); Xc = Xr - mu; Yc = Y - Y.mean(0)
        B = np.linalg.solve(Xc.T @ Xc + 1e-2 * np.trace(Xc.T @ Xc) / Xc.shape[1] * np.eye(Xc.shape[1]), Xc.T @ Yc)
        M = rank_k(B[H:].T, k)
        out.append((torch.tensor(M, dtype=torch.float32), torch.tensor(snd.reshape(-1, H).mean(0), dtype=torch.float32)))
    return out


def fit_interventional(m, X, k, eps=0.05, reps=6):
    n = X.shape[0]; dA, dB = {0: [], 1: []}, {0: [], 1: []}
    _, _, _, _, sL, sR = rollout(m, X, drive=True)
    with torch.no_grad():
        for _ in range(reps):
            t = int(rng.integers(6, X.shape[1] - 2))
            hL = torch.tensor(sL[:, t - 1]); hR = torch.tensor(sR[:, t - 1])
            xL = X[:, t, :W].reshape(n, -1); xR = X[:, t, W:].reshape(n, -1)
            base = step(m, hL, hR, xL, xR)
            for snd in (0, 1):
                d = torch.randn(n, H) * eps
                pert = (d, torch.zeros_like(d)) if snd == 0 else (torch.zeros_like(d), d)
                # perturb the SENDER's state as seen by the receiver through the link (the sender's own update ignored)
                new = step(m, hL, hR, xL, xR, pert=pert)
                rcv = 1 if snd == 0 else 0
                dA[snd].append(d.numpy()); dB[snd].append((new[rcv] - base[rcv]).numpy())
    out = []
    for snd in (1, 0):                                   # R->L first (sender R = 1), then L->R
        A = np.concatenate(dA[snd]); B = np.concatenate(dB[snd])
        Mf = np.linalg.lstsq(A, B, rcond=None)[0].T
        out.append((torch.tensor(rank_k(Mf, k), dtype=torch.float32),
                    torch.tensor((sR if snd == 1 else sL).reshape(-1, H).mean(0), dtype=torch.float32)))
    return out


if __name__ == '__main__':
    DRIVE['amp'] = AMP
    m = Pair(D); m.load_state_dict(torch.load(os.path.join(R, f'model_d{D}.pt'))); m.eval()
    Xfit, _, _ = world(600, seed=41); Xte, _, Pte = world(1000, seed=43)
    res = {}
    for k in (2, 4, 8):
        bc = fit_correlational(m, Xfit, k); bi = fit_interventional(m, Xfit, k)
        bx = fit_correlational(m, Xfit, k, own_history=False)
        for nm, kw in (('intact', {}), ('cut', dict(cut=True)), ('cut+corrBMI', dict(cut=True, bmi=bc)),
                       ('cut+xcorrBMI', dict(cut=True, bmi=bx)), ('cut+intervBMI', dict(cut=True, bmi=bi))):
            if k != 2 and nm in ('intact', 'cut'):
                continue
            fL, fR, pL, pR, _, _ = rollout(m, Xte, **kw)
            wake = consistency(fL, fR, pL, pR, 10)[0]
            dr = rollout(m, Xte, dream_from=10, **kw)
            dream = consistency(*dr[:4], 10)[0]
            key = nm if nm in ('intact', 'cut') else f'{nm}_k{k}'
            er, me = entry_ratio(Xte, pL, pR, Pte)
            # sanity: an unstable BMI makes predictions saturate; then "consistency" is spurious. Report the absolute
            # prediction error and how often the receivers' edge pixels look occupied at all (true rate ~ 0.1-0.2)
            occ = float(np.mean([(pR[:, 10:, :2].max(-1) > 0.5).mean(), (pL[:, 10:, -2:].max(-1) > 0.5).mean()]))
            res[key] = dict(entry_ratio=er, mean_err=me, edge_occupied=occ, wake=wake, dream=dream)
            print(key, res[key], flush=True)
    json.dump(res, open(os.path.join(R, f'bmi_calibration_d{D}' + (f'_drive{AMP}' if AMP else '') + '.json'), 'w'), indent=1)
