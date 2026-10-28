"""H8 / neural routing (docs/02 section 4.6): replace one hemisphere of an intact brain by a separately developed
'machine' hemisphere and connect it to the remaining biological hemisphere. Which way of choosing the connection
gives the biological hemisphere back an integrated world?

Brain:    the intact d = 16 pair (model_d16.pt). Its left hemisphere L ('biological', frozen) has learned to read
          callosal messages m = U V h_R from its original partner R.
Machine:  the right hemisphere R' of the d = 0 model (model_d0.pt): trained on the same right half-field, but never
          connected to anything. Its code is unrelated to R's. It runs on its own (no input from L).
Link:     L receives m(t) = g(h_R'(t-1)) in place of the cut callosum. Only L's side is evaluated: does L again
          predict objects entering from the machine's half (T5) and render them consistently (T8, R'->L direction)?

Ways to choose g (all fitted with the same budget of world sequences):
  R1 selectivity   Watanabe's proposal: show the same stimuli to the original R and to R', pair each R' unit with the
                   R unit of most similar response (Hungarian assignment on the correlation matrix), rescale each
                   unit linearly, and send U V (rescaled, permuted h_R') - i.e. "plug each machine unit into the cut
                   fibre with the most similar tuning".
  R4 stitching     population version: ridge regression h_R ~ A h_R' + b on the same paired stimuli, send U V (A h_R' + b).
                   (Model stitching / representational alignment.)
  R2 backprop      learn a rank-4 map W = P Q (h_R' -> message) by gradient descent on L's prediction error, L and R'
                   frozen. Not possible in a real brain (needs gradients through the biological hemisphere): the
                   optimum to compare against.
  R3 closed loop   the same rank-4 map tuned by SPSA (simultaneous perturbation): perturb W, measure L's prediction
                   error twice, step along the difference. Uses only the error of the biological side - what a
                   stimulating/recording BMI could do in closed loop, searching blindly in H*k dimensions.
  R5 align+loop    realistic two-stage: align by regression first (R4), then tune only k scalar gains, one per
                   leading message direction, in closed loop from the same scalar error.
References: intact (L with its own R), cut (m = 0).

usage: python routing.py RESULTS_DIR [budget_steps]
"""
import sys, os, json, time, numpy as np, torch
from scipy.optimize import linear_sum_assignment
R_ = sys.argv[1] if len(sys.argv) > 1 else 'prototype/results'; STEPS_ = int(sys.argv[2]) if len(sys.argv) > 2 else 600
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], R_, '1']
exec(open(os.path.join(here, 'toy_split_brain.py')).read().split("if __name__")[0])
torch.manual_seed(0); np.random.seed(0)
K_LINK = int(os.environ.get('K_LINK', '4'))


def load(d):
    m = Pair(d); m.load_state_dict(torch.load(os.path.join(R_, f'model_d{d}.pt'))); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


BIO = load(16); MAC = load(0)
L, R, Rm = BIO.L, BIO.R, MAC.R


def run_R_alone(hemi, X):
    """right hemisphere states h(t) with no message input (n, T, H)"""
    n = X.shape[0]; h = torch.zeros(n, H); out = []
    for t in range(X.shape[1]):
        h = hemi.gru(torch.cat([hemi.enc(X[:, t, W:].reshape(n, -1)), torch.zeros(n, H)], 1), h); out.append(h)
    return torch.stack(out, 1)


def run_intact_states(X):
    n = X.shape[0]; hL = torch.zeros(n, H); hR = torch.zeros(n, H); sR = []
    for t in range(X.shape[1]):
        mLR, mRL = BIO.msg(0, hL), BIO.msg(1, hR)
        hL = L.gru(torch.cat([L.enc(X[:, t, :W].reshape(n, -1)), mRL], 1), hL)
        hR = R.gru(torch.cat([R.enc(X[:, t, W:].reshape(n, -1)), mLR], 1), hR); sR.append(hR)
    return torch.stack(sR, 1)


def run_L(X, msg_fn, sender_states):
    """L driven by msg_fn(sender state at t-1); returns L's predictions (n, T-1, W*C)"""
    n = X.shape[0]; hL = torch.zeros(n, H); preds = []; prev = torch.zeros(n, H)
    for t in range(X.shape[1] - 1):
        m = msg_fn(prev) if msg_fn is not None else torch.zeros(n, H)
        hL = L.gru(torch.cat([L.enc(X[:, t, :W].reshape(n, -1)), m], 1), hL)
        preds.append(L.dec(hL)); prev = sender_states[:, t]
    return torch.stack(preds, 1)


def loss_L(X, msg_fn, S):
    p = run_L(X, msg_fn, S)
    return ((p - X[:, 1:, :W].reshape(X.shape[0], X.shape[1] - 1, -1)) ** 2).mean()


def evaluate(msg_fn, sender='machine', n=1000):
    X, ids, P = world(n, seed=77)
    with torch.no_grad():
        S = run_R_alone(Rm, X) if sender == 'machine' else run_intact_states(X)
        p = run_L(X, msg_fn, S).reshape(n, T - 1, W, C).numpy()
    Xn = X.numpy(); err = ((p - Xn[:, 1:, :W]) ** 2).mean((2, 3))
    ent = [err[(P[:, t, k] == W) & (P[:, t + 1, k] == W - 1), t] for t in range(T - 1) for k in range(K)]
    ent = np.concatenate(ent)
    # T8, R -> L direction: object at the right half's first pixel at t, gone from it at t+1; consistent if L's
    # prediction for t+1 shows the same type at its last (or second-to-last) pixel
    occ = Xn[:, :, W:].max(-1) > 0.5; typ = Xn[:, :, W:].argmax(-1); q = p.max(-1) > 0.5; u = p.argmax(-1)
    hit = tot = 0
    for t in range(10, T - 2):
        ev = occ[:, t, 0] & ~occ[:, t + 1, 0]
        ok = ev & ((q[:, t, W - 1] & (u[:, t, W - 1] == typ[:, t, 0])) | (q[:, t, W - 2] & (u[:, t, W - 2] == typ[:, t, 0])))
        hit += ok.sum(); tot += ev.sum()
    return dict(entry_ratio=float(ent.mean() / err[:, 5:].mean()), mean_err=float(err[:, 5:].mean()),
                consistency=float(hit / max(tot, 1)))


def paired_states(n_seq):
    X, _, _ = world(n_seq, seed=91)
    with torch.no_grad():
        return run_intact_states(X).reshape(-1, H).numpy(), run_R_alone(Rm, X).reshape(-1, H).numpy()


def uv(h):                                  # the biological callosal projection L reads: U V h
    return BIO.msg(1, h)


def fit_R1(n_seq):
    hR, hM = paired_states(n_seq)
    zR = (hR - hR.mean(0)) / (hR.std(0) + 1e-6); zM = (hM - hM.mean(0)) / (hM.std(0) + 1e-6)
    Cm = zR.T @ zM / len(zR)
    ri, ci = linear_sum_assignment(-np.abs(Cm))              # R unit ri[k] <- machine unit ci[k]
    a = np.zeros(H); b = np.zeros(H); perm = np.zeros(H, int)
    for i, j in zip(ri, ci):
        perm[i] = j; a[i], b[i] = np.polyfit(hM[:, j], hR[:, i], 1)
    a, b, perm = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32), torch.tensor(perm)
    return lambda h: uv(a * h[:, perm] + b), dict(mean_abs_match_r=float(np.abs(Cm[ri, ci]).mean()))


def fit_R4(n_seq):
    hR, hM = paired_states(n_seq)
    mu, Xc = hM.mean(0), hM - hM.mean(0)
    A = np.linalg.solve(Xc.T @ Xc + 1e-2 * np.trace(Xc.T @ Xc) / H * np.eye(H), Xc.T @ (hR - hR.mean(0)))
    A_, c_ = torch.tensor(A, dtype=torch.float32), torch.tensor(hR.mean(0) - mu @ A, dtype=torch.float32)
    r2 = 1 - ((Xc @ A - (hR - hR.mean(0))) ** 2).sum() / ((hR - hR.mean(0)) ** 2).sum()
    return lambda h: uv(h @ A_ + c_), dict(r2_alignment=float(r2))


def lowrank():
    Pm = torch.zeros(H, K_LINK); Qm = torch.randn(K_LINK, H) * 0.1
    return Pm, Qm


def fit_R2(steps, batch=32):
    Pm, Qm = lowrank(); Pm.requires_grad_(True); Qm.requires_grad_(True)
    opt = torch.optim.Adam([Pm, Qm], 3e-3)
    for s in range(steps):
        X, _, _ = world(batch, seed=5000 + s)
        with torch.no_grad(): S = run_R_alone(Rm, X)
        l = loss_L(X, lambda h: (h @ Qm.T) @ Pm.T, S); opt.zero_grad(); l.backward(); opt.step()
    Pf, Qf = Pm.detach(), Qm.detach()
    return lambda h: (h @ Qf.T) @ Pf.T


def fit_R3(iters, batch=32, c=float(os.environ.get('SPSA_C', '0.05')), lr=float(os.environ.get('SPSA_LR', '0.5'))):
    """SPSA with a common random batch for the two evaluations; Adam on the SPSA gradient estimate.
    The BMI only ever reads one scalar per evaluation (L's prediction error), as a closed-loop implant would."""
    Pm, Qm = lowrank(); theta = torch.cat([Pm.ravel(), Qm.ravel()]); nP = Pm.numel()
    unpack = lambda th: (th[:nP].view(H, K_LINK), th[nP:].view(K_LINK, H))
    m_ = torch.zeros_like(theta); v_ = torch.zeros_like(theta)
    with torch.no_grad():
        for s in range(iters):
            X, _, _ = world(batch, seed=5000 + s); S = run_R_alone(Rm, X)
            D = torch.randint(0, 2, theta.shape).float() * 2 - 1
            lp = loss_L(X, (lambda P_, Q_: (lambda h: (h @ Q_.T) @ P_.T))(*unpack(theta + c * D)), S)
            lm = loss_L(X, (lambda P_, Q_: (lambda h: (h @ Q_.T) @ P_.T))(*unpack(theta - c * D)), S)
            g = (lp - lm) / (2 * c) * D
            m_ = 0.9 * m_ + 0.1 * g; v_ = 0.999 * v_ + 0.001 * g ** 2          # Adam on the SPSA estimate
            theta = theta - lr * 0.001 * (m_ / (1 - 0.9 ** (s + 1))) / (torch.sqrt(v_ / (1 - 0.999 ** (s + 1))) + 1e-8)
    Pf, Qf = unpack(theta)
    return lambda h: (h @ Qf.T) @ Pf.T


def fit_R5(n_seq, iters, batch=32, c=0.1, lr=0.2, k=4):
    """Realistic two-stage BMI: (1) align the machine's population to the cut partner by regression on paired
    stimuli (R4, no gradients through the brain); (2) tune only k scalar gains, one per leading message direction,
    in closed loop from L's prediction error alone (SPSA). The search space is k, not H*k."""
    base, info = fit_R4(n_seq)
    hR, hM = paired_states(n_seq)
    with torch.no_grad():
        M = base(torch.tensor(hM[:4000], dtype=torch.float32))                  # messages the aligned BMI would send
    V = torch.linalg.svd(M - M.mean(0), full_matrices=False)[2][:k]             # leading message directions
    gains = torch.ones(k)
    def fn(g):
        return lambda h: base(h) @ (torch.eye(H) + V.T @ torch.diag(g - 1) @ V)
    m_ = torch.zeros(k); v_ = torch.zeros(k)
    with torch.no_grad():
        for s_ in range(iters):
            X, _, _ = world(batch, seed=7000 + s_); S = run_R_alone(Rm, X)
            D = torch.randint(0, 2, (k,)).float() * 2 - 1
            lp = loss_L(X, fn(gains + c * D), S); lm = loss_L(X, fn(gains - c * D), S)
            gr = (lp - lm) / (2 * c) * D
            m_ = 0.9 * m_ + 0.1 * gr; v_ = 0.999 * v_ + 0.001 * gr ** 2
            gains = gains - lr * (m_ / (1 - 0.9 ** (s_ + 1))) / (torch.sqrt(v_ / (1 - 0.999 ** (s_ + 1))) + 1e-8)
    return fn(gains), {**info, 'gains': gains.tolist()}


if __name__ == '__main__':
    fn = os.path.join(R_, 'routing.json'); res = json.load(open(fn)) if os.path.exists(fn) else {}
    def done(k, v):
        res[k] = v; json.dump(res, open(fn, 'w'), indent=1); print(k, v, flush=True)
    if 'intact' not in res: done('intact', evaluate(uv, sender='bio'))
    if 'cut' not in res: done('cut', evaluate(None))
    n_seq = STEPS_ * 32 // T                                    # same number of world frames as the learners see
    if 'R1_selectivity' not in res:
        f, info = fit_R1(max(n_seq, 200)); done('R1_selectivity', {**evaluate(f), **info})
    if 'R4_stitching' not in res:
        f, info = fit_R4(max(n_seq, 200)); done('R4_stitching', {**evaluate(f), **info})
    if 'R2_backprop' not in res:
        t = time.time(); f = fit_R2(STEPS_); done('R2_backprop', {**evaluate(f), 'sec': time.time() - t})
    if 'R5_align_then_loop' not in res:
        t = time.time(); f, info = fit_R5(max(n_seq, 200), STEPS_)
        done('R5_align_then_loop', {**evaluate(f), **info, 'sec': time.time() - t})
    tag = f"_k{K_LINK}" if K_LINK != 4 else ''
    key = 'R3_closed_loop' + tag + (f"_x{os.environ['SPSA_X']}" if os.environ.get('SPSA_X') else '')
    if key not in res:
        t = time.time(); f = fit_R3(int(os.environ.get('SPSA_X', '3')) * STEPS_)
        done(key, {**evaluate(f), 'sec': time.time() - t, 'k_link': K_LINK, 'lr': float(os.environ.get('SPSA_LR', '0.5')),
                   'c': float(os.environ.get('SPSA_C', '0.05'))})
