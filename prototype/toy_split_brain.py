"""Toy split-brain world model: does the benchmark separate the conditions, on a CPU, in minutes?

World: a 1-D strip of 2W pixels (left half seen only by hemisphere L, right half only by R). K objects, each with an
identity (one of C types, one-hot channel) and a velocity of +-1 pixel/step, bouncing at the ends, so objects cross
the midline. Each hemisphere is a small recurrent world model (GRU) trained to predict its own next half-frame.

Conditions
  none        no link (split brain)
  link-d      low-rank callosal link: m_{A->B}(t) = U V h_A(t-1), rank d, added to B's GRU input
Read-outs on frozen states (linear probes, trained on held-out sequences)
  own         a hemisphere's probe sees only its own state
  committee   sum of the two hemispheres' independently trained logits (no mixing of states; mouse B1)
  shared      one probe on the concatenated states [h_L, h_R] (a learned shared controller)

Measures
  T5 entry error   prediction error of a hemisphere at the step an object enters its half from the other half,
                   relative to the error at ordinary steps (needs knowledge from the other hemisphere)
  T1 detection     "is there any object of type 0 anywhere?"  (committee should suffice)
  T2 same/diff     "is the leftmost-half object the same type as the right-half object?" (needs integration)
  silencing        zero h_A for 5 steps; B's state change = receiving response. Compare the receiving subspace with
                   the correlated subspace of normal activity (reduced-rank ridge from h_A to h_B), as in the mouse
                   Phase 2: fraction of the reliable receiving response inside the top-3 correlated directions.

usage: python toy_split_brain.py OUTDIR [steps]
"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn

torch.set_num_threads(2)
OUT = sys.argv[1]; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
W, C, K, T, H = 12, 4, 2, 32, 48
rng = np.random.default_rng(0)


def world(n, T=T, seed=None):
    r = np.random.default_rng(seed)
    X = np.zeros((n, T, 2 * W, C), np.float32)
    ids = r.integers(0, C, (n, K)); pos = r.integers(0, 2 * W, (n, K)); vel = r.choice([-1, 1], (n, K))
    P = np.zeros((n, T, K), int)
    for t in range(T):
        for k in range(K):
            X[np.arange(n), t, pos[:, k], ids[:, k]] = 1.0
        P[:, t] = pos
        nxt = pos + vel
        bounce = (nxt < 0) | (nxt >= 2 * W)
        vel = np.where(bounce, -vel, vel); pos = np.clip(pos + vel, 0, 2 * W - 1)
    X += r.normal(0, 0.05, X.shape).astype(np.float32)
    return torch.tensor(X), ids, P


class Hemi(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(W * C, 64), nn.ReLU())
        self.gru = nn.GRUCell(64 + H, H)
        self.dec = nn.Linear(H, W * C)


class Pair(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.L, self.R = Hemi(), Hemi(); self.d = d
        if d:
            self.V = nn.ModuleList([nn.Linear(H, d, bias=False) for _ in range(2)])
            self.U = nn.ModuleList([nn.Linear(d, H, bias=False) for _ in range(2)])

    def msg(self, i, h):                        # message from hemisphere i (0 = L, 1 = R) to the other
        return self.U[i](self.V[i](h)) if self.d else torch.zeros_like(h)

    def forward(self, X, silence=None, sil_t=(0, 0)):
        n = X.shape[0]; hL = torch.zeros(n, H); hR = torch.zeros(n, H); preds, states = [], []
        for t in range(X.shape[1]):
            xL = X[:, t, :W].reshape(n, -1); xR = X[:, t, W:].reshape(n, -1)
            mLR, mRL = self.msg(0, hL), self.msg(1, hR)
            hL_new = self.L.gru(torch.cat([self.L.enc(xL), mRL], 1), hL)
            hR_new = self.R.gru(torch.cat([self.R.enc(xR), mLR], 1), hR)
            if silence is not None and sil_t[0] <= t < sil_t[1]:
                if silence == 0: hL_new = torch.zeros_like(hL_new)
                else: hR_new = torch.zeros_like(hR_new)
            hL, hR = hL_new, hR_new
            preds.append((self.L.dec(hL), self.R.dec(hR))); states.append((hL, hR))
        return preds, states


def train(d, steps):
    torch.manual_seed(1)
    m = Pair(d); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for s in range(steps):
        X, _, _ = world(32, seed=1000 + s)
        preds, _ = m(X)
        loss = 0
        for t in range(T - 1):
            loss = loss + ((preds[t][0] - X[:, t + 1, :W].reshape(32, -1)) ** 2).mean() + \
                   ((preds[t][1] - X[:, t + 1, W:].reshape(32, -1)) ** 2).mean()
        opt.zero_grad(); (loss / (T - 1)).backward(); opt.step()
    return m


def evaluate(m, n=4000):
    X, ids, P = world(n, seed=7)
    with torch.no_grad():
        preds, states = m(X)
    err = np.zeros((n, T - 1, 2))
    for t in range(T - 1):
        err[:, t, 0] = ((preds[t][0] - X[:, t + 1, :W].reshape(n, -1)) ** 2).mean(1).numpy()
        err[:, t, 1] = ((preds[t][1] - X[:, t + 1, W:].reshape(n, -1)) ** 2).mean(1).numpy()
    # entry events: object at t in one half, at t+1 in the other
    entry, ordinary = [], []
    for t in range(T - 1):
        for k in range(K):
            a, b = P[:, t, k], P[:, t + 1, k]
            intoR = (a == W - 1) & (b == W); intoL = (a == W) & (b == W - 1)
            entry += list(err[intoR, t, 1]) + list(err[intoL, t, 0])
    ordinary = err[:, 5:].mean()
    hL = torch.stack([s[0] for s in states], 1).numpy(); hR = torch.stack([s[1] for s in states], 1).numpy()
    return dict(entry_err=float(np.mean(entry)), ordinary_err=float(ordinary),
                entry_ratio=float(np.mean(entry) / ordinary)), (hL, hR, ids, P)


def probes(hL, hR, ids, P, t=-1):
    """small nonlinear (MLP) probes on states at time t; 2-fold cross-validation. Same/different of two categorical
    identities is not linearly separable, so linear probes would fail even with full information."""
    from sklearn.neural_network import MLPClassifier
    def LogisticRegression(max_iter=2000, C=1.0, class_weight=None):
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=max_iter, random_state=0)
    n = len(ids)
    pos = P[:, t]; leftk = pos < W
    # T1: any object of type 0 anywhere
    y1 = (ids == 0).any(1).astype(int)
    # T2: trials with exactly one object in each half; same type?
    one_each = leftk.sum(1) == 1
    kL = np.argmax(leftk, 1); kR = np.argmax(~leftk, 1)
    y2 = (ids[np.arange(n), kL] == ids[np.arange(n), kR]).astype(int)
    out = {}
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    f = np.arange(n) % 2
    for task, y, msk in (('T1', y1, np.ones(n, bool)), ('T2', y2, one_each)):
        res = {}
        for name in ('own_L', 'own_R', 'committee', 'shared'):
            acc = []
            for k in (0, 1):
                tr, te = (f != k) & msk, (f == k) & msk
                # balance the training classes by subsampling the majority class
                i1, i0 = np.where(tr & (y == 1))[0], np.where(tr & (y == 0))[0]
                mn = min(len(i1), len(i0)); sel = np.r_[np.random.default_rng(k).permutation(i1)[:mn], np.random.default_rng(k).permutation(i0)[:mn]]
                tr = np.zeros(n, bool); tr[sel] = True
                if name == 'shared':
                    Z = np.c_[hL[:, t], hR[:, t]]
                    p = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced').fit(Z[tr], y[tr]).predict(Z[te])
                elif name == 'committee':
                    gL = LogisticRegression(max_iter=2000, class_weight='balanced').fit(hL[tr, t], y[tr]); gR = LogisticRegression(max_iter=2000, class_weight='balanced').fit(hR[tr, t], y[tr])
                    p = ((logit(gL.predict_proba(hL[te, t])[:, 1]) + logit(gR.predict_proba(hR[te, t])[:, 1])) > 0).astype(int)
                else:
                    Z = hL[:, t] if name == 'own_L' else hR[:, t]
                    p = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Z[tr], y[tr]).predict(Z[te])
                yt = y[te]; acc.append(0.5 * (np.mean(p[yt == 1] == 1) + np.mean(p[yt == 0] == 0)))   # balanced accuracy
            res[name] = float(np.mean(acc))
        res['chance'] = 0.5
        out[task] = res
    return out


def silencing(m, n=400, which=0, t0=20, t1=25):
    """receiving response of the other hemisphere vs its correlated subspace with the silenced one"""
    X, _, _ = world(n, seed=11)
    with torch.no_grad():
        _, s0 = m(X); _, s1 = m(X, silence=which, sil_t=(t0, t1))
    A0 = torch.stack([s[which] for s in s0], 1).numpy(); B0 = torch.stack([s[1 - which] for s in s0], 1).numpy()
    B1 = torch.stack([s[1 - which] for s in s1], 1).numpy()
    D = (B1 - B0)[:, t0:t1]                                     # trials x time x H (deterministic model: no noise)
    # split trials in halves for a reliability-style estimate
    D1, D2 = D[::2].reshape(-1, H), D[1::2].reshape(-1, H)
    # correlated subspace from normal activity, trial-to-trial residuals (reduced-rank ridge A -> B)
    XA = A0[:, 5:].reshape(-1, H); YB = B0[:, 5:].reshape(-1, H)
    XA = XA - XA.mean(0); YB = YB - YB.mean(0)
    Wr = np.linalg.solve(XA.T @ XA + 1e-1 * np.trace(XA.T @ XA) / H * np.eye(H), XA.T @ YB)
    Vc = np.linalg.svd(XA @ Wr, full_matrices=False)[2][:3].T
    def frac(V, A, B): return np.sum((A @ V) * (B @ V)) / np.sum(A * B)
    Ur = np.linalg.svd(D1, full_matrices=False)[2][:3].T
    return dict(f_corr3=float(frac(Vc, D1, D2)), f_rec3=float(frac(Ur, D1, D2)), f_rand3=3 / H,
                resp_norm=float(np.linalg.norm(D) / np.sqrt(D.size)))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for d in (0, 1, 2, 4, 16):
        t = time.time(); m = train(d, STEPS)
        torch.save(m.state_dict(), os.path.join(OUT, f'model_d{d}.pt'))
        ev, (hL, hR, ids, P) = evaluate(m)
        pr = probes(hL, hR, ids, P)
        r = dict(**ev, probes=pr, train_sec=time.time() - t)
        if d:
            r['silence_L'] = silencing(m, which=0); r['silence_R'] = silencing(m, which=1)
        results[f'd{d}'] = r
        print(d, json.dumps(r), flush=True)
    json.dump(results, open(os.path.join(OUT, 'toy_results.json'), 'w'), indent=1)
