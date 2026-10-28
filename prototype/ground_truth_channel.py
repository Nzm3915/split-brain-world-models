"""Ground-truth test of the mouse methods: in a system whose callosal channel is known, does silencing recover it,
and does the correlation of normal activity?

For a trained linked model (rank-d link A -> B), the true channel is the set of directions in B's state space that
the message moves: T = span of dh_B/dm * U over the operating points (Jacobian of B's GRU update with respect to
the message, times the message's output weights U). Three estimates of a d-dimensional subspace of B's state:
  receiving    top-d directions of B's response when A's state is zeroed for 5 steps (as in the mouse Phase 2)
  correlated   top-d directions of B predicted from A by reduced-rank ridge regression of normal activity
               (Semedo et al. 2019; mouse Phase 2)
  B's own PCs  top-d principal components of B's normal activity (a baseline)
Similarity to T: mean squared cosine of the principal angles, ||T' S||_F^2 / d (1 = identical, d/H = random).

usage: python ground_truth_channel.py RESULTS_DIR
"""
import sys, os, json, numpy as np, torch
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else 'prototype/results', '1']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_split_brain.py')).read().split("if __name__")[0])
R = sys.argv[1]


def orth(M, k):
    return np.linalg.svd(M, full_matrices=False)[0][:, :k]


def true_channel(m, X, src=0):
    """Jacobian of the receiver's state update with respect to the incoming message, times U; src = sender"""
    n = X.shape[0]; H_ = H
    with torch.no_grad():
        _, states = m(X)
    cols = []
    dst = m.R if src == 0 else m.L
    for t in range(5, X.shape[1], 3):
        hS = states[t - 1][src]; hD = states[t - 1][1 - src]
        xD = (X[:, t, W:] if src == 0 else X[:, t, :W]).reshape(n, -1)
        enc = dst.enc(xD).detach()
        U = m.U[src].weight.detach()                                  # H x d
        msg = m.msg(src, hS).detach().requires_grad_(True)
        h_new = dst.gru(torch.cat([enc, msg], 1), hD.detach())
        J = []
        for i in range(H_):
            g = torch.autograd.grad(h_new[:, i].sum(), msg, retain_graph=True)[0]   # n x H (d/dmsg)
            J.append(g)
        J = torch.stack(J, 1)                                          # n x H(out) x H(msg)
        cols.append(torch.einsum('noh,hd->nod', J, U).reshape(-1, H_, U.shape[1]).permute(1, 0, 2).reshape(H_, -1).numpy())
    return np.concatenate(cols, 1)


def estimates(m, X, src=0, k=1):
    with torch.no_grad():
        _, s0 = m(X); _, s1 = m(X, silence=src, sil_t=(20, 25))
    A0 = torch.stack([s[src] for s in s0], 1).numpy(); B0 = torch.stack([s[1 - src] for s in s0], 1).numpy()
    B1 = torch.stack([s[1 - src] for s in s1], 1).numpy()
    D = (B1 - B0)[:, 20:25].reshape(-1, H)
    XA = A0[:, 5:].reshape(-1, H); YB = B0[:, 5:].reshape(-1, H); XA = XA - XA.mean(0); YB = YB - YB.mean(0)
    Wr = np.linalg.solve(XA.T @ XA + 1e-1 * np.trace(XA.T @ XA) / H * np.eye(H), XA.T @ YB)
    return dict(receiving=orth(D.T, k), correlated=np.linalg.svd(XA @ Wr, full_matrices=False)[2][:k].T,
                own_pcs=np.linalg.svd(YB, full_matrices=False)[2][:k].T)


if __name__ == '__main__':
    out = {}
    X, _, _ = world(300, seed=21)
    for d in (1, 2, 4, 16):
        f = os.path.join(R, f'model_d{d}.pt')
        if not os.path.exists(f): continue
        m = Pair(d); m.load_state_dict(torch.load(f)); m.eval()
        res = {}
        for src, nm in ((0, 'L_to_R'), (1, 'R_to_L')):
            k = min(d, 4)
            T = orth(true_channel(m, X, src), k)
            est = estimates(m, X, src, k)
            res[nm] = {key: float(np.sum((T.T @ S) ** 2) / k) for key, S in est.items()}
            res[nm]['random'] = k / H
        out[f'd{d}'] = res
        print(d, json.dumps(res), flush=True)
    json.dump(out, open(os.path.join(R, 'ground_truth_channel.json'), 'w'), indent=1)
