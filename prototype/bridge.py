"""Pillar 2 of docs/02 section 6: the intervention-calibrated bridge as one function, so that it can be applied to
any pair of frozen modules, not just to the toy hemispheres.

    bridge = fit_bridge(sender, receiver, probe, k=4)

The caller supplies three callables and keeps everything else:
    sender(x)            -> the sender's state for input x               (frozen, may be a black box)
    receiver(x, m)       -> the receiver's state given input x and a message m of size H_r   (frozen)
    probe()              -> a batch of inputs (x_sender, x_receiver) drawn from normal operation
The bridge is a rank-k linear map from the sender's state to the receiver's message port, fitted from
*interventions only*:

  1. receiving directions. Push random messages m into the receiver and record how its state moves:
         dR = receiver(x, m) - receiver(x, 0)        ->  J_r (H_r x H_r), the port's effect
  2. sending directions. Perturb the sender's state and record how the message it would emit changes, or, when the
     sender has no port to read, perturb its state directly and record the state change: dS.
  3. solve   min_W || J_r W dS - dR_target ||,  rank(W) <= k
     where dR_target is what the receiver should have received: the response that the intact partner produced.
     With no intact partner available (two modules that were never connected), the target is replaced by the
     sender's state projected on its own leading directions, which is what R4 does in routing.py.

Correlational baselines for the same interface: Granger-style regression and plain cross-correlation.

This file is deliberately written against callables rather than against toy_split_brain.Pair, so the same code can
be pointed at two pretrained encoders later (docs/02 section 6.3, M2).

usage: python bridge.py RESULTS_DIR     (self-test against the toy models, reproducing routing.py's numbers)
"""
import sys, os, json, numpy as np, torch
here = os.path.dirname(os.path.abspath(__file__))
R_ = sys.argv[1] if len(sys.argv) > 1 else 'prototype/results'
sys.argv = [sys.argv[0], R_, '1']
exec(open(os.path.join(here, 'toy_split_brain.py')).read().split("if __name__")[0])


def rank_k(M, k):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def port_response(receiver, batch, H_r, eps=0.05, reps=8, seed=0):
    """J_r: how a message at the port moves the receiver's state (the receiving directions)"""
    g = torch.Generator().manual_seed(seed); A, B = [], []
    for _ in range(reps):
        m = torch.randn(batch[1].shape[0], H_r, generator=g) * eps
        A.append(m.numpy()); B.append((receiver(batch[1], m) - receiver(batch[1], torch.zeros_like(m))).numpy())
    return np.linalg.lstsq(np.concatenate(A), np.concatenate(B), rcond=None)[0]      # (H_r, H_r)


def fit_bridge(sender_states, target_states, J_r, k=4, ridge=1e-2):
    """W maps the sender's state to the port so that J_r W s reproduces the target movement of the receiver.
    sender_states (n, H_s), target_states (n, H_r): both measured under interventions."""
    X = sender_states - sender_states.mean(0); Y = target_states - target_states.mean(0)
    # row-vector convention throughout: dR = dm @ J_r, so the message that would have produced dR is dR @ pinv(J_r)
    T = Y @ np.linalg.pinv(J_r)
    G = X.T @ X + ridge * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1])
    W = np.linalg.solve(G, X.T @ T)
    return rank_k(W.T, k)                                      # (H_r, H_s)


def fit_correlational(sender_states, receiver_states, k=4, own_history=None, ridge=1e-2):
    """baseline: the receiver's state regressed on the sender's, optionally conditioning on its own past"""
    X = sender_states if own_history is None else np.c_[own_history, sender_states]
    X = X - X.mean(0); Y = receiver_states - receiver_states.mean(0)
    G = X.T @ X + ridge * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1])
    B = np.linalg.solve(G, X.T @ Y)
    M = B[-sender_states.shape[1]:].T
    return rank_k(M, k)


if __name__ == '__main__':
    # self-test: the toy d16 pair, with L as the receiver and R as the sender, exactly as routing.py's 'intact' case
    m = Pair(16); m.load_state_dict(torch.load(os.path.join(R_, 'model_d16.pt'))); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    X, _, _ = world(400, seed=61); n = X.shape[0]
    hL = torch.zeros(n, H); hR = torch.zeros(n, H); SL, SR = [], []
    with torch.no_grad():
        for t in range(T):
            mLR, mRL = m.msg(0, hL), m.msg(1, hR)
            hL = m.L.gru(torch.cat([m.L.enc(X[:, t, :W].reshape(n, -1)), mRL], 1), hL)
            hR = m.R.gru(torch.cat([m.R.enc(X[:, t, W:].reshape(n, -1)), mLR], 1), hR)
            SL.append(hL.clone()); SR.append(hR.clone())
    # average the port response and the interventions over several time points: one time point gives a noisy
    # linear picture of a gated (nonlinear) update
    g = torch.Generator().manual_seed(1); dS, dR = [], []; Js = []
    for t0 in (8, 12, 16, 20, 24):
        xL = X[:, t0, :W].reshape(n, -1)
        receiver = (lambda h_prev: (lambda x, msg: m.L.gru(torch.cat([m.L.enc(x), msg], 1), h_prev)))(SL[t0 - 1])
        Js.append(port_response(receiver, (None, xL), H, seed=t0))
        with torch.no_grad():
            for _ in range(8):
                d = torch.randn(n, H, generator=g) * 0.05
                dS.append(d.numpy())
                dR.append((receiver(xL, m.msg(1, SR[t0 - 1] + d)) - receiver(xL, m.msg(1, SR[t0 - 1]))).numpy())
    J_r = np.mean(Js, 0)
    Wi = fit_bridge(np.concatenate(dS), np.concatenate(dR), J_r, k=4)
    # correlational baseline on the same states
    Sr = torch.stack(SR, 1).reshape(-1, H).numpy(); Sl = torch.stack(SL, 1).reshape(-1, H).numpy()
    Wc = fit_correlational(Sr[:-n], Sl[n:], k=4, own_history=Sl[:-n])
    # how well does each reproduce the true port map U V of the intact link?
    true = (m.U[1].weight @ m.V[1].weight).numpy()
    cos = lambda A, B: float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))
    # a rank-4 bridge cannot match a rank-16 map: the ceiling is the best rank-4 approximation of the true map
    sv = np.linalg.svd(true, compute_uv=False)
    ceiling = float(np.sqrt((sv[:4] ** 2).sum() / (sv ** 2).sum()))
    out = dict(intervention_vs_true=cos(Wi, true), correlation_vs_true=cos(Wc, true), rank4_ceiling=ceiling,
               intervention_over_ceiling=cos(Wi, true) / ceiling, correlation_over_ceiling=cos(Wc, true) / ceiling,
               scale_intervention=float(np.linalg.norm(Wi) / np.linalg.norm(true)),
               scale_correlation=float(np.linalg.norm(Wc) / np.linalg.norm(true)))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(R_, 'bridge_selftest.json'), 'w'), indent=1)
