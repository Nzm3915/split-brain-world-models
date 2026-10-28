"""T8 (dream consistency) on the toy models: do the two hemispheres render the same object crossing the midline
when both run on their own predictions, with no observations?

Wake: observations for all steps; at each step each hemisphere predicts its next half-frame.
Dream: observations for the first 10 steps, then each hemisphere is fed its own predicted half-frame (open loop).
Event: in the sender's frame at time t an object sits at the edge pixel next to the midline, moving toward it, and is
gone from that pixel at t+1. Consistent if the receiver's predicted frame for t+1 shows an object of the SAME type at
its edge pixel (or one pixel further). Decoding a frame: pixel occupied if its max channel exceeds 0.5, type = argmax.
C = fraction of consistent events; Delta = C(link) - C(no link), for wake and dream.

usage: python dream_consistency.py RESULTS_DIR
"""
import sys, os, json, numpy as np, torch
R = sys.argv[1] if len(sys.argv) > 1 else 'prototype/results'
sys.argv = [sys.argv[0], R, '1']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_split_brain.py')).read().split("if __name__")[0])


def run(m, X, dream_from=None):
    n = X.shape[0]; hL = torch.zeros(n, H); hR = torch.zeros(n, H)
    framesL, framesR = [], []                       # what each hemisphere "sees" (obs or own prediction) at t
    predL, predR = [], []
    xL = X[:, 0, :W].reshape(n, -1); xR = X[:, 0, W:].reshape(n, -1)
    with torch.no_grad():
        for t in range(X.shape[1] - 1):
            if dream_from is None or t < dream_from:
                xL = X[:, t, :W].reshape(n, -1); xR = X[:, t, W:].reshape(n, -1)
            framesL.append(xL); framesR.append(xR)
            mLR, mRL = m.msg(0, hL), m.msg(1, hR)
            hL = m.L.gru(torch.cat([m.L.enc(xL), mRL], 1), hL); hR = m.R.gru(torch.cat([m.R.enc(xR), mLR], 1), hR)
            pL, pR = m.L.dec(hL), m.R.dec(hR)
            predL.append(pL); predR.append(pR)
            xL, xR = pL, pR                              # used only in the dream phase
    f = lambda L: torch.stack(L, 1).reshape(n, len(L), W, C).numpy()
    return f(framesL), f(framesR), f(predL), f(predR)


def decode(fr):
    occ = fr.max(-1) > 0.5; typ = fr.argmax(-1)
    return occ, typ


def consistency(frL, frR, prL, prR, t0):
    oL, tL = decode(frL); oR, tR = decode(frR); qL, uL = decode(prL); qR, uR = decode(prR)
    hit, tot = 0, 0
    T_ = frL.shape[1]
    for t in range(t0, T_ - 1):
        # L -> R: object at L's pixel W-1 at t, absent at t+1 in L's own frame
        ev = oL[:, t, W - 1] & ~oL[:, t + 1, W - 1]
        ok = ev & ((qR[:, t, 0] & (uR[:, t, 0] == tL[:, t, W - 1])) | (qR[:, t, 1] & (uR[:, t, 1] == tL[:, t, W - 1])))
        hit += ok.sum(); tot += ev.sum()
        ev = oR[:, t, 0] & ~oR[:, t + 1, 0]
        ok = ev & ((qL[:, t, W - 1] & (uL[:, t, W - 1] == tR[:, t, 0])) | (qL[:, t, W - 2] & (uL[:, t, W - 2] == tR[:, t, 0])))
        hit += ok.sum(); tot += ev.sum()
    return float(hit / max(tot, 1)), int(tot)


if __name__ == '__main__':
    X, _, _ = world(1000, seed=31)
    out = {}
    for d in (0, 1, 2, 4, 16):
        f = os.path.join(R, f'model_d{d}.pt')
        if not os.path.exists(f): continue
        m = Pair(d); m.load_state_dict(torch.load(f)); m.eval()
        cw, nw = consistency(*run(m, X), 10)
        cd, nd = consistency(*run(m, X, dream_from=10), 10)
        out[f'd{d}'] = dict(wake=cw, dream=cd, n_wake=nw, n_dream=nd)
        print(d, out[f'd{d}'], flush=True)
    for d in list(out):
        if d != 'd0' and 'd0' in out:
            out[d]['delta_wake'] = out[d]['wake'] - out['d0']['wake']; out[d]['delta_dream'] = out[d]['dream'] - out['d0']['dream']
    json.dump(out, open(os.path.join(R, 'dream_consistency.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
