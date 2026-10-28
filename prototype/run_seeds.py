"""Replicate the toy results over training seeds (the analogue of 'mouse as the unit' in 01).

For each link rank d and each seed: train, then compute T5 entry ratio, T1/T2 probes, the ground-truth channel test
(receiving vs correlated estimate) and wake/dream consistency. Results: SEEDDIR/seed_results.json

usage: python run_seeds.py SEEDDIR [steps]
"""
import sys, os, json, time, numpy as np, torch
OUTD = sys.argv[1]; STEPS_ = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], OUTD, str(STEPS_)]
exec(open(os.path.join(here, 'toy_split_brain.py')).read().split("if __name__")[0])
src_gt = open(os.path.join(here, 'ground_truth_channel.py')).read()
exec(src_gt[src_gt.index('def orth('):src_gt.index("if __name__ == '__main__'")], globals())
src_dream = open(os.path.join(here, 'dream_consistency.py')).read()
exec(src_dream[src_dream.index('def run('):src_dream.index("if __name__ == '__main__'")], globals())


def train_seed(d, steps, seed):
    torch.manual_seed(seed)
    m = Pair(d); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for s in range(steps):
        X, _, _ = world(32, seed=100000 * seed + s)
        preds, _ = m(X)
        loss = 0
        for t in range(T - 1):
            loss = loss + ((preds[t][0] - X[:, t + 1, :W].reshape(32, -1)) ** 2).mean() + \
                   ((preds[t][1] - X[:, t + 1, W:].reshape(32, -1)) ** 2).mean()
        opt.zero_grad(); (loss / (T - 1)).backward(); opt.step()
    return m


if __name__ == '__main__':
    os.makedirs(OUTD, exist_ok=True)
    fn = os.path.join(OUTD, 'seed_results.json')
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    Xg, _, _ = world(300, seed=21); Xd, _, _ = world(1000, seed=31)
    for seed in (11, 12, 13):
        for d in (0, 2, 16):
            key = f'd{d}_s{seed}'
            if key in res: continue
            t = time.time(); m = train_seed(d, STEPS_, seed); m.eval()
            ev, (hL, hR, ids, P) = evaluate(m)
            r = dict(entry_ratio=ev['entry_ratio'], probes=probes(hL, hR, ids, P))
            r['wake'], _ = consistency(*run(m, Xd), 10); r['dream'], _ = consistency(*run(m, Xd, dream_from=10), 10)
            if d:
                k = min(d, 4); gt = {}
                for src, nm in ((0, 'L_to_R'), (1, 'R_to_L')):
                    Tm = orth(true_channel(m, Xg, src), k); est = estimates(m, Xg, src, k)
                    gt[nm] = {kk: float(np.sum((Tm.T @ S) ** 2) / k) for kk, S in est.items()}; gt[nm]['random'] = k / H
                r['ground_truth'] = gt
            r['sec'] = time.time() - t
            res[key] = r; json.dump(res, open(fn, 'w'), indent=1)
            print(key, json.dumps(r), flush=True)
