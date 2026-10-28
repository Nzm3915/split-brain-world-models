"""Export frames from the trained toy models for the interactive demo (docs/02 section 6.4).

For a handful of trials and for each condition, store what the world actually shows and what each hemisphere
predicts for the next step, as small integers: for every pixel, the predicted object type (0 = empty, 1..C) and the
confidence quantised to 0..9. Conditions:
  intact_d16, cut (d16 with the messages zeroed), bmi_corr / bmi_interv (cut + the two calibrated BMIs, rank 4),
  link1, link4 (weaker links), and for each of these a 'dream' version (observations stop after step 10).
Output: demo/frames.json (a few hundred kB), consumed by the HTML page.

usage: python export_demo.py RESULTS_DIR OUT_JSON [n_trials]
"""
import sys, os, json, numpy as np, torch
R_ = sys.argv[1]; OUTJ = sys.argv[2]; NT = int(sys.argv[3]) if len(sys.argv) > 3 else 6
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], R_, '1']
_src = open(os.path.join(here, 'bmi_calibration.py')).read()
exec(_src[:_src.index("\nif __name__ == '__main__':")])


def quant(frame):
    """(n, T, W, C) -> list of [type, conf] per pixel, type 0 = empty"""
    conf = frame.max(-1); typ = frame.argmax(-1) + 1
    typ = np.where(conf > 0.5, typ, 0)
    q = np.clip((conf * 9).round(), 0, 9).astype(int)
    return np.stack([typ, q], -1)


if __name__ == '__main__':
    torch.manual_seed(0)
    m16 = Pair(16); m16.load_state_dict(torch.load(os.path.join(R_, 'model_d16.pt'))); m16.eval()
    Xfit, _, _ = world(600, seed=41)
    bc = fit_correlational(m16, Xfit, 4); bi = fit_interventional(m16, Xfit, 4)
    X, ids, P = world(NT, seed=123)
    out = dict(W=W, C=C, T=T - 1, n_trials=NT, conditions={})
    conds = [('intact', m16, {}), ('cut', m16, dict(cut=True)),
             ('bmi_corr', m16, dict(cut=True, bmi=bc)), ('bmi_interv', m16, dict(cut=True, bmi=bi))]
    for d in (1, 4):
        md = Pair(d); md.load_state_dict(torch.load(os.path.join(R_, f'model_d{d}.pt'))); md.eval()
        conds.append((f'link{d}', md, {}))
    for name, mdl, kw in conds:
        for mode, dream in (('wake', None), ('dream', 10)):
            fL, fR, pL, pR, _, _ = rollout(mdl, X, dream_from=dream, **kw)
            er, me = entry_ratio(X, pL, pR, P)
            pack = lambda q: [''.join(f'{t}{c}' for t, c in row.reshape(-1, 2)) for tr in q for row in [tr.reshape(tr.shape[0], -1)]]
            flat = lambda q: [[''.join(f'{v[0]}{v[1]}' for v in step) for step in tr] for tr in q]
            out['conditions'][f'{name}_{mode}'] = dict(
                seen=flat(np.concatenate([quant(fL), quant(fR)], 2)),
                predL=flat(quant(pL)), predR=flat(quant(pR)),
                truth=flat(quant(X.numpy()[:, 1:])),
                entry_ratio=round(er, 2), mean_err=round(me, 5),
                consistency=round(consistency(fL, fR, pL, pR, 10)[0], 3))
            print(name, mode, out['conditions'][f'{name}_{mode}']['entry_ratio'],
                  out['conditions'][f'{name}_{mode}']['consistency'], flush=True)
    os.makedirs(os.path.dirname(OUTJ) or '.', exist_ok=True)
    json.dump(out, open(OUTJ, 'w'), separators=(',', ':'))
    print('wrote', OUTJ, os.path.getsize(OUTJ) // 1024, 'kB')
