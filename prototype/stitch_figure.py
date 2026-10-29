"""Figure for docs/02 section 6.2: what the two ways of calibrating a bridge actually look like.

Encoder A is joined to decoder B through a thin linear bridge, both frozen. The bridge is fitted either from the
statistics of normal activity (correlational) or from interventions, on a recording that is increasingly
contaminated by shared variance the bridge cannot see. The figure shows the images that come out.

usage: python stitch_figure.py OUTDIR [steps]
"""
import sys, os, numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'prototype/results/stitch'
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
src = open(os.path.join(here, 'stitch_frozen.py')).read()
exec(src[:src.index("\nif __name__")])

INK, MUTED, BLUE, ORANGE = '#0b0b0b', '#8a8984', '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.titleweight': 'bold'})

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    mA, _ = train(1, STEPS, 0); mB, _ = train(2, STEPS, 0)
    xf = images(2000, 777); xt = images(8, 2024); xe = images(2000, 888)   # xe: the set the MSE is quoted on
    amps = [0.0, 1.0, 2.0, 4.0]
    rows = [('world', None)] + [(f'correlational', a) for a in amps] + [(f'interventional', a) for a in amps]
    fig, ax = plt.subplots(len(rows), 8, figsize=(8.6, 1.08 * len(rows)))
    for r, (kind, amp) in enumerate(rows):
        if kind == 'world':
            imgs = xt.numpy()
        else:
            globals()['NUIS'] = amp
            if kind == 'correlational':
                W, muA, muB = fit_correlational(mA, mB, xf, k=16)
            else:
                W = fit_interventional(mA, mB, xf, k=16)
                # the bridge also needs an offset; each module's own mean latent is a single-module quantity,
                # so it is available without any joint statistics
                with torch.no_grad():
                    muA = mA.encode(xf).numpy().mean(0); muB = mB.encode(xf).numpy().mean(0)
            with torch.no_grad():
                z = mA.encode(xt).numpy()
                z = (z - (muA if muA is not None else 0)) @ W + (muB if muB is not None else 0)
                imgs = mB.dec(torch.tensor(z, dtype=torch.float32)).numpy()
        if kind == 'world':
            err = 0.0
        else:                                  # quote the error on 2000 held-out images, not on the 8 shown
            with torch.no_grad():
                ze = mA.encode(xe).numpy()
                ze = (ze - (muA if muA is not None else 0)) @ W + (muB if muB is not None else 0)
                err = float(((mB.dec(torch.tensor(ze, dtype=torch.float32)) - xe) ** 2).mean())
        for c in range(8):
            a_ = ax[r, c]; a_.imshow(imgs[c].reshape(S, S), cmap='magma', vmin=0, vmax=1); a_.axis('off')
        lab = 'the world' if kind == 'world' else f'{kind}\nnuisance {amp:g}   MSE {err:.3f}'
        ax[r, 0].set_ylabel(lab, rotation=0, ha='right', va='center', fontsize=7.5,
                            color=INK if kind == 'world' else (BLUE if kind == 'interventional' else ORANGE))
        ax[r, 0].axis('on'); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        for sp in ax[r, 0].spines.values(): sp.set_visible(False)
    fig.suptitle('Encoder A joined to decoder B by a thin bridge, both frozen:\n'
                 'what each way of calibrating the bridge produces as the recording is contaminated',
                 fontsize=9.5, fontweight='bold', x=0.56)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, 'fig_stitch.png'), dpi=150)
    print('saved', os.path.join(OUT, 'fig_stitch.png'))
