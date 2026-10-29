"""Frames for the pillar-2b demo: what each calibration makes of the same image.

Wires A's trunk into B's classifier through each of the three bridges and records, per image, what A says on its
own and what comes out the other side. Writes a JSON with a thumbnail of each image and the labels, so the page
can show the comparison without any model in the browser.

usage: python export_stitch_demo.py OUT_JSON DATA_DIR [n_fit] [n_show]
"""
import sys, os, io, json, base64, numpy as np, torch
here = os.path.dirname(os.path.abspath(__file__))
OUTJ = sys.argv[1] if len(sys.argv) > 1 else 'demo/stitch_frames.json'
DATA = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/data_untrusted'
N_FIT = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
N_SHOW = int(sys.argv[4]) if len(sys.argv) > 4 else 36
_src = open(os.path.join(here, 'stitch_pretrained.py')).read()
sys.argv = [sys.argv[0], 'prototype/results/stitch', DATA]
exec(_src[:_src.index("\nif __name__ == '__main__':")])

CIFAR = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
RANK = 128          # a bridge narrow enough to be a real constraint, where the three fits differ most


def categories(name):
    import torchvision.models as M
    return {'resnet18': M.ResNet18_Weights, 'resnet34': M.ResNet34_Weights,
            'mobilenet_v3_small': M.MobileNet_V3_Small_Weights}[name].IMAGENET1K_V1.meta['categories']


def thumbs(data_dir, idx, px=96):
    from torchvision import datasets
    from PIL import Image
    ds = datasets.CIFAR10(data_dir, train=False, download=False)
    out = []
    for i in idx:
        im = Image.fromarray(ds.data[i]).resize((px, px), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, format='PNG')
        out.append(dict(png='data:image/png;base64,' + base64.b64encode(b.getvalue()).decode(),
                        cifar=CIFAR[int(ds.targets[i])]))
    return out


def predict(headB, zA, W, muA, muB):
    with torch.no_grad():
        z = (zA - muA) @ W + muB
        return headB(torch.tensor(z, dtype=torch.float32)).numpy()


if __name__ == '__main__':
    nameA, nameB = 'resnet18', 'resnet34'
    cats = categories(nameA)
    trunkA, headA, ZA = split_model(nameA)
    trunkB, headB, ZB = split_model(nameB)
    xf = images(N_FIT, os.path.join(DATA, 'imagenette2-160', 'train') if os.path.isdir(os.path.join(DATA, 'imagenette2-160', 'train')) else DATA, seed=1)
    zAf = batched(trunkA, xf).numpy(); zBf = batched(trunkB, xf).numpy()
    h = len(zAf) // 2
    JA = head_jacobian(headA, torch.tensor(zAf[:256]), seed=3)
    JB = head_jacobian(headB, torch.tensor(zBf[h:h + 256]), seed=4)
    Z = zAf[:h] - zAf[:h].mean(0); CA = Z.T @ Z / len(Z)
    Wi = fit_interventional_input_metric(JA, JB, CA, RANK)
    Wc, muA, muB = fit_correlational(zAf, zBf, RANK, ridge=1e-2)
    Wp, pA, pB = fit_procrustes(zAf, zBf, RANK)
    # the images shown: photographs not used in the fit, a few from each of the ten classes
    import glob
    from PIL import Image
    files = sorted(glob.glob(os.path.join(DATA, '**', 'val', '**', '*.JPEG'), recursive=True))
    by_cls = {}
    for f in files:
        by_cls.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    rng = np.random.default_rng(5); per = max(N_SHOW // len(by_cls), 1)
    shown = [f for c in sorted(by_cls) for f in rng.permutation(by_cls[c])[:per]][:N_SHOW]
    arr, th = [], []
    for f in shown:
        im = Image.open(f).convert('RGB'); w, h = im.size; sc = (PX + 18) / min(w, h)
        im = im.resize((max(PX, int(round(w * sc))), max(PX, int(round(h * sc)))), Image.BILINEAR)
        w, h = im.size; l, t = (w - PX) // 2, (h - PX) // 2; im = im.crop((l, t, l + PX, t + PX))
        arr.append(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)
        b_ = io.BytesIO(); im.resize((112, 112), Image.LANCZOS).save(b_, format='JPEG', quality=82)
        th.append('data:image/jpeg;base64,' + base64.b64encode(b_.getvalue()).decode())
    xs = (torch.from_numpy(np.stack(arr)) - MEAN) / STD
    zAe = batched(trunkA, xs).numpy()
    own = batched(headA, torch.tensor(zAe)).numpy()
    lg = dict(interventional=predict(headB, zAe, Wi, zAf[:h].mean(0), zBf[h:].mean(0)),
              correlational=predict(headB, zAe, Wc, muA, muB),
              procrustes=predict(headB, zAe, Wp, pA, pB))
    frames = []
    for j in range(len(shown)):
        f = dict(png=th[j], sender=cats[int(own[j].argmax())])
        for k, v in lg.items():
            f[k] = cats[int(v[j].argmax())]
            f[k + '_agree'] = bool(v[j].argmax() == own[j].argmax())
        frames.append(f)
    res = json.load(open(os.environ.get('STITCH_JSON', 'prototype/results/stitch/stitch_pretrained_photos.json')))
    pair = res['resnet18->resnet34']
    out = dict(pair=f'{nameA} -> {nameB}', rank=RANK, latent=[ZA, ZB], n_fit=N_FIT,
               floor=pair['floor_top1'],
               scores={m: pair['by_nuisance']['0.0'][str(RANK)][m]['top1_agreement_with_A']
                       for m in pair['by_nuisance']['0.0'][str(RANK)]},
               frames=frames)
    os.makedirs(os.path.dirname(OUTJ) or '.', exist_ok=True)
    json.dump(out, open(OUTJ, 'w'))
    for m in ('interventional', 'correlational', 'procrustes'):
        print(f'{m:16s} agreement on the {len(pick)} shown images: '
              f"{np.mean([f[m + '_agree'] for f in frames]):.2f}  (full eval set: {out['scores'].get('interv_inputmetric' if m == 'interventional' else m, float('nan')):.3f})", flush=True)
    print('wrote', OUTJ, round(os.path.getsize(OUTJ) / 1024), 'kB')
