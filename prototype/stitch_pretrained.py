"""Item 2b of the review log: the intervention-calibrated bridge applied to networks nobody here trained.

prototype/stitch_frozen.py and stitch_mismatch.py showed the law on autoencoders trained in this repository. The
objection that remains is that those models were made by the same hand. Here the two modules are public
ImageNet-pretrained networks, downloaded as they are and never touched:

    image --[A's trunk]--> zA --[bridge W, rank k]--> zB' --[B's classifier head]--> 1000 logits

Neither network is fine-tuned, no gradient passes through either, and only W is fitted. The question is the same
one the mouse project asks of a brain-machine interface: with what has to be *recorded* to calibrate the link,
which calibration survives?

  correlational    zB ~ zA by ridge regression over normal images, truncated to rank k (standard stitching)
  interventional   match the effect of a latent perturbation on the output instead of the latents themselves:
                     JA = d logits_A / d zA   (how A's OWN head reads a direction in A's latent)
                     JB = d logits_B / d zB
                   choose W so that B's head reads what A meant:  JB W ~ JA,  rank(W) <= k.
                   Uses only perturbations of the two modules, never their joint statistics.
  procrustes       orthogonal Procrustes on paired latents (the usual alignment baseline)
  corr+metric      the control that decides what the interventional fit is actually buying: the SAME correlational
                   regression, but with the rank truncated in the metric of B's head rather than in latent space,
                   i.e. told which latent directions B's output is sensitive to. Unconstrained least squares is
                   unaffected by a weighting on the output side; under a rank constraint it is not. If this closes
                   the gap, the lesson is not "intervene" but "weight by what the receiver does with the signal" -
                   and intervention is how a link gets that weighting without a joint recording.

Scored against what the bridge is supposed to preserve - A's own verdict on the image - as the top-1 agreement
with A's prediction and the cosine of the centred logits. The nuisance condition is the same as in stitch_frozen:
the *recording used to fit* is contaminated by shared variance the bridge cannot see (zA + s uA, zB + s uB with
the same u per image), while evaluation always uses the clean trunk. The interventional fit never sees the
recording, so the contamination cannot reach it.

  interv/unpaired  the interventional fit again, with the two operating points taken from disjoint halves of the
                   images, so that no image is ever passed through both networks. If this matches the paired
                   version, the bridge needs no joint recording at all.
  interv/inputmetric  the unpaired interventional fit again, but with the rank truncated in the metric of A's own
                   latent covariance instead of in the raw matrix. A supplies that covariance by itself, so this is
                   still unpaired. It is here because the plain interventional fit loses badly at low rank, and the
                   suspicion is that its truncation throws away functionally important directions merely because
                   they carry little weight in the Jacobian.

Two controls, because a clean result deserves suspicion:

  the correlational baseline is given a ridge sweep and reported at its best setting, so the comparison is not
  against a strawman whose regularisation happened to be wrong;

  interv/wrongframe repeats the interventional fit after a fixed random permutation of B's output classes - the
                   engineer believing the wrong correspondence between the two output spaces. If the interventional
                   advantage comes from knowing that correspondence, this collapses to the floor while the
                   correlational fit, which never uses the output frame, does not move. That is the honest
                   statement of what the method needs: not a shared representation, but a shared output.

Pairs: resnet18 <-> resnet34 (both heads linear, both latents 512: the matched case) and
       resnet18 <-> mobilenet_v3_small (different family, 512 <-> 576, a nonlinear head: the mismatched case).

A caveat that belongs in the result, not after it: these networks share an output space, because both were trained
on ImageNet and class k means the same thing to each. The interventional fit uses exactly that - it compares the
two modules' effects in a common currency. Correlational stitching instead tries to match representations, which
have no such shared frame. That is the mouse project's claim in another setting: what a link has to carry is fixed
by what the receiver does with it, and calibration needs a shared output, not a shared code.

Images: CIFAR-10 upscaled to 128 px, used only as a stream of natural images - the ImageNet labels these models
emit on them are beside the point, since the reference is model A's own output, not a ground truth.

usage: python stitch_pretrained.py OUTDIR DATA_DIR [n_fit] [n_eval]
"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn

OUT = sys.argv[1] if len(sys.argv) > 1 else 'prototype/results/stitch'
DATA = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/data_untrusted'
N_FIT = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
N_EVAL = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
PX = 128
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NUIS_LEVELS = (0.0, 1.0, 4.0)
torch.set_num_threads(max(1, os.cpu_count() or 1))


def images(n, data_dir, px=PX, seed=0):
    """n CIFAR-10 test images, upscaled and normalised the way the pretrained models expect"""
    from torchvision import datasets
    ds = datasets.CIFAR10(data_dir, train=False, download=False)
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    x = torch.tensor(ds.data[idx].transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
    x = nn.functional.interpolate(x, size=(px, px), mode='bilinear', align_corners=False)
    return (x - MEAN) / STD


def split_model(name):
    """(trunk -> pooled feature vector, head -> logits, latent size) for a pretrained network, frozen"""
    import torchvision.models as M
    spec = {'resnet18': (M.resnet18, M.ResNet18_Weights.IMAGENET1K_V1),
            'resnet34': (M.resnet34, M.ResNet34_Weights.IMAGENET1K_V1),
            'mobilenet_v3_small': (M.mobilenet_v3_small, M.MobileNet_V3_Small_Weights.IMAGENET1K_V1)}[name]
    m = spec[0](weights=spec[1]).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    if name.startswith('resnet'):
        body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4, m.avgpool)
        trunk = lambda x: body(x).flatten(1)
        head, z = m.fc, m.fc.in_features
    else:
        trunk = lambda x: m.avgpool(m.features(x)).flatten(1)
        head, z = m.classifier, m.classifier[0].in_features
    return trunk, head, z


def batched(fn, x, bs=64):
    with torch.no_grad():
        return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


def head_jacobian(head, z0, eps=0.05, reps=24, seed=0):
    """how the logits move per unit of latent, averaged over the data: (Z, 1000). Measured rather than read off
    the weights, because one of the heads is nonlinear."""
    g = torch.Generator().manual_seed(seed); A, B = [], []
    with torch.no_grad():
        base = head(z0)
        for _ in range(reps):
            d = torch.randn(z0.shape, generator=g) * eps * z0.std()
            A.append(d.numpy()); B.append((head(z0 + d) - base).numpy())
    return np.linalg.lstsq(np.concatenate(A), np.concatenate(B), rcond=None)[0]


def rank_k(M, k):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


NUIS_DIRS = 8


def contaminate(zA, zB, amp, seed=3):
    """the recording used to fit the bridge carries variance the bridge cannot see: the same NUIS_DIRS latent
    components enter both recordings per image, scaled to a multiple of each latent's own spread"""
    if not amp:
        return zA, zB
    g = np.random.default_rng(seed); m = NUIS_DIRS
    dA = g.normal(size=(m, zA.shape[1])) / np.sqrt(m)
    dB = g.normal(size=(m, zB.shape[1])) / np.sqrt(m)
    u = g.normal(size=(zA.shape[0], m))
    return zA + amp * zA.std(0) * (u @ dA), zB + amp * zB.std(0) * (u @ dB)


RIDGES = (1e-4, 1e-2, 1e0)


def fit_correlational(zA, zB, k, ridge=1e-2):
    X = zA - zA.mean(0); Y = zB - zB.mean(0)
    G = X.T @ X + ridge * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1])
    return rank_k(np.linalg.solve(G, X.T @ Y), k), zA.mean(0), zB.mean(0)


def fit_procrustes(zA, zB, k):
    X = zA - zA.mean(0); Y = zB - zB.mean(0)
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    return rank_k(U @ Vt, k), zA.mean(0), zB.mean(0)


def fit_correlational_metric(zA, zB, JB, k, ridge=1e-2):
    """correlational, but the rank-k truncation is taken in the metric of B's head: minimise ||(zB - zA W) JB||.
    JB = U S V^T, so with W' = W U S the objective is an ordinary regression of zB U S on zA, and the rank of W'
    is the rank of W; the solution is mapped back through the pseudo-inverse of U S."""
    U, S, _ = np.linalg.svd(JB, full_matrices=False)
    T = U * S                                           # (ZB, min(ZB,1000)); a bijection on its range
    X = zA - zA.mean(0); Yt = (zB - zB.mean(0)) @ T
    G = X.T @ X + ridge * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1])
    Wp = rank_k(np.linalg.solve(G, X.T @ Yt), k)
    return Wp @ np.linalg.pinv(T), zA.mean(0), zB.mean(0)


def fit_interventional(JA, JB, k):
    # row convention: dzB = dzA @ W, dlogits = dzB @ JB; want JB^T W^T = JA^T
    return rank_k(np.linalg.lstsq(JB.T, JA.T, rcond=None)[0].T, k)


def fit_interventional_input_metric(JA, JB, CA, k):
    """the interventional fit, truncated in the metric of A's own latent spread rather than in the raw matrix.

    At full rank the two agree; under a rank constraint they do not, and that is where the plain interventional fit
    loses to the correlational one. The error that matters is the one produced on the inputs that actually occur,
    ||CA^(1/2) (W JB - JA)||, so A's latent covariance whitens the problem: with W' = CA^(1/2) W the objective is
    an ordinary least squares, the rank of W' is the rank of W, and the solution maps back.
    CA comes from A alone, so this stays unpaired - B never sees the images A saw."""
    w, V = np.linalg.eigh(CA)
    w = np.clip(w, 1e-12, None)
    H = V * np.sqrt(w); Hi = V / np.sqrt(w)                      # CA^(1/2) = H V^T, and its inverse
    Wp = rank_k(np.linalg.lstsq(JB.T, (H @ V.T @ JA).T, rcond=None)[0].T, k)
    return Hi @ V.T @ Wp


def evaluate(headB, zA_eval, logitsA, W, muA, muB):
    with torch.no_grad():
        zB = (zA_eval - (muA if muA is not None else 0)) @ W + (muB if muB is not None else 0)
        lg = headB(torch.tensor(zB, dtype=torch.float32)).numpy()
    top1 = float((lg.argmax(1) == logitsA.argmax(1)).mean())
    a = lg - lg.mean(1, keepdims=True); b = logitsA - logitsA.mean(1, keepdims=True)
    cos = float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))
    return dict(top1_agreement_with_A=top1, logit_cosine=cos)


def run_pair(nameA, nameB, xf, xe, out):
    t0 = time.time()
    trunkA, headA, ZA = split_model(nameA)
    trunkB, headB, ZB = split_model(nameB)
    zAf = batched(trunkA, xf).numpy(); zBf = batched(trunkB, xf).numpy()
    zAe = batched(trunkA, xe).numpy()
    logitsA = batched(headA, torch.tensor(zAe)).numpy()
    print(f'{nameA} -> {nameB}: latents {ZA} -> {ZB}, {len(xf)} fit / {len(xe)} eval images '
          f'[{round(time.time()-t0)} s]', flush=True)
    # the interventional fit needs only the two heads' responses to perturbations at their operating points
    JA = head_jacobian(headA, torch.tensor(zAf[:256]), seed=1)
    JB = head_jacobian(headB, torch.tensor(zBf[:256]), seed=2)
    # the same, with the two operating points taken from DISJOINT halves of the images, so that no image is ever
    # passed through both networks: the bridge is then fitted without any paired recording whatsoever
    h = len(zAf) // 2
    JAu = head_jacobian(headA, torch.tensor(zAf[:256]), seed=3)
    JBu = head_jacobian(headB, torch.tensor(zBf[h:h + 256]), seed=4)
    Z = zAf[:h] - zAf[:h].mean(0); CAu = Z.T @ Z / len(Z)        # A's own latent covariance: no pairing involved
    perm = np.random.default_rng(7).permutation(JBu.shape[1])    # the control: the wrong output correspondence
    ks = [k for k in (8, 32, 128, min(ZA, ZB)) if k <= min(ZA, ZB)]
    res = dict(pair=f'{nameA}->{nameB}', ZA=ZA, ZB=ZB, n_fit=len(xf), n_eval=len(xe), ranks=ks)
    # a floor: B's most common prediction, i.e. what a bridge that carries nothing would score
    with torch.no_grad():
        lgB = headB(torch.tensor(zBf[:512])).numpy()
    mode = np.bincount(lgB.argmax(1), minlength=1000).argmax()
    res['floor_top1'] = float((logitsA.argmax(1) == mode).mean())
    res['by_nuisance'] = {}
    for amp in NUIS_LEVELS:
        cA, cB = contaminate(zAf, zBf, amp)
        row = {}
        for k in ks:
            # the baseline gets a ridge sweep and is scored at its best setting
            best = None
            for rg in RIDGES:
                Wc_, muA_, muB_ = fit_correlational(cA, cB, k, ridge=rg)
                ev = evaluate(headB, zAe, logitsA, Wc_, muA_, muB_)
                if best is None or ev['top1_agreement_with_A'] > best[0]['top1_agreement_with_A']:
                    best = (dict(ev, ridge=rg), Wc_, muA_, muB_)
            corr_ev, Wc, muA, muB = best
            Wp, pA, pB = fit_procrustes(cA, cB, k) if ZA == ZB else (None, None, None)
            Wi = fit_interventional(JA, JB, k)
            Wm, mA, mB = fit_correlational_metric(cA, cB, JB, k)
            Wu = fit_interventional(JAu, JBu, k)
            Wim = fit_interventional_input_metric(JAu, JBu, CAu, k)
            Wwf = fit_interventional_input_metric(JAu, JBu[:, perm], CAu, k)
            row[k] = dict(correlational=corr_ev,
                          corr_metric=evaluate(headB, zAe, logitsA, Wm, mA, mB),
                          interventional=evaluate(headB, zAe, logitsA, Wi, zAf.mean(0), zBf.mean(0)),
                          interv_unpaired=evaluate(headB, zAe, logitsA, Wu, zAf[:h].mean(0), zBf[h:].mean(0)),
                          interv_inputmetric=evaluate(headB, zAe, logitsA, Wim, zAf[:h].mean(0), zBf[h:].mean(0)),
                          interv_wrongframe=evaluate(headB, zAe, logitsA, Wwf, zAf[:h].mean(0), zBf[h:].mean(0)))
            if Wp is not None:
                row[k]['procrustes'] = evaluate(headB, zAe, logitsA, Wp, pA, pB)
        res['by_nuisance'][amp] = row
        for k0 in ks:
            line = '  '.join(f"{n[:4]} {v['top1_agreement_with_A']:.3f}/{v['logit_cosine']:+.3f}"
                             for n, v in row[k0].items())
            print(f'  nuisance {amp:3.1f} rank {k0:3d}: {line}', flush=True)
        print(f'    [{round(time.time()-t0)} s]', flush=True)
    out[res['pair']] = res
    json.dump(out, open(os.path.join(OUT, 'stitch_pretrained.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    xf = images(N_FIT, DATA, seed=1); xe = images(N_EVAL, DATA, seed=2)
    out = {}
    run_pair('resnet18', 'resnet34', xf, xe, out)
    run_pair('resnet34', 'resnet18', xf, xe, out)                  # the same pair the other way round
    run_pair('resnet18', 'mobilenet_v3_small', xf, xe, out)
    run_pair('mobilenet_v3_small', 'resnet18', xf, xe, out)
    print('done', round(time.time() - t0), 's', flush=True)
