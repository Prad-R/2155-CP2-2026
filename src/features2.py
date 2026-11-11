"""Extra rotation/flip-invariant features (v2), on top of features.py.

Longer-range pair counts, neighbourhood densities, distance histograms, cluster
detail, and "access" features (does a cell have every other use nearby?).
"""
import numpy as np
from scipy import ndimage
from features import features, _pair_counts, DMAN_OTHER, R, C, PAIRS, INVARIANT, group_cols

DCH = np.maximum(np.abs(R[:, None] - R[None]), np.abs(C[:, None] - C[None])).astype(np.int8)
DCH_O = DCH.copy(); np.fill_diagonal(DCH_O, 99)
OFFS = {"o02": [(0, 2), (2, 0)], "o12": [(1, 2), (1, -2), (2, 1), (2, -1)], "o22": [(2, 2), (2, -2)],
        "o03": [(0, 3), (3, 0)], "o13": [(1, 3), (1, -3), (3, 1), (3, -1)]}
S8 = np.ones((3, 3))


def _shift_pairs(g, dr, dc):
    a = g[:, :7 - dr, max(0, -dc):7 - max(0, dc)]
    b = g[:, dr:, max(0, dc):7 + min(0, dc)]
    return _pair_counts(a, b)


def features2(g):
    g = np.asarray(g); n = len(g); f = g.reshape(n, 49)
    oh = f[:, :, None] == np.arange(5); ohf = oh.astype(np.float32)
    cnt = ohf.sum(1); cs = np.maximum(cnt, 1)
    parts, names = [], []
    for k, offs in OFFS.items():
        parts.append(sum(_shift_pairs(g, dr, dc) for dr, dc in offs)); names += [f"{k}_{a}{b}" for a, b in PAIRS]
    # neighbourhood density: mean number of b cells within Chebyshev radius r of an a cell
    for r in (1, 2):
        W = ((DCH <= r) & (DCH > 0)).astype(np.float32)            # 49,49
        nb = ohf.transpose(0, 2, 1) @ W                             # n,5(b),49: b-cells near each cell
        dens = np.einsum("nbc,nca->nab", nb, ohf) / cs[:, :, None]  # n,a,b
        parts.append(dens.reshape(n, 25)); names += [f"dens{r}_{a}{b}" for a in range(5) for b in range(5)]
    acc_all, worst = np.ones((n, 49), bool), np.zeros((n, 49), np.float32)
    for b in range(5):
        dm = np.minimum(np.where(oh[:, None, :, b], DMAN_OTHER[None], 99).min(2), 13).astype(np.float32)
        dc = np.minimum(np.where(oh[:, None, :, b], DCH_O[None], 99).min(2), 7).astype(np.float32)
        notb = ~oh[:, :, b]
        acc_all &= (dm <= 2) | ~notb
        worst = np.maximum(worst, np.where(notb, dm, 0))
        for a in range(5):
            ma = ohf[:, :, a]
            parts.append(np.stack([((dm <= 3) * ma).sum(1), ((dm <= 1) * ma).sum(1) / cs[:, a], ((dm <= 2) * ma).sum(1) / cs[:, a],
                                   (dc * ma).sum(1) / cs[:, a], ((dc <= 1) * ma).sum(1)], 1))
            names += [f"{s}_{a}{b}" for s in ("le3", "fr1", "fr2", "chmean", "ch1")]
    # access: cells with every other use within Manhattan 2; worst distance to any other use
    parts.append(np.stack([(acc_all * ohf[:, :, a]).sum(1) for a in range(5)], 1)); names += [f"accall_{a}" for a in range(5)]
    parts.append(np.stack([(worst * ohf[:, :, a]).sum(1) / cs[:, a] for a in range(5)], 1)); names += [f"worst_{a}" for a in range(5)]
    parts.append(np.stack([acc_all.sum(1), worst.mean(1), worst.max(1)], 1)); names += ["accall", "worst_mean", "worst_max"]
    # clusters: isolated cells, 8-connected clusters, 2x2 blocks, corners
    cl = np.zeros((n, 15), np.float32)
    for i in range(n):
        for b in range(5):
            m = g[i] == b
            lab, k = ndimage.label(m)
            if k: cl[i, 3 * b] = (np.bincount(lab.ravel())[1:] == 1).sum()
            lab, k = ndimage.label(m, S8)
            if k: cl[i, 3 * b + 1] = k; cl[i, 3 * b + 2] = np.bincount(lab.ravel())[1:].max()
    parts.append(cl); names += [f"{s}_{b}" for b in range(5) for s in ("isolated", "ncl8", "maxcl8")]
    o4 = ohf.reshape(n, 7, 7, 5)
    parts.append((o4[:, :-1, :-1] * o4[:, 1:, :-1] * o4[:, :-1, 1:] * o4[:, 1:, 1:]).sum((1, 2))); names += [f"block2_{b}" for b in range(5)]
    parts.append(o4[:, [0, 0, 6, 6], [0, 6, 0, 6]].sum(1)); names += [f"corner_{b}" for b in range(5)]
    return np.concatenate([p.astype(np.float32) for p in parts], 1), names


def features_inv2(g):
    """invariant v1 features + v2 -> X"""
    X1, groups = features(g)
    return np.concatenate([X1[:, group_cols(groups, INVARIANT)], features2(g)[0]], 1)
