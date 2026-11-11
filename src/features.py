"""Wide-net hand features for 7x7 zoning grids.

groups (name -> columns) are returned so models can use subsets, e.g. only the
rotation/reflection-invariant groups.
"""
import numpy as np
from scipy import ndimage

IDX = np.arange(49)
R, C = IDX // 7, IDX % 7
DMAN = (np.abs(R[:, None] - R[None]) + np.abs(C[:, None] - C[None])).astype(np.int8)
DMAN_OTHER = DMAN.copy()
np.fill_diagonal(DMAN_OTHER, 99)  # nearest *other* cell
RING = np.minimum.reduce([R, C, 6 - R, 6 - C])  # 0 (edge) .. 3 (centre)
PAIRS = [(a, b) for a in range(5) for b in range(a, 5)]
PAIRCODE = np.array([a * 5 + b for a, b in PAIRS])
INVARIANT = ["count", "adj4", "adjd", "dist", "comp", "ring"]
DMAX = 13


def _pair_counts(a, b):
    n = len(a)
    a = a.reshape(n, -1).astype(np.int16)
    b = b.reshape(n, -1).astype(np.int16)
    code = np.minimum(a, b) * 5 + np.maximum(a, b)
    return np.stack([(code == c).sum(1) for c in PAIRCODE], 1).astype(np.float32)


def _components(g):
    n = len(g)
    out = np.zeros((n, 10), np.float32)  # per type: n components, largest size
    for i in range(n):
        for b in range(5):
            lab, k = ndimage.label(g[i] == b)
            if k:
                out[i, 2 * b] = k
                out[i, 2 * b + 1] = np.bincount(lab.ravel())[1:].max()
    return out


def features(g):
    """g: (n,7,7) ints in 0..4 -> (X float32, groups dict name->slice)."""
    g = np.asarray(g)
    n = len(g)
    f = g.reshape(n, 49)
    oh = f[:, :, None] == np.arange(5)  # n,49,5 bool
    ohf = oh.astype(np.float32)
    parts = {}
    parts["onehot"] = ohf.reshape(n, -1)
    cnt = ohf.sum(1)
    parts["count"] = cnt
    parts["adj4"] = np.concatenate(
        [_pair_counts(g[:, :, :-1], g[:, :, 1:]), _pair_counts(g[:, :-1, :], g[:, 1:, :])], 0
    ).reshape(2, n, -1).sum(0)
    parts["adjd"] = np.concatenate(
        [_pair_counts(g[:, :-1, :-1], g[:, 1:, 1:]), _pair_counts(g[:, :-1, 1:], g[:, 1:, :-1])], 0
    ).reshape(2, n, -1).sum(0)
    # distance from each cell to the nearest (other) cell of type b
    dist = []
    for b in range(5):
        d = np.where(oh[:, None, :, b], DMAN_OTHER[None], 99).min(2)  # n,49
        d = np.minimum(d, DMAX).astype(np.float32)
        for a in range(5):
            ma = ohf[:, :, a]
            ca = np.maximum(cnt[:, a], 1)
            dist.append((d * ma).sum(1) / ca)          # mean distance a -> nearest b
            dist.append((d * ma).max(1))               # worst-case distance
            dist.append(((d <= 1) * ma).sum(1))        # a-cells touching a b
            dist.append(((d <= 2) * ma).sum(1))        # a-cells within 2 of a b
    parts["dist"] = np.stack(dist, 1)
    parts["comp"] = _components(g)
    parts["ring"] = np.concatenate([ohf[:, RING == r].sum(1) for r in range(4)], 1)
    g4 = ohf.reshape(n, 7, 7, 5)
    parts["row"] = g4.sum(2).reshape(n, -1)
    parts["col"] = g4.sum(1).reshape(n, -1)
    groups, cols, start = {}, [], 0
    for k, v in parts.items():
        groups[k] = (start, start + v.shape[1])
        start += v.shape[1]
        cols.append(v.astype(np.float32))
    return np.concatenate(cols, 1), groups


def _chunk(g):
    return features(g)[0]


def features_parallel(g, procs=8, chunk=4000):
    from multiprocessing import Pool
    chunks = [g[i:i + chunk] for i in range(0, len(g), chunk)]
    if procs <= 1 or len(chunks) == 1:
        return np.concatenate([_chunk(c) for c in chunks], 0)
    with Pool(procs) as p:
        return np.concatenate(p.map(_chunk, chunks), 0)


def group_cols(groups, names):
    return np.concatenate([np.arange(*groups[k]) for k in names])
