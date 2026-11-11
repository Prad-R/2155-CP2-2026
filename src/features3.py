"""Wide rule-recovery basis (v3): a complete pairwise-distance basis plus nearest-distance,
neighbour-threshold and component features. Any rule that is a sum over cell pairs of a
function of (types, distance), or a sum over cells of a function of the distance to the
nearest cell of some type, is exactly linear in these columns.  features3(g) -> (X, names)"""
import numpy as np
from scipy import ndimage

T = 5
RC = np.array([(r, c) for r in range(7) for c in range(7)])
DR, DC = np.abs(RC[:, None, 0] - RC[None, :, 0]), np.abs(RC[:, None, 1] - RC[None, :, 1])
MAN, CHE, EUC = (DR + DC).astype(np.float32), np.maximum(DR, DC).astype(np.float32), np.sqrt(DR ** 2 + DC ** 2).astype(np.float32)
RING = np.minimum.reduce([RC[:, 0], RC[:, 1], 6 - RC[:, 0], 6 - RC[:, 1]])


def features3(g):
    g = np.asarray(g).reshape(-1, 7, 7)
    n = len(g)
    O = (g[:, None] == np.arange(T)[None, :, None, None]).astype(np.float32)       # n,5,7,7
    Of = O.reshape(n, T, 49)
    parts, names = [], []
    # 1. pair counts for every offset class (min|d|, max|d|) and unordered type pair
    cls = {}
    for dr in range(0, 7):
        for dc in range(-6, 7):
            if dr == 0 and dc <= 0:
                continue
            a = O[:, :, :7 - dr, max(0, -dc):7 - max(0, dc)]
            b = O[:, :, dr:, max(0, dc):7 + min(0, dc)]
            M = np.einsum("nahw,nbhw->nab", a, b)
            M = M + M.transpose(0, 2, 1)
            k = (min(dr, abs(dc)), max(dr, abs(dc)))
            cls[k] = cls.get(k, 0) + M
    iu = np.triu_indices(T)
    for k, M in sorted(cls.items()):
        Mu = M[:, iu[0], iu[1]].copy()
        Mu[:, iu[0] == iu[1]] /= 2
        parts.append(Mu); names += [f"pair{k[0]}{k[1]}_{i}{j}" for i, j in zip(*iu)]
    # 2. distance from every cell to the nearest cell of each type (self excluded)
    near = {}
    for nm, D, cap in (("man", MAN, 13.0), ("che", CHE, 7.0), ("euc", EUC, 9.0)):
        Dx = D.copy(); np.fill_diagonal(Dx, np.inf)
        dn = np.empty((n, T, 49), np.float32)
        for b in range(T):
            dn[:, b] = np.where(Of[:, b][:, None, :] > 0, Dx[None], np.inf).min(2)
        near[nm] = np.minimum(dn, cap)
    for a in range(T):
        for b in range(T):
            m = Of[:, a]
            dm = near["man"][:, b]
            cols = [((dm == d) * m).sum(1) for d in range(1, 7)] + [((dm >= 7) * m).sum(1)]
            cols += [(dm * m).sum(1), (near["che"][:, b] * m).sum(1), (near["euc"][:, b] * m).sum(1), (dm * m).max(1), (m / dm).sum(1),
                     ((near["che"][:, b] == 1) * m).sum(1), ((near["che"][:, b] <= 2) * m).sum(1)]
            parts.append(np.stack(cols, 1))
            names += [f"near{a}{b}_{s}" for s in ["m1", "m2", "m3", "m4", "m5", "m6", "m7p", "msum", "csum", "esum", "mmax", "minv", "c1", "c2"]]
    # 3. cells of type a with at least k neighbours of type b (4- and 8-neighbourhood)
    P = np.pad(O, ((0, 0), (0, 0), (1, 1), (1, 1)))
    n4 = P[:, :, :-2, 1:-1] + P[:, :, 2:, 1:-1] + P[:, :, 1:-1, :-2] + P[:, :, 1:-1, 2:]
    n8 = n4 + P[:, :, :-2, :-2] + P[:, :, :-2, 2:] + P[:, :, 2:, :-2] + P[:, :, 2:, 2:]
    for a in range(T):
        for b in range(T):
            cols = [(O[:, a] * (n4[:, b] >= k)).sum((1, 2)) for k in (1, 2, 3, 4)] + [(O[:, a] * (n8[:, b] >= k)).sum((1, 2)) for k in (1, 2, 3, 4, 6)]
            parts.append(np.stack(cols, 1)); names += [f"nb{a}{b}_{s}" for s in ["4k1", "4k2", "4k3", "4k4", "8k1", "8k2", "8k3", "8k4", "8k6"]]
    # 4. connected components
    comp = np.zeros((n, T, 8), np.float32)
    s8 = np.ones((3, 3))
    for i in range(n):
        for t in range(T):
            for j, st in enumerate((None, s8)):
                lab, k = ndimage.label(g[i] == t, structure=st)
                if k:
                    sz = np.bincount(lab.ravel())[1:]
                    comp[i, t, 4 * j:4 * j + 4] = (k, sz.max(), (sz ** 2).sum(), (sz == 1).sum())
    parts.append(comp.reshape(n, -1)); names += [f"comp{t}_{c}{s}" for t in range(T) for c in ("4", "8") for s in ("n", "max", "sq", "single")]
    # 5. counts, products, rings
    cnt = Of.sum(2)
    parts.append(cnt); names += [f"cnt{t}" for t in range(T)]
    parts.append(cnt[:, iu[0]] * cnt[:, iu[1]]); names += [f"cc{i}{j}" for i, j in zip(*iu)]
    parts.append(np.stack([(Of[:, t] * (RING == r)).sum(1) for t in range(T) for r in range(4)], 1)); names += [f"ring{t}_{r}" for t in range(T) for r in range(4)]
    X = np.concatenate([p.astype(np.float32) for p in parts], 1)
    assert X.shape[1] == len(names)
    return X, names
