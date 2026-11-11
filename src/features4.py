"""v4 basis: geodesic (path) distances with blocked district types, and per-cell nonlinear
combinations of local densities.  features4(g) -> (X, names)"""
import itertools
import numpy as np

T = 5


def _bfs(src, passable, cap=30):
    """src, passable: (n,7,7) bool. Path distance (4-moves) from the nearest src cell; paths may only
    step on passable cells (the destination cell itself may be anything). -> (n,7,7) float, cap if unreachable."""
    n = len(src)
    dist = np.where(src, 0.0, np.inf).astype(np.float32)
    front = src.copy()
    reach = src.copy()
    for d in range(1, 49):
        P = np.pad(front, ((0, 0), (1, 1), (1, 1)))
        nb = P[:, :-2, 1:-1] | P[:, 2:, 1:-1] | P[:, 1:-1, :-2] | P[:, 1:-1, 2:]
        new = nb & ~reach
        if not new.any():
            break
        dist[new] = d
        reach |= new
        front = new & passable          # only passable cells propagate further
    return np.minimum(dist, cap)


def features4(g):
    g = np.asarray(g).reshape(-1, 7, 7)
    n = len(g)
    O = g[:, None] == np.arange(T)[None, :, None, None]                      # n,5,7,7 bool
    parts, names = [], []
    # geodesic distance from cells of type a to the nearest b, with a set of blocked types
    for b in range(T):
        others = [t for t in range(T) if t != b]
        for r in (1, 2):
            for blocked in itertools.combinations(others, r):
                passable = ~np.isin(g, blocked)
                d = _bfs(O[:, b], passable)
                for a in range(T):
                    if a == b:
                        continue
                    m = O[:, a]
                    cnt = np.maximum(m.sum((1, 2)), 1)
                    s = (d * m).sum((1, 2)).astype(np.float32)
                    cols = [s, s / cnt, ((d >= 30) & m).sum((1, 2)), ((d <= 1) & m).sum((1, 2)), ((d <= 2) & m).sum((1, 2)), ((d <= 3) & m).sum((1, 2)), np.where(m, d, 0).max((1, 2))]
                    parts.append(np.stack(cols, 1))
                    bl = "".join(map(str, blocked))
                    names += [f"geo{a}to{b}_x{bl}_{s_}" for s_ in ["sum", "mean", "unreach", "le1", "le2", "le3", "max"]]
    # area reachable / region sizes: number of cells reachable from type-b cells without crossing blocked type x
    # per-cell nonlinear combos of local densities
    Of = O.astype(np.float32)
    P1 = np.pad(Of, ((0, 0), (0, 0), (2, 2), (2, 2)))
    def dens(offs):
        return sum(P1[:, :, 2 + dr:9 + dr, 2 + dc:9 + dc] for dr, dc in offs)
    n4 = dens([(-1, 0), (1, 0), (0, -1), (0, 1)])
    n8 = dens([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)])
    m2 = dens([(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if 0 < abs(dr) + abs(dc) <= 2])
    c2 = dens([(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0)])
    for nm, D in (("n4", n4), ("n8", n8), ("m2", m2), ("c2", c2)):
        for a in range(T):
            m = Of[:, a]
            for b, c in itertools.combinations(range(T), 2):
                x, y = D[:, b], D[:, c]
                cols = [(m * np.minimum(x, y)).sum((1, 2)), (m * x * y).sum((1, 2)), (m * (x > y)).sum((1, 2)), (m * (x < y)).sum((1, 2)), (m * np.maximum(x - y, 0)).sum((1, 2)), (m * np.maximum(y - x, 0)).sum((1, 2))]
                parts.append(np.stack(cols, 1)); names += [f"loc{nm}_{a}_{b}{c}_{s}" for s in ["min", "prod", "gt", "lt", "pos", "neg"]]
            for b in range(T):
                x = D[:, b]
                parts.append(np.stack([(m * x * x).sum((1, 2)), (m * (x == 0)).sum((1, 2)), (m * np.sqrt(x)).sum((1, 2))], 1)); names += [f"loc{nm}_{a}_{b}_{s}" for s in ["sq", "zero", "sqrt"]]
    X = np.concatenate([p.astype(np.float32) for p in parts], 1)
    assert X.shape[1] == len(names), (X.shape, len(names))
    return X, names
