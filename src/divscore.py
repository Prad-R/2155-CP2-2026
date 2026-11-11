"""Exact (integer) diversity score, leaderboard-score decoding, and subset selection.

Leaderboard score = S / 49000 where S = sum over valid grids of the sum of Hamming
counts to the 10 nearest other valid grids (requires >= 11 valid grids).
"""
import itertools
import numpy as np

K = 10
DENOM = 49 * K * 100


def hamming_matrix(a, b=None):
    a = np.asarray(a).reshape(len(a), -1).astype(np.int8)
    b = a if b is None else np.asarray(b).reshape(len(b), -1).astype(np.int8)
    D = np.zeros((len(a), len(b)), np.int16)
    for s in range(0, len(a), 256):
        D[s:s + 256] = (a[s:s + 256, None, :] != b[None, :, :]).sum(2)
    return D


def int_score_from_D(D, k=K):
    """D: square int Hamming matrix of the valid grids."""
    n = len(D)
    if n < 2:
        return 0
    D = D.astype(np.int32).copy()
    np.fill_diagonal(D, 10**6)
    kk = min(k, n - 1)
    return int(np.partition(D, kk - 1, axis=1)[:, :kk].sum())


def int_score(grids, valid=None):
    grids = np.asarray(grids)
    if valid is not None:
        grids = grids[np.asarray(valid, bool)]
    return int_score_from_D(hamming_matrix(grids))


def score(grids, valid=None):
    return int_score(grids, valid) / DENOM


def decode(grids, observed, max_invalid=3, p_valid=None, chunk=20000):
    """Find all sets of <= max_invalid invalid grids that reproduce the observed score.

    observed: leaderboard float (or exact integer S). Returns dict with the target
    integer, the all-valid integer and a list of (invalid_index_tuple, log_prior).
    """
    grids = np.asarray(grids)
    n = len(grids)
    target = int(observed) if float(observed) > 1.5 else int(round(float(observed) * DENOM))
    D = hamming_matrix(grids).astype(np.int32)
    np.fill_diagonal(D, 10**6)
    m = K + max_invalid
    nb = np.argsort(D, axis=1, kind="stable")[:, :m]          # n, m neighbour ids
    nd = np.take_along_axis(D, nb, 1)                           # n, m distances
    full = int(nd[:, :K].sum())
    sols = []
    lp = None
    if p_valid is not None:
        pv = np.clip(np.asarray(p_valid, float), 1e-6, 1 - 1e-6)
        lp = np.log1p(-pv) - np.log(pv)                         # log-odds of being invalid
    for r in range(0, max_invalid + 1):
        combos = np.array(list(itertools.combinations(range(n), r)), dtype=np.int16).reshape(-1, r) if r else np.zeros((1, 0), np.int16)
        for s in range(0, len(combos), chunk):
            c = combos[s:s + chunk]
            S = np.zeros((len(c), n), bool)
            if r:
                S[np.arange(len(c))[:, None], c] = True
            rem = S[:, nb]                                      # h, n, m neighbour removed?
            keep = ~rem
            first = keep & (np.cumsum(keep, 2) <= K)
            tot = ((first * nd[None]).sum(2) * ~S).sum(1)
            for h in np.where(tot == target)[0]:
                t = tuple(int(x) for x in c[h])
                sols.append((t, float(lp[list(t)].sum()) if lp is not None else 0.0))
    sols.sort(key=lambda x: (len(x[0]), -x[1]))
    return {"target": target, "all_valid": full, "solutions": sols}


def greedy_then_swap(Dpool, n=100, iters=200000, seed=0, weights=None, init=None, log=None):
    """Pick n items from a pool (Hamming matrix Dpool) maximising the 10-NN score.

    Farthest-point style greedy start, then annealed single swaps on the exact objective.
    """
    rng = np.random.RandomState(seed)
    P = len(Dpool)
    D = Dpool.astype(np.int32)
    if init is None:
        sel = [int(rng.randint(P))]
        # greedy: add the item maximising the sum of its K smallest distances to the chosen set
        while len(sel) < n:
            sub = D[:, sel]
            kk = min(K, len(sel))
            v = np.partition(sub, kk - 1, axis=1)[:, :kk].sum(1).astype(float) / kk
            v[sel] = -1
            sel.append(int(np.argmax(v)))
    else:
        sel = list(init)
    sel = np.array(sel)

    def f(s):
        return int_score_from_D(D[np.ix_(s, s)])

    cur = best = f(sel)
    best_sel = sel.copy()
    insel = np.zeros(P, bool); insel[sel] = True
    T0 = 4.0
    for it in range(iters):
        T = T0 * (1 - it / iters) + 1e-9
        i = rng.randint(n)
        j = rng.randint(P)
        if insel[j]:
            continue
        old = sel[i]
        sel[i] = j
        new = f(sel)
        if new >= cur or rng.rand() < np.exp((new - cur) / T):
            insel[old], insel[j] = False, True
            cur = new
            if cur > best:
                best, best_sel = cur, sel.copy()
        else:
            sel[i] = old
        if log and it % 20000 == 0:
            log(f"swap it={it} cur={cur/DENOM:.4f} best={best/DENOM:.4f}")
    return best_sel, best
