"""Build 100-grid sets from confirmed-valid grids only (results/viz/designs.json).

Strategies: farthest-point picks inside each cluster (character / layout clusters,
proportional or equal quotas), global farthest-point, and exact swap local search
(replace a set member by a pool grid whenever the exact 10-NN score rises).
A set made only of confirmed-valid grids scores exactly what is computed here.

  python -u src/pick_valid.py            -> results/pick_valid.json, results/pick_valid/<name>.npy
"""
import json, os, sys
import numpy as np
from common import ROOT
from divscore import hamming_matrix, int_score, DENOM, K

status = sys.argv[1].split(",") if len(sys.argv) > 1 else ["confirmed"]
d = [x for x in json.load(open(os.path.join(ROOT, "results/viz/designs.json"))) if x["st"] in status]
P = np.array([[int(c) for c in x["g"]] for x in d], dtype=np.int8)
cc = np.array([x["cc"] for x in d]); cl = np.array([x["cl"] for x in d])
sets = [x["set"] for x in d]
allsets = [x["sets"] for x in d]
N = len(P)
D = hamming_matrix(P).astype(np.int32)
print(f"pool {N} grids ({'+'.join(status)}); per set {dict(zip(*np.unique(sets, return_counts=True)))}", flush=True)
print("character clusters", dict(zip(*np.unique(cc, return_counts=True))), "layout clusters", dict(zip(*np.unique(cl, return_counts=True))), flush=True)


def score(idx):
    return int_score(P[idx].reshape(-1, 7, 7))


def farthest(cand, k, seed=0):
    """greedy maximin inside cand (indices); start from the grid farthest from the rest on average"""
    cand = np.asarray(cand)
    if k <= 0:
        return []
    if k >= len(cand):
        return list(cand)
    sub = D[np.ix_(cand, cand)]
    rng = np.random.RandomState(seed)
    first = int(np.argmax(sub.mean(1))) if seed == 0 else int(rng.randint(len(cand)))
    pick = [first]
    md = sub[first].copy()
    for _ in range(k - 1):
        md[pick] = -1
        j = int(np.argmax(md))
        pick.append(j)
        md = np.minimum(md, sub[j])
    return list(cand[pick])


def quotas(lab, mode):
    u, n = np.unique(lab, return_counts=True)
    q = n * 100.0 / n.sum() if mode == "prop" else np.full(len(u), 100.0 / len(u))
    q = np.minimum(np.floor(q).astype(int), n)
    while q.sum() < 100:                       # hand out the remainder to clusters with room
        room = np.where(q < n)[0]
        q[room[np.argmax((n - q)[room])]] += 1
    return dict(zip(u, q))


def per_cluster(lab, mode):
    out = []
    for c, k in quotas(lab, mode).items():
        out += farthest(np.where(lab == c)[0], int(k))
    return out


def local_search(idx, name):
    """steepest exact swap search; returns improved index list"""
    idx = list(idx)
    cur = score(idx)
    improved = True
    rnd = 0
    while improved:
        improved = False
        rnd += 1
        for pos in np.random.RandomState(rnd).permutation(100):
            inset = np.zeros(N, bool); inset[idx] = True
            out = np.where(~inset)[0]
            Ds = D[np.ix_(idx, idx)].copy()
            np.fill_diagonal(Ds, 10**6)
            rows = D[np.ix_(out, idx)].copy()          # m,100
            rows[:, pos] = 10**6
            Dc = np.repeat(Ds[None], len(out), 0)
            Dc[:, pos, :] = rows
            Dc[:, :, pos] = rows
            sc = np.partition(Dc, K - 1, axis=2)[:, :, :K].sum((1, 2))
            j = int(np.argmax(sc))
            if sc[j] > cur:
                idx[pos], cur, improved = int(out[j]), int(sc[j]), True
        print(f"  {name}: round {rnd} score {cur/DENOM:.6f}", flush=True)
    assert score(idx) == cur
    return idx


starts = {
    "char_prop": per_cluster(cc, "prop"),
    "char_equal": per_cluster(cc, "equal"),
    "layout_prop": per_cluster(cl, "prop"),
    "layout_equal": per_cluster(cl, "equal"),
    "global_far": farthest(np.arange(N), 100),
}
for s in sorted({t for a in allsets for t in a}):
    m = [i for i, a in enumerate(allsets) if s in a]
    if len(m) == 100:
        starts["set_" + s] = m
os.makedirs(os.path.join(ROOT, "results/pick_valid"), exist_ok=True)
res = {}
for name, idx in starts.items():
    assert len(set(idx)) == 100, len(idx)
    s0 = score(idx)
    print(f"{name}: picked score {s0/DENOM:.6f}", flush=True)
    idx2 = local_search(idx, name)
    s1 = score(idx2)
    src = dict(zip(*np.unique([sets[i] for i in idx2], return_counts=True)))
    res[name] = {"picked": s0 / DENOM, "picked_int": s0, "after_swaps": s1 / DENOM, "after_swaps_int": s1,
                 "sources": {k: int(v) for k, v in src.items()},
                 "char_clusters": {int(k): int(v) for k, v in zip(*np.unique(cc[idx2], return_counts=True))}}
    np.save(os.path.join(ROOT, "results/pick_valid", f"{'_'.join(status)}_{name}.npy"), P[idx2].reshape(100, 7, 7))
    print(f"{name}: after swaps {s1/DENOM:.10f} (int {s1}) sources {src}", flush=True)
json.dump(res, open(os.path.join(ROOT, "results", f"pick_valid_{'_'.join(status)}.json"), "w"), indent=1)
