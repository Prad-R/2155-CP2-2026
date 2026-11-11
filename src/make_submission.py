"""Build a 100-grid submission from a candidate pool, maximising the 10-NN diversity
score, and log it (results/submission_log.jsonl).

  python -u src/make_submission.py --name S1 --source dataset --blend b0 --pmin 0.99 --purpose "..."
  python -u src/make_submission.py --name S2 --source cache/cands_x.npz --pmin 0.99 --purpose "..."

A candidate .npz must hold `grids` (n,7,7) and `pvalid` (n,).
"""
import argparse, hashlib, json, os, time
import numpy as np

from common import ROOT, load_data
from divscore import hamming_matrix, greedy_then_swap, int_score, DENOM

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--source", default="dataset")
p.add_argument("--blend", default="b0")
p.add_argument("--pmin", type=float, default=0.99)
p.add_argument("--pmax", type=float, default=2.0)
p.add_argument("--maxpool", type=int, default=6000)
p.add_argument("--iters", type=int, default=150000)
p.add_argument("--exclude", default="", help="comma list of earlier submission .npy files whose grids are excluded")
p.add_argument("--purpose", default="")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
log = lambda *a: print(*a, flush=True)

if args.source == "dataset":
    grids, _ = load_data()
    pv = np.load(os.path.join(ROOT, "cache", f"ds_pvalid_{args.blend}.npy"))
    ids = np.arange(len(grids))
else:
    z = np.load(os.path.join(ROOT, args.source))
    grids, pv = z["grids"], z["pvalid"]
    ids = -np.ones(len(grids), int)

keep = np.where((pv >= args.pmin) & (pv < args.pmax))[0]
keep = keep[np.argsort(-pv[keep])][:args.maxpool]
G = grids[keep].astype(np.int8)
_, ui = np.unique(G.reshape(len(G), -1), axis=0, return_index=True)
ui.sort(); keep, G = keep[ui], G[ui]
if args.exclude:
    ex = np.concatenate([np.load(f) for f in args.exclude.split(",")]).reshape(-1, 49)
    dup = (hamming_matrix(G, ex) == 0).any(1)
    keep, G = keep[~dup], G[~dup]
log(f"pool: {len(G)} candidates with pvalid >= {args.pmin}")
assert len(G) >= 100, "pool too small"

D = hamming_matrix(G)
sel, best = greedy_then_swap(D, 100, iters=args.iters, seed=args.seed, log=log)
sub = G[sel].astype(int)
assert sub.shape == (100, 7, 7) and sub.dtype == int and sub.min() >= 0 and sub.max() <= 4
assert len(np.unique(sub.reshape(100, -1), axis=0)) == 100
S = int_score(sub)
assert S == best
ts = time.strftime("%Y%m%d_%H%M%S")
fn = os.path.join("submissions", f"{args.name}_{ts}.npy")
np.save(os.path.join(ROOT, fn), sub)
h = hashlib.sha1(sub.astype(np.int8).tobytes()).hexdigest()[:8]
entry = dict(name=args.name, file=fn, sha=h, time=ts, source=args.source, pmin=args.pmin, pool=len(G),
             expected_int=S, expected_score=round(S / DENOM, 10), expected_valid=float(pv[keep[sel]].sum()),
             min_pvalid=float(pv[keep[sel]].min()), dataset_ids=ids[keep[sel]].tolist(),
             pvalid=[round(float(x), 5) for x in pv[keep[sel]]],
             purpose=args.purpose, returned_score=None, decode=None)
with open(os.path.join(ROOT, "results", "submission_log.jsonl"), "a") as f:
    f.write(json.dumps(entry) + "\n")
log(f"saved {fn} sha={h} all-valid score={S/DENOM:.10f} (int {S}) expected #valid={entry['expected_valid']:.1f}")
