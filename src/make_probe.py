"""Targeted labelling: certified set with n grids replaced by risky ones; every outcome has a unique score.
  python -u src/make_probe.py --name S14 --core submissions/S8_x.npy --risky ckpt/z3/final.npz --lo 0 --hi 0.95 --n 10
Writes submissions/<name>_<ts>.npy, results/probes/<name>.json (score -> failed risky grids), and a log entry.
"""
import argparse, hashlib, itertools, json, os, time
import numpy as np
from common import ROOT
from divscore import hamming_matrix, int_score, DENOM
p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--core", required=True); p.add_argument("--risky", required=True)
p.add_argument("--lo", type=float, default=0.0); p.add_argument("--hi", type=float, default=0.95)
p.add_argument("--n", type=int, default=10); p.add_argument("--skip", type=int, default=0, help="offset into the risky list (for a second, disjoint probe)")
p.add_argument("--pfile", default=""); p.add_argument("--pkey", default="b0"); p.add_argument("--lowest", action="store_true"); p.add_argument("--tries", type=int, default=60)
p.add_argument("--purpose", default=""); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
core = np.load(os.path.join(ROOT, a.core)).astype(np.int8)
z = np.load(os.path.join(ROOT, a.risky)); rg = z["grids"].astype(np.int8)
rp = np.load(os.path.join(ROOT, a.pfile))[a.pkey] if a.pfile else z["pvalid"]
cand = np.where((rp >= a.lo) & (rp < a.hi))[0]; cand = cand[np.argsort(rp[cand])]
pick = cand[a.skip:][:a.n] if a.lowest else cand[a.skip::max((len(cand) - a.skip) // a.n, 1)][:a.n]
assert len(pick) == a.n, (len(cand), len(pick))
assert (hamming_matrix(rg[pick], core) > 0).all()
rng = np.random.RandomState(a.seed)
bestc = None
for attempt in range(a.tries):
    # which core grids to drop: random, biased to those closest to the new grids
    d = hamming_matrix(core, rg[pick]).min(1) + rng.rand(len(core)) * (2 * attempt)
    drop = np.argsort(d)[:a.n]
    keepc = np.setdiff1d(np.arange(len(core)), drop)
    sub = np.concatenate([core[keepc], rg[pick]]).astype(int)
    table = {}
    for r in range(a.n + 1):
        for c in itertools.combinations(range(a.n), r):
            v = np.ones(100, bool); v[[100 - a.n + i for i in c]] = False
            table.setdefault(int_score(sub, v), []).append(list(c))
    ncol = 2 ** a.n - len(table)
    if bestc is None or ncol < bestc[0]:
        bestc = (ncol, keepc, sub, table)
    if ncol == 0:
        break
ncol, keepc, sub, table = bestc
amb = [v for v in table.values() if len(v) > 1]
print(f"{ncol} collisions; ambiguous scores {len(amb)}, largest group {max([len(v) for v in amb], default=1)}; ambiguous among <=2 failures: {sum(1 for v in amb if sum(len(c) <= 2 for c in v) > 1)}", flush=True)
assert len(np.unique(sub.reshape(100, -1), axis=0)) == 100 and sub.min() >= 0 and sub.max() <= 4
ts = time.strftime("%Y%m%d_%H%M%S"); fn = os.path.join("submissions", f"{a.name}_{ts}.npy")
np.save(os.path.join(ROOT, fn), sub); S = int_score(sub)
json.dump(dict(file=fn, core=a.core, core_kept=keepc.tolist(), risky=a.risky, risky_idx=pick.tolist(), risky_pvalid=rp[pick].tolist(),
               table={str(k): v for k, v in table.items()}), open(os.path.join(ROOT, "results", "probes", a.name + ".json"), "w"))
pv = [1.0] * 90 + [round(float(x), 5) for x in rp[pick]]
entry = dict(name=a.name, file=fn, sha=hashlib.sha1(sub.astype(np.int8).tobytes()).hexdigest()[:8], time=ts, source=f"probe: {a.core} + {a.risky}", pmin=a.lo, pool=100,
             expected_int=S, expected_score=round(S / DENOM, 10), expected_valid=float(sum(pv)), min_pvalid=float(min(pv)), dataset_ids=[-1] * 100, pvalid=pv,
             purpose=a.purpose, returned_score=None, decode=None)
open(os.path.join(ROOT, "results", "submission_log.jsonl"), "a").write(json.dumps(entry) + "\n")
print(f"saved {fn} all-valid {S/DENOM:.10f} (int {S}); risky pvalid {np.round(rp[pick],3).tolist()}; all-10-fail score {min(table)/DENOM:.4f}; {len(table)} distinct outcomes", flush=True)
