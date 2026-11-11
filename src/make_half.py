"""Split probe: locate the failures of a submitted set whose failure count is known.
Keeps part of the set, fills the rest with certified grids; the returned score, together with the
first score, narrows down which kept grids failed.
  python -u src/make_half.py --name S16 --set submissions/S13_x.npy --pv ckpt/z3/final.npz --obs 35821 --core submissions/S8_x.npy
"""
import argparse, hashlib, json, os, time
import numpy as np
from common import ROOT
from divscore import hamming_matrix, int_score, decode, DENOM
p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--set", required=True); p.add_argument("--pv", required=True)
p.add_argument("--obs", type=int, required=True); p.add_argument("--core", required=True)
p.add_argument("--maxfail", type=int, default=4); p.add_argument("--keep", type=int, default=50); p.add_argument("--tries", type=int, default=12)
p.add_argument("--exclude", default="", help="comma list of set indices already tested (kept in an earlier probe)")
p.add_argument("--hyp", default="", help="existing hypothesis file to reuse/filter"); p.add_argument("--purpose", default=""); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
S = np.load(os.path.join(ROOT, a.set)).astype(np.int8)
z = np.load(os.path.join(ROOT, a.pv)); _m = hamming_matrix(S, z["grids"]); assert (_m.min(1) == 0).all(); pv = z["pvalid"][_m.argmin(1)]   # submission order differs from the run order
core = np.load(os.path.join(ROOT, a.core)).astype(np.int8)
hf = a.hyp or os.path.join(ROOT, "results", "probes", f"{a.name}_hyp.json")
if os.path.exists(hf):
    H = json.load(open(hf))
else:
    d = decode(S, a.obs, max_invalid=a.maxfail, p_valid=pv)
    H = [dict(idx=list(s), lp=lp) for s, lp in d["solutions"]]
    json.dump(H, open(hf, "w"))
nf = np.bincount([len(h["idx"]) for h in H]); print(f"{len(H)} failure sets reproduce the first score; by size {nf.tolist()}", flush=True)
lp = np.array([h["lp"] for h in H]); w = np.exp(lp - lp.max()); w /= w.sum()
marg = np.zeros(100)
for h, wi in zip(H, w): marg[h["idx"]] += wi
print("most suspected grids (index: posterior):", [(int(i), round(float(marg[i]), 2)) for i in np.argsort(-marg)[:12]], flush=True)
rng = np.random.RandomState(a.seed); excl = [int(x) for x in a.exclude.split(",") if x]
avail = np.setdiff1d(np.arange(100), excl)
best = None
for t in range(a.tries):
    # keep the most suspected grids spread at random between halves
    o = avail[np.argsort(-(marg[avail] + 0.02 * rng.rand(len(avail))))]
    K = np.sort(np.concatenate([o[t % 2::2][:a.keep]]) if len(avail) > a.keep else avail)
    okc = np.where(hamming_matrix(core, S[K]).min(1) > 0)[0]
    C = rng.choice(okc, 100 - len(K), replace=False)
    sub = np.concatenate([S[K], core[C]]).astype(int)
    pos = {int(k): i for i, k in enumerate(K)}
    out = {}
    for h, wi in zip(H, w):
        f = tuple(sorted(pos[i] for i in h["idx"] if i in pos))
        if f not in out:
            v = np.ones(100, bool); v[list(f)] = False
            out[f] = [int_score(sub, v), 0.0]
        out[f][1] += wi
    sc = {}
    for f, (s, wi) in out.items(): sc.setdefault(s, []).append((f, wi))
    # expected probability that the probe pins down the kept failures uniquely
    uniq = sum(max(x[1] for x in v) for v in sc.values())
    if best is None or uniq > best[0]: best = (uniq, K, C, sub, sc, len(out))
uniq, K, C, sub, sc, nout = best
assert len(np.unique(sub.reshape(100, -1), axis=0)) == 100
ts = time.strftime("%Y%m%d_%H%M%S"); fn = os.path.join("submissions", f"{a.name}_{ts}.npy"); np.save(os.path.join(ROOT, fn), sub); Sall = int_score(sub)
json.dump(dict(file=fn, set=a.set, kept=K.tolist(), core=a.core, core_idx=C.tolist(), hyp=hf,
               table={str(s): [dict(failed_set_idx=[int(K[i]) for i in f], weight=wi) for f, wi in v] for s, v in sc.items()}),
          open(os.path.join(ROOT, "results", "probes", a.name + ".json"), "w"))
pvs = [round(float(x), 5) for x in pv[K]] + [1.0] * len(C)
entry = dict(name=a.name, file=fn, sha=hashlib.sha1(sub.astype(np.int8).tobytes()).hexdigest()[:8], time=ts, source=f"split probe: {a.set} kept {len(K)} + {a.core}", pmin=0, pool=100,
             expected_int=Sall, expected_score=round(Sall / DENOM, 10), expected_valid=float(sum(pvs)), min_pvalid=float(min(pvs)), dataset_ids=[-1] * 100, pvalid=pvs, purpose=a.purpose, returned_score=None, decode=None)
open(os.path.join(ROOT, "results", "submission_log.jsonl"), "a").write(json.dumps(entry) + "\n")
print(f"saved {fn} all-valid {Sall/DENOM:.10f} (int {Sall}); {nout} possible outcomes for the kept grids, {len(sc)} distinct scores; chance the score pins them down: {uniq:.2f}", flush=True)
