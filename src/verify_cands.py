"""Score search candidates with the independent GBM ensemble, blend with the CNN
predictions saved by search.py, and write a candidate file for make_submission.py.

  python -u src/verify_cands.py --search s0 --blend b0 --out cache/cands_s0.npz
"""
import argparse, glob, json, os
import numpy as np

from common import ROOT, load_data
from surrogate import GBMEns
from blend import blend_apply, p_valid
from divscore import hamming_matrix

p = argparse.ArgumentParser()
p.add_argument("--search", default="s0")
p.add_argument("--blend", default="b0")
p.add_argument("--out", default="")
p.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
args = p.parse_args()

cfg = json.load(open(os.path.join(ROOT, "results", f"blend_{args.blend}.json")))
B = cfg["blend"]
files = sorted(glob.glob(os.path.join(ROOT, "ckpt", args.search, "round_*.npz")))
Z = [np.load(f) for f in files]
G = np.concatenate([z["grids"] for z in Z]); Mc = np.concatenate([z["mean"] for z in Z]); Sc = np.concatenate([z["std"] for z in Z])
_, ui = np.unique(G.reshape(len(G), -1), axis=0, return_index=True)
G, Mc, Sc = G[ui], Mc[ui], Sc[ui]
print(f"{len(files)} rounds, {len(G)} unique candidates", flush=True)

Pg = GBMEns(cfg["gbm"]).predict(G, procs=args.procs)          # n,4,5
Mg, Sg = Pg.mean(2), Pg.std(2)
pred = blend_apply(B, Mg, Mc)
pv, pa = p_valid(B, pred)
both_min = np.minimum(Mg, Mc).min(1)                           # weakest advisor under the more pessimistic model
print("CNN mean per advisor", Mc.mean(0).round(3), "GBM mean per advisor", Mg.mean(0).round(3))
print("GBM min-advisor >= m:", {m: int((Mg.min(1) >= m).sum()) for m in (0.75, 0.8, 0.85, 0.9)})
print("both models min >= m:", {m: int((both_min >= m).sum()) for m in (0.75, 0.8, 0.85, 0.9)})
print("P(valid) >= p:", {q: int((pv >= q).sum()) for q in (0.5, 0.9, 0.95, 0.99, 0.999)})
print("weakest advisor (blend) counts:", np.bincount(pa.argmin(1), minlength=4).tolist())
# distance to the nearest dataset grid among the model-valid dataset grids (how far out of distribution)
grids, _ = load_data()
dsp = np.load(os.path.join(ROOT, "cache", f"ds_pvalid_{args.blend}.npy"))
ref = grids[np.argsort(-dsp)[:2000]]
dmin = np.concatenate([hamming_matrix(G[s:s + 2000], ref).min(1) for s in range(0, len(G), 2000)])
print("Hamming distance to nearest of top-2000 dataset grids: quantiles", np.quantile(dmin, [0, .25, .5, .75, 1]).tolist())
out = args.out or os.path.join("cache", f"cands_{args.search}.npz")
np.savez(os.path.join(ROOT, out), grids=G, pvalid=pv.astype(np.float32), pred=pred.astype(np.float32),
         gbm=Mg, cnn=Mc, gbm_std=Sg, cnn_std=Sc, both_min=both_min, dmin=dmin)
print("saved", out)
