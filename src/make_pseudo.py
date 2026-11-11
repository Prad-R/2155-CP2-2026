"""Pseudo-labels (z space) for unlabelled dataset grids from the z0 fold models.
cache/pseudo_<blend>.npz: idx (n,), P (n,4,5) = blended prediction of fold-k models (k-th teacher never saw fold k)."""
import argparse, json, os
import numpy as np
from common import ROOT, load_data
from surrogate import GBMEnsZ
p = argparse.ArgumentParser(); p.add_argument("--blend", default="z0"); p.add_argument("--n", type=int, default=100000); p.add_argument("--procs", type=int, default=8)
args = p.parse_args()
cfg = json.load(open(os.path.join(ROOT, "results", f"blend_{args.blend}.json"))); B = cfg["blend"]
grids, ratings = load_data()
unl = np.where(np.isnan(ratings).all(1))[0]
idx = np.sort(np.random.RandomState(0).choice(unl, args.n, replace=False))
Pg = GBMEnsZ(cfg["gbm"]).predict(grids[idx], procs=args.procs)
P = np.zeros_like(Pg)
for a in range(4):
    for k in range(5):
        pc = np.load(os.path.join(ROOT, "ckpt", cfg["cnn"], f"a{a}_aug1_f{k}_all.npy"), mmap_mode="r")[idx]
        P[:, a, k] = B[a]["w_gbm"] * Pg[:, a, k] + B[a]["w_cnn"] * pc + B[a]["b"]
np.savez(os.path.join(ROOT, "cache", f"pseudo_{args.blend}.tmp.npz"), idx=idx, P=P)
os.replace(os.path.join(ROOT, "cache", f"pseudo_{args.blend}.tmp.npz"), os.path.join(ROOT, "cache", f"pseudo_{args.blend}.npz"))
print("saved", P.shape, "mean z per advisor", P.mean((0, 2)).round(3), "share predicted valid per advisor", (P.mean(2) >= 0.6745).mean(0).round(3))
