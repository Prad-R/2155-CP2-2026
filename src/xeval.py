"""Score finished sets under several model generations.
  python -u src/xeval.py z1,z2,z3,z4 b0,z0   -> results/xeval.json + per-grid results/xeval_<tag>.npz
"""
import json, os, sys
import numpy as np
from scipy.stats import norm
from common import ROOT
from surrogate import load_surrogates
from blend import blend_apply
from divscore import int_score, DENOM
tags, blends = sys.argv[1].split(","), sys.argv[2].split(",")
G = {t: np.load(os.path.join(ROOT, "ckpt", t, "final.npz"))["grids"].astype(np.int8) for t in tags}
out, per = {}, {t: {} for t in tags}
for b in blends:
    cfg, cnn, gbm, TH = load_surrogates(b); B = cfg["blend"]; sig = np.array([x["sigma"] for x in B])
    for t in tags:
        pred = blend_apply(B, gbm.predict(G[t]).mean(2), cnn.predict(G[t]).mean(2))
        pa = norm.cdf((pred - TH) / sig); pv = pa.prod(1)
        per[t][b] = pv
        out.setdefault(t, {"score": int_score(G[t]) / DENOM})[b] = dict(exp_fail=float((1 - pv).sum()), below95=int((pv < .95).sum()), below80=int((pv < .8).sum()), below50=int((pv < .5).sum()), minp=float(pv.min()), weakest=np.bincount(pa.argmin(1), minlength=4).tolist())
        print(t, b, f"score {out[t]['score']:.4f}", out[t][b], flush=True)
for t in tags:
    np.savez(os.path.join(ROOT, "results", f"xeval_{t}.npz"), **per[t])
json.dump(out, open(os.path.join(ROOT, "results", "xeval.json"), "w"), indent=1)
