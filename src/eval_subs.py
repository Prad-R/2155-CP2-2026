"""Check blend configs against what the leaderboard told us about each submission."""
import glob, json, os, sys
import numpy as np
from scipy.stats import norm
from common import ROOT
from surrogate import load_surrogates
from blend import blend_apply
rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "submission_log.jsonl"))]
fails = {"S1": 1, "S2": 0, "S3": 0, "S4": 2, "S5": 0, "S6": 2, "S7": 2, "S8": 0, "S9": 2, "S10": 4}
for tag in sys.argv[1:]:
    cfg, cnn, gbm, TH = load_surrogates(tag); B = cfg["blend"]; sig = np.array([b["sigma"] for b in B])
    print(f"== blend {tag}: per submission: failed (true) | expected failures | grids P<0.95 | P<0.5 | min P | weakest advisor counts W/T/Tr/B")
    for r in rows:
        g = np.load(os.path.join(ROOT, r["file"]))
        pred = blend_apply(B, gbm.predict(g).mean(2), cnn.predict(g).mean(2))
        pa = norm.cdf((pred - TH) / sig); pv = pa.prod(1)
        print(f"  {r['name']:4s} {fails.get(r['name'],'?')} | {(1-pv).sum():5.1f} | {(pv<0.95).sum():3d} | {(pv<0.5).sum():3d} | {pv.min():.3f} | {np.bincount(pa.argmin(1), minlength=4).tolist()}", flush=True)
