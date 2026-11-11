"""Blend z-space GBM + CNN on OOF predictions -> results/blend_<tag>.json (space = z)."""
import argparse, json, os, pickle
import numpy as np
from scipy.special import ndtr
from sklearn.metrics import r2_score
from common import ROOT, ADVISORS

p = argparse.ArgumentParser()
p.add_argument("--gbm", default="gbmz"); p.add_argument("--cnn", default="cnnz"); p.add_argument("--tag", default="z0")
args = p.parse_args()
ZT = 0.6744897501960817          # z of 0.75
B = []
for a in range(4):
    d = pickle.load(open(os.path.join(ROOT, "ckpt", args.gbm, f"cv_a{a}.pkl"), "rb")); z, y, og = d["z"], d["y"], d["oof"]
    oc = np.load(os.path.join(ROOT, "ckpt", args.cnn, f"a{a}_aug1_oof.npy"))
    A = np.stack([og, oc, np.ones_like(og)], 1); co = np.linalg.lstsq(A, z, rcond=None)[0]; pb = A @ co
    hi = pb >= 0.385; res = z[hi] - pb[hi]
    pv = pb >= ZT
    st = dict(w_gbm=float(co[0]), w_cnn=float(co[1]), b=float(co[2]), sigma=float(np.sqrt(np.mean(res ** 2))), r2_z=float(r2_score(z, pb)),
              r2_y=float(r2_score(y, ndtr(pb))), r2_y_gbm=float(r2_score(y, ndtr(og))), r2_y_cnn=float(r2_score(y, ndtr(oc))),
              sigma_y_hi=float(np.sqrt(np.mean((y[hi] - ndtr(pb[hi])) ** 2))), corr_resid=float(np.corrcoef(z - og, z - oc)[0, 1]),
              prec_at_thresh=float((y[pv] >= 0.75).mean()), recall_at_thresh=float(pv[y >= 0.75].mean()))
    for m in (1.0, 1.5, 2.0):
        s_ = pb >= ZT + m * st["sigma"]; st[f"n_{m}sig"] = int(s_.sum()); st[f"prec_{m}sig"] = float((y[s_] >= 0.75).mean())
    B.append(st); print(ADVISORS[a], json.dumps({k: round(v, 4) for k, v in st.items()}))
json.dump(dict(space="z", thresh=ZT, gbm=args.gbm, cnn=args.cnn, augs=[1, 1, 1, 1], blend=B), open(os.path.join(ROOT, "results", f"blend_{args.tag}.json"), "w"), indent=1)
