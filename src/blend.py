"""Blend GBM + CNN surrogates on out-of-fold predictions, estimate error near the
threshold, and rank all dataset grids by estimated probability of being valid.

  python -u src/blend.py --gbm gbm0 --cnn cnn0 --augs 1,1,1,1 --tag b0
"""
import argparse, json, os, pickle
import numpy as np
from scipy.stats import norm
from sklearn.metrics import r2_score

from common import ROOT, ADVISORS, THRESH, load_data, atomic_save


def blend_fit(gbm, cnn, augs, fset="all"):
    """Returns per-advisor dict(w, b, sigma) fitted on OOF predictions."""
    out = []
    for a in range(4):
        d = pickle.load(open(os.path.join(ROOT, "ckpt", gbm, f"cv_a{a}_{fset}.pkl"), "rb"))
        y, og = d["y"], d["oof"]
        oc = np.load(os.path.join(ROOT, "ckpt", cnn, f"a{a}_aug{augs[a]}_oof.npy"))
        A = np.stack([og, oc, np.ones_like(og)], 1)
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        pb = A @ coef
        hi = pb >= 0.65                                   # error where it matters
        res = y[hi] - pb[hi]
        st = dict(w_gbm=float(coef[0]), w_cnn=float(coef[1]), b=float(coef[2]),
                  r2=float(r2_score(y, pb)), rmse=float(np.sqrt(np.mean((y - pb) ** 2))),
                  sigma=float(np.sqrt(np.mean(res ** 2))), bias_hi=float(res.mean()),
                  q01=float(np.quantile(res, 0.01)), q05=float(np.quantile(res, 0.05)),
                  r2_gbm=float(r2_score(y, og)), r2_cnn=float(r2_score(y, oc)),
                  corr_resid=float(np.corrcoef(y - og, y - oc)[0, 1]))
        for m in (0.75, 0.8, 0.85, 0.9):
            pv = pb >= m
            st[f"n>={m}"] = int(pv.sum()); st[f"prec@{m}"] = float((y[pv] >= THRESH).mean())
        out.append(st)
    return out


def blend_apply(B, Pg, Pc):
    """Pg, Pc: (n,4) fold-mean predictions -> blended (n,4)."""
    return np.stack([B[a]["w_gbm"] * Pg[:, a] + B[a]["w_cnn"] * Pc[:, a] + B[a]["b"] for a in range(4)], 1)


def p_valid(B, pred, extra_sigma=0.0):
    z = np.stack([(pred[:, a] - THRESH) / np.hypot(B[a]["sigma"], extra_sigma) for a in range(4)], 1)
    return norm.cdf(z).prod(1), norm.cdf(z)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gbm", default="gbm0"); p.add_argument("--cnn", default="cnn0")
    p.add_argument("--augs", default="1,1,1,1"); p.add_argument("--tag", default="b0")
    args = p.parse_args()
    augs = [int(x) for x in args.augs.split(",")]
    B = blend_fit(args.gbm, args.cnn, augs)
    for a in range(4):
        print(ADVISORS[a], json.dumps({k: round(v, 4) for k, v in B[a].items()}))
    json.dump(dict(gbm=args.gbm, cnn=args.cnn, augs=augs, blend=B),
              open(os.path.join(ROOT, "results", f"blend_{args.tag}.json"), "w"), indent=1)
    Pg = np.load(os.path.join(ROOT, "ckpt", args.gbm, "pred_all.npy")).mean(2)
    Pc = np.stack([np.mean([np.load(os.path.join(ROOT, "ckpt", args.cnn, f"a{a}_aug{augs[a]}_f{k}_all.npy"))
                            for k in range(5)], 0) for a in range(4)], 1)
    pred = blend_apply(B, Pg, Pc)
    pv, pa = p_valid(B, pred)
    atomic_save(os.path.join(ROOT, "cache", f"ds_pred_{args.tag}.npy"), pred.astype(np.float32))
    atomic_save(os.path.join(ROOT, "cache", f"ds_pvalid_{args.tag}.npy"), pv.astype(np.float32))
    print("dataset: blended min-advisor >= m:", {m: int((pred.min(1) >= m).sum()) for m in (0.75, 0.8, 0.85, 0.9)})
    print("  both models >= m on all advisors:", {m: int(((Pg.min(1) >= m) & (Pc.min(1) >= m)).sum()) for m in (0.75, 0.8, 0.85, 0.9)})
    print("  P(valid) >= p:", {q: int((pv >= q).sum()) for q in (0.5, 0.8, 0.9, 0.95, 0.99)})
    top = np.argsort(-pv)[:100]
    print("  top-100 by P(valid): expected valid %.1f, min p %.3f, weakest advisor counts %s" % (
        pv[top].sum(), pv[top].min(), np.bincount(pa[top].argmin(1), minlength=4).tolist()))
    print("  top-100 mean blended pred per advisor", pred[top].mean(0).round(3).tolist())
