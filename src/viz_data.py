"""Collect every submitted grid with what we know about it, for the design visualisations.
-> results/viz/designs.json   (run on a compute node; uses the z0 models for per-advisor predictions)"""
import json, os
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from common import ROOT
from divscore import hamming_matrix, K
from features import features, group_cols
from surrogate import load_surrogates
from blend import blend_apply

rows = {r["name"]: r for r in map(json.loads, open(os.path.join(ROOT, "results", "submission_log.jsonl")))}
FAILS = {"S1": 1, "S2": 0, "S3": 0, "S4": 2, "S5": 0, "S6": 2, "S7": 2, "S8": 0, "S9": 2, "S10": 4, "S11": 2, "S13": 4}
G, meta, seen = [], [], {}
def add(g, s, status, nfail, own):
    k = g.astype(np.int8).tobytes()
    if k in seen:
        meta[seen[k]]["sets"].append(s); return
    seen[k] = len(G); G.append(g.astype(np.int8).reshape(49)); meta.append(dict(set=s, sets=[s], status=status, nfail=nfail, co=int(own)))
def own_contrib(sub):
    D = hamming_matrix(sub).astype(np.int32); np.fill_diagonal(D, 10**6)
    return np.partition(D, K - 1, axis=1)[:, :K].sum(1)
order = ["S2", "S3", "S5", "S8", "S14", "S15", "S1", "S4", "S6", "S7", "S9", "S10", "S11", "S13"]      # confirmed sets first so duplicates keep the stronger tag
for s in order:
    sub = np.load(os.path.join(ROOT, rows[s]["file"])); oc = own_contrib(sub)
    if s in ("S14", "S15"):
        d = rows[s]["decode"]
        for i in range(92, 100):
            j = i - 92
            if s == "S14": st = "invalid" if j in d["failed_risky"] else "confirmed"
            else: st = "invalid" if j in d["certain_invalid_risky"] else "unresolved"
            add(sub[i], s, st, 0, oc[i])
        continue
    for i in range(100):
        add(sub[i], s, "confirmed" if FAILS[s] == 0 else "near", FAILS[s], oc[i])
G = np.array(G); n = len(G)
print("grids", n, {k: sum(m["status"] == k for m in meta) for k in ("confirmed", "near", "invalid", "unresolved")}, flush=True)
# pooled contribution among grids not known to be invalid
ok = np.array([m["status"] in ("confirmed", "near") for m in meta])
D = hamming_matrix(G).astype(np.int32); Dp = D.copy(); np.fill_diagonal(Dp, 10**6); Dp[:, ~ok] = 10**6
cp = np.partition(Dp, K - 1, axis=1)[:, :K].sum(1)
# model view (second generation)
cfg, cnn, gbm, TH = load_surrogates("z0"); B = cfg["blend"]; sig = np.array([b["sigma"] for b in B])
g3 = G.reshape(-1, 7, 7)
pred = blend_apply(B, gbm.predict(g3).mean(2), cnn.predict(g3).mean(2))
pa = norm.cdf((pred - TH) / sig); yhat = ndtr(pred)
# two notions of similarity
X1, groups = features(g3)
Xc = X1[:, group_cols(groups, ["count", "adj4"])].astype(np.float64); Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
Xl = (G[:, :, None] == np.arange(5)[None, None]).reshape(n, -1).astype(np.float64)
out = {}
for nm, X, kw in (("l", Xl, dict(metric="precomputed", init="random")), ("c", Xc, dict(init="pca"))):
    emb = TSNE(n_components=2, perplexity=35, random_state=0, **kw).fit_transform(D.astype(float) / 49 if nm == "l" else X)
    best = None
    for k in range(4, 9):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        s = silhouette_score(X, lab)
        print(nm, "k", k, "silhouette", round(float(s), 3), np.bincount(lab).tolist(), flush=True)
        if best is None or s > best[0] + 0.005: best = (s, lab)
    lab = best[1]; rank = np.argsort(np.argsort(-np.bincount(lab)))        # number clusters by size, 1 = largest
    emb = (emb - emb.min(0)) / (emb.max(0) - emb.min(0))
    out[nm] = (emb, rank[lab] + 1)
recs = []
for i in range(n):
    m = meta[i]
    recs.append(dict(g="".join(map(str, G[i])), set=m["set"], sets=m["sets"], st=m["status"], nf=m["nfail"], co=m["co"], cp=int(cp[i]) if ok[i] else None,
                     cnt=np.bincount(G[i], minlength=5).tolist(), y=[round(float(v), 3) for v in yhat[i]], pv=round(float(pa[i].prod()), 3), tight=int(pa[i].argmin()),
                     xl=round(float(out["l"][0][i, 0]), 4), yl=round(float(out["l"][0][i, 1]), 4), cl=int(out["l"][1][i]),
                     xc=round(float(out["c"][0][i, 0]), 4), yc=round(float(out["c"][0][i, 1]), 4), cc=int(out["c"][1][i])))
os.makedirs(os.path.join(ROOT, "results", "viz"), exist_ok=True)
json.dump(recs, open(os.path.join(ROOT, "results", "viz", "designs.json"), "w"), separators=(",", ":"))
print("S5 own-set contributions sum", sum(r["co"] for r in recs if r["set"] == "S5"), "(logged 36903)", flush=True)
