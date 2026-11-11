"""Loadable surrogate ensembles (CNN folds on GPU, LightGBM folds on CPU)."""
import os, pickle
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from common import ROOT


class Block(nn.Module):
    def __init__(s, w):
        super().__init__()
        s.c1, s.c2 = nn.Conv2d(w, w, 3, padding=1), nn.Conv2d(w, w, 3, padding=1)
        s.n1, s.n2 = nn.BatchNorm2d(w), nn.BatchNorm2d(w)

    def forward(s, x):
        h = F.gelu(s.n1(s.c1(x)))
        return F.gelu(x + s.n2(s.c2(h)))


class Net(nn.Module):
    def __init__(s, w=128, blocks=6):
        super().__init__()
        s.inp = nn.Conv2d(5, w, 3, padding=1)
        s.blocks = nn.Sequential(*[Block(w) for _ in range(blocks)])
        s.head = nn.Sequential(nn.Linear(3 * w, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(s, g):
        x = F.one_hot(g.long(), 5).permute(0, 3, 1, 2).float()
        x = s.blocks(F.gelu(s.inp(x)))
        x = torch.cat([x.mean((2, 3)), x.amax((2, 3)), x.amin((2, 3))], 1)
        return s.head(x).squeeze(1)


class CNNEns:
    """augs: per-advisor aug flag (which trained variant to load)."""

    def __init__(s, tag="cnn0", augs=(1, 1, 1, 1), dev=None, width=128, blocks=6, folds=5):
        s.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        s.models = []
        for a in range(4):
            ms = []
            for k in range(folds):
                m = Net(width, blocks).to(s.dev)
                m.load_state_dict(torch.load(os.path.join(ROOT, "ckpt", tag, f"a{a}_aug{augs[a]}_f{k}.pt"),
                                             map_location=s.dev))
                ms.append(m.eval())
            s.models.append(ms)

    @torch.no_grad()
    def predict_t(s, g, bs=32768):
        """g: torch (B,7,7) on device -> (B,4,folds) tensor."""
        out = []
        for i in range(0, len(g), bs):
            gb = g[i:i + bs]
            out.append(torch.stack([torch.stack([m(gb) for m in ms], 1) for ms in s.models], 1))
        return torch.cat(out)

    def predict(s, g):
        return s.predict_t(torch.as_tensor(np.asarray(g), device=s.dev)).cpu().numpy()


class GBMEns:
    def __init__(s, tag="gbm0", fset="all"):
        s.d = [pickle.load(open(os.path.join(ROOT, "ckpt", tag, f"cv_a{a}_{fset}.pkl"), "rb")) for a in range(4)]

    def predict(s, g, procs=8):
        """-> (B,4,5)"""
        from features import features_parallel
        X = features_parallel(np.asarray(g), procs=procs)
        P = np.zeros((len(g), 4, 5), np.float32)
        for a, d in enumerate(s.d):
            Xa = X[:, d["cols"]]
            for k, m in enumerate(d["models"]):
                P[:, a, k] = m.predict(Xa)
        return P


def _feat2_chunk(g):
    from features2 import features_inv2
    return features_inv2(g)


class GBMEnsZ:
    """LightGBM on invariant features v1+v2, predicting z (inverse-normal of the score)."""

    def __init__(s, tag="gbmz"):
        s.d = [pickle.load(open(os.path.join(ROOT, "ckpt", tag, f"cv_a{a}.pkl"), "rb")) for a in range(4)]

    def predict(s, g, procs=1):
        g = np.asarray(g)
        if procs > 1 and len(g) > 2000:
            from multiprocessing import Pool
            with Pool(procs) as p:
                X = np.concatenate(p.map(_feat2_chunk, [g[i:i + 2000] for i in range(0, len(g), 2000)]))
        else:
            X = _feat2_chunk(g)
        P = np.zeros((len(g), 4, 5), np.float32)
        for a, d in enumerate(s.d):
            for k, m in enumerate(d["models"]):
                P[:, a, k] = m.predict(X)
        return P


def load_surrogates(blend_tag):
    """-> cfg, cnn ensemble, gbm ensemble, threshold in the model's output space"""
    import json
    cfg = json.load(open(os.path.join(ROOT, "results", f"blend_{blend_tag}.json")))
    cnn = CNNEns(cfg["cnn"], cfg["augs"])
    gbm = GBMEnsZ(cfg["gbm"]) if cfg.get("space") == "z" else GBMEns(cfg["gbm"])
    return cfg, cnn, gbm, cfg.get("thresh", 0.75)
