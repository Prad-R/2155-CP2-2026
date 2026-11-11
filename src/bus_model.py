"""Interim Business model: un-augmented sum-pooled CNNs (the advisor is NOT rotation/flip symmetric) + free count function.
Target = level index of the Business label (central labels).  A grid passes when the ensemble prediction is at least
THRESH_LEVEL + margin.   Train:  python -u src/bus_model.py train
"""
import os, sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

from common import ROOT, load_data, folds_for

OUT = os.path.join(ROOT, "ckpt", "busx")


class SumNet(nn.Module):
    def __init__(s, nconv=1, w=96, k=3):
        super().__init__()
        layers, c = [], 5
        for i in range(nconv):
            layers += [nn.Conv2d(c, w, k, padding=k // 2), nn.GELU()]; c = w
        layers += [nn.Conv2d(c, w, 1), nn.GELU(), nn.Conv2d(w, w, 1), nn.GELU(), nn.Conv2d(w, 1, 1)]
        s.f = nn.Sequential(*layers)
        s.cnt = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        s.b = nn.Parameter(torch.zeros(1))

    def forward(s, g):
        x = F.one_hot(g.long(), 5).permute(0, 3, 1, 2).float()
        return s.f(x).sum((1, 2, 3)) / 7 + s.b + s.cnt(x.sum((2, 3)) / 10).squeeze(1)


CFGS = [dict(nconv=1), dict(nconv=2)]


def levels():
    G, Y = load_data()
    idx = np.where(~np.isnan(Y[:, 3]))[0]
    y = Y[idx, 3]
    vals, lev = np.unique(y, return_inverse=True)
    thr = int(np.searchsorted(vals, 0.75))                      # first level whose label >= 0.75
    return G[idx], y, lev, thr


class BusEns:
    def __init__(s, dev=None):
        s.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        meta = json.load(open(os.path.join(OUT, "meta.json")))
        s.mu, s.sd, s.thr, s.sigma = meta["mu"], meta["sd"], meta["thr_level"], meta["oof_sd"]
        s.models = []
        for c, kw in enumerate(CFGS):
            for k in range(5):
                m = SumNet(**kw).to(s.dev)
                m.load_state_dict(torch.load(os.path.join(OUT, f"c{c}_f{k}.pt"), map_location=s.dev))
                s.models.append(m.eval())

    @torch.no_grad()
    def predict(s, g):
        """-> (mean level, sd across the ensemble) for grids (n,7,7) or (n,49)"""
        gt = torch.as_tensor(np.asarray(g).reshape(-1, 7, 7), device=s.dev)
        p = torch.stack([m(gt) for m in s.models]) * s.sd + s.mu
        return p.mean(0).cpu().numpy(), p.std(0).cpu().numpy()


def train():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g, y, lev, thr = levels()
    cen = (y > 0.03) & (y < 0.97)
    gc, lc = g[cen], lev[cen].astype(np.float64)
    mu, sd = lc.mean(), lc.std()
    t = torch.as_tensor((lc - mu) / sd, dtype=torch.float32, device=dev)
    Gt = torch.as_tensor(gc, device=dev)
    f = folds_for(len(gc), 5, 0)
    oof = np.zeros((len(CFGS), len(gc)))
    for c, kw in enumerate(CFGS):
        for k in range(5):
            tr, te = torch.as_tensor(f != k, device=dev), torch.as_tensor(f == k, device=dev)
            torch.manual_seed(100 * c + k)
            m = SumNet(**kw).to(dev)
            epochs, bs, lr = 400, 256, 2e-3
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)
            idx = torch.where(tr)[0]; n = len(idx); steps = epochs * (n // bs)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
            m.train()
            for ep in range(epochs):
                perm = idx[torch.randperm(n, device=dev)]
                for i in range(n // bs):
                    b = perm[i * bs:(i + 1) * bs]
                    loss = F.mse_loss(m(Gt[b]), t[b])
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            m.eval()
            with torch.no_grad():
                p = m(Gt)
            oof[c, f == k] = (p[te].cpu().numpy() * sd + mu)
            torch.save(m.state_dict(), os.path.join(OUT, f"c{c}_f{k}.pt"))
            print(f"cfg {c} fold {k}: held-out resid sd {float((p - t)[te].std()) * sd:.2f} levels, train {float((p - t)[tr].std()) * sd:.2f}", flush=True)
    res = lc - oof.mean(0)
    near = np.abs(lc - thr) < 60
    print(f"blend of the two configs (OOF): resid sd {res.std():.2f} levels; near threshold (|level-thr|<60, n={near.sum()}): sd {res[near].std():.2f}, "
          f"quantiles {np.quantile(res[near], [.001, .01, .05, .5, .95, .99, .999]).round(1)}")
    print(f"threshold level {thr}; levels total {lev.max() + 1}")
    json.dump(dict(mu=float(mu), sd=float(sd), thr_level=thr, oof_sd=float(res[near].std())), open(os.path.join(OUT, "meta.json"), "w"))


if __name__ == "__main__":
    train()
