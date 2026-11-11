"""CNN surrogates: per advisor, 5-fold CV, with and without D4 (rotation/flip) augmentation,
then prediction on all 500k grids. Resumable per (advisor, aug, fold) unit.

  python -u src/train_cnn.py --tag cnn0 [--quick]
"""
import argparse, json, os, time
import numpy as np
from scipy.special import ndtri, ndtr
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import r2_score

from common import ROOT, ADVISORS, THRESH, load_data, atomic_save, folds_for

p = argparse.ArgumentParser()
p.add_argument("--tag", default="cnn0")
p.add_argument("--ztarget", action="store_true", help="train on the inverse-normal of the label (proxy for the raw score)")
p.add_argument("--quick", action="store_true")
p.add_argument("--epochs", type=int, default=120)
p.add_argument("--width", type=int, default=128)
p.add_argument("--blocks", type=int, default=6)
p.add_argument("--augs", default="0,1")
args = p.parse_args()
if args.quick:
    args.epochs = 3

out = os.path.join(ROOT, "ckpt", args.tag + ("_quick" if args.quick else ""))
os.makedirs(out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
log("device", dev)


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

    def forward(s, g):  # g: (B,7,7) long
        x = F.one_hot(g.long(), 5).permute(0, 3, 1, 2).float()
        x = s.blocks(F.gelu(s.inp(x)))
        x = torch.cat([x.mean((2, 3)), x.amax((2, 3)), x.amin((2, 3))], 1)
        return s.head(x).squeeze(1)


def aug_d4(g):
    k = np.random.randint(4)
    g = torch.rot90(g, k, (1, 2))
    if np.random.rand() < 0.5:
        g = g.flip(2)
    return g


def predict(model, g, bs=8192):
    model.eval()
    o = []
    with torch.no_grad():
        for i in range(0, len(g), bs):
            o.append(model(g[i:i + bs]).float().cpu())
    return torch.cat(o).numpy()


def train(gtr, ytr, gte, yte, aug, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net(args.width, args.blocks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    bs = 128
    steps = args.epochs * (len(gtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(gtr), device=dev)
        for i in range(len(gtr) // bs):
            b = perm[i * bs:(i + 1) * bs]
            gb = aug_d4(gtr[b]) if aug else gtr[b]
            loss = F.mse_loss(model(gb), ytr[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return model


grids, ratings = load_data()
lab = ~np.isnan(ratings)
G = torch.from_numpy(grids).to(dev)
nall = 20000 if args.quick else len(grids)
augs = [int(x) for x in args.augs.split(",")]
summary = {}
for a in range(4):
    idx = np.where(lab[:, a])[0]
    if args.quick:
        idx = idx[:600]
    y = ratings[idx, a].astype(np.float32)
    if args.ztarget:
        y = ndtri(np.clip(y, 0.0005, 0.9995)).astype(np.float32)
    fold = folds_for(len(idx))
    gi, yi = G[idx], torch.from_numpy(y).to(dev)
    for aug in augs:
        oof = np.zeros(len(idx), np.float32)
        for k in range(5):
            unit = os.path.join(out, f"a{a}_aug{aug}_f{k}")
            if not os.path.exists(unit + "_oof.npy"):
                tr, te = torch.from_numpy(fold != k).to(dev), torch.from_numpy(fold == k).to(dev)
                model = train(gi[tr], yi[tr], gi[te], yi[te], aug, seed=k)
                pall = predict(model, G[:nall])
                atomic_save(unit + "_all.npy", pall.astype(np.float32))
                torch.save(model.state_dict(), unit + ".pt.tmp"); os.replace(unit + ".pt.tmp", unit + ".pt")
                atomic_save(unit + "_oof.npy", predict(model, gi[te]))  # written last = unit done
            oof[fold == k] = np.load(unit + "_oof.npy")
            log(ADVISORS[a], f"aug{aug} fold{k} r2={r2_score(y[fold==k], oof[fold==k]):.4f}")
        atomic_save(os.path.join(out, f"a{a}_aug{aug}_oof.npy"), oof)
        st = {"r2": float(r2_score(y, oof)), "rmse": float(np.sqrt(np.mean((oof - y) ** 2)))}
        if args.ztarget:
            st["r2_y"] = float(r2_score(ndtr(y), ndtr(oof)))
            _hi = oof >= 0.385
            st["sigma_z_hi"] = float(np.sqrt(np.mean((oof[_hi] - y[_hi]) ** 2)))
        for mrg in (0.75, 0.80, 0.85, 0.90):
            pv = oof >= mrg
            st[f"n_pred>={mrg}"] = int(pv.sum())
            st[f"prec@{mrg}"] = float((y[pv] >= THRESH).mean()) if pv.any() else None
        summary[f"{ADVISORS[a]}/aug{aug}"] = st
        log(ADVISORS[a], f"aug{aug}", json.dumps(st))

json.dump(summary, open(os.path.join(ROOT, "results", f"{args.tag}{'_quick' if args.quick else ''}_summary.json"), "w"), indent=1)
open(os.path.join(out, "DONE"), "w").write("ok\n")
log("DONE")
