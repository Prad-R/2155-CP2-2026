"""Multi-task CNN: one shared trunk, four advisor heads, z targets, D4 augmentation.
Optionally adds pseudo-labelled unlabelled grids (self-training / distillation of the z0 blend).

  python -u src/train_mt.py --tag mt0
  python -u src/train_mt.py --tag mt1 --pseudo cache/pseudo_z0.npz --pw 0.3
Per fold k: ckpt/<tag>/f{k}.pt, f{k}_oof.npy (n_labels_total,) for the labels whose fold == k. Resumable.
"""
import argparse, json, os, pickle, time
import numpy as np
from scipy.special import ndtri, ndtr
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import r2_score
from common import ROOT, ADVISORS, load_data, atomic_save, folds_for
from train_cnn_net import Block

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True); p.add_argument("--epochs", type=int, default=120)
p.add_argument("--width", type=int, default=160); p.add_argument("--blocks", type=int, default=6)
p.add_argument("--pseudo", default=""); p.add_argument("--pw", type=float, default=0.3); p.add_argument("--pseudo-epochs", type=int, default=30)
p.add_argument("--quick", action="store_true")
args = p.parse_args()
if args.quick: args.epochs, args.pseudo_epochs = 3, 1
out = os.path.join(ROOT, "ckpt", args.tag + ("_quick" if args.quick else "")); os.makedirs(out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


class MTNet(nn.Module):
    def __init__(s, w, blocks):
        super().__init__()
        s.inp = nn.Conv2d(5, w, 3, padding=1); s.blocks = nn.Sequential(*[Block(w) for _ in range(blocks)])
        s.head = nn.Sequential(nn.Linear(3 * w, 384), nn.GELU(), nn.Dropout(0.1), nn.Linear(384, 4))

    def forward(s, g):
        x = F.one_hot(g.long(), 5).permute(0, 3, 1, 2).float()
        x = s.blocks(F.gelu(s.inp(x)))
        return s.head(torch.cat([x.mean((2, 3)), x.amax((2, 3)), x.amin((2, 3))], 1))


def aug(g):
    g = torch.rot90(g, np.random.randint(4), (1, 2))
    return g.flip(2) if np.random.rand() < 0.5 else g


grids, ratings = load_data()
G = torch.from_numpy(grids).to(dev)
pg, pa, pz, pf = [], [], [], []
for a in range(4):
    idx = np.where(~np.isnan(ratings[:, a]))[0]
    pg.append(idx); pa.append(np.full(len(idx), a)); pz.append(ndtri(np.clip(ratings[idx, a], 0.0005, 0.9995))); pf.append(folds_for(len(idx)))
pg, pa, pz, pf = [np.concatenate(x) for x in (pg, pa, pz, pf)]
PG, PA, PZ = torch.from_numpy(pg).to(dev), torch.from_numpy(pa).to(dev), torch.from_numpy(pz).float().to(dev)
ps = np.load(os.path.join(ROOT, args.pseudo)) if args.pseudo else None
if ps is not None:
    SI, SP = torch.from_numpy(ps["idx"]).to(dev), torch.from_numpy(ps["P"]).float().to(dev)     # (n,), (n,4,5)
    if args.quick: SI, SP = SI[:3000], SP[:3000]

oof = np.zeros(len(pg), np.float32)
for k in range(5):
    unit = os.path.join(out, f"f{k}")
    if not os.path.exists(unit + "_oof.npy"):
        torch.manual_seed(k); np.random.seed(k)
        tr = torch.from_numpy(np.where(pf != k)[0]).to(dev); te = np.where(pf == k)[0]
        model = MTNet(args.width, args.blocks).to(dev)
        bs = 256
        phases = ([("pseudo", args.pseudo_epochs)] if ps is not None else []) + [("real", args.epochs)]
        for ph, ep in phases:
            nb = (len(SI) if ph == "pseudo" else len(tr)) // bs
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3 if ph == "pseudo" else (1e-3 if ps is not None else 2e-3), total_steps=ep * nb, pct_start=0.1)
            for e in range(ep):
                model.train()
                perm = torch.randperm(len(tr), device=dev); sperm = torch.randperm(len(SI), device=dev) if ps is not None else None
                for i in range(nb):
                    b = tr[perm[(i * bs) % len(tr):(i * bs) % len(tr) + bs]]
                    loss = F.mse_loss(model(aug(G[PG[b]])).gather(1, PA[b][:, None]).squeeze(1), PZ[b])
                    if ps is not None and (ph == "pseudo" or True):
                        sb = sperm[(i * bs) % len(SI):(i * bs) % len(SI) + bs]
                        w = 1.0 if ph == "pseudo" else args.pw
                        loss = (loss if ph == "real" else 0.3 * loss) + w * F.mse_loss(model(aug(G[SI[sb]])), SP[sb][:, :, k])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            pr = torch.cat([model(G[PG[torch.from_numpy(te[i:i + 4096]).to(dev)]]) for i in range(0, len(te), 4096)]).cpu().numpy()
        torch.save(model.state_dict(), unit + ".pt.tmp"); os.replace(unit + ".pt.tmp", unit + ".pt")
        atomic_save(unit + "_oof.npy", pr[np.arange(len(te)), pa[te]].astype(np.float32))
    oof[pf == k] = np.load(unit + "_oof.npy")
    log(f"fold {k}: " + " ".join(f"{ADVISORS[a][:5]} {r2_score(pz[(pf==k)&(pa==a)], oof[(pf==k)&(pa==a)]):.4f}" for a in range(4)))

cfg = json.load(open(os.path.join(ROOT, "results", "blend_z0.json")))
summary = {}
for a in range(4):
    m = pa == a; z, o = pz[m], oof[m]; y = ndtr(z)
    atomic_save(os.path.join(out, f"a{a}_oof.npy"), o)
    d = pickle.load(open(os.path.join(ROOT, "ckpt", cfg["gbm"], f"cv_a{a}.pkl"), "rb")); oc = np.load(os.path.join(ROOT, "ckpt", cfg["cnn"], f"a{a}_aug1_oof.npy"))
    st = {}
    for nm, cols in [("mt alone", [o]), ("cnnz alone", [oc]), ("gbmz+cnnz (z0)", [d["oof"], oc]), ("gbmz+mt", [d["oof"], o]), ("gbmz+cnnz+mt", [d["oof"], oc, o])]:
        A_ = np.stack(cols + [np.ones_like(z)], 1); pb = A_ @ np.linalg.lstsq(A_, z, rcond=None)[0]; hi = pb >= 0.385
        st[nm] = dict(r2_z=round(float(r2_score(z, pb)), 4), r2_y=round(float(r2_score(y, ndtr(pb))), 4), sigma_z_hi=round(float(np.sqrt(np.mean((z[hi] - pb[hi]) ** 2))), 4))
    summary[ADVISORS[a]] = st; log(ADVISORS[a], json.dumps(st))
json.dump(summary, open(os.path.join(ROOT, "results", f"{args.tag}{'_quick' if args.quick else ''}_summary.json"), "w"), indent=1)
