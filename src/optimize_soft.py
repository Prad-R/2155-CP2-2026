"""Set optimisation with a soft validity penalty that hardens over the run.

Start from any set (e.g. the unconstrained optimum, ckpt/u1/final.npz) and maximise
    J = S_int - lam * sum_i deficit_i,   deficit_i = max(0, log(pmin) - log P(valid_i))
with lam rising geometrically from --lam0 to --lam1 over the first --soft-frac of the
steps; after that the constraint is hard (a grid's deficit may never increase).
Moves are single-cell changes plus optional two-cell swaps (--nswap). All neighbours of the chosen grid are scored
exactly for the diversity gain and by the CNN ensemble for the validity change; the
best few are confirmed with the GBM-blended model. Resumable.

  python -u src/optimize_soft.py --tag w1 --init ckpt/u1/final.npz --steps 150000
"""
import argparse, json, os, signal, time
import numpy as np
from scipy.stats import norm

from common import ROOT, THRESH
from surrogate import load_surrogates
from blend import blend_apply
from divscore import hamming_matrix, int_score, DENOM, K

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--init", required=True)
p.add_argument("--blend", default="b0")
p.add_argument("--pmin", type=float, default=0.95)
p.add_argument("--blend2", default="", help="second model generation that must also accept each grid")
p.add_argument("--pmin2", type=float, default=0.9)
p.add_argument("--lam-hard", type=float, default=0.0, help="penalty weight in the hard phase (default: lam1)")
p.add_argument("--lam0", type=float, default=0.3)
p.add_argument("--lam1", type=float, default=100.0)
p.add_argument("--soft-frac", type=float, default=0.7)
p.add_argument("--t0", type=float, default=1.0)
p.add_argument("--t1", type=float, default=0.1)
p.add_argument("--topk", type=int, default=8)
p.add_argument("--nswap", type=int, default=0, help="random two-cell swaps tried per step, besides the 196 single-cell changes")
p.add_argument("--steps", type=int, default=150000)
p.add_argument("--ckpt-every", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--quick", action="store_true")
args = p.parse_args()
if args.quick:
    args.steps, args.ckpt_every = 300, 100

out = os.path.join(ROOT, "ckpt", args.tag + ("_quick" if args.quick else ""))
os.makedirs(out, exist_ok=True)
t_start = time.time()
log = lambda *a: print(f"[{time.time()-t_start:7.1f}s]", *a, flush=True)
cfg, cnn, gbm, TH = load_surrogates(args.blend)
B = cfg["blend"]
SIG = np.array([b["sigma"] for b in B])
WSUM = np.array([b["w_gbm"] + b["w_cnn"] for b in B])
LP = np.log(args.pmin)
if args.blend2:
    cfg2, cnn2, gbm2, TH2 = load_surrogates(args.blend2)
    B2 = cfg2["blend"]
    SIG2 = np.array([b["sigma"] for b in B2])
    WSUM2 = np.array([b["w_gbm"] + b["w_cnn"] for b in B2])
    LP2 = np.log(args.pmin2)


def deficit2(pred2):
    return np.maximum(0.0, LP2 - norm.logcdf((pred2 - TH2) / SIG2).sum(-1))


def models2(g):
    mc = cnn2.predict(g.reshape(-1, 7, 7)).mean(2)
    mg = gbm2.predict(g.reshape(-1, 7, 7), procs=1).mean(2)
    return mc, blend_apply(B2, mg, mc)


def logpv(pred):
    return norm.logcdf((pred - TH) / SIG).sum(-1)


def deficit(pred):
    return np.maximum(0.0, LP - logpv(pred))


def models(g):
    """-> cnn mean (n,4), blended pred (n,4)"""
    mc = cnn.predict(g.reshape(-1, 7, 7)).mean(2)
    mg = gbm.predict(g.reshape(-1, 7, 7), procs=1).mean(2)
    return mc, blend_apply(B, mg, mc)


def total_scores(Dc):
    return np.partition(Dc, K - 1, axis=2)[:, :, :K].sum((1, 2))


state_f = os.path.join(out, "state.npz")
if os.path.exists(os.path.join(out, "DONE")):
    log("already done"); raise SystemExit
if os.path.exists(state_f):
    z = np.load(state_f)
    G, step0, best, bestG = z["G"], int(z["step"]), int(z["best"]), z["bestG"]
    log(f"resumed at step {step0}")
else:
    G = np.load(os.path.join(ROOT, args.init))
    G = (G["grids"] if args.init.endswith(".npz") else G).astype(np.int8).reshape(100, 49)
    step0, best, bestG = 0, -1, G.copy()
rng = np.random.RandomState(args.seed + step0)
MC, PRED = models(G)
DEF = deficit(PRED)
if args.blend2:
    MC2, PRED2 = models2(G)
    DEFB = deficit2(PRED2)
    DEF = DEF + DEFB
    log(f"second model {args.blend2}: grids below pmin2 {int((DEFB>0).sum())} deficit {DEFB.sum():.2f}")
DEFA = DEF - (DEFB if args.blend2 else 0)
log(f"start score {int_score(G.reshape(-1,7,7))/DENOM:.4f}  grids below pmin {int((DEF>0).sum())}  total deficit {DEF.sum():.1f}")


def save(step):
    tmp = state_f + ".tmp.npz"
    np.savez(tmp, G=G, step=step, best=best, bestG=bestG)
    os.replace(tmp, state_f)


stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))

n = 100
D = hamming_matrix(G).astype(np.int32)
np.fill_diagonal(D, 10**6)
cur = int(total_scores(D[None])[0])
if best < 0 and not (DEF > 0).any():
    best, bestG = cur, G.copy()
cell = np.repeat(np.arange(49), 4)
shift = np.tile(np.arange(1, 5), 49)
nsoft = max(int(args.steps * args.soft_frac), 1)
acc = 0
step = step0 - 1
for step in range(step0, args.steps):
    if stop["flag"]:
        break
    hard = step >= nsoft
    lam = (args.lam_hard or args.lam1) if hard else args.lam0 * (args.lam1 / args.lam0) ** (step / nsoft)
    T = args.t0 * (args.t1 / args.t0) ** (step / args.steps)
    if step % 2000 == 0 and step > step0:
        log(f"step {step}: {'hard' if hard else 'soft'} lam {lam:.2f} T {T:.2f} score {cur/DENOM:.4f} below-pmin {int((DEF>0).sum())} deficit {DEF.sum():.2f} "
            f"best-feasible {best/DENOM:.4f} accepted {acc}")
    if step % args.ckpt_every == 0 and step > step0:
        save(step)
    # pick a grid: low contributors, and grids still below the validity margin
    contrib = np.partition(D, K - 1, axis=1)[:, :K].sum(1)
    w = np.exp(-(contrib - contrib.min()) / 15.0) + 0.5 * (DEF > 0)
    i = rng.choice(n, p=w / w.sum())
    old = G[i, cell]
    new = (old + shift) % 5
    rows = D[i][None] - (G[:, cell] != old[None]).T + (G[:, cell] != new[None]).T
    cand = np.repeat(G[i][None], 196, 0)
    cand[np.arange(196), cell] = new
    if args.nswap:
        # two-cell swaps inside the grid (district counts unchanged)
        c1, c2 = rng.randint(49, size=3 * args.nswap), rng.randint(49, size=3 * args.nswap)
        ok = G[i, c1] != G[i, c2]
        c1, c2 = c1[ok][:args.nswap], c2[ok][:args.nswap]
        a1, a2 = G[i, c1], G[i, c2]
        rows2 = (D[i][None] - (G[:, c1] != a1[None]).T - (G[:, c2] != a2[None]).T
                 + (G[:, c1] != a2[None]).T + (G[:, c2] != a1[None]).T)
        cand2 = np.repeat(G[i][None], len(c1), 0)
        cand2[np.arange(len(c1)), c1] = a2
        cand2[np.arange(len(c1)), c2] = a1
        rows, cand = np.concatenate([rows, rows2]), np.concatenate([cand, cand2])
    M = len(cand)
    rows[:, i] = 10**6
    Dc = np.repeat(D[None], M, 0)
    Dc[:, i, :] = rows
    Dc[:, :, i] = rows
    gain = (total_scores(Dc) - cur).astype(float)
    # cheap validity estimate for all 196: shift the blended prediction by the CNN change
    mc = cnn.predict(cand.reshape(-1, 7, 7)).mean(2)
    dd_apx = deficit(PRED[i][None] + (mc - MC[i][None]) * WSUM[None]) - DEFA[i]
    if args.blend2:
        mc2 = cnn2.predict(cand.reshape(-1, 7, 7)).mean(2)
        dd_apx = dd_apx + deficit2(PRED2[i][None] + (mc2 - MC2[i][None]) * WSUM2[None]) - DEFB[i]
    u_apx = gain - lam * dd_apx
    top = np.argsort(-(u_apx + T * rng.gumbel(size=M)))[:args.topk]
    mck, predk = models(cand[top])
    dak = deficit(predk)
    dd = dak - DEFA[i]
    if args.blend2:
        mc2k, pred2k = models2(cand[top])
        dbk = deficit2(pred2k)
        dd = dd + dbk - DEFB[i]
    u = gain[top] - lam * dd
    if hard:
        u[dd > 1e-9] = -np.inf
    j = int(np.argmax(u))
    if not (u[j] >= -T * rng.exponential()):
        continue
    k = top[j]
    G[i], MC[i], PRED[i], DEF[i], DEFA[i] = cand[k], mck[j], predk[j], DEF[i] + dd[j], dak[j]
    if args.blend2:
        MC2[i], PRED2[i], DEFB[i] = mc2k[j], pred2k[j], dbk[j]
    D[i, :] = rows[k]
    D[:, i] = rows[k]
    cur += int(gain[k])
    acc += 1
    if cur > best and not (DEF > 0).any():
        best, bestG = cur, G.copy()

save(step + 1 if not stop["flag"] else step)
assert int_score(G.reshape(100, 7, 7)) == cur
if not stop["flag"]:
    feas = best > 0
    Gf = (bestG if feas else G).reshape(100, 7, 7)
    _, pred = models(Gf)
    pvf = np.exp(logpv(pred))
    sc = int_score(Gf)
    tmp = os.path.join(out, "final.tmp.npz")
    extra = {}
    if args.blend2:
        _, p2 = models2(Gf)
        extra["pvalid2"] = np.exp(norm.logcdf((p2 - TH2) / SIG2).sum(-1))
        log(f"second model {args.blend2}: pvalid min {extra['pvalid2'].min():.3f} mean {extra['pvalid2'].mean():.3f} expected failures {(1-extra['pvalid2']).sum():.1f}")
    np.savez(tmp, grids=Gf, pvalid=pvf, pred=pred, score=sc, **extra)
    os.replace(tmp, os.path.join(out, "final.npz"))
    log(f"final {'feasible' if feas else 'NOT FEASIBLE'} score {sc/DENOM:.10f} (int {sc}) pvalid min {pvf.min():.3f} mean {pvf.mean():.3f} below-pmin {int((pvf < args.pmin - 1e-9).sum())}")
    open(os.path.join(out, "DONE"), "w").write("ok\n")
