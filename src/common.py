"""Shared helpers: data loading, atomic saves, CV splits."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVISORS = ["Wellness", "Tax", "Transportation", "Business"]
THRESH = 0.75


def load_data():
    grids = np.concatenate(
        [np.load(os.path.join(ROOT, "datasets", f"grids_{i}.npy")) for i in range(5)]
    ).astype(np.int8)
    ratings = np.load(os.path.join(ROOT, "datasets", "scores.npy"))
    return grids, ratings


def atomic_save(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def folds_for(n, k=5, seed=0):
    """Deterministic fold id (0..k-1) for each of n labelled samples."""
    rng = np.random.RandomState(seed)
    f = np.arange(n) % k
    rng.shuffle(f)
    return f


def d4(g):
    """All 8 rotations/reflections of grids shaped (..., 7, 7)."""
    out = []
    for k in range(4):
        r = np.rot90(g, k, axes=(-2, -1))
        out.append(r)
        out.append(r[..., ::-1])
    return out
