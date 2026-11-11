"""Exact advisor rules recovered from the labels (see UPDATES.md, 2026-11-11).

Raw integer scores for Wellness, Tax, Transportation and Business; a grid is valid for an advisor iff
raw >= THRESH[advisor].  Label thresholds read off the 5,000 labels per advisor (first raw score whose label is
>= 0.75): Wellness 70, Tax 16, Transportation -187, Business -1157/-1156.  Tax and Business occasionally show two
adjacent labels for one raw value (float arithmetic in the organisers' code), so THRESH is one step inside for those.

Conventions: g[r, c], row 0 on top. "left of" = smaller column, "above" = smaller row.
d(a->b) = Manhattan distance from a cell of type a to the nearest cell of type b.
Types: 0 R(esidential) 1 I(ndustrial) 2 C(ommercial) 3 P(ark) 4 O(ffice).
The rules are only verified for grids that contain at least one R, I, C and P.
"""
import numpy as np

_RC = np.array([(r, c) for r in range(7) for c in range(7)])
MAN = (np.abs(_RC[:, None, 0] - _RC[None, :, 0]) + np.abs(_RC[:, None, 1] - _RC[None, :, 1])).astype(np.int32)
BIG = 1000
NAMES = ["Wellness", "Tax", "Transportation", "Business"]
THRESH = np.array([70, 17, -187, -1155])           # safe thresholds (see module docstring)
THRESH_LABEL = np.array([70, 16, -187, -1157])     # first raw value with a label >= 0.75 in the data


def _left(A, B):
    return (A[:, :, :-1] & B[:, :, 1:]).sum((1, 2))


def _above(A, B):
    return (A[:, :-1, :] & B[:, 1:, :]).sum((1, 2))


def raw_scores(g):
    """g: (n,7,7) int array -> raw (n,4) int64 [Wellness, Tax, Transportation, Business], ok (n,) bool (R, I, C, P all present)."""
    g = np.asarray(g).reshape(-1, 7, 7)
    n = len(g)
    T = [(g == k) for k in range(5)]
    R, I, C, P, O = T
    Tf = [t.reshape(n, 49) for t in T]
    cnt = np.stack([t.sum(1) for t in Tf], 1).astype(np.int64)
    ok = (cnt[:, :4] > 0).all(1)
    dn = []
    for b in range(4):                                               # nearest R, I, C, P
        d = np.where(Tf[b][:, None, :], MAN[None], BIG).min(2)       # n,49
        dn.append(d.astype(np.int64))
    dR, dI, dC, dP = dn
    Rf, If, Cf, Pf, Of = [t.astype(np.int64) for t in Tf]
    well = ((Rf * (-3 * dC ** 2 - dP ** 2)).sum(1) + (Pf * 2 * dI).sum(1)
            + 20 * _left(P, R) + 20 * (_left(P, P) + _above(P, P)) + 15 * _above(R, P) + 4 * _above(R, I))
    tax = (10 * cnt[:, 0] + 3 * cnt[:, 2] - 20 * cnt[:, 3]
           - 2 * (_left(R, I) + _left(I, R) + _above(R, I) + _above(I, R))
           + 8 * _left(P, R) + 3 * _left(R, P) + 10 * _above(R, P) + 3 * _above(P, R))
    trans = (-2 * (Rf * dC).sum(1) - (If * dR).sum(1) - 5 * (If * dC).sum(1) - 5 * (Cf * dI).sum(1) - (Of * dR).sum(1)
             + 3 * (_left(I, I) + _above(I, I)) + 3 * (_left(C, C) + _above(C, C))
             + cnt[:, 0] - R[:, :, 0].sum(1))
    nR, nI, nC, nP, nO = cnt.T
    bus = (-24 * nI - 37 * nC - 32 * nP - 22 * nO - 8 * np.maximum(0, 10 * nR - 3 * nI - 3 * nC - 4 * nO)
           + 10 * (_left(I, I) + _above(I, I)) - 2 * C[:, :, 1:].sum((1, 2)) - 5 * C[:, :-1, :].sum((1, 2)))
    return np.stack([well, tax, trans, bus], 1).astype(np.int64), ok


def valid(g, thresh=THRESH):
    raw, ok = raw_scores(g)
    return ok & (raw >= np.asarray(thresh)[None]).all(1)
