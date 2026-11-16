"""Recover hands that the two views could not triangulate, from the hand's own measured shape.

When one hand hides the other, the detector's keypoints in the two views disagree and the hand fails the stereo
checks, or it is found in one view only. For such a frame this module takes the hand's 3D shape from the nearest frame
where it was measured, and moves it as a rigid body until it lands on the 2D keypoints of whichever views have them.
Its size is known from the measured frame, so even one view fixes its distance.

The result is an estimate, not a measurement: the fingers keep the pose they had in the measured frame. Joints placed
this way get status 3, so they can be told apart from triangulated (1) and gap-filled (2) joints, and left out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stereo import StereoRig, _rotmat, _runs, savgol

PLACED = 3  # kp3d_status of a joint placed from the hand's shape in a nearby frame


@dataclass
class FillSettings:
    reach: float = 5.0  # seconds from the nearest measured frame of that hand; further away nothing is placed
    huber_px: float = 0.005  # share of the image height: residuals above it count less
    inlier_px: float = 0.012  # share of the image height: a keypoint within this of the fitted hand agrees
    min_inliers: int = 12  # agreeing keypoints, over both views
    min_share: float = 0.5  # of the keypoints used, the share that must agree
    single_view: bool = True  # when the two views disagree, accept a fit to one of them
    max_step: float = 0.12  # metres the wrist may move from the frame the fit started from
    max_turn: float = 60.0  # degrees the hand may turn from the frame the fit started from
    min_depth: float = 0.1
    max_depth: float = 1.5


def _pose(P0: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The points P0 (21, 3) turned by the rotation vector p[:3] about their centroid and shifted by p[3:]."""
    c = P0.mean(0)
    return (P0 - c) @ _rotmat(p[:3]).T + c + p[3:]


def _residuals(rig: StereoRig, P: np.ndarray, obs: list) -> np.ndarray:
    out = []
    for view, uv, use in obs:
        with np.errstate(all="ignore"):
            out.append((rig.project(P[use], view) - uv[use]).ravel())
    return np.concatenate(out)


def fit_rigid(rig: StereoRig, P0: np.ndarray, obs: list, s: FillSettings):
    """Move the hand P0 (21, 3) rigidly onto 2D keypoints. `obs`: [(view, uv (21, 2), use (21,) bool)].
    Returns (points (21, 3), agreeing keypoints, keypoints used, their median distance in pixels), or the reason
    it was rejected as a string."""
    huber, limit = s.huber_px * rig.height, s.inlier_px * rig.height
    p = np.zeros(6)
    steps = np.array([1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5])
    for _ in range(10):
        r = _residuals(rig, _pose(P0, p), obs)
        if not np.isfinite(r).all():
            return "not finite"
        J = np.stack([(_residuals(rig, _pose(P0, p + np.eye(6)[k] * steps[k]), obs) - r) / steps[k] for k in range(6)], 1)
        if not np.isfinite(J).all():
            return "not finite"
        w = np.minimum(1.0, huber / np.maximum(np.abs(r), 1e-9))
        A = np.einsum("ni,n,nj->ij", J, w, J) + 1e-6 * np.eye(6)
        step = np.linalg.solve(A, np.einsum("ni,n,n->i", J, w, r))
        p = p - step
        if np.abs(step).max() < 1e-6:
            break
    P = _pose(P0, p)
    r = _residuals(rig, P, obs)
    if not np.isfinite(r).all():
        return "not finite"
    dist = np.linalg.norm(r.reshape(-1, 2), axis=1)
    inliers = dist < limit
    if inliers.sum() < s.min_inliers or inliers.mean() < s.min_share:
        return "keypoints disagree"
    if np.linalg.norm(p[3:]) > s.max_step or np.degrees(np.linalg.norm(p[:3])) > s.max_turn:
        return "moved too far"
    if not s.min_depth < P[0, 2] < s.max_depth:
        return "depth"
    return P, int(inliers.sum()), len(dist), float(np.median(dist[inliers]))


def _pick(rig: StereoRig, view: int, kp: np.ndarray, conf: np.ndarray, P0: np.ndarray, other: np.ndarray | None):
    """The detection in this view that belongs to the hand at P0: the one whose keypoints lie nearest to where P0 is
    expected, unless they lie nearer to the other hand. `kp` (2, 21, 2), `conf` (2, 21). Returns (uv, use) or None."""
    with np.errstate(all="ignore"):
        expect = rig.project(P0, view)
        expect_other = rig.project(other, view) if other is not None else None
    if not np.isfinite(expect).all():
        return None
    size = float(np.linalg.norm(expect.max(0) - expect.min(0)))
    best = None
    for slot in range(2):
        use = conf[slot] > 0
        if use.sum() < 8:
            continue
        d = float(np.median(np.linalg.norm(kp[slot][use] - expect[use], axis=1)))
        if d > 0.6 * size:
            continue
        if expect_other is not None and np.isfinite(expect_other).all():
            if float(np.median(np.linalg.norm(kp[slot][use] - expect_other[use], axis=1))) < d:
                continue
        if best is None or d < best[0]:
            best = (d, kp[slot].astype(float), use)
    return best[1:] if best else None


def fill_hands(rig: StereoRig, X: np.ndarray, valid: np.ndarray, views, fps: float,
               settings: FillSettings | None = None, counts: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Place hands in the frames where `valid` is False.

    `X` (N, 2, 21, 3) cleaned joints (NaN where unknown), `valid` (N, 2), `views` per camera
    (kp (N, 2, 21, 2), conf (N, 2, 21), score (N, 2)) as given to label_stereo (slots as the detector labelled them).
    A frame is filled from its neighbour in time, forwards from the last measured frame and backwards from the next
    one; where both reach it, the fit with more agreeing keypoints is kept. Returns the points with the placed hands
    added, and `placed` (N, 2). `counts`, if given, is filled with how the attempts ended (both passes together)."""
    s = settings or FillSettings()
    n = len(valid)
    reach = max(1, round(s.reach * fps))
    whole = valid & np.isfinite(X[..., 0]).all(-1)  # frames that can lend their shape
    out, placed = X.copy(), np.zeros((n, 2), bool)
    score = np.zeros((n, 2), int)
    counts = counts if counts is not None else {}

    def note(what):
        counts[what] = counts.get(what, 0) + 1

    for hand in range(2):
        if not whole[:, hand].any():
            continue
        for order in (range(n), range(n - 1, -1, -1)):
            last, since = None, 0
            for i in order:
                if whole[i, hand]:
                    last, since = X[i, hand], 0
                    continue
                since += 1
                if valid[i, hand]:
                    continue  # measured, but with joints missing: neither filled nor a source
                if last is None or since > reach:
                    note("no measured frame in reach")
                    last = None
                    continue
                other = X[i, 1 - hand] if valid[i, 1 - hand] else None
                obs = []
                for view, (kp, conf, _) in enumerate(views):
                    hit = _pick(rig, view, kp[i], conf[i], last, other)
                    if hit is not None:
                        obs.append((view, *hit))
                fit = fit_rigid(rig, last, obs, s) if obs else "no keypoints near the hand"
                if fit == "keypoints disagree" and len(obs) == 2 and s.single_view:
                    # one view may show the hand badly (hidden behind the other hand): try each view alone
                    alone = [f for f in (fit_rigid(rig, last, [o], s) for o in obs) if not isinstance(f, str)]
                    if alone:
                        fit = max(alone, key=lambda f: f[1])
                        note("placed from one view after the two disagreed")
                if isinstance(fit, str):
                    note(fit)  # the next frame starts again from the last hand that was measured or placed
                    continue
                note("placed")
                P, agree, _, _ = fit
                last = P
                if agree > score[i, hand]:
                    out[i, hand], placed[i, hand], score[i, hand] = P, True, agree
    return out, placed


def smooth_placed(X: np.ndarray, placed: np.ndarray, fps: float, window: float = 0.3) -> np.ndarray:
    """Smooth the placed hands over time together with their measured neighbours (Savitzky-Golay over `window`
    seconds). Measured frames keep their values."""
    X = X.copy()
    win = max(5, round(window * fps) | 1)
    for hand in range(2):
        if not placed[:, hand].any():
            continue
        for a, b in _runs(np.isfinite(X[:, hand]).all((1, 2))):
            if b - a >= win and placed[a:b, hand].any():
                smooth = savgol(X[a:b, hand].reshape(b - a, -1), win).reshape(b - a, 21, 3)
                rows = np.flatnonzero(placed[a:b, hand])
                X[a + rows, hand] = smooth[rows]
    return X
