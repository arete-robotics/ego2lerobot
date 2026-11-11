"""Stereo hand labels: calibration, rectification, triangulation with match checks, and clean-up over time.

Calibration file (JSON; pixels and metres; OpenCV conventions; camera axes x right, y down, z forward):

    {"image_width": 1920, "image_height": 1080,
     "model": "pinhole",                       # or "fisheye" (Kannala-Brandt, distortion = [k1, k2, k3, k4])
     "left":  {"fx": 1050, "fy": 1050, "cx": 960, "cy": 540, "distortion": [k1, k2, p1, p2, k3]},
     "right": {"fx": 1046, "fy": 1046, "cx": 955, "cy": 538, "distortion": [k1, k2, p1, p2, k3]},
     "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],   # rotation from the left camera frame to the right camera frame
     "T": [-0.063, 0, 0]}                      # metres: a point X in the left frame is R X + T in the right frame

For a pair that is already rectified, give only image_width, image_height, fx, fy, cx, cy and baseline (metres).
All 3D output is in the left camera frame.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .camera import Camera
from .hands import BONES, HandTrack

NEEDED = (0, 4, 5, 8, 17)  # wrist, thumb tip, index knuckle, index tip, little knuckle: palm frame and pinch
REASONS = {-1: "no hand", 0: "triangulated", 1: "one view only", 2: "epipolar", 3: "depth", 4: "hand size", 5: "few joints"}


def _radtan(xy: np.ndarray, d: tuple) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV radial-tangential model: distorted = radial * xy + tangential."""
    k1, k2, p1, p2, k3 = (tuple(d) + (0.0,) * 5)[:5]
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    tangential = np.stack([2 * p1 * x * y + p2 * (r2 + 2 * x * x), p1 * (r2 + 2 * y * y) + 2 * p2 * x * y], -1)
    return radial[:, None], tangential


@dataclass(frozen=True)
class Lens:
    """One camera. pinhole: distortion k1 k2 p1 p2 [k3] (OpenCV); fisheye: k1 k2 k3 k4 (Kannala-Brandt)."""

    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "pinhole"
    distortion: tuple[float, ...] = ()

    @classmethod
    def from_dict(cls, d: dict, model: str) -> Lens:
        return cls(float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]), model,
                   tuple(float(k) for k in d.get("distortion", ())))

    def scaled(self, sx: float, sy: float) -> Lens:
        return replace(self, fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy)

    def _fisheye(self) -> Camera:
        return Camera(0, 0, self.fx, self.fy, self.cx, self.cy, "fisheye", (tuple(self.distortion) + (0.0,) * 4)[:4])

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (n, 2) → undistorted normalized coordinates (x/z, y/z)."""
        if self.model == "fisheye":
            return self._fisheye().undistort(uv)
        xy0 = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy], -1)
        xy = xy0
        if any(self.distortion):
            for _ in range(30):
                radial, tangential = _radtan(xy, self.distortion)
                xy = (xy0 - tangential) / radial
        return xy

    def pixels(self, xy: np.ndarray) -> np.ndarray:
        """Undistorted normalized coordinates (n, 2) → pixels. Inverse of `rays`."""
        if self.model == "fisheye":
            return self._fisheye().project(np.column_stack([xy, np.ones(len(xy))]))
        if any(self.distortion):
            radial, tangential = _radtan(xy, self.distortion)
            xy = radial * xy + tangential
        return np.stack([self.fx * xy[:, 0] + self.cx, self.fy * xy[:, 1] + self.cy], -1)

    def vector(self) -> np.ndarray:
        """fx fy cx cy d0 d1 d2 d3 d4, as stored per frame in the dataset."""
        return np.array([self.fx, self.fy, self.cx, self.cy, *(tuple(self.distortion) + (0.0,) * 5)[:5]], np.float32)

    def to_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy, "distortion": list(self.distortion)}


def _rotvec(R: np.ndarray) -> np.ndarray:
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-12:
        return np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle)) * angle


def _rotmat(v: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _rotate(R: np.ndarray, X: np.ndarray) -> np.ndarray:
    """R applied to each row of X (n, 3). einsum, not BLAS: Accelerate's matmul raises spurious FP warnings here."""
    return np.einsum("ij,nj->ni", R, X)


def _rectify(rays: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    v = _rotate(rotation, np.column_stack([rays, np.ones(len(rays))]))
    return v[:, :2] / v[:, 2:3]


class StereoRig:
    """A calibrated horizontal stereo pair. Rectification follows Bouguet (as OpenCV's stereoRectify): each camera turns
    by half the relative rotation, then both turn so the baseline lies along x and epipolar lines become rows."""

    def __init__(self, width: int, height: int, left: Lens, right: Lens, R=np.eye(3), T=(-0.06, 0.0, 0.0)):
        self.width, self.height, self.left, self.right = int(width), int(height), left, right
        self.R = np.asarray(R, float).reshape(3, 3)
        self.T = np.asarray(T, float).reshape(3)
        if not (self.T[0] < 0 and abs(self.T[0]) > abs(self.T[1])):
            raise ValueError("expected a horizontal pair with the right camera to the right of the left one (T[0] < 0)")
        half = _rotmat(-0.5 * _rotvec(self.R))
        t = half @ self.T
        axis = np.cross(t, [-1.0, 0.0, 0.0])
        n = np.linalg.norm(axis)
        align = _rotmat(axis / n * np.arccos(np.clip(-t[0] / np.linalg.norm(t), -1, 1))) if n > 1e-12 else np.eye(3)
        self.R1 = align @ half.T  # left camera → rectified frame
        self.R2 = align @ half  # right camera → rectified frame; R2 R = R1 and R2 T = (−baseline, 0, 0)
        self.baseline = float(np.linalg.norm(t))
        self.focal = (left.fy + right.fy) / 2  # turns rectified normalized errors into pixels

    @classmethod
    def from_json(cls, path: str | Path, width: int | None = None, height: int | None = None) -> StereoRig:
        d = json.loads(Path(path).read_text())
        model = d.get("model", "pinhole")
        if model not in ("pinhole", "fisheye"):
            raise ValueError(f"unsupported camera model {model!r}; use pinhole or fisheye")
        if "baseline" in d:
            lens = Lens.from_dict(d, model)
            rig = cls(d["image_width"], d["image_height"], lens, lens, np.eye(3), (-float(d["baseline"]), 0.0, 0.0))
        else:
            rig = cls(d["image_width"], d["image_height"], Lens.from_dict(d["left"], model),
                      Lens.from_dict(d["right"], model), d["R"], d["T"])
        return rig.scaled(width, height) if width and height else rig

    def scaled(self, width: int, height: int) -> StereoRig:
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return StereoRig(width, height, self.left.scaled(sx, sy), self.right.scaled(sx, sy), self.R, self.T)

    def to_dict(self) -> dict:
        return {"image_width": self.width, "image_height": self.height, "model": self.left.model,
                "left": self.left.to_dict(), "right": self.right.to_dict(), "R": self.R.tolist(), "T": self.T.tolist(),
                "baseline_m": self.baseline, "rectify_left": self.R1.tolist(), "rectify_right": self.R2.tolist()}

    def project(self, X: np.ndarray, view: int = 0) -> np.ndarray:
        """Points (n, 3) in the left camera frame → pixels in the left (0) or right (1) original image."""
        if view:
            X = _rotate(self.R, X) + self.T
        return (self.right if view else self.left).pixels(X[:, :2] / X[:, 2:3])

    def triangulate(self, uv_left: np.ndarray, uv_right: np.ndarray):
        """Matching pixels (n, 2) in the two original images → points (n, 3) in the left camera frame, the row
        disagreement after rectification (epipolar error) and the mean reprojection error over both images, in pixels."""
        a = _rectify(self.left.rays(uv_left), self.R1)
        b = _rectify(self.right.rays(uv_right), self.R2)
        with np.errstate(all="ignore"):  # parallel or crossed rays give inf or negative depth; the checks reject them
            z = self.baseline / (a[:, 0] - b[:, 0])
            X = _rotate(self.R1.T, np.stack([a[:, 0] * z, (a[:, 1] + b[:, 1]) / 2 * z, z], -1))  # → left camera
            reprojection = (np.linalg.norm(self.project(X, 0) - uv_left, axis=-1)
                            + np.linalg.norm(self.project(X, 1) - uv_right, axis=-1)) / 2
        return X, np.abs(a[:, 1] - b[:, 1]) * self.focal, reprojection


@dataclass
class Checks:
    """When a stereo match counts. Pixel limits are shares of the image height (0.01 = 10.8 px at 1080 rows)."""

    max_epipolar: float = 0.01  # a hand's median row disagreement after rectification; above it the match is wrong
    max_reprojection: float = 0.006  # one joint's mean reprojection error; above it the joint is dropped
    min_depth: float = 0.1  # metres from the left camera
    max_depth: float = 1.5
    max_depth_spread: float = 0.15  # a joint further than this from its hand's median depth is dropped
    palm_size: tuple[float, float] = (0.04, 0.15)  # mean wrist–index, wrist–little, index–little knuckle distance
    min_joints: int = 8  # joints that must pass for the hand to count as triangulated


def match_hand(rig: StereoRig, uv_l, ok_l, uv_r, ok_r, checks: Checks):
    """Triangulate one hand seen in both views: (21, 3) points with NaN for dropped joints, the median epipolar and
    reprojection errors in pixels, and a reason code (see REASONS)."""
    X = np.full((21, 3), np.nan)
    both = ok_l & ok_r
    if both.sum() < checks.min_joints:
        return X, np.nan, np.nan, 5
    P, epipolar, reprojection = rig.triangulate(uv_l[both], uv_r[both])
    epi = float(np.median(epipolar))
    if epi > checks.max_epipolar * rig.height:
        return X, epi, np.nan, 2
    z = P[:, 2]
    zm = np.median(z)
    if not checks.min_depth < zm < checks.max_depth:
        return X, epi, np.nan, 3
    keep = ((reprojection <= checks.max_reprojection * rig.height) & (z > checks.min_depth) & (z < checks.max_depth)
            & (np.abs(z - zm) < checks.max_depth_spread))
    if keep.sum() < checks.min_joints:
        return X, epi, np.nan, 5
    X[np.flatnonzero(both)[keep]] = P[keep]
    if np.isfinite(X[[0, 5, 17], 0]).all():
        palm = np.mean([np.linalg.norm(X[a] - X[b]) for a, b in ((0, 5), (0, 17), (5, 17))])
        if not checks.palm_size[0] < palm < checks.palm_size[1]:
            return np.full((21, 3), np.nan), epi, np.nan, 4
    return X, epi, float(np.median(reprojection[keep])), 0


def match_frame(rig: StereoRig, left, right, checks: Checks):
    """One frame. `left` and `right` are (kp (2, 21, 2), conf (2, 21), score (2,)) with slot 0 the left hand.
    Handedness can differ between the views, so both pairings are tried; the one with more good matches wins, then the
    one with the lower epipolar error. Returns points (2, 21, 3), epipolar (2,), reprojection (2,), reason (2,) and,
    per hand, the slot it came from in each view."""
    (kl, cl, sl), (kr, cr, sr) = left, right
    best = None
    for pairs in (((0, 0), (1, 1)), ((0, 1), (1, 0))):
        found = [(a, b, *match_hand(rig, kl[a], cl[a] > 0, kr[b], cr[b] > 0, checks))
                 for a, b in pairs if sl[a] > 0 and sr[b] > 0]
        good = [f for f in found if f[-1] == 0]
        key = (len(good), -sum(f[3] for f in good))
        if best is None or key > best[0]:
            best = (key, found)
    found = best[1]
    slots = [a if a == b or sl[a] >= sr[b] else b for a, b, *_ in found]  # trust the more confident handedness
    if len(set(slots)) < len(slots):
        slots = [a for a, *_ in found]
    X = np.full((2, 21, 3), np.nan)
    epi, rep = np.full(2, np.nan), np.full(2, np.nan)
    reason = np.where((sl > 0) | (sr > 0), 1, -1)
    order_l, order_r = [0, 1], [0, 1]
    for s, (a, b, x, e, r, why) in zip(slots, found):
        X[s], epi[s], rep[s], reason[s] = x, e, r, why
        order_l[s], order_l[1 - s] = a, 1 - a
        order_r[s], order_r[1 - s] = b, 1 - b
    return X, epi, rep, reason, order_l, order_r


def savgol(y: np.ndarray, window: int, order: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0 (odd window); the ends use the polynomial fitted to the first/last window."""
    half = window // 2
    A = np.vander(np.arange(-half, half + 1), order + 1, increasing=True)
    fit = np.linalg.pinv(A)  # window values → polynomial coefficients
    out = y.copy()
    out[half:len(y) - half] = sliding_window_view(y, window, axis=0) @ fit[0]
    out[:half] = A[:half] @ (fit @ y[:window])
    out[len(y) - half:] = A[half + 1:] @ (fit @ y[-window:])
    return out


def _runs(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def clean(X: np.ndarray, fps: float, max_gap: float = 0.35, window: float = 0.3, spike: float = 0.05,
          bone_weight: float = 0.3):
    """Temporal clean-up of triangulated joints (N, 2, 21, 3): drop spikes (further than `spike` metres from the
    5-frame median), fill gaps up to `max_gap` seconds linearly, Savitzky-Golay smoothing over `window` seconds, then
    pull each bone towards its median length over the clip. Returns points and a status per joint:
    0 unknown, 1 triangulated in this frame, 2 filled across a gap."""
    X = X.copy()
    padded = np.pad(X, ((2, 2), (0, 0), (0, 0), (0, 0)), constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(sliding_window_view(padded, 5, axis=0), axis=-1)
    X[np.linalg.norm(X - median, axis=-1) > spike] = np.nan
    status = np.isfinite(X[..., 0]).astype(np.int8)
    gap, win = max(1, round(max_gap * fps)), max(5, round(window * fps) | 1)
    for s in range(2):
        for j in range(21):
            P = X[:, s, j]
            idx = np.flatnonzero(np.isfinite(P[:, 0]))
            for a, b in zip(idx[:-1], idx[1:]):
                if 1 < b - a <= gap + 1:
                    P[a + 1:b] = P[a] + (np.arange(1, b - a) / (b - a))[:, None] * (P[b] - P[a])
                    status[a + 1:b, s, j] = 2
            for a, b in _runs(np.isfinite(P[:, 0])):
                if b - a >= win:
                    P[a:b] = savgol(P[a:b], win)
    for s in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lengths = {bone: np.nanmedian(np.linalg.norm(X[:, s, bone[1]] - X[:, s, bone[0]], axis=-1)) for bone in BONES}
        for _ in range(4):
            for p, q in BONES:
                v = X[:, s, q] - X[:, s, p]
                n = np.linalg.norm(v, axis=-1, keepdims=True)
                corr = np.nan_to_num((n - lengths[(p, q)]) / np.maximum(n, 1e-6) * v * 0.5 * bone_weight)
                X[:, s, p] += corr
                X[:, s, q] -= corr
    return X, status


@dataclass
class StereoLabels:
    """Hand labels for one stereo clip.

    kp2d:         (N, 2, 2, 21, 3) view (left, right camera) × hand × joint × (u, v, confidence), original pixels;
                  hand slots follow the 3D hands
    kp3d:         (N, 2, 21, 3) metres in the left camera frame, NaN where unknown
    status:       (N, 2, 21) 0 unknown, 1 triangulated in this frame, 2 filled across a short gap
    valid:        (N, 2) at least 15 joints known, including wrist, knuckles and thumb and index tips
    epipolar:     (N, 2) median row disagreement of the matched hand after rectification, pixels
    reprojection: (N, 2) median reprojection error of the kept joints, pixels
    reason:       (N, 2) see REASONS
    """

    kp2d: np.ndarray
    kp3d: np.ndarray
    status: np.ndarray
    valid: np.ndarray
    epipolar: np.ndarray
    reprojection: np.ndarray
    reason: np.ndarray
    fps: float
    width: int
    height: int

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: str | Path) -> StereoLabels:
        d = np.load(path)
        return cls(*(d[k] for k in ("kp2d", "kp3d", "status", "valid", "epipolar", "reprojection", "reason")),
                   float(d["fps"]), int(d["width"]), int(d["height"]))

    def as_track(self) -> HandTrack:
        """A HandTrack whose score is 1 where the hand has stereo labels, for episode segmentation."""
        n = len(self.valid)
        return HandTrack(np.zeros((n, 2, 21, 2), np.float32), np.zeros((n, 2, 21, 3), np.float32),
                         self.valid.astype(np.float32), self.fps, self.width, self.height)


def label_stereo(rig: StereoRig, views, fps: float, checks: Checks | None = None) -> StereoLabels:
    """`views`: per camera (kp (N, 2, 21, 2) pixels, conf (N, 2, 21), score (N, 2)) → StereoLabels."""
    checks = checks or Checks()
    n = min(len(views[0][2]), len(views[1][2]))
    X = np.full((n, 2, 21, 3), np.nan)
    epi, rep = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    reason = np.full((n, 2), -1, np.int8)
    kp2d = np.zeros((n, 2, 2, 21, 3))
    for i in range(n):
        frame = [(kp[i].astype(float), conf[i], score[i]) for kp, conf, score in views]
        X[i], epi[i], rep[i], reason[i], *orders = match_frame(rig, *frame, checks)
        for v, ((kp, conf, _), order) in enumerate(zip(frame, orders)):
            kp2d[i, v] = np.concatenate([kp[order], conf[order][..., None]], -1)
    X, status = clean(X, fps)
    known = np.isfinite(X[..., 0])
    valid = (known.sum(-1) >= 15) & known[..., NEEDED].all(-1)
    X[~valid] = np.nan
    status[~valid] = 0
    return StereoLabels(kp2d, X, status, valid, epi, rep, reason, fps, rig.width, rig.height)


def _q(a: np.ndarray, q: float, digits: int = 3):
    return round(float(np.percentile(a, q)), digits) if len(a) else None


def _final_reprojection(labels: StereoLabels, frames, rig: StereoRig) -> np.ndarray:
    """Distance (pixels, mean of both views) between the final 3D joints and the detections they were triangulated from."""
    measured = labels.status[frames] == 1
    if not measured.any():
        return np.array([])
    kp2d, kp3d = labels.kp2d[frames], labels.kp3d[frames][measured]
    return np.mean([np.linalg.norm(rig.project(kp3d, v) - kp2d[:, v][measured][:, :2], axis=-1) for v in (0, 1)], 0)


def summarize(parts) -> dict:
    """Triangulation statistics for QC over a list of (StereoLabels, frame indices, StereoRig)."""
    cat = lambda key: np.concatenate([getattr(lab, key)[idx] for lab, idx, _ in parts])  # noqa: E731
    kp2d, kp3d, status, valid, reason = cat("kp2d"), cat("kp3d"), cat("status"), cat("valid"), cat("reason")
    seen = kp2d[..., 2].max(-1) > 0  # (n, view, hand)
    both = int(np.minimum(seen[:, 0].sum(-1), seen[:, 1].sum(-1)).sum())
    ok = reason == 0
    final = np.concatenate([_final_reprojection(*part) for part in parts])
    depth = kp3d[..., 0, 2][valid]
    joints = status[valid]
    return {
        "frames": len(valid),
        "hand_frames_left_view": int(seen[:, 0].sum()),
        "hand_frames_right_view": int(seen[:, 1].sum()),
        "hand_frames_both_views": both,
        "hand_frames_triangulated": int(ok.sum()),
        "valid_triangulation_rate": round(int(ok.sum()) / both, 3) if both else None,
        "hand_frames_labelled": int(valid.sum()),
        "rejected": {REASONS[k]: int((reason == k).sum()) for k in (1, 2, 3, 4, 5)},
        "epipolar_px_median": _q(cat("epipolar")[ok], 50, 2),
        "reprojection_px_median": _q(cat("reprojection")[ok], 50, 2),
        "reprojection_px_p90": _q(cat("reprojection")[ok], 90, 2),
        "final_reprojection_px_median": _q(final, 50, 2),
        "joints_in_labelled_hands": {"triangulated": round(float((joints == 1).mean()), 3) if len(joints) else None,
                                     "filled": round(float((joints == 2).mean()), 3) if len(joints) else None},
        "wrist_depth_m": {"min": _q(depth, 0), "p5": _q(depth, 5), "median": _q(depth, 50), "p95": _q(depth, 95),
                          "max": _q(depth, 100)},
    }


def per_frame(labels: StereoLabels, frames) -> dict:
    """Per-frame statistics for the given frame indices (JSON-ready; null where not applicable)."""
    def rounded(a):
        return np.where(np.isfinite(a), np.round(a, 2), None).tolist()
    seen = labels.kp2d[frames][..., 2].max(-1) > 0
    return {"frame": [int(i) for i in frames],
            "hands_left_view": seen[:, 0].sum(-1).tolist(), "hands_right_view": seen[:, 1].sum(-1).tolist(),
            "reason": labels.reason[frames].tolist(),
            "triangulated_joints": (labels.status[frames] == 1).sum(-1).tolist(),
            "epipolar_px": rounded(labels.epipolar[frames]), "reprojection_px": rounded(labels.reprojection[frames]),
            "valid": labels.valid[frames].astype(int).tolist()}
