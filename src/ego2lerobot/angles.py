"""Finger joint angles from the 21 hand joints, for retargeting to robot hands.

Per hand: 15 bend angles (5 fingers x base, middle, end joint), 4 spread angles between neighbouring
fingers, and one closure value. Angles are in radians; 0 is a straight joint. They are only as good as
the 3D joints they come from, so they are written for measured (stereo) hands.
"""
from __future__ import annotations

import numpy as np

FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
           "ring": (13, 14, 15, 16), "little": (17, 18, 19, 20)}
ANGLE_NAMES = ([f"{f}_{j}" for f in FINGERS for j in ("base", "middle", "end")]
               + ["spread_thumb_index", "spread_index_middle", "spread_middle_ring", "spread_ring_little", "closure"])


def _angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between vectors along the last axis; NaN where either is missing or has no length."""
    den = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("...i,...i->...", u, v) / den
    return np.arccos(np.clip(cos, -1.0, 1.0))


def joint_angles(kp: np.ndarray) -> np.ndarray:
    """kp (..., 21, 3) in any metric frame -> (..., 20) in the order of ANGLE_NAMES.

    Bend: the angle between the bone before a joint and the bone after it. For the four fingers the base
    joint is the knuckle (bone before it: wrist to knuckle); for the thumb it is the joint nearest the wrist.
    Spread: the angle between the first bones of two neighbouring fingers.
    Closure: mean bend of the base and middle joints of the four fingers over 90 degrees, clipped to 0..1
    (0 open hand, 1 fist)."""
    kp = np.asarray(kp, float)
    wrist = kp[..., 0, :]
    bend, first = [], []
    for a, b, c, d in FINGERS.values():
        A, B, C, D = kp[..., a, :], kp[..., b, :], kp[..., c, :], kp[..., d, :]
        bend += [_angle(A - wrist, B - A), _angle(B - A, C - B), _angle(C - B, D - C)]
        first.append(B - A)
    spread = [_angle(first[i], first[i + 1]) for i in range(4)]
    four = np.stack([bend[3 * f + j] for f in range(1, 5) for j in (0, 1)], -1)
    closure = np.clip(four.mean(-1) / (np.pi / 2), 0.0, 1.0)
    return np.stack(bend + spread + [closure], -1)
