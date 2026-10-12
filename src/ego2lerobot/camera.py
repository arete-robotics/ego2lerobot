"""Camera models: turn pixel coordinates into undistorted normalized coordinates."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Camera:
    """Pinhole or OpenCV fisheye (Kannala–Brandt, k1–k4) intrinsics for a width × height image."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "pinhole"  # "pinhole" (no distortion) or "fisheye"
    k: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_json(cls, path: str | Path, width: int | None = None, height: int | None = None) -> Camera:
        """Read intrinsics like Egocentric-10K's intrinsics.json (model, image_width, image_height, fx, fy, cx, cy, k1–k4)."""
        d = json.loads(Path(path).read_text())
        cam = cls(width=int(d.get("image_width", width or 0)), height=int(d.get("image_height", height or 0)),
                  fx=float(d["fx"]), fy=float(d["fy"]), cx=float(d["cx"]), cy=float(d["cy"]),
                  model=d.get("model", "pinhole"), k=tuple(float(d.get(f"k{i}", 0.0)) for i in range(1, 5)))
        if cam.model not in ("pinhole", "fisheye"):
            raise ValueError(f"unsupported camera model {cam.model!r}; use pinhole or fisheye")
        return cam.scaled(width, height) if width and height and (width, height) != (cam.width, cam.height) else cam

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_degrees: float) -> Camera:
        """A distortion-free guess for videos without calibration, from the horizontal field of view."""
        f = (width / 2) / math.tan(math.radians(hfov_degrees) / 2)
        return cls(width=width, height=height, fx=f, fy=f, cx=width / 2, cy=height / 2)

    def scaled(self, width: int, height: int) -> Camera:
        sx, sy = width / self.width, height / self.height
        return replace(self, width=width, height=height, fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy)

    def vector(self) -> np.ndarray:
        """fx fy cx cy k1 k2 k3 k4, as stored per frame in the dataset."""
        return np.array([self.fx, self.fy, self.cx, self.cy, *self.k], np.float32)

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) → normalized undistorted coordinates (N, 2), i.e. (X/Z, Y/Z) in OpenCV camera axes."""
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        if self.model == "pinhole":
            return np.stack([x, y], -1)
        k1, k2, k3, k4 = self.k
        theta_d = np.sqrt(x * x + y * y)
        theta = theta_d.copy()
        for _ in range(10):  # Newton on θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸) = θd
            t2 = theta * theta
            f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
            df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
            theta = theta - f / df
        scale = np.where(theta_d > 1e-9, np.tan(theta) / np.maximum(theta_d, 1e-9), 1.0)
        return np.stack([x * scale, y * scale], -1)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Camera-frame points (N, 3) → pixels (N, 2). Inverse of `undistort` for points in front of the camera."""
        a, b = points[:, 0] / points[:, 2], points[:, 1] / points[:, 2]
        if self.model == "fisheye":
            k1, k2, k3, k4 = self.k
            r = np.sqrt(a * a + b * b)
            theta = np.arctan(r)
            t2 = theta * theta
            theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            s = np.where(r > 1e-9, theta_d / np.maximum(r, 1e-9), 1.0)
            a, b = a * s, b * s
        return np.stack([self.fx * a + self.cx, self.fy * b + self.cy], -1)
