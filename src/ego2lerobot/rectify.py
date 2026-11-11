"""Hand detection on a rectilinear view of fisheye video, with keypoints mapped back to the original pixels.

MediaPipe's hand models were trained mostly on ordinary (rectilinear) images. In ultra-wide fisheye video, hands away
from the image centre are bent and squeezed and are found less often. `PinholeView` resamples every frame to a pinhole
view of the same orientation with a chosen horizontal field of view, MediaPipe runs on that, and each keypoint is
mapped back through the fisheye model. The rest of the pipeline is unchanged: keypoints stay in original-video
pixels and the dataset keeps the original image and intrinsics.

What falls outside the pinhole view (the far edges and corners of the fisheye frame) is not searched. A narrower view
keeps the centre sharper; a wider one reaches further out but shrinks the centre. 100–110° is a reasonable start for a
~120° fisheye.
"""
from __future__ import annotations

import numpy as np

from .camera import Camera


class PinholeView:
    """Remap frames decoded at `size` from a fisheye `camera` to a pinhole view of `hfov` degrees, same size."""

    def __init__(self, camera: Camera, hfov: float = 105.0, width: int = 960):
        if camera.model != "fisheye":
            raise ValueError("--rectify needs a fisheye camera (--intrinsics with model fisheye, --camera or --fisheye-hfov)")
        import cv2  # OpenCV (Apache 2.0) is installed with mediapipe

        self._remap, self._linear, self._border = cv2.remap, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
        self.source = camera
        dw = min(width, camera.width)
        dh = round(camera.height * dw / camera.width / 2) * 2
        self.size = (dw, dh)
        self.pinhole = Camera.from_fov(dw, dh, hfov)
        self.tag = f"pinhole{hfov:g}"
        u, v = np.meshgrid(np.arange(dw, dtype=np.float64), np.arange(dh, dtype=np.float64))
        src = camera.scaled(dw, dh).project(self._points(np.column_stack([u.ravel(), v.ravel()])))
        self.map_x = src[:, 0].reshape(dh, dw).astype(np.float32)
        self.map_y = src[:, 1].reshape(dh, dw).astype(np.float32)

    def _points(self, uv: np.ndarray) -> np.ndarray:
        """Pinhole-view pixels (N, 2) → camera-frame points on the z = 1 plane (N, 3)."""
        rays = self.pinhole.undistort(np.asarray(uv, np.float64))
        return np.column_stack([rays, np.ones(len(rays))])

    def image(self, rgb: np.ndarray) -> np.ndarray:
        """A frame decoded at `size` (H, W, 3) → the pinhole view (H, W, 3); outside the fisheye frame is black."""
        return self._remap(rgb, self.map_x, self.map_y, self._linear, borderMode=self._border)

    def to_source(self, uv: np.ndarray) -> np.ndarray:
        """Pinhole-view pixels (N, 2) → pixels in the original fisheye video (N, 2)."""
        return self.source.project(self._points(uv))
