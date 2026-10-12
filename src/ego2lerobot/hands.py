"""Hand keypoints from video, lifted to 3D, and wrist/grip pseudo-actions."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from .camera import Camera

SIDES = ("left", "right")
JOINTS = ("wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip", "index_tip",
          "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
          "little_mcp", "little_pip", "little_dip", "little_tip")
BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17))

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DETECT_WIDTH = 960


def model_path() -> Path:
    """MediaPipe Hand Landmarker (Apache 2.0), downloaded once to ~/.cache/ego2lerobot."""
    path = Path.home() / ".cache" / "ego2lerobot" / "hand_landmarker.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(path)
    return path


@dataclass
class HandTrack:
    """Per-frame detections for one video; index 0 is the left hand, 1 the right.

    kp:    (N, 2, 21, 2) pixels in the original video
    world: (N, 2, 21, 3) MediaPipe metric hand shape, metres, centred on the hand
    score: (N, 2) handedness confidence, 0 where the hand was not found
    """

    kp: np.ndarray
    world: np.ndarray
    score: np.ndarray
    fps: float
    width: int
    height: int

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, kp=self.kp, world=self.world, score=self.score, fps=self.fps, width=self.width, height=self.height)

    @classmethod
    def load(cls, path: str | Path) -> HandTrack:
        d = np.load(path)
        return cls(d["kp"], d["world"], d["score"], float(d["fps"]), int(d["width"]), int(d["height"]))

    def slice(self, start: int, end: int) -> HandTrack:
        return HandTrack(self.kp[start:end], self.world[start:end], self.score[start:end], self.fps, self.width, self.height)

    def summary(self) -> dict:
        s = self.score
        return {"frames": len(s), "any_hand": round(float((s.max(1) > 0).mean()), 3), "two_hands": round(float((s.min(1) > 0).mean()), 3)}


def detect(video: str | Path, min_confidence: float = 0.4, progress: bool = True) -> HandTrack:
    """Run MediaPipe Hand Landmarker (video mode, tracking between frames) on every frame."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path())), running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=min_confidence, min_hand_presence_confidence=min_confidence, min_tracking_confidence=min_confidence)
    container = av.open(str(video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    fps = float(stream.average_rate)
    width, height = stream.codec_context.width, stream.codec_context.height
    dw = min(DETECT_WIDTH, width)
    dh = round(height * dw / width / 2) * 2
    kp, world, score = [], [], []
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for n, frame in enumerate(container.decode(stream)):
            image = frame.to_ndarray(format="rgb24", width=dw, height=dh)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=image), int(n * 1000 / fps))
            f_kp = np.zeros((2, 21, 2), np.float32)
            f_world = np.zeros((2, 21, 3), np.float32)
            f_score = np.zeros(2, np.float32)
            for lms, wlms, hand in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
                # Checked on Egocentric-10K: hands labelled "Left" sit in the left half of the frame 85% of the time,
                # so the labels already match a non-mirrored first-person view.
                side = 0 if hand[0].category_name == "Left" else 1
                if hand[0].score <= f_score[side]:
                    continue
                f_kp[side] = [(p.x * width, p.y * height) for p in lms]
                f_world[side] = [(p.x, p.y, p.z) for p in wlms]
                f_score[side] = hand[0].score
            kp.append(f_kp)
            world.append(f_world)
            score.append(f_score)
            if progress and n and n % 3000 == 0:
                print(f"{Path(video).name}: {n} frames", flush=True)
    container.close()
    if not kp:
        raise ValueError(f"{video}: no video frames could be decoded")
    return HandTrack(np.array(kp), np.array(world), np.array(score), fps, width, height)


def place_hand(world: np.ndarray, rays: np.ndarray) -> np.ndarray | None:
    """Put the metric hand shape `world` (21, 3) in the camera frame so that it projects onto `rays` (21, 2).

    Solves (Xj + tx) − aj (Zj + tz) = 0 and (Yj + ty) − bj (Zj + tz) = 0 for the translation t by least squares.
    The hand's scale comes from MediaPipe's average hand size, so depth carries a few centimetres of error.
    """
    a, b = rays[:, 0], rays[:, 1]
    n = len(world)
    A = np.zeros((2 * n, 3))
    y = np.zeros(2 * n)
    A[:n, 0] = 1
    A[:n, 2] = -a
    y[:n] = a * world[:, 2] - world[:, 0]
    A[n:, 1] = 1
    A[n:, 2] = -b
    y[n:] = b * world[:, 2] - world[:, 1]
    t, *_ = np.linalg.lstsq(A, y, rcond=None)
    return world + t if t[2] > 0.05 else None


def end_effector(p: np.ndarray) -> np.ndarray:
    """Wrist position, palm rotation as its first two columns (6D) and thumb–index pinch distance: 10 numbers.

    Palm frame: x from the little-finger knuckle to the index knuckle, z the palm normal, y = z × x.
    """
    wrist, index, little = p[0], p[5], p[17]
    x = index - little
    z = np.cross(index - wrist, little - wrist)
    if np.linalg.norm(x) < 1e-6 or np.linalg.norm(z) < 1e-9:
        return np.zeros(10)
    x = x / np.linalg.norm(x)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.concatenate([wrist, x, y, [np.linalg.norm(p[4] - p[8])]])


def frame_labels(track: HandTrack, i: int, camera: Camera, out_width: int, out_height: int):
    """Labels for frame i: kp2d (2, 21, 3) in the output image, kp3d (2, 21, 3), valid (2,), ee (2, 10)."""
    sx, sy = out_width / track.width, out_height / track.height
    kp2d = np.zeros((2, 21, 3))
    kp3d = np.zeros((2, 21, 3))
    valid = np.zeros(2)
    ee = np.zeros((2, 10))
    for s in range(2):
        score = float(track.score[i, s])
        if score <= 0:
            continue
        uv = track.kp[i, s].astype(np.float64)
        p3 = place_hand(track.world[i, s].astype(np.float64), camera.undistort(uv))
        if p3 is None:
            continue
        kp2d[s] = np.column_stack([uv[:, 0] * sx, uv[:, 1] * sy, np.full(21, score)])
        kp3d[s] = p3
        valid[s] = 1
        ee[s] = end_effector(p3)
    return kp2d, kp3d, valid, ee
