"""Stereo video input (two synchronised files, or one side-by-side file), hand detection in both views with MediaPipe,
and the optional RTMPose second model fused per view."""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from .export import frames_between
from .hands import HandTrack, model_path

RTMPOSE_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
               "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip")


class StereoVideo:
    """Left and right views. With one file, its left half is the left camera and its right half the right camera.
    Two files must start together at the same frame rate; frames are paired by index."""

    def __init__(self, left: str | Path, right: str | Path | None = None):
        self.left, self.right = Path(left), Path(right) if right else None
        with av.open(str(self.left)) as c:
            s = c.streams.video[0]
            self.rate, w, self.height = s.average_rate, s.codec_context.width, s.codec_context.height
        self.fps = float(self.rate)
        self.width = w if self.right else w // 2
        if self.right:
            with av.open(str(self.right)) as c:
                s = c.streams.video[0]
                if (s.codec_context.width, s.codec_context.height) != (w, self.height) or float(s.average_rate) != self.fps:
                    raise ValueError(f"{self.left.name} and {self.right.name} differ in size or frame rate")

    @property
    def name(self) -> str:
        return self.left.stem

    def size(self, width: int | None = None) -> tuple[int, int]:
        width = width or self.width
        return width, round(width * self.height / self.width / 2) * 2

    def frames(self, start: int = 0, end: int | None = None, width: int | None = None):
        """Yield (frame index, left RGB, right RGB) for start ≤ i < end, each view `width` wide."""
        width, height = self.size(width)
        end = 10**9 if end is None else end
        if self.right is None:
            for i, image in frames_between(self.left, start, end, self.fps, 2 * width, height):
                yield i, image[:, :width], image[:, width:]
            return
        gl = frames_between(self.left, start, end, self.fps, width, height)
        gr = frames_between(self.right, start, end, self.fps, width, height)
        a, b = next(gl, None), next(gr, None)
        while a is not None and b is not None:
            if a[0] == b[0]:
                yield a[0], a[1], b[1]
                a, b = next(gl, None), next(gr, None)
            elif a[0] < b[0]:
                a = next(gl, None)
            else:
                b = next(gr, None)


def _slots(result, width: int, height: int):
    """MediaPipe result → kp (2, 21, 2), world (2, 21, 3), score (2,), slot 0 the left hand. Two hands with the same
    handedness label are told apart by their position in the image."""
    hands = [(h[0].category_name == "Left", h[0].score, lms, w)
             for lms, w, h in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness)]
    if len(hands) == 2 and hands[0][0] == hands[1][0]:
        hands.sort(key=lambda h: np.mean([p.x for p in h[2]]))
        sides = [0, 1]
    else:
        sides = [0 if h[0] else 1 for h in hands]
    kp, world, score = np.zeros((2, 21, 2), np.float32), np.zeros((2, 21, 3), np.float32), np.zeros(2, np.float32)
    for side, (_, sc, lms, wlms) in zip(sides, hands):
        if sc > score[side]:
            kp[side] = [(p.x * width, p.y * height) for p in lms]
            world[side] = [(p.x, p.y, p.z) for p in wlms]
            score[side] = sc
    return kp, world, score


def detect_stereo(video: StereoVideo, detect_width: int = 960, min_confidence: float = 0.4) -> list[HandTrack]:
    """MediaPipe Hand Landmarker (video mode) on both views in one pass → [left view track, right view track]."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    def landmarker():
        return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path())), running_mode=vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=min_confidence, min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence))

    empty = (np.zeros((2, 21, 2), np.float32), np.zeros((2, 21, 3), np.float32), np.zeros(2, np.float32))
    rows = ([], [])
    with landmarker() as lm_left, landmarker() as lm_right:
        for i, *images in video.frames(width=min(detect_width, video.width)):
            for lm, image, out in zip((lm_left, lm_right), images, rows):
                result = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)),
                                             int(i * 1000 / video.fps))
                out.extend([empty] * (i - len(out)))  # frames the decoder skipped
                out.append(_slots(result, video.width, video.height))
    if not rows[0]:
        raise ValueError(f"{video.name}: no video frames could be decoded")
    return [HandTrack(*(np.array(x) for x in zip(*out)), video.fps, video.width, video.height) for out in rows]


def _boxes(track: HandTrack, max_gap: int = 15) -> np.ndarray:
    """Square boxes (N, 2, 4) around the MediaPipe hands, padded by 35 %; frames without a hand borrow the nearest box
    within `max_gap` frames, so the second model can recover short MediaPipe drop-outs."""
    n = len(track.score)
    boxes = np.full((n, 2, 4), np.nan)
    lo, hi = track.kp.min(2), track.kp.max(2)  # (N, 2, 2)
    centre, half = (lo + hi) / 2, 0.675 * np.maximum((hi - lo).max(-1), 0.03 * track.height)
    found = track.score > 0
    boxes[found] = np.concatenate([centre - half[..., None], centre + half[..., None]], -1)[found]
    for s in range(2):
        ok = np.flatnonzero(found[:, s])
        if not len(ok):
            continue
        nearest = ok[np.clip(np.searchsorted(ok, np.arange(n)), 0, len(ok) - 1)]
        before = ok[np.clip(np.searchsorted(ok, np.arange(n)) - 1, 0, len(ok) - 1)]
        nearest = np.where(np.abs(before - np.arange(n)) < np.abs(nearest - np.arange(n)), before, nearest)
        fill = ~found[:, s] & (np.abs(nearest - np.arange(n)) <= max_gap)
        boxes[fill, s] = boxes[nearest[fill], s]
    return boxes


def rtmpose_views(video: StereoVideo, tracks: list[HandTrack]) -> list[tuple[np.ndarray, np.ndarray]]:
    """RTMPose-m hand keypoints (rtmlib, ONNX on CPU) inside boxes from the MediaPipe tracks, for both views:
    [(kp (N, 2, 21, 2), confidence (N, 2, 21))].

    The rtmlib code and MMPose are Apache 2.0, but these weights were trained on hand datasets that include
    research-only ones (FreiHAND, RHD), so the second model is off by default."""
    from rtmlib import RTMPose

    pose = RTMPose(onnx_model=RTMPOSE_URL, model_input_size=(256, 256), backend="onnxruntime", device="cpu")
    n = len(tracks[0].score)
    boxes = [_boxes(t) for t in tracks]
    out = [(np.zeros((n, 2, 21, 2)), np.zeros((n, 2, 21))) for _ in tracks]
    for i, *images in video.frames():
        if i >= n:
            break
        for v, image in enumerate(images):
            idx = [s for s in range(2) if np.isfinite(boxes[v][i, s, 0])]
            if idx:
                kp, score = pose(np.ascontiguousarray(image[:, :, ::-1]), bboxes=[boxes[v][i, s].tolist() for s in idx])
                for m, s in enumerate(idx):
                    out[v][0][i, s], out[v][1][i, s] = kp[m], np.clip(score[m], 0, 1)
    return out


def fuse(track: HandTrack, rtm: tuple[np.ndarray, np.ndarray] | None = None, agree: float = 0.15):
    """One view's 2D keypoints → (kp (N, 2, 21, 2), confidence (N, 2, 21), hand score (N, 2)).

    MediaPipe alone gives each joint its hand's score. With RTMPose, a joint is the confidence-weighted mean of the two
    models where they agree (within `agree` × hand size), RTMPose where it is confident (> 0.5) and MediaPipe missed
    or disagrees, and MediaPipe otherwise."""
    present = (track.score > 0)[..., None] & np.ones(21, bool)
    kp = track.kp.astype(float)
    conf = np.where(present, track.score[..., None], 0.0)
    if rtm is None:
        return kp, conf, track.score.astype(float)
    r_kp, r_conf = rtm
    size = np.maximum((kp.max(2) - kp.min(2)).max(-1), 30.0)[..., None]  # (N, 2, 1) pixels
    r_ok = r_conf > 0.3
    close = present & r_ok & (np.linalg.norm(kp - r_kp, axis=-1) < agree * size)
    use_r = r_ok & (r_conf > 0.5) & ~close
    w = 0.6
    mean = (w * kp + r_conf[..., None] * r_kp) / (w + r_conf[..., None])
    out = np.where(close[..., None], mean, np.where(use_r[..., None], r_kp, kp))
    out_conf = np.where(close, np.maximum(conf, r_conf), np.where(use_r, r_conf, conf))
    score = np.where(track.score > 0, track.score, np.where(use_r.any(-1), out_conf.mean(-1), 0.0))
    return out, out_conf, score


def caption_video(video: StereoVideo, cache: Path, width: int = 640) -> Path:
    """A video of the left view for the captioner: the left file itself, or a small copy cut from a side-by-side file."""
    if video.right is not None:
        return video.left
    path = cache / f"{video.name}.left.mp4"
    if not path.exists():
        w, h = video.size(width)
        with av.open(str(path), "w") as out:
            stream = out.add_stream("h264", rate=video.rate)
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            for _, left, _ in video.frames(width=w):
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(left), format="rgb24")):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
    return path
