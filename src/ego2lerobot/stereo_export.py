"""Write stereo episodes as a LeRobot v3 dataset: both camera streams, the calibration, triangulated hands, the IMU
when given, per-frame triangulation statistics, and preview videos."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .export import LEFT_COLOUR, RIGHT_COLOUR
from .export import features as mono_features
from .hands import BONES, JOINTS, SIDES, end_effector
from .segments import Segment
from .stereo import REASONS, StereoLabels, StereoRig, per_frame, summarize
from .stereo_video import StereoVideo


class Imu:
    """IMU samples from a CSV with a header row. The first column is `t`, seconds since the first video frame; the other
    columns (for example gx, gy, gz in rad/s and ax, ay, az in m/s²) are passed through unchanged."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        header = [h.strip() for h in self.path.open().readline().split(",")]
        if header[0] != "t":
            raise ValueError(f"{self.path.name}: the first column must be t (seconds since the first video frame)")
        data = np.loadtxt(self.path, delimiter=",", skiprows=1, ndmin=2)
        data = data[np.argsort(data[:, 0], kind="stable")]
        self.names, self.t, self.values = header[1:], data[:, 0], data[:, 1:]

    def mean(self, t0: float, t1: float) -> np.ndarray:
        """Mean of the samples with t0 ≤ t < t1 (zeros when there are none)."""
        a, b = np.searchsorted(self.t, [t0, t1])
        return self.values[a:b].mean(0) if b > a else np.zeros(len(self.names))

    def write(self, path: Path, t0: float, t1: float) -> None:
        a, b = np.searchsorted(self.t, [t0, t1])
        np.savetxt(path, np.column_stack([self.t[a:b], self.values[a:b]]), delimiter=",", fmt="%.6g",
                   header=",".join(["t", *self.names]), comments="")


@dataclass
class StereoEpisode:
    video: StereoVideo
    rig: StereoRig  # at the video's size
    labels: StereoLabels
    segment: Segment
    task: str
    imu: Imu | None = None
    info: dict = field(default_factory=dict)


def features(width: int, height: int, imu_names=()) -> dict:
    mono = mono_features(width, height)
    image = mono["observation.images.ego"]
    lens = ["fx", "fy", "cx", "cy", "d0", "d1", "d2", "d3", "d4"]
    out = {
        "observation.images.left": image, "observation.images.right": dict(image),
        "observation.intrinsics.left": {"dtype": "float32", "shape": (9,), "names": lens},
        "observation.intrinsics.right": {"dtype": "float32", "shape": (9,), "names": lens},
        "observation.extrinsics": {"dtype": "float32", "shape": (12,),
                                   "names": [f"r{i}{j}" for i in range(3) for j in range(3)] + ["tx", "ty", "tz"]},
        "observation.hands.kp2d.left": mono["observation.hands.kp2d"],
        "observation.hands.kp2d.right": dict(mono["observation.hands.kp2d"]),
        "observation.hands.kp3d": mono["observation.hands.kp3d"],
        "observation.hands.kp3d_status": {"dtype": "float32", "shape": (42,), "names": [f"{s}_{j}" for s in SIDES for j in JOINTS]},
        **{k: mono[k] for k in ("observation.hands.valid", "observation.hands.ee", "action")},
    }
    if imu_names:
        out["observation.imu"] = {"dtype": "float32", "shape": (len(imu_names),), "names": list(imu_names)}
    return out


def frame_labels(labels: StereoLabels, i: int, sx: float, sy: float) -> dict:
    kp2d = labels.kp2d[i].copy()
    kp2d[..., 0] *= sx
    kp2d[..., 1] *= sy
    valid = labels.valid[i]
    ee = np.stack([end_effector(labels.kp3d[i, s]) if valid[s] else np.zeros(10) for s in range(2)])
    f32 = lambda a: np.asarray(a, np.float32).reshape(-1)  # noqa: E731
    return {"observation.hands.kp2d.left": f32(kp2d[0]), "observation.hands.kp2d.right": f32(kp2d[1]),
            "observation.hands.kp3d": f32(np.nan_to_num(labels.kp3d[i])), "observation.hands.kp3d_status": f32(labels.status[i]),
            "observation.hands.valid": f32(valid), "observation.hands.ee": f32(ee)}


class Preview:
    """Both views side by side with the stereo hand skeletons reprojected into each, and a panel with the 3D wrist
    position and grip in the left camera frame. Hands seen but not triangulated are drawn thin and grey."""

    PANEL = 64

    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.width, self.height = width, height
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("h264", rate=fps)
        self.stream.width, self.stream.height, self.stream.pix_fmt = 2 * width, height + self.PANEL, "yuv420p"
        self.font = ImageFont.load_default(size=max(12, height // 28))

    def add(self, left: np.ndarray, right: np.ndarray, labels: StereoLabels, i: int, rig: StereoRig, seconds: float):
        """`rig` at the preview size."""
        canvas = Image.new("RGB", (2 * self.width, self.height + self.PANEL), (18, 18, 20))
        sx, sy = self.width / labels.width, self.height / labels.height
        for v, image in enumerate((left, right)):
            view = Image.fromarray(image)
            draw = ImageDraw.Draw(view)
            for s, colour in enumerate((LEFT_COLOUR, RIGHT_COLOUR)):
                if labels.valid[i, s]:
                    known = np.isfinite(labels.kp3d[i, s, :, 0])
                    pts = np.full((21, 2), np.nan)
                    pts[known] = rig.project(labels.kp3d[i, s][known], v)
                    self._skeleton(draw, pts, colour, 3, labels.status[i, s])
                elif labels.kp2d[i, v, s, :, 2].max() > 0:
                    self._skeleton(draw, labels.kp2d[i, v, s, :, :2] * (sx, sy), (150, 150, 150), 1, None)
            draw.text((8, 6), ("Left camera", "Right camera")[v], font=self.font, fill=(255, 255, 255))
            canvas.paste(view, (v * self.width, 0))
        draw = ImageDraw.Draw(canvas)
        for s, (name, colour) in enumerate((("Left hand", LEFT_COLOUR), ("Right hand", RIGHT_COLOUR))):
            y = self.height + 6 + s * (self.PANEL // 2 - 2)
            if labels.valid[i, s]:
                p = labels.kp3d[i, s]
                text = (f"{name}  wrist x {p[0, 0]:+.2f}  y {p[0, 1]:+.2f}  z {p[0, 2]:.2f} m   "
                        f"grip {np.linalg.norm(p[4] - p[8]) * 100:.1f} cm   ({(labels.status[i, s] == 1).sum()}/21 joints triangulated)")
            else:
                text = f"{name}  {REASONS[int(labels.reason[i, s])]}"
            draw.text((10, y), text, font=self.font, fill=colour if labels.valid[i, s] else (150, 150, 150))
        draw.text((2 * self.width - 260, self.height + 6), f"{seconds:6.1f} s   stereo, left camera frame",
                  font=self.font, fill=(150, 150, 155))
        for packet in self.stream.encode(av.VideoFrame.from_image(canvas)):
            self.container.mux(packet)

    @staticmethod
    def _skeleton(draw, pts, colour, width, status):
        for a, b in BONES:
            if np.isfinite(pts[[a, b]]).all():
                draw.line([tuple(pts[a]), tuple(pts[b])], fill=colour, width=width)
        if status is not None:
            for j in np.flatnonzero(np.isfinite(pts[:, 0])):
                u, v = pts[j]
                draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(255, 255, 255) if status[j] == 1 else (140, 140, 140))

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def write_stereo_dataset(episodes: list[StereoEpisode], out: str | Path, repo_id: str, width: int = 640, step: int = 2,
                         preview_dir: str | Path | None = None, previews: int = 12) -> dict:
    """Write `episodes` to a fresh LeRobot dataset at `out`, keeping every `step`-th frame, plus meta/calibration.json,
    meta/triangulation_stats.json, meta/episodes_info.json and meta/imu/episode_NNNNNN.csv."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not episodes:
        raise ValueError("no episodes to write")
    first = episodes[0]
    width, height = first.video.size(width)
    fps = round(first.video.fps / step)
    imu_names = first.imu.names if first.imu else []
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features(width, height, imu_names), root=out,
                                    robot_type="human_hands", use_videos=True)
    every = max(1, len(episodes) // max(1, previews)) if preview_dir else 0
    meta, stats, parts, imu_slices = [], [], [], []
    for n, ep in enumerate(episodes):
        video, lab = ep.video, ep.labels
        if round(video.fps / step) != fps or video.size(width) != (width, height) or (ep.imu.names if ep.imu else []) != imu_names:
            print(f"skipping {video.name}: frame rate, aspect ratio or IMU columns differ from the first episode")
            continue
        stored = ep.rig.scaled(width, height)
        constants = {"observation.intrinsics.left": stored.left.vector(), "observation.intrinsics.right": stored.right.vector(),
                     "observation.extrinsics": np.concatenate([stored.R.ravel(), stored.T]).astype(np.float32)}
        preview = None
        if every and n % every == 0:
            Path(preview_dir).mkdir(parents=True, exist_ok=True)
            preview = Preview(Path(preview_dir) / f"episode_{len(meta):06d}.mp4", width, height, fps)
        rows, kept = [], []
        for i, left, right in video.frames(ep.segment.start, ep.segment.end, width):
            if (i - ep.segment.start) % step or i >= len(lab.valid):
                continue
            row = {"observation.images.left": left, "observation.images.right": right, **constants,
                   **frame_labels(lab, i, width / lab.width, height / lab.height)}
            if ep.imu:
                row["observation.imu"] = ep.imu.mean((i - step / 2) / video.fps, (i + step / 2) / video.fps).astype(np.float32)
            rows.append(row)
            kept.append(i)
            if preview:
                preview.add(left, right, lab, i, stored, i / video.fps)
        if preview:
            preview.close()
        if not rows:
            continue
        for m, row in enumerate(rows):
            row["action"] = rows[min(m + 1, len(rows) - 1)]["observation.hands.ee"]
            row["task"] = ep.task
            dataset.add_frame(row)
        dataset.save_episode()
        index = len(meta)
        if ep.imu:
            imu_slices.append((index, ep.imu, ep.segment.start / video.fps, ep.segment.end / video.fps))
        source = {"left": video.left.name, "right": video.right.name if video.right else None,
                  "side_by_side": video.right is None, "start_frame": ep.segment.start, "end_frame": ep.segment.end}
        parts.append((lab, kept, ep.rig))
        stats.append({"episode_index": index, "source": source, "summary": summarize(parts[-1:]),
                      "per_frame": per_frame(lab, kept)})
        valid = lab.valid[kept]
        meta.append({"episode_index": index, "task": ep.task, "frames": len(rows),
                     "any_hand": round(float(valid.any(1).mean()), 3), "two_hands": round(float(valid.all(1).mean()), 3),
                     "source": source, **ep.info})
    dataset.finalize()
    meta_dir = out / "meta"
    for index, imu, t0, t1 in imu_slices:
        (meta_dir / "imu").mkdir(exist_ok=True)
        imu.write(meta_dir / "imu" / f"episode_{index:06d}.csv", t0, t1)
    calibration = {"original": first.rig.to_dict(), "stored": first.rig.scaled(width, height).to_dict(),
                   "frame": "points in metres in the left camera frame: x right, y down, z forward"}
    (meta_dir / "calibration.json").write_text(json.dumps(calibration, indent=1))
    (meta_dir / "episodes_info.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    overall = summarize(parts) if parts else {}
    (meta_dir / "triangulation_stats.json").write_text(json.dumps({
        "about": "Per stored frame and hand: reason codes " + json.dumps({str(k): v for k, v in REASONS.items()})
                 + "; pixel errors at the original video size.", "summary": overall, "episodes": stats}, indent=None))
    summary = {"episodes": len(meta), "frames": sum(m["frames"] for m in meta), "fps": fps,
               "minutes": round(sum(m["frames"] for m in meta) / fps / 60, 1), **{k: overall.get(k) for k in (
                   "valid_triangulation_rate", "reprojection_px_median")}}
    print(json.dumps(summary))
    return summary
