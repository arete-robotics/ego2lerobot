"""Write stereo episodes as a LeRobot v3 dataset: both camera streams, the calibration, triangulated hands, the head
pose and the hands in a world frame when given, the IMU when given, per-frame triangulation statistics, and preview
videos."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .angles import ANGLE_NAMES, joint_angles
from .export import LEFT_COLOUR, RIGHT_COLOUR, remove_empty_dirs
from .export import features as mono_features
from .hands import BONES, JOINTS, SIDES, end_effector
from .segments import Segment
from .stereo import REASONS, Checks, StereoLabels, StereoRig, per_frame, summarize
from .stereo_video import StereoVideo, h264_stream


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
    head: object | None = None  # odometry.HeadPose for the whole source clip


POSE_NAMES = ["x", "y", "z", "r1x", "r1y", "r1z", "r2x", "r2y", "r2z"]


def features(width: int, height: int, imu_names=(), head: bool = False) -> dict:
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
        "observation.hands.joint_angles": {"dtype": "float32", "shape": (2 * len(ANGLE_NAMES),),
                                           "names": [f"{s}_{a}" for s in SIDES for a in ANGLE_NAMES]},
        **{k: mono[k] for k in ("observation.hands.valid", "observation.hands.ee", "action")},
    }
    if imu_names:
        out["observation.imu"] = {"dtype": "float32", "shape": (len(imu_names),), "names": list(imu_names)}
    if head:
        out["observation.head.pose"] = {"dtype": "float32", "shape": (9,), "names": POSE_NAMES}
        out["observation.head.valid"] = {"dtype": "float32", "shape": (1,), "names": ["valid"]}
        out["observation.hands.kp3d_world"] = dict(mono["observation.hands.kp3d"])
        out["observation.hands.ee_world"] = dict(mono["observation.hands.ee"])
    return out


def head_labels(head, labels: StereoLabels, i: int) -> dict:
    """The head pose of frame i (left camera → world: position and the first two columns of the rotation) and the
    hands moved into that world frame. While the pose is not solved it is held and `valid` is 0; the world-frame
    hands are then zeros, as they are for a hand that is not in 3D."""
    f32 = lambda a: np.asarray(a, np.float32).reshape(-1)  # noqa: E731
    solved = i < len(head.status) and head.status[i] == 1
    j = min(i, len(head.status) - 1)
    world, ee = np.zeros((2, 21, 3)), np.zeros((2, 10))
    if solved:
        for s in range(2):
            if labels.valid[i, s]:
                world[s] = np.nan_to_num(head.to_world(labels.kp3d[i, s], i))
                ee[s] = end_effector(world[s])
    return {"observation.head.pose": f32(head.vector(j)), "observation.head.valid": f32([float(solved)]),
            "observation.hands.kp3d_world": f32(world), "observation.hands.ee_world": f32(ee)}


def frame_labels(labels: StereoLabels, i: int, sx: float, sy: float) -> dict:
    kp2d = labels.kp2d[i].copy()
    kp2d[..., 0] *= sx
    kp2d[..., 1] *= sy
    valid = labels.valid[i]
    ee = np.stack([end_effector(labels.kp3d[i, s]) if valid[s] else np.zeros(10) for s in range(2)])
    f32 = lambda a: np.asarray(a, np.float32).reshape(-1)  # noqa: E731
    return {"observation.hands.kp2d.left": f32(kp2d[0]), "observation.hands.kp2d.right": f32(kp2d[1]),
            "observation.hands.kp3d": f32(np.nan_to_num(labels.kp3d[i])), "observation.hands.kp3d_status": f32(labels.status[i]),
            "observation.hands.joint_angles": f32(np.nan_to_num(joint_angles(labels.kp3d[i]))),
            "observation.hands.valid": f32(valid), "observation.hands.ee": f32(ee)}


class Preview:
    """Both views side by side with the stereo hand skeletons reprojected into each, and a panel with the 3D wrist
    position and grip in the left camera frame. Hands seen but not triangulated are drawn thin and grey."""

    PANEL = 64

    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.width, self.height = width, height
        self.container = av.open(str(path), "w")
        self.stream = h264_stream(self.container, fps, 2 * width, height + self.PANEL)
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


def checks_dict(checks: Checks, rig: StereoRig) -> dict:
    """The triangulation checks as given (pixel limits as shares of the image height) and in pixels for `rig`."""
    return {**asdict(checks), "image_height": rig.height, "max_epipolar_px": round(checks.max_epipolar * rig.height, 2),
            "max_reprojection_px": round(checks.max_reprojection * rig.height, 2)}


def write_grasp_events(dataset: str | Path, grasp: dict) -> None:
    """meta/grasp_events.json for a stereo dataset: the same layout as `run` writes, from the triangulated joints."""
    from . import signals
    about = ("Per hand, frames where the grip closes (grasp) or opens (release). Grip aperture = thumb-tip to index-tip "
             "distance / wrist to middle-finger knuckle distance, both from the triangulated 3D joints; closed below "
             "`close`, open above `open`, a state must last `min_state_s`; no event is emitted across a gap longer than "
             "`max_gap_s` without a triangulated hand. Frames and times refer to the source video.")
    (Path(dataset) / "meta" / "grasp_events.json").write_text(json.dumps(
        {"about": about, "source": "stereo triangulation",
         "thresholds": {"close": signals.CLOSE, "open": signals.OPEN, "min_state_s": signals.MIN_STATE_S,
                        "max_gap_s": signals.MAX_GAP_S}, "videos": grasp}, indent=1))


def write_stereo_dataset(episodes: list[StereoEpisode], out: str | Path, repo_id: str, width: int = 640, step: int = 2,
                         preview_dir: str | Path | None = None, previews: int = 12, checks: Checks | None = None) -> dict:
    """Write `episodes` to a fresh LeRobot dataset at `out`, keeping every `step`-th frame, plus meta/calibration.json,
    meta/triangulation_stats.json, meta/episodes_info.json and meta/imu/episode_NNNNNN.csv. `checks` (the ones given
    to label_stereo) are recorded in triangulation_stats.json for the QC report."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not episodes:
        raise ValueError("no episodes to write")
    first = episodes[0]
    width, height = first.video.size(width)
    fps = round(first.video.fps / step)
    imu_names = first.imu.names if first.imu else []
    with_head = all(ep.head is not None for ep in episodes)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features(width, height, imu_names, with_head), root=out,
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
            if with_head:
                row.update(head_labels(ep.head, lab, i))
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
        source = {"video": video.left.name, "left": video.left.name, "right": video.right.name if video.right else None,
                  "side_by_side": video.right is None, "start_frame": ep.segment.start, "end_frame": ep.segment.end}
        parts.append((lab, kept, ep.rig))
        stats.append({"episode_index": index, "source": source, "summary": summarize(parts[-1:]),
                      "per_frame": per_frame(lab, kept)})
        valid = lab.valid[kept]
        meta.append({"episode_index": index, "task": ep.task, "frames": len(rows),
                     "any_hand": round(float(valid.any(1).mean()), 3), "two_hands": round(float(valid.all(1).mean()), 3),
                     "source": source, **ep.info})
        if with_head:
            in_range = [i for i in kept if i < len(ep.head.status)]
            meta[-1]["head_pose"] = {
                "tracked": round(float((ep.head.status[in_range] == 1).mean()), 3) if in_range else 0.0,
                "parts": sorted({int(p) for p in ep.head.part[in_range]})}
    dataset.finalize()
    remove_empty_dirs(out / "images")
    meta_dir = out / "meta"
    for index, imu, t0, t1 in imu_slices:
        (meta_dir / "imu").mkdir(exist_ok=True)
        imu.write(meta_dir / "imu" / f"episode_{index:06d}.csv", t0, t1)
    calibration = {"original": first.rig.to_dict(), "stored": first.rig.scaled(width, height).to_dict(),
                   "frame": "points in metres in the left camera frame: x right, y down, z forward"}
    rigs = {}
    for ep in episodes:  # one entry per source clip when clips came with their own calibration
        rigs.setdefault(ep.video.left.name, ep.rig.to_dict())
    if any(r != calibration["original"] for r in rigs.values()):
        calibration["note"] = "clips have their own calibration; `original` and `stored` are those of the first clip"
        calibration["clips"] = rigs
    (meta_dir / "calibration.json").write_text(json.dumps(calibration, indent=1))
    if with_head:
        clips = {}
        for ep in episodes:
            clips.setdefault(ep.video.left.name, {**ep.head.summary(), **ep.head.checks})
        (meta_dir / "head_pose.json").write_text(json.dumps({
            "about": "observation.head.pose is the pose of the left camera in a world frame that is fixed for one source "
                     "clip: x y z in metres, then the first two columns of the rotation (left camera → world). It comes "
                     "from stereo visual odometry without loop closure, so it drifts over a long clip. Where "
                     "observation.head.valid is 0 the pose is held from the last solved frame. After a break "
                     "(`breaks`) the motion in between is unknown: poses are comparable only within one part "
                     "(`head_pose.parts` in episodes_info.json). observation.hands.kp3d_world and ee_world are the "
                     "hands in that world frame.",
            "clips": clips}, indent=1))
    (meta_dir / "episodes_info.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    overall = summarize(parts) if parts else {}
    (meta_dir / "triangulation_stats.json").write_text(json.dumps({
        "about": "Per stored frame and hand: reason codes " + json.dumps({str(k): v for k, v in REASONS.items()})
                 + "; pixel errors at the original video size.", "summary": overall,
        "checks": checks_dict(checks, first.rig) if checks else None, "episodes": stats}, indent=None))
    summary = {"episodes": len(meta), "frames": sum(m["frames"] for m in meta), "fps": fps,
               "minutes": round(sum(m["frames"] for m in meta) / fps / 60, 1), **{k: overall.get(k) for k in (
                   "valid_triangulation_rate", "reprojection_px_median")}}
    print(json.dumps(summary))
    return summary
