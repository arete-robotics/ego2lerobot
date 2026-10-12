"""Write episodes as a LeRobot v3 dataset, with optional preview videos."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from .camera import Camera
from .hands import BONES, JOINTS, SIDES, HandTrack, frame_labels
from .segments import Segment

LEFT_COLOUR, RIGHT_COLOUR = (224, 161, 0), (75, 172, 198)


@dataclass
class Episode:
    video: Path
    camera: Camera
    track: HandTrack
    segment: Segment
    task: str
    info: dict = field(default_factory=dict)  # sub-steps, objects, source, ... (written to meta/episodes_info.json)


def features(width: int, height: int) -> dict:
    kp2d = [f"{s}_{j}_{a}" for s in SIDES for j in JOINTS for a in ("u", "v", "score")]
    kp3d = [f"{s}_{j}_{a}" for s in SIDES for j in JOINTS for a in ("x", "y", "z")]
    ee = [f"{s}_{a}" for s in SIDES for a in ("x", "y", "z", "r1x", "r1y", "r1z", "r2x", "r2y", "r2z", "pinch")]
    return {
        "observation.images.ego": {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]},
        "observation.intrinsics": {"dtype": "float32", "shape": (8,), "names": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]},
        "observation.hands.kp2d": {"dtype": "float32", "shape": (126,), "names": kp2d},
        "observation.hands.kp3d": {"dtype": "float32", "shape": (126,), "names": kp3d},
        "observation.hands.valid": {"dtype": "float32", "shape": (2,), "names": list(SIDES)},
        "observation.hands.ee": {"dtype": "float32", "shape": (20,), "names": ee},
        "action": {"dtype": "float32", "shape": (20,), "names": ee},
    }


def draw_hands(image: Image.Image, kp2d: np.ndarray, valid: np.ndarray, min_score: float = 0.3) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for s, colour in enumerate((LEFT_COLOUR, RIGHT_COLOUR)):
        if not valid[s]:
            continue
        pts = kp2d[s]
        for a, b in BONES:
            if pts[a, 2] >= min_score and pts[b, 2] >= min_score:
                draw.line([tuple(pts[a, :2]), tuple(pts[b, :2])], fill=colour, width=3)
        for u, v, c in pts:
            if c >= min_score:
                draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(255, 255, 255))
    return image


def frames_between(video: Path, start: int, end: int, fps: float, width: int, height: int):
    """Yield (frame index, RGB array) for frames start ≤ i < end."""
    container = av.open(str(video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
    container.seek(int(max(0.0, origin + start / fps - 1.0) / stream.time_base), stream=stream)
    for frame in container.decode(stream):
        i = round((float(frame.pts * stream.time_base) - origin) * fps)
        if i >= end:
            break
        if i >= start:
            yield i, frame.to_ndarray(format="rgb24", width=width, height=height)
    container.close()


def write_dataset(episodes: list[Episode], out: str | Path, repo_id: str, width: int = 640, step: int = 2,
                  preview_dir: str | Path | None = None, previews: int = 12) -> dict:
    """Write `episodes` to a fresh LeRobot dataset at `out`, keeping every `step`-th frame."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not episodes:
        raise ValueError("no episodes to write")
    first = episodes[0]
    height = round(width * first.track.height / first.track.width / 2) * 2
    fps = round(first.track.fps / step)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features(width, height), root=out,
                                    robot_type="human_hands", use_videos=True)
    every = max(1, len(episodes) // max(1, previews)) if preview_dir else 0
    meta = []
    for n, ep in enumerate(episodes):
        if round(ep.track.fps / step) != fps or round(width * ep.track.height / ep.track.width / 2) * 2 != height:
            print(f"skipping {ep.video.name}: frame rate or aspect ratio differs from the first episode")
            continue
        camera = ep.camera.scaled(width, height)
        rows = []
        writer = None
        if every and n % every == 0:
            Path(preview_dir).mkdir(parents=True, exist_ok=True)
            container = av.open(str(Path(preview_dir) / f"episode_{len(meta):06d}.mp4"), "w")
            stream = container.add_stream("h264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            writer = (container, stream)
        for i, image in frames_between(ep.video, ep.segment.start, ep.segment.end, ep.track.fps, width, height):
            if (i - ep.segment.start) % step:
                continue
            kp2d, kp3d, valid, ee = frame_labels(ep.track, i, ep.camera, width, height)
            rows.append({
                "observation.images.ego": image,
                "observation.intrinsics": camera.vector(),
                "observation.hands.kp2d": kp2d.reshape(-1).astype(np.float32),
                "observation.hands.kp3d": kp3d.reshape(-1).astype(np.float32),
                "observation.hands.valid": valid.astype(np.float32),
                "observation.hands.ee": ee.reshape(-1).astype(np.float32),
            })
            if writer:
                for packet in writer[1].encode(av.VideoFrame.from_image(draw_hands(Image.fromarray(image), kp2d, valid))):
                    writer[0].mux(packet)
        if writer:
            for packet in writer[1].encode():
                writer[0].mux(packet)
            writer[0].close()
        if not rows:
            continue
        for m, row in enumerate(rows):
            row["action"] = rows[min(m + 1, len(rows) - 1)]["observation.hands.ee"]
            row["task"] = ep.task
            dataset.add_frame(row)
        dataset.save_episode()
        valid = np.array([r["observation.hands.valid"] for r in rows])
        meta.append({"episode_index": len(meta), "task": ep.task, "frames": len(rows),
                     "any_hand": round(float((valid.sum(1) > 0).mean()), 3), "two_hands": round(float((valid.sum(1) == 2).mean()), 3),
                     "source": {"video": ep.video.name, "start_frame": ep.segment.start, "end_frame": ep.segment.end}, **ep.info})
    dataset.finalize()
    (out / "meta" / "episodes_info.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    summary = {"episodes": len(meta), "frames": sum(m["frames"] for m in meta), "fps": fps,
               "minutes": round(sum(m["frames"] for m in meta) / fps / 60, 1)}
    print(json.dumps(summary))
    return summary
