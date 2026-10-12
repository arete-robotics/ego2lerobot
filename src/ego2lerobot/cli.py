"""ego2lerobot: first-person video → LeRobot dataset with hand keypoints, wrist actions and language.

  ego2lerobot run VIDEO [VIDEO ...] --out DATASET [--intrinsics cam.json | --hfov 100] [--caption]
  ego2lerobot detect VIDEO HANDS.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av

from .camera import Camera
from .export import Episode, write_dataset
from .hands import HandTrack, detect
from .segments import find_segments, spread


def video_size(video: Path) -> tuple[int, int]:
    with av.open(str(video)) as c:
        s = c.streams.video[0].codec_context
        return s.width, s.height


def camera_for(video: Path, args) -> Camera:
    width, height = video_size(video)
    if args.intrinsics:
        return Camera.from_json(args.intrinsics, width, height)
    return Camera.from_fov(width, height, args.hfov)


def cached_track(video: Path, cache: Path) -> HandTrack:
    path = cache / f"{video.stem}.hands.npz"
    if path.exists():
        return HandTrack.load(path)
    track = detect(video)
    track.save(path)
    print(json.dumps({"video": video.name, **track.summary()}), flush=True)
    return track


def run(args) -> None:
    cache = args.cache or args.out.with_name(args.out.name + "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    captioner = None
    if args.caption:
        from .caption import Captioner
        captioner = Captioner(args.caption_model)
    episodes = []
    for video in args.videos:
        camera = camera_for(video, args)
        track = cached_track(video, cache)
        segments = find_segments(track, args.min_seconds, args.max_seconds, min_coverage=args.min_coverage)
        if args.per_video:
            segments = spread(segments, args.per_video)
        for seg in segments:
            info = {"seconds": round(seg.seconds(track.fps), 2), "hand_coverage": seg.hand_coverage}
            task = args.default_task
            if captioner:
                caption = captioner(video, seg.start, seg.end, track.fps)
                if caption.get("task", "").strip().lower() == "idle":
                    continue
                task = caption.get("task") or task
                info.update({k: caption[k] for k in ("steps", "objects") if k in caption})
            episodes.append(Episode(video, camera, track, seg, task.strip(), info))
        print(f"{video.name}: {len(segments)} episodes", flush=True)
    write_dataset(episodes, args.out, args.repo_id, args.width, args.step, args.preview)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ego2lerobot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="videos → LeRobot dataset")
    r.add_argument("videos", nargs="+", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--repo-id", default="local/ego2lerobot")
    cam = r.add_mutually_exclusive_group()
    cam.add_argument("--intrinsics", type=Path, help="JSON with fx fy cx cy (and model fisheye + k1–k4)")
    cam.add_argument("--hfov", type=float, default=100.0, help="horizontal field of view when uncalibrated (default 100°)")
    r.add_argument("--caption", action="store_true", help="instructions from Qwen3-VL (needs ego2lerobot[caption], Apple silicon)")
    r.add_argument("--caption-model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    r.add_argument("--default-task", default="manipulate objects with the hands")
    r.add_argument("--min-seconds", type=float, default=3.0)
    r.add_argument("--max-seconds", type=float, default=15.0)
    r.add_argument("--min-coverage", type=float, default=0.7, help="share of frames that must show a hand")
    r.add_argument("--per-video", type=int, default=0, help="keep at most N episodes per video, spread out (0 = all)")
    r.add_argument("--width", type=int, default=640)
    r.add_argument("--step", type=int, default=2, help="keep every Nth frame (2: 30 → 15 fps)")
    r.add_argument("--preview", type=Path, help="folder for preview videos with the hand skeleton drawn")
    r.add_argument("--cache", type=Path, help="where hand detections are kept (default: OUT_cache)")

    d = sub.add_parser("detect", help="hand detections for one video")
    d.add_argument("video", type=Path)
    d.add_argument("out", type=Path)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args)
    else:
        track = detect(args.video)
        track.save(args.out)
        print(json.dumps(track.summary()))


if __name__ == "__main__":
    main()
