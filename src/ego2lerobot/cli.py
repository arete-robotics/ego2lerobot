"""ego2lerobot: first-person video → LeRobot dataset with hand keypoints, wrist actions and language.

  ego2lerobot run VIDEO [VIDEO ...] --out DATASET [--intrinsics cam.json | --hfov 100 | --camera action-ultrawide] [--caption]
  ego2lerobot detect VIDEO HANDS.npz
  ego2lerobot prepare VIDEO [VIDEO ...] --out PROXY_DIR [--width 1920 --fps 30]   (4K/60 fps → lighter proxy)
  ego2lerobot qc DATASET                                                          (qc_report.json / .md / .html)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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
    if getattr(args, "camera", None):
        return Camera.preset(args.camera, width, height)
    if getattr(args, "fisheye_hfov", None):
        return Camera.from_fisheye_fov(width, height, args.fisheye_hfov)
    return Camera.from_fov(width, height, args.hfov)


def track_path(video: Path, cache: Path, view=None) -> Path:
    return cache / f"{video.stem}{'.' + view.tag if view else ''}.hands.npz"


def cached_track(video: Path, cache: Path, view=None) -> HandTrack:
    path = track_path(video, cache, view)
    if path.exists():
        return HandTrack.load(path)
    track = detect(video, view=view)
    track.save(path)
    print(json.dumps({"video": video.name, **track.summary()}), flush=True)
    return track


def run(args) -> None:
    cache = args.cache or args.out.with_name(args.out.name + "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    captioner = None
    if args.caption:
        from .caption import Captioner
        captioner = Captioner(args.caption_model, backend=args.caption_backend, url=args.caption_url)
    episodes, videos, dropped = [], [], []
    for video in args.videos:
        camera = camera_for(video, args)
        view = None
        if args.rectify:
            from .rectify import PinholeView
            view = PinholeView(camera, args.rectify)
        track = cached_track(video, cache, view)
        videos.append({"name": video.name, "path": str(video.resolve()),
                       "track": str(track_path(video, cache, view).resolve()), "camera": asdict(camera)})
        segments = find_segments(track, args.min_seconds, args.max_seconds, min_coverage=args.min_coverage)
        if args.per_video:
            segments = spread(segments, args.per_video)
        for seg in segments:
            info = {"seconds": round(seg.seconds(track.fps), 2), "hand_coverage": seg.hand_coverage}
            task = args.default_task
            if captioner:
                caption = captioner(video, seg.start, seg.end, track.fps)
                if caption.get("task", "").strip().lower() == "idle":
                    dropped.append({"video": video.name, "start": seg.start, "end": seg.end, "reason": "caption said idle"})
                    continue
                task = caption.get("task") or task
                info.update({k: caption[k] for k in ("steps", "objects") if k in caption})
            episodes.append(Episode(video, camera, track, seg, task.strip(), info))
        print(f"{video.name}: {len(segments)} episodes", flush=True)
    write_dataset(episodes, args.out, args.repo_id, args.width, args.step, args.preview)
    from .qc import write_report, write_run_log
    settings = {"min_seconds": args.min_seconds, "max_seconds": args.max_seconds, "gap_seconds": 1.0,
                "min_coverage": args.min_coverage, "per_video": args.per_video, "step": args.step, "width": args.width,
                "rectify": args.rectify}
    write_run_log(args.out, settings, videos, dropped, bool(args.caption), args.default_task)
    report = write_report(args.out, cache)
    print(f"QC report: {args.out / 'qc_report.md'} ({report['overall']['flagged_episodes']} episodes with warnings)")


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
    cam.add_argument("--camera", choices=["action-ultrawide"], help="nominal lens preset for an uncalibrated camera (see camera.PRESETS)")
    cam.add_argument("--fisheye-hfov", type=float, help="uncalibrated fisheye: nominal equidistant lens with this horizontal FOV")
    r.add_argument("--rectify", type=float, nargs="?", const=105.0,
                   help="fisheye only: detect hands on a pinhole view of this horizontal FOV (default 105°), map back")
    r.add_argument("--caption", action="store_true", help="instructions from Qwen3-VL (see --caption-backend)")
    r.add_argument("--caption-backend", default="auto", choices=["auto", "mlx", "transformers", "openai"],
                   help="auto: openai if --caption-url is given, else mlx on Apple silicon, else transformers")
    r.add_argument("--caption-url", help="OpenAI-compatible server, e.g. http://localhost:8000/v1 (vLLM)")
    r.add_argument("--caption-model", help="model id (default depends on the backend; see caption.DEFAULT_MODELS)")
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

    p = sub.add_parser("prepare", help="downscaled, frame-rate-reduced proxies of large videos (e.g. 4K at 60 fps)")
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True, help="folder for PROXY.mp4 and PROXY.source.json")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--fps", type=float, default=30.0, help="keep every k-th frame so the rate is about this")

    q = sub.add_parser("qc", help="quality report for a dataset written by run")
    q.add_argument("dataset", type=Path)
    q.add_argument("--cache", type=Path, help="hand detections (default: DATASET_cache)")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args)
    elif args.cmd == "prepare":
        from .prepare import prepare
        for video in args.videos:
            print(json.dumps(prepare(video, args.out / f"{video.stem}.mp4", args.width, args.fps)), flush=True)
    elif args.cmd == "qc":
        from .qc import write_report
        report = write_report(args.dataset, args.cache)
        print(json.dumps(report["overall"]))
    else:
        track = detect(args.video)
        track.save(args.out)
        print(json.dumps(track.summary()))


if __name__ == "__main__":
    main()
