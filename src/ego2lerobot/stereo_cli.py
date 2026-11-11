"""`ego2lerobot stereo`: head-mounted stereo video and its calibration → LeRobot dataset with triangulated hands.

  ego2lerobot stereo LEFT.mp4 RIGHT.mp4 --calib stereo.json --out DATASET [--imu imu.csv] [--preview DIR]
  ego2lerobot stereo --sbs PAIR1.mp4 PAIR2.mp4 --calib stereo.json --out DATASET
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DESCRIPTION = __doc__ + """
The calibration format is described in ego2lerobot/stereo.py (intrinsics and distortion per camera, R and T from the
left to the right camera in metres, or fx fy cx cy and baseline for a rectified pair). IMU CSV: a header row, first
column t in seconds since the first video frame, other columns passed through."""


def add_parser(sub) -> None:
    s = sub.add_parser("stereo", help="stereo videos + calibration → LeRobot dataset with triangulated hands",
                       description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("videos", nargs="+", type=Path, help="LEFT RIGHT [LEFT RIGHT ...], or side-by-side files with --sbs")
    s.add_argument("--calib", type=Path, required=True, help="stereo calibration JSON, shared by all clips")
    s.add_argument("--sbs", action="store_true", help="each video holds both views side by side, left camera on the left")
    s.add_argument("--imu", type=Path, nargs="+", help="one IMU CSV per clip, in the same order")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--repo-id", default="local/ego2lerobot-stereo")
    s.add_argument("--rtmpose", action="store_true",
                   help="add RTMPose-m hand keypoints as a second model (needs ego2lerobot[rtmpose]); off by default "
                        "because its weights were trained on research-only datasets")
    s.add_argument("--max-depth", type=float, default=1.5, help="farthest plausible hand, metres (default 1.5)")
    s.add_argument("--caption", action="store_true", help="instructions from Qwen3-VL on the left view (ego2lerobot[caption])")
    s.add_argument("--caption-model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    s.add_argument("--default-task", default="manipulate objects with the hands")
    s.add_argument("--min-seconds", type=float, default=3.0)
    s.add_argument("--max-seconds", type=float, default=15.0)
    s.add_argument("--min-coverage", type=float, default=0.7, help="share of frames that must have a labelled hand")
    s.add_argument("--per-video", type=int, default=0, help="keep at most N episodes per clip, spread out (0 = all)")
    s.add_argument("--width", type=int, default=640, help="stored width of each view")
    s.add_argument("--step", type=int, default=2, help="keep every Nth frame (2: 30 → 15 fps)")
    s.add_argument("--detect-width", type=int, default=960, help="width each view is resized to for MediaPipe")
    s.add_argument("--preview", type=Path, help="folder for side-by-side preview videos")
    s.add_argument("--cache", type=Path, help="where detections are kept (default: OUT_cache)")


def views_for(video, cache: Path, rtmpose: bool, detect_width: int):
    """Per-view fused 2D keypoints, with MediaPipe (and RTMPose) detections cached in `cache`."""
    from .hands import HandTrack
    from .stereo_video import detect_stereo, fuse, rtmpose_views

    paths = [cache / f"{video.name}.{v}.hands.npz" for v in ("left", "right")]
    if all(p.exists() for p in paths):
        tracks = [HandTrack.load(p) for p in paths]
    else:
        tracks = detect_stereo(video, detect_width)
        for track, path in zip(tracks, paths):
            track.save(path)
    rtm = None
    if rtmpose:
        import numpy as np
        path = cache / f"{video.name}.rtmpose.npz"
        if path.exists():
            d = np.load(path)
            rtm = [(d["kp_left"], d["conf_left"]), (d["kp_right"], d["conf_right"])]
        else:
            rtm = rtmpose_views(video, tracks)
            np.savez_compressed(path, kp_left=rtm[0][0], conf_left=rtm[0][1], kp_right=rtm[1][0], conf_right=rtm[1][1])
    return [fuse(track, rtm[v] if rtm else None) for v, track in enumerate(tracks)]


def run(args) -> None:
    from .segments import find_segments, spread
    from .stereo import Checks, StereoRig, label_stereo, summarize
    from .stereo_export import Imu, StereoEpisode, write_stereo_dataset
    from .stereo_video import StereoVideo, caption_video

    if args.sbs:
        clips = [(v, None) for v in args.videos]
    elif len(args.videos) % 2:
        raise SystemExit("give videos as LEFT RIGHT pairs, or use --sbs for side-by-side files")
    else:
        clips = list(zip(args.videos[::2], args.videos[1::2]))
    if args.imu and len(args.imu) != len(clips):
        raise SystemExit(f"--imu needs one file per clip ({len(clips)})")
    cache = args.cache or args.out.with_name(args.out.name + "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    captioner = None
    if args.caption:
        from .caption import Captioner
        captioner = Captioner(args.caption_model)
    checks = Checks(max_depth=args.max_depth)
    episodes = []
    for k, (left, right) in enumerate(clips):
        video = StereoVideo(left, right)
        rig = StereoRig.from_json(args.calib, video.width, video.height)
        labels = label_stereo(rig, views_for(video, cache, args.rtmpose, args.detect_width), video.fps, checks)
        labels.save(cache / f"{video.name}.stereo.npz")
        print(json.dumps({"clip": video.name, **summarize([(labels, slice(None), rig)])}), flush=True)
        segments = find_segments(labels.as_track(), args.min_seconds, args.max_seconds, min_coverage=args.min_coverage)
        if args.per_video:
            segments = spread(segments, args.per_video)
        imu = Imu(args.imu[k]) if args.imu else None
        for seg in segments:
            info = {"seconds": round(seg.seconds(video.fps), 2), "hand_coverage": seg.hand_coverage}
            task = args.default_task
            if captioner:
                caption = captioner(caption_video(video, cache), seg.start, seg.end, video.fps)
                if caption.get("task", "").strip().lower() == "idle":
                    continue
                task = caption.get("task") or task
                info.update({k_: caption[k_] for k_ in ("steps", "objects") if k_ in caption})
            episodes.append(StereoEpisode(video, rig, labels, seg, task.strip(), imu, info))
        print(f"{video.name}: {len(segments)} episodes", flush=True)
    write_stereo_dataset(episodes, args.out, args.repo_id, args.width, args.step, args.preview)
