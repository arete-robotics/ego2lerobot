"""`ego2lerobot stereo`: head-mounted stereo video and its calibration → LeRobot dataset with triangulated hands.

  ego2lerobot stereo LEFT.mp4 RIGHT.mp4 --calib stereo.json --out DATASET [--imu imu.csv] [--preview DIR] [--caption]
                     [--segment action|fixed]
  ego2lerobot stereo --sbs PAIR1.mp4 PAIR2.mp4 --calib stereo.json --out DATASET

Sub-steps, grasp and release events and captions work as in `run`, with the wrist motion and grip measured by
triangulation; the QC report (qc_report.md) adds a stereo section.
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
    s.add_argument("--max-epipolar", type=float, default=0.01,
                   help="largest median row disagreement of a matched hand after rectification, as a share of the image "
                        "height (default 0.01: 10.8 px at 1080 rows)")
    s.add_argument("--max-reprojection", type=float, default=0.006,
                   help="largest reprojection error of one joint, as a share of the image height (default 0.006)")
    s.add_argument("--caption", action="store_true", help="instructions from Qwen3-VL on the left view (see run --caption)")
    s.add_argument("--caption-backend", default="auto", choices=["auto", "mlx", "transformers", "openai"],
                   help="auto: openai if --caption-url is given, else mlx on Apple silicon, else transformers")
    s.add_argument("--caption-url", help="OpenAI-compatible server, e.g. http://localhost:8000/v1 (vLLM)")
    s.add_argument("--caption-model", help="model id (default depends on the backend; see caption.DEFAULT_MODELS)")
    s.add_argument("--default-task", default="manipulate objects with the hands")
    s.add_argument("--segment", choices=["action", "fixed"], default="action",
                   help="action: cut at action boundaries found in the triangulated wrist motion and grip (checked by "
                        "the captioner, or two signals agreeing without --caption); fixed: equal splits as before")
    s.add_argument("--min-seconds", type=float, help="shortest step (default 2 s with --segment action, 3 s with fixed)")
    s.add_argument("--max-seconds", type=float, default=15.0, help="longest step; longer ones are split")
    s.add_argument("--merge-agreeing", action="store_true", help="merge touching steps whose captions agree (see run)")
    s.add_argument("--drop-unverified", action="store_true",
                   help="leave out steps whose caption fails verification (default: keep them, marked verified: false)")
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


def with_3d_coverage(seg, labels):
    """`seg` with hand_coverage and two_hands as shares of frames with hands labelled in 3D (not just seen)."""
    from dataclasses import replace
    any_hand, both = labels.coverage(seg.start, seg.end)
    return replace(seg, hand_coverage=any_hand, two_hands=both)


def action_steps(left_video, labels, track, events: dict, captioner, args, min_seconds: float, entry: dict):
    """--segment action for one stereo clip: as cli.action_steps, on the measured track (StereoLabels.step_track).

    Boundaries come from segments.find_steps (checked by the captioner on the left view); hand presence is from the
    2D detections, so a stretch where triangulation failed is a gap, not a hand leaving the view. Hand coverage is the
    share of frames with a hand labelled in 3D; steps below --min-coverage are dropped."""
    from .segments import find_steps, merge_agreeing
    judge = None
    if captioner:
        def judge(frame, lo, hi):
            return captioner.boundary(left_video, track, frame, lo, hi)
    steps, entry["boundaries"] = find_steps(track, judge, min_seconds, args.max_seconds, events=events)
    steps = [with_3d_coverage(s, labels) for s in steps]
    confirmed = sum(bool(d.get("new_step")) for d in entry["boundaries"])
    print(f"{entry['name']}: {len(entry['boundaries'])} candidate cuts checked, {confirmed} confirmed, {len(steps)} steps",
          flush=True)
    low = [s for s in steps if s.hand_coverage < args.min_coverage]
    steps = [s for s in steps if s.hand_coverage >= args.min_coverage]
    firsts = [None] * len(steps)
    if captioner:
        from .caption import captions_agree
        firsts = [captioner.caption(left_video, track, s.start, s.end) for s in steps]
        if args.merge_agreeing:
            steps, firsts, merged = merge_agreeing(steps, firsts, captions_agree, track, args.max_seconds)
            steps = [with_3d_coverage(s, labels) for s in steps]
            firsts = [captioner.caption(left_video, track, s.start, s.end) if m else f
                      for s, f, m in zip(steps, firsts, merged)]
            print(f"{entry['name']}: {sum(merged)} steps merged from neighbours with agreeing captions", flush=True)
    entry["steps"] = [{"start": s.start, "end": s.end, "hand_coverage": s.hand_coverage, "cuts": s.cuts}
                      for s in sorted(low + steps, key=lambda s: s.start)]
    return list(zip(steps, firsts))


def run(args) -> None:
    from .cli import events_between, portable
    from .qc import write_report, write_run_log
    from .segments import find_segments, spread
    from .signals import grasp_events
    from .stereo import Checks, StereoRig, label_stereo, summarize
    from .stereo_export import Imu, StereoEpisode, checks_dict, write_grasp_events, write_stereo_dataset
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
    action = args.segment == "action"
    min_seconds = args.min_seconds if args.min_seconds is not None else (2.0 if action else 3.0)
    captioner = None
    if args.caption:
        from .caption import Captioner, StepCaptioner
        captioner = (StepCaptioner if action else Captioner)(args.caption_model, backend=args.caption_backend,
                                                             url=args.caption_url)
    checks = Checks(max_epipolar=args.max_epipolar, max_reprojection=args.max_reprojection, max_depth=args.max_depth)
    episodes, videos, dropped, grasp = [], [], [], {}
    for k, (left, right) in enumerate(clips):
        video = StereoVideo(left, right)
        rig = StereoRig.from_json(args.calib, video.width, video.height)
        labels = label_stereo(rig, views_for(video, cache, args.rtmpose, args.detect_width), video.fps, checks)
        labels.save(cache / f"{video.name}.stereo.npz")
        print(json.dumps({"clip": video.name, **summarize([(labels, slice(None), rig)])}), flush=True)
        track = labels.step_track(rig)
        events = grasp_events(track)
        name = video.left.name
        grasp[name] = {"fps": video.fps, **events}
        entry = {"name": name, "path": portable(video.left, args.out),
                 "right": portable(video.right, args.out) if video.right else None, "side_by_side": video.right is None,
                 "track": f"{video.name}.stereo.npz", "checks": checks_dict(checks, rig)}
        left_video = caption_video(video, cache) if captioner else None
        if action:
            items = action_steps(left_video, labels, track, events, captioner, args, min_seconds, entry)
        else:
            items = [(seg, None) for seg in find_segments(labels.as_track(), min_seconds, args.max_seconds,
                                                          min_coverage=args.min_coverage)]
        videos.append(entry)
        if args.per_video:
            items = spread(items, args.per_video)
        imu = Imu(args.imu[k]) if args.imu else None
        for seg, first in items:  # captions and drops as in cli.run
            info = {"seconds": round(seg.seconds(video.fps), 2), "hand_coverage": seg.hand_coverage}
            if action:
                info["cuts"] = seg.cuts
            task = args.default_task
            if captioner:
                if not action:
                    caption = captioner(left_video, seg.start, seg.end, video.fps)
                elif first["task"] == "idle":
                    caption = first
                else:
                    caption = captioner.verify(left_video, track, seg.start, seg.end, first)
                reason = None
                if caption.get("task", "").strip().lower() == "idle":
                    reason = "caption said idle"
                elif args.drop_unverified and caption.get("verified") is False:
                    reason = "caption not verified"
                if reason:
                    dropped.append({"video": name, "start": seg.start, "end": seg.end, "reason": reason,
                                    "task": caption.get("task")})
                    continue
                task = caption.get("task") or task
                info.update({k_: caption[k_] for k_ in ("steps", "object", "objects", "verified", "confidence",
                                                         "check", "caption_problems") if k_ in caption})
            info["grasp_events"] = events_between(events, seg.start, seg.end, video.fps)
            episodes.append(StereoEpisode(video, rig, labels, seg, task.strip(), imu, info))
        print(f"{video.name}: {len(items)} {'steps' if action else 'episodes'}", flush=True)
    if not episodes:
        raise SystemExit("no episode has enough frames with triangulated hands; see the statistics above "
                         "(check the calibration, or lower --min-coverage / --min-seconds)")
    write_stereo_dataset(episodes, args.out, args.repo_id, args.width, args.step, args.preview, checks=checks)
    write_grasp_events(args.out, grasp)
    settings = {"stereo": True, "segment": args.segment, "min_seconds": min_seconds, "max_seconds": args.max_seconds,
                "gap_seconds": 1.0, "min_coverage": args.min_coverage, "per_video": args.per_video, "step": args.step,
                "width": args.width, "drop_unverified": args.drop_unverified, "merge_agreeing": args.merge_agreeing,
                "checks": checks_dict(checks, rig), "calibration": args.calib.name}
    write_run_log(args.out, settings, videos, dropped, bool(args.caption), args.default_task)
    report = write_report(args.out, cache)
    print(f"QC report: {args.out / 'qc_report.md'} ({report['overall']['flagged_episodes']} episodes with warnings)")
