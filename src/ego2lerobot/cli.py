"""ego2lerobot: first-person video → LeRobot dataset with hand keypoints, wrist actions and language.

  ego2lerobot run VIDEO [VIDEO ...] --out DATASET [--intrinsics cam.json | --hfov 100 | --camera action-ultrawide] [--caption]
                  [--segment action|fixed]
  ego2lerobot detect VIDEO HANDS.npz
  ego2lerobot prepare VIDEO [VIDEO ...] --out PROXY_DIR [--width 1920 --fps 30]   (4K/60 fps → lighter proxy)
  ego2lerobot qc DATASET                                                          (qc_report.json / .md / .html)
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import av

from .camera import Camera
from .export import Episode, write_dataset
from .hands import HandTrack, detect
from .segments import find_segments, find_steps, merge_agreeing, spread
from .signals import grasp_events
from . import stereo_cli


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


def portable(path: Path, base: Path) -> str:
    """`path` relative to `base` when it sits next to it (at most one level up), else the file name only.

    meta/run.json travels with the dataset, so it must not carry absolute local paths (user names, folder names)."""
    rel = os.path.relpath(Path(path).resolve(), Path(base).resolve())
    return Path(path).name if os.path.isabs(rel) or rel.startswith(os.pardir + os.sep + os.pardir) else rel


def events_between(events: dict, start: int, end: int, fps: float) -> list[dict]:
    """Grasp and release events inside [start, end), with times in seconds from `start`."""
    return sorted(({"hand": side, "event": e["event"], "t": round((e["frame"] - start) / fps, 3)}
                   for side, evs in events.items() for e in evs if start <= e["frame"] < end), key=lambda e: e["t"])


def action_steps(video: Path, track: HandTrack, events: dict, captioner, args, min_seconds: float, entry: dict):
    """--segment action: steps that follow the work, each with its first caption (None without --caption, and with
    --step-vocab FILE unless --merge-agreeing needs it).

    Cuts come from find_steps (checked by the captioner). Steps with too little hand in view are dropped. With
    --merge-agreeing, touching neighbours whose first captions agree (same kind of action on the same object) are
    merged and captioned again.
    Adds the candidate cuts ("boundaries") and every step found ("steps") to `entry` for meta/run.json."""
    judge = None
    if captioner:
        def judge(frame, lo, hi):
            return captioner.boundary(video, track, frame, lo, hi)
    steps, entry["boundaries"] = find_steps(track, judge, min_seconds, args.max_seconds, events=events)
    confirmed = sum(bool(d.get("new_step")) for d in entry["boundaries"])
    print(f"{video.name}: {len(entry['boundaries'])} candidate cuts checked, {confirmed} confirmed, {len(steps)} steps",
          flush=True)
    low = [s for s in steps if s.hand_coverage < args.min_coverage]
    steps = [s for s in steps if s.hand_coverage >= args.min_coverage]
    firsts = [None] * len(steps)
    vocab_file = getattr(args, "step_vocab", None) not in (None, "auto")
    if captioner and not (vocab_file and not args.merge_agreeing):
        from .caption import captions_agree
        firsts = [captioner.caption(video, track, s.start, s.end) for s in steps]
        if args.merge_agreeing:
            steps, firsts, merged = merge_agreeing(steps, firsts, captions_agree, track, args.max_seconds)
            firsts = [captioner.caption(video, track, s.start, s.end) if m else f
                      for s, f, m in zip(steps, firsts, merged)]
            print(f"{video.name}: {sum(merged)} steps merged from neighbours with agreeing captions", flush=True)
    entry["steps"] = [{"start": s.start, "end": s.end, "hand_coverage": s.hand_coverage, "cuts": s.cuts}
                      for s in sorted(low + steps, key=lambda s: s.start)]
    return list(zip(steps, firsts))


def run(args) -> None:
    cache = args.cache or args.out.with_name(args.out.name + "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    action = args.segment == "action"
    min_seconds = args.min_seconds if args.min_seconds is not None else (2.0 if action else 3.0)
    captioner = None
    if args.caption:
        from .caption import Captioner, StepCaptioner
        captioner = (StepCaptioner if action else Captioner)(args.caption_model, backend=args.caption_backend,
                                                             url=args.caption_url)
    typer, vocab = None, None
    if args.step_vocab and not (args.caption and action):
        raise SystemExit("--step-vocab needs --caption and --segment action")
    if args.step_vocab and args.step_vocab != "auto":
        from .vocab import StepTyper, load_vocab
        vocab = {"source": "file", "file": portable(Path(args.step_vocab), args.out), "types": load_vocab(args.step_vocab)}
        if not vocab["types"]:
            raise SystemExit(f"{args.step_vocab}: no step types (one per line)")
        typer = StepTyper(captioner, vocab["types"])

    def prepare_video(video):
        camera = camera_for(video, args)
        view = None
        if args.rectify:
            from .rectify import PinholeView
            view = PinholeView(camera, args.rectify)
        track = cached_track(video, cache, view)
        events = grasp_events(track)
        entry = {"name": video.name, "path": portable(video, args.out), "track": track_path(video, cache, view).name,
                 "camera": asdict(camera)}
        side = video.with_suffix(".source.json")  # written by `ego2lerobot prepare`
        if side.exists():
            src = json.loads(side.read_text())
            entry["original"] = {k: src[k] for k in ("source", "width", "height", "fps", "seconds", "codec") if k in src}
        if action:
            items = action_steps(video, track, events, captioner, args, min_seconds, entry)
        else:
            items = [(seg, None) for seg in find_segments(track, min_seconds, args.max_seconds, min_coverage=args.min_coverage)]
        return camera, track, events, entry, items

    prepared = {}
    if args.step_vocab == "auto":  # all first captions are needed before the list can be built
        prepared = {video: prepare_video(video) for video in args.videos}
        typer, vocab = auto_vocab(captioner, [f for *_, items in prepared.values() for _, f in items], args.out)
    episodes, videos, dropped, grasp = [], [], [], {}
    for video in args.videos:
        camera, track, events, entry, items = prepared.pop(video) if video in prepared else prepare_video(video)
        grasp[video.name] = {"fps": track.fps, **events}
        videos.append(entry)
        if args.per_video:
            items = spread(items, args.per_video)
        for seg, first in items:
            info = {"seconds": round(seg.seconds(track.fps), 2), "hand_coverage": seg.hand_coverage}
            if action:
                info["cuts"] = seg.cuts
            task = args.default_task
            if captioner:
                if not action:
                    caption = captioner(video, seg.start, seg.end, track.fps)
                elif typer:
                    caption = typer(video, track, seg.start, seg.end, first)
                elif first["task"] == "idle":
                    caption = first
                else:
                    caption = captioner.verify(video, track, seg.start, seg.end, first)
                reason = None
                if caption.get("task", "").strip().lower() == "idle":
                    reason = "caption said idle"
                elif args.drop_unverified and caption.get("verified") is False:
                    reason = "caption not verified"
                if reason:
                    dropped.append({"video": video.name, "start": seg.start, "end": seg.end, "reason": reason,
                                    "task": caption.get("task")})
                    continue
                task = caption.get("task") or task
                info.update({k: caption[k] for k in ("steps", "object", "objects", "step_type", "verified",
                                                      "confidence", "check", "caption_problems") if k in caption})
            info["grasp_events"] = events_between(events, seg.start, seg.end, track.fps)
            episodes.append(Episode(video, camera, track, seg, task.strip(), info))
        print(f"{video.name}: {len(items)} {'steps' if action else 'episodes'}", flush=True)
    write_dataset(episodes, args.out, args.repo_id, args.width, args.step, args.preview)
    write_grasp_events(args.out, grasp)
    from .qc import write_report, write_run_log
    settings = {"segment": args.segment, "min_seconds": min_seconds, "max_seconds": args.max_seconds, "gap_seconds": 1.0,
                "min_coverage": args.min_coverage, "per_video": args.per_video, "step": args.step, "width": args.width,
                "rectify": args.rectify, "drop_unverified": args.drop_unverified, "merge_agreeing": args.merge_agreeing}
    if vocab:
        settings["step_vocab"] = "auto" if vocab["source"] == "auto" else vocab["file"]
        write_step_vocab(args.out, vocab)
    write_run_log(args.out, settings, videos, dropped, bool(args.caption), args.default_task)
    report = write_report(args.out, cache)
    print(f"QC report: {args.out / 'qc_report.md'} ({report['overall']['flagged_episodes']} episodes with warnings)")


def vocab_path(out: Path) -> Path:
    """Where --step-vocab auto saves the list it built: next to the dataset, so that a re-run does not delete it."""
    return out.with_name(out.name + "_step_vocab.txt")


def auto_vocab(captioner, firsts: list, out: Path):
    """--step-vocab auto: (StepTyper or None, metadata) from the first captions of every step of every video."""
    from .vocab import StepTyper, build_vocab, save_vocab
    built = build_vocab(captioner, firsts)
    path = vocab_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_vocab(path, built["types"], [f"Step types built by --step-vocab auto from {len(built['captions'])} step captions.",
                                      "Edit (one imperative step type per line) and re-run with --step-vocab FILE."])
    if built["types"]:
        print(f"step vocabulary ({len(built['types'])} types) saved to {path}:", *built["types"], sep="\n  ", flush=True)
    else:
        print("step vocabulary: no step type passed the checks; using free captions", flush=True)
    vocab = {"source": "auto", "file": path.name, **built}
    return (StepTyper(captioner, built["types"]) if built["types"] else None), vocab


def write_step_vocab(dataset: Path, vocab: dict) -> None:
    """meta/step_vocab.json: the list of step types and how many delivered episodes have each."""
    counts = [0] * (len(vocab["types"]) + 1)
    for ep in json.loads((dataset / "meta" / "episodes_info.json").read_text()):
        counts[ep.get("step_type") or 0] += 1
    about = ("Closed list of step types. In meta/episodes_info.json, step_type is the 1-based number of an "
             "episode's type in `types` (its task is that type's text), or 0 when the two picks did not agree on a "
             "type and the episode has a free caption instead (verified: false).")
    (dataset / "meta" / "step_vocab.json").write_text(json.dumps(
        {"about": about, **vocab, "episodes_per_type": {"0": counts[0], **{str(k): c for k, c in enumerate(counts[1:], 1)}}},
        indent=1, ensure_ascii=False))


def write_grasp_events(dataset: Path, grasp: dict) -> None:
    """meta/grasp_events.json: per video and hand, when the hand closes (grasp) and opens (release)."""
    from . import signals
    about = ("Per hand, frames where the grip closes (grasp) or opens (release). Grip aperture = thumb-tip to index-tip "
             "distance / palm length from MediaPipe's metric hand shape; closed below `close`, open above `open`, a state "
             "must last `min_state_s`; no event is emitted across a gap longer than `max_gap_s` without the hand. "
             "Frames and times refer to the source video.")
    (dataset / "meta" / "grasp_events.json").write_text(json.dumps(
        {"about": about, "thresholds": {"close": signals.CLOSE, "open": signals.OPEN, "min_state_s": signals.MIN_STATE_S,
                                        "max_gap_s": signals.MAX_GAP_S}, "videos": grasp}, indent=1))


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
    r.add_argument("--segment", choices=["action", "fixed"], default="action",
                   help="action: cut at action boundaries (hand signals checked by the captioner, or signals alone "
                        "without --caption); fixed: the earlier equal splits and one-question captions, to reproduce "
                        "datasets made before sub-steps")
    r.add_argument("--min-seconds", type=float, help="shortest step (default 2 s with --segment action, 3 s with fixed)")
    r.add_argument("--max-seconds", type=float, default=15.0, help="longest step; longer ones are split")
    r.add_argument("--merge-agreeing", action="store_true",
                   help="merge touching steps whose captions agree (fewer, longer steps; on the evaluation clips this "
                        "cost recall and caption accuracy, so it is off by default)")
    r.add_argument("--step-vocab", metavar="FILE|auto",
                   help="label each step with one type from a closed list instead of a free caption: a text file, one "
                        "imperative step type per line, or 'auto' to build the list from the video (saved as "
                        "OUT_step_vocab.txt); needs --caption")
    r.add_argument("--drop-unverified", action="store_true",
                   help="leave out steps whose caption fails verification (default: keep them, marked verified: false)")
    r.add_argument("--min-coverage", type=float, default=0.7, help="share of frames that must show a hand")
    r.add_argument("--per-video", type=int, default=0, help="keep at most N episodes per video, spread out (0 = all)")
    r.add_argument("--width", type=int, default=640)
    r.add_argument("--step", type=int, default=2, help="keep every Nth frame (2: 30 → 15 fps)")
    r.add_argument("--preview", type=Path, help="folder for preview videos with the hand skeleton drawn")
    r.add_argument("--cache", type=Path, help="where hand detections are kept (default: OUT_cache)")

    d = sub.add_parser("detect", help="hand detections for one video")
    d.add_argument("video", type=Path)
    d.add_argument("out", type=Path)
    stereo_cli.add_parser(sub)

    p = sub.add_parser("prepare", help="downscaled, frame-rate-reduced proxies of large videos (e.g. 4K at 60 fps)")
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True, help="folder for PROXY.mp4 and PROXY.source.json")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--fps", type=float, default=30.0, help="keep every k-th frame so the rate is about this")

    q = sub.add_parser("qc", help="quality report for a dataset written by run")
    q.add_argument("dataset", type=Path)
    q.add_argument("--cache", type=Path, help="hand detections (default: DATASET_cache)")

    args = ap.parse_args(argv)
    if args.cmd == "stereo":
        return stereo_cli.run(args)
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
