"""Automatic quality report for a dataset written by `ego2lerobot run`: qc_report.json, qc_report.md, qc_report.html.

Everything is computed from the pipeline's own output: the hand tracks in the cache, the delivered frames in
data/*.parquet, meta/episodes_info.json, and meta/run.json (settings and dropped segments, written by `run`).
There is no ground truth here. A high detection rate means MediaPipe reported a hand, not that the hand is right.
"""
from __future__ import annotations

import datetime
import html
import json
from pathlib import Path

import numpy as np

from .hands import HandTrack
from .segments import find_segments, hand_runs

DISCLAIMER = ("These are automatic checks of the pipeline's own output, with no ground truth. They point to likely "
              "problems; they do not measure accuracy. Thresholds are rules of thumb.")
DIST_RANGE = (0.15, 1.2)  # metres from the camera; a wrist outside this is implausible for a worn camera
BORDER = 0.01  # a hand is "cut" if a keypoint lies within this share of the image width of the border, or outside
JUMP_M = 0.05  # a wrist that leaves the line between its neighbouring frames by more than this has jumped
MIN_RANGE_S = 1.0  # unusable ranges shorter than this are not listed
FLAGS = {"missing": 0.2, "cut": 0.3, "distance": 0.1, "jumps": 0.05}  # per-episode warning thresholds (shares)
DEFAULTS = {"min_seconds": 3.0, "max_seconds": 15.0, "gap_seconds": 1.0, "min_coverage": 0.7}


# ---------- metrics on arrays ----------

def border_cut(kp: np.ndarray, width: int, height: int, margin: float = BORDER) -> np.ndarray:
    """kp (N, 21, 2) pixels → (N,) True where some keypoint is within `margin`·width of the border or outside."""
    m = margin * width
    u, v = kp[..., 0], kp[..., 1]
    return ((u < m) | (u > width - m) | (v < m) | (v > height - m)).any(-1)


def wrist_residual(p: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Distance of each point from the midpoint of its two neighbours, |p[t] − (p[t−1] + p[t+1]) / 2|.

    This is half the second difference (acceleration / fps²): near zero for smooth motion, large for jitter and
    jumps. NaN where the frame or a neighbour is not valid. p (N, D), valid (N,) → (N,).
    """
    r = np.full(len(p), np.nan)
    if len(p) >= 3:
        ok = valid[:-2] & valid[1:-1] & valid[2:]
        d = np.linalg.norm(p[1:-1] - (p[:-2] + p[2:]) / 2, axis=-1)
        r[1:-1] = np.where(ok, d, np.nan)
    return r


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of the True stretches of a boolean array."""
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _share(mask: np.ndarray, of: np.ndarray | None = None) -> float | None:
    if of is not None:
        mask = mask[of]
    return round(float(np.mean(mask)), 3) if len(mask) else None


def frame_checks(kp2d: np.ndarray, kp3d: np.ndarray, valid: np.ndarray, width: int, height: int) -> dict:
    """Per-frame, per-hand checks on delivered frames: kp2d (N, 2, 21, 3), kp3d (N, 2, 21, 3), valid (N, 2)."""
    valid = valid > 0
    cut = np.stack([border_cut(kp2d[:, s, :, :2], width, height) for s in range(2)], 1) & valid
    dist = np.linalg.norm(kp3d[:, :, 0], axis=-1)
    far = ((dist < DIST_RANGE[0]) | (dist > DIST_RANGE[1])) & valid
    px = np.stack([wrist_residual(kp2d[:, s, 0, :2], valid[:, s]) for s in range(2)], 1)
    m = np.stack([wrist_residual(kp3d[:, s, 0], valid[:, s]) for s in range(2)], 1)
    z = np.stack([wrist_residual(kp3d[:, s, 0, 2:], valid[:, s]) for s in range(2)], 1)
    return {"valid": valid, "cut": cut, "far": far, "jitter_px": px, "jitter_m": m, "jitter_z": z}


def hand_summary(c: dict, s: int) -> dict:
    """Shares and jitter for hand s (0 left, 1 right) from `frame_checks` output."""
    v = c["valid"][:, s]
    px, m, z = c["jitter_px"][:, s], c["jitter_m"][:, s], c["jitter_z"][:, s]
    has = ~np.isnan(m)
    return {
        "detected": _share(v),
        "cut_by_border": _share(c["cut"][:, s], v),
        "distance_implausible": _share(c["far"][:, s], v),
        "jitter_px": round(float(np.nanmedian(px)), 2) if has.any() else None,
        "jitter_mm": round(float(np.nanmedian(m)) * 1000, 1) if has.any() else None,
        "jitter_depth_mm": round(float(np.nanmedian(z)) * 1000, 1) if has.any() else None,
        "jumps": _share(m[has] > JUMP_M) if has.any() else None,
    }


def episode_flags(hands: list[dict], any_hand: float | None) -> list[str]:
    flags = []
    if any_hand is not None and 1 - any_hand > FLAGS["missing"]:
        flags.append(f"no hand in {1 - any_hand:.0%} of frames")
    checks = (("cut_by_border", "cut", "cut by border"), ("distance_implausible", "distance", "implausible distance"),
              ("jumps", "jumps", "wrist jumps"))
    for name, h in zip("LR", hands):
        for key, limit, label in checks:
            if h[key] is not None and h[key] > FLAGS[limit]:
                flags.append(f"{name} {label} {h[key]:.0%}")
    return flags


def problem_ranges(c: dict, fps: float, offset_s: float) -> list[dict]:
    """Stretches of at least MIN_RANGE_S inside an episode where no detected hand is usable."""
    out = []
    for key, reason in (("valid", "no hand detected"), ("cut", "every detected hand cut by border"),
                        ("far", "implausible wrist distance")):
        if key == "valid":
            bad = ~c["valid"].any(1)
        else:
            bad = c["valid"].any(1) & ~(c["valid"] & ~c[key]).any(1)
        for a, b in runs(bad):
            if (b - a) / fps >= MIN_RANGE_S:
                out.append({"start_s": round(offset_s + a / fps, 2), "end_s": round(offset_s + b / fps, 2), "reason": reason})
    return out


# ---------- segment audit on the full hand track ----------

def audit_segments(track: HandTrack, settings: dict, exported: set, idle: set) -> list[dict]:
    """Every hand-activity stretch in a video and what happened to it: kept, or dropped and why."""
    s = {**DEFAULTS, **settings}
    fps = track.fps
    out = []
    for a, b in hand_runs(track, s["gap_seconds"]):
        if (b - a) / fps < s["min_seconds"]:
            out.append({"start": a, "end": b, "kept": False, "kind": "too short",
                        "reason": f"hands in view only {(b - a) / fps:.1f} s (< {s['min_seconds']:g} s)"})
    for seg in find_segments(track, s["min_seconds"], s["max_seconds"], s["gap_seconds"], 0.0):
        key = (seg.start, seg.end)
        if key in exported:
            kind, reason = None, None
        elif seg.hand_coverage < s["min_coverage"]:
            kind, reason = "low hand coverage", f"hand coverage {seg.hand_coverage:.2f} < {s['min_coverage']:g}"
        elif key in idle:
            kind = reason = "caption said idle"
        else:
            kind, reason = "not selected", "not selected (--per-video) or skipped at export"
        out.append({"start": seg.start, "end": seg.end, "kept": reason is None, "kind": kind, "reason": reason,
                    "hand_coverage": seg.hand_coverage})
    for d in out:
        d["start_s"], d["end_s"] = round(d["start"] / fps, 2), round(d["end"] / fps, 2)
    return sorted(out, key=lambda d: d["start"])


def video_ranges(track: HandTrack, segments: list[dict]) -> list[dict]:
    """Time not covered by a kept episode, as ranges with the reason (no hands, or why the segment was dropped)."""
    fps = track.fps
    n = len(track.score)
    reason = np.array(["no hand detected"] * n, dtype=object)
    for d in segments:
        reason[d["start"]:d["end"]] = d["reason"]
    out = []
    i = 0
    while i < n:
        j = i
        while j < n and reason[j] == reason[i]:
            j += 1
        if reason[i] is not None and (j - i) / fps >= MIN_RANGE_S:
            out.append({"start_s": round(i / fps, 2), "end_s": round(j / fps, 2), "reason": reason[i]})
        i = j
    return out


# ---------- reading a dataset ----------

def _load_frames(dataset: Path) -> dict[int, dict[str, np.ndarray]]:
    """Per episode index: kp2d (N, 2, 21, 3), kp3d (N, 2, 21, 3), valid (N, 2) from data/*/*.parquet."""
    import pyarrow.parquet as pq

    cols = {"observation.hands.kp2d": (2, 21, 3), "observation.hands.kp3d": (2, 21, 3), "observation.hands.valid": (2,)}
    parts = []
    for f in sorted(dataset.glob("data/*/*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", *cols])
        d = {"episode_index": t.column("episode_index").to_numpy(), "frame_index": t.column("frame_index").to_numpy()}
        for c, shape in cols.items():
            d[c] = np.asarray(t.column(c).combine_chunks().flatten(), np.float64).reshape(-1, *shape)
        parts.append(d)
    if not parts:
        return {}
    allf = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out = {}
    for e in np.unique(allf["episode_index"]):
        idx = np.flatnonzero(allf["episode_index"] == e)
        idx = idx[np.argsort(allf["frame_index"][idx])]
        out[int(e)] = {"kp2d": allf["observation.hands.kp2d"][idx], "kp3d": allf["observation.hands.kp3d"][idx],
                       "valid": allf["observation.hands.valid"][idx]}
    return out


def _video_facts(entry: dict, track: HandTrack | None) -> dict:
    facts = {"video": entry.get("name") or Path(entry.get("path", "?")).name}
    if track is not None:
        facts.update(width=track.width, height=track.height, fps=round(track.fps, 3), frames=len(track.score),
                     seconds=round(len(track.score) / track.fps, 2))
    side = Path(entry.get("path", "")).with_suffix(".source.json")
    if entry.get("path") and side.exists():  # written by `ego2lerobot prepare`
        src = json.loads(side.read_text())
        facts["original"] = {k: src[k] for k in ("source", "width", "height", "fps", "seconds", "codec") if k in src}
    return facts


def build_report(dataset: str | Path, cache: str | Path | None = None) -> dict:
    dataset = Path(dataset)
    info = json.loads((dataset / "meta" / "info.json").read_text())
    episodes = json.loads((dataset / "meta" / "episodes_info.json").read_text())
    run_path = dataset / "meta" / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    settings = {**DEFAULTS, **run.get("settings", {})}
    fps = float(info["fps"])
    h, w = info["features"]["observation.images.ego"]["shape"][:2]
    cache = Path(cache) if cache else dataset.with_name(dataset.name + "_cache")
    entries = run.get("videos") or [{"name": n, "track": str(cache / f"{Path(n).stem}.hands.npz")}
                                    for n in dict.fromkeys(e["source"]["video"] for e in episodes)]
    tracks = {}
    for entry in entries:
        entry.setdefault("name", Path(entry.get("path", "?")).name)
        tpath = Path(entry["track"]) if entry.get("track") else None
        tracks[entry["name"]] = HandTrack.load(tpath) if tpath and tpath.exists() else None
    frames = _load_frames(dataset)

    report_eps, ranges, all_checks = [], [], []
    for e in episodes:
        c = frames.get(e["episode_index"])
        if c is None:
            continue
        checks = frame_checks(c["kp2d"], c["kp3d"], c["valid"], w, h)
        all_checks.append(checks)
        track = tracks.get(e["source"]["video"])
        start_s = e["source"]["start_frame"] / track.fps if track is not None else None
        hands = [hand_summary(checks, s) for s in range(2)]
        any_hand = _share(checks["valid"].any(1))
        steps = e.get("steps") or []
        report_eps.append({
            "episode_index": e["episode_index"], "video": e["source"]["video"],
            "start_s": round(start_s, 2) if start_s is not None else None, "seconds": round(len(checks["valid"]) / fps, 2),
            "frames": len(checks["valid"]), "any_hand": any_hand, "left": hands[0], "right": hands[1],
            "task": e.get("task"), "steps": len(steps) if isinstance(steps, list) else 0,
            "captioned": bool(run.get("caption")) and e.get("task") != run.get("default_task"),
            "flags": episode_flags(hands, any_hand),
        })
        if start_s is not None:
            ranges += [{"video": e["source"]["video"], **r} for r in problem_ranges(checks, fps, start_s)]

    videos, dropped = [], []
    for entry in entries:
        name = entry["name"]
        track = tracks[name]
        facts = _video_facts(entry, track)
        if track is not None:
            score = track.score > 0
            cut = np.stack([border_cut(track.kp[:, s], track.width, track.height) for s in range(2)], 1) & score
            facts.update(left_detected=_share(score[:, 0]), right_detected=_share(score[:, 1]),
                         any_hand=_share(score.any(1)), both_hands=_share(score.all(1)),
                         left_cut=_share(cut[:, 0], score[:, 0]), right_cut=_share(cut[:, 1], score[:, 1]))
            exported = {(e["source"]["start_frame"], e["source"]["end_frame"]) for e in episodes if e["source"]["video"] == name}
            idle = {(d["start"], d["end"]) for d in run.get("dropped", []) if d.get("video") == name}
            segs = audit_segments(track, settings, exported, idle)
            dropped += [{"video": name, **{k: d[k] for k in ("start_s", "end_s", "kind", "reason")}} for d in segs if not d["kept"]]
            ranges += [{"video": name, **r} for r in video_ranges(track, segs)]
        videos.append(facts)

    overall = {"videos": len(videos), "source_minutes": round(sum(v.get("seconds", 0) for v in videos) / 60, 2),
               "episodes": len(report_eps), "frames": sum(e["frames"] for e in report_eps),
               "episode_minutes": round(sum(e["frames"] for e in report_eps) / fps / 60, 2), "fps": fps,
               "image": f"{w}x{h}"}
    if all_checks:
        merged = {k: np.concatenate([c[k] for c in all_checks]) for k in all_checks[0]}
        overall["any_hand"] = _share(merged["valid"].any(1))
        overall["both_hands"] = _share(merged["valid"].all(1))
        overall["left"], overall["right"] = hand_summary(merged, 0), hand_summary(merged, 1)
    lengths = [e["seconds"] for e in report_eps]
    overall["episode_seconds"] = {"min": min(lengths), "median": round(float(np.median(lengths)), 2), "max": max(lengths)} if lengths else None
    overall["captions"] = {"enabled": bool(run.get("caption")), "captioned": sum(e["captioned"] for e in report_eps),
                           "with_steps": sum(e["steps"] > 0 for e in report_eps), "episodes": len(report_eps),
                           "idle_dropped": sum(d["kind"] == "caption said idle" for d in dropped)}
    reasons: dict[str, int] = {}
    for d in dropped:
        reasons[d["kind"]] = reasons.get(d["kind"], 0) + 1
    overall["dropped_segments"] = {"count": len(dropped), "by_reason": reasons}
    overall["flagged_episodes"] = sum(bool(e["flags"]) for e in report_eps)
    from . import __version__
    return {
        "dataset": dataset.name, "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "tool": f"ego2lerobot {__version__}", "disclaimer": DISCLAIMER,
        "thresholds": {"wrist_distance_m": list(DIST_RANGE), "border_margin": BORDER, "jump_m": JUMP_M,
                       "min_range_s": MIN_RANGE_S, "episode_flags": FLAGS},
        "settings": settings, "overall": overall, "videos": videos, "episodes": report_eps,
        "dropped_segments": dropped, "unusable_ranges": sorted(ranges, key=lambda r: r["start_s"] - r["end_s"]),  # longest first
    }


# ---------- rendering ----------

def _pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


def _num(x, unit="") -> str:
    return "–" if x is None else f"{x:g}{unit}"


def _clock(s) -> str:
    return "–" if s is None else f"{int(s // 60)}:{s % 60:04.1f}"


def to_markdown(r: dict, max_rows: int = 15) -> str:
    o = r["overall"]
    L, R = o.get("left", {}), o.get("right", {})
    cap = o["captions"]
    captions = (f"{cap['captioned']}/{cap['episodes']} episodes, {cap['with_steps']} with sub-steps, "
                f"{cap['idle_dropped']} dropped as idle") if cap["enabled"] else "off (default task text)"
    drops = o["dropped_segments"]
    drop_text = str(drops["count"]) + (" (" + ", ".join(f"{k}: {v}" for k, v in drops["by_reason"].items()) + ")"
                                       if drops["count"] else "")
    length = o["episode_seconds"] or {}
    lines = [f"# QC report: {r['dataset']}", "",
             f"Generated {r['generated']} by {r['tool']}.", "",
             f"**{r['disclaimer']}**", "",
             "## Summary", "", "| Check | Result |", "|---|---|",
             f"| Source video | {o['videos']} file(s), {o['source_minutes']} min |",
             f"| Episodes | {o['episodes']} ({o['episode_minutes']} min at {o['fps']:g} fps, {o['image']}); "
             f"length {_num(length.get('min'))}–{_num(length.get('max'))} s, median {_num(length.get('median'))} s |",
             f"| Hand detected (episode frames) | left {_pct(L.get('detected'))}, right {_pct(R.get('detected'))}, "
             f"any {_pct(o.get('any_hand'))}, both {_pct(o.get('both_hands'))} |",
             f"| Hand cut by image border | left {_pct(L.get('cut_by_border'))}, right {_pct(R.get('cut_by_border'))} |",
             f"| Wrist distance outside {DIST_RANGE[0]}–{DIST_RANGE[1]} m | left {_pct(L.get('distance_implausible'))}, "
             f"right {_pct(R.get('distance_implausible'))} |",
             f"| Wrist jitter (median) | left {_num(L.get('jitter_px'), ' px')} / {_num(L.get('jitter_mm'), ' mm')} "
             f"(depth {_num(L.get('jitter_depth_mm'), ' mm')}), right {_num(R.get('jitter_px'), ' px')} / "
             f"{_num(R.get('jitter_mm'), ' mm')} (depth {_num(R.get('jitter_depth_mm'), ' mm')}) |",
             f"| Wrist jumps > {JUMP_M * 100:g} cm | left {_pct(L.get('jumps'))}, right {_pct(R.get('jumps'))} |",
             f"| Captions | {captions} |",
             f"| Dropped segments | {drop_text} |",
             f"| Episodes with warnings | {o['flagged_episodes']} of {o['episodes']} |", ""]
    lines += ["## Source videos", "", "| Video | Original | Processed | Any hand | Left | Right | Cut L / R |", "|---|---|---|---|---|---|---|"]
    for v in r["videos"]:
        orig = v.get("original")
        o_txt = f"{orig['width']}×{orig['height']} {orig['fps']:g} fps {orig.get('codec', '')}, {orig['seconds']} s" if orig else "same"
        p_txt = f"{v.get('width')}×{v.get('height')} {_num(v.get('fps'))} fps, {_num(v.get('seconds'))} s"
        lines.append(f"| {v['video']} | {o_txt} | {p_txt} | {_pct(v.get('any_hand'))} | {_pct(v.get('left_detected'))} | "
                     f"{_pct(v.get('right_detected'))} | {_pct(v.get('left_cut'))} / {_pct(v.get('right_cut'))} |")
    lines += ["", "## Episodes", "", "| # | Start | s | Left | Right | Cut L / R | Jitter L / R | Task | Warnings |",
              "|---|---|---|---|---|---|---|---|---|"]
    for e in r["episodes"]:
        lj = f"{_num(e['left']['jitter_px'])}/{_num(e['right']['jitter_px'])} px"
        task = (e["task"] or "").replace("|", "/")
        lines.append(f"| {e['episode_index']} | {_clock(e['start_s'])} | {e['seconds']:g} | {_pct(e['left']['detected'])} | "
                     f"{_pct(e['right']['detected'])} | {_pct(e['left']['cut_by_border'])} / {_pct(e['right']['cut_by_border'])} | "
                     f"{lj} | {task} | {'; '.join(e['flags']) or 'none'} |")
    if r["dropped_segments"]:
        lines += ["", "## Dropped segments", "", "| Video | From | To | Reason |", "|---|---|---|---|"]
        lines += [f"| {d['video']} | {_clock(d['start_s'])} | {_clock(d['end_s'])} | {d['reason']} |" for d in r["dropped_segments"]]
    lines += ["", f"## Unusable ranges (longest {max_rows}; all in qc_report.json)", ""]
    if r["unusable_ranges"]:
        lines += ["| Video | From | To | s | Reason |", "|---|---|---|---|---|"]
        lines += [f"| {u['video']} | {_clock(u['start_s'])} | {_clock(u['end_s'])} | {u['end_s'] - u['start_s']:.1f} | {u['reason']} |"
                  for u in r["unusable_ranges"][:max_rows]]
    else:
        lines.append("None of at least 1 s.")
    lines += ["", "## How to read this", "",
              "- *Detected*: MediaPipe reported the hand and its 3D placement succeeded. Episode figures use the delivered frames.",
              f"- *Cut by border*: a keypoint lies within {BORDER:.0%} of the image width of the edge (or outside); "
              "keypoints of a cut hand are partly guessed.",
              "- *Wrist distance*: from the average hand size, not a depth sensor; few-centimetre errors are normal.",
              "- *Jitter*: median distance of the wrist from the midpoint of its neighbouring frames (half the "
              f"frame-to-frame acceleration), in pixels of the delivered {o['image']} image and in millimetres in 3D. "
              "*Depth* is the part along the viewing axis; when it makes up most of the 3D figure, the noise is in "
              "the distance estimate rather than in the image. It includes real motion, so compare deliveries "
              "rather than expect zero. "
              f"*Jumps* are frames where this exceeds {JUMP_M * 100:g} cm (often a depth jump or a left/right swap).",
              "- *Unusable ranges*: time outside kept episodes (with the reason it was not kept), and stretches inside "
              "episodes where no hand is usable.", ""]
    return "\n".join(lines)


def to_html(markdown: str, title: str) -> str:
    """The markdown above as a standalone page (headings, tables, bullets, paragraphs only)."""
    def inline(text):
        text = html.escape(text)
        for mark, tag in (("**", "strong"), ("*", "em")):
            parts = text.split(mark)
            text = "".join(p if i % 2 == 0 else f"<{tag}>{p}</{tag}>" for i, p in enumerate(parts)) if len(parts) % 2 else text
        return text

    body, table, items = [], [], []

    def flush():
        if table:
            rows = [r for r in table if not set(r.replace("|", "").strip()) <= set("-")]
            cells = [[inline(c.strip()) for c in r.strip().strip("|").split("|")] for r in rows]
            body.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in cells[0]) + "</tr>"
                        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells[1:]) + "</table>")
            table.clear()
        if items:
            body.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for line in markdown.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        if line.startswith("- "):
            items.append(line[2:])
            continue
        flush()
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            body.append(f"<h{level}>{inline(line.lstrip('#').strip())}</h{level}>")
        elif line.strip():
            body.append(f"<p>{inline(line)}</p>")
    flush()
    style = ("body{font:14px/1.45 -apple-system,Segoe UI,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}"
             "table{border-collapse:collapse;margin:8px 0 16px;font-size:13px}td,th{border:1px solid #ccc;padding:3px 7px;text-align:left}"
             "th{background:#f3f3f3}h1{font-size:22px}h2{font-size:17px;margin-top:22px}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{style}</style></head><body>{''.join(body)}</body></html>")


def write_report(dataset: str | Path, cache: str | Path | None = None) -> dict:
    """Write qc_report.json, qc_report.md and qc_report.html into `dataset`; return the report."""
    dataset = Path(dataset)
    report = build_report(dataset, cache)
    (dataset / "qc_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    md = to_markdown(report)
    (dataset / "qc_report.md").write_text(md)
    (dataset / "qc_report.html").write_text(to_html(md, f"QC report: {report['dataset']}"))
    return report


def write_run_log(dataset: str | Path, settings: dict, videos: list[dict], dropped: list[dict],
                  caption: bool, default_task: str) -> None:
    """meta/run.json: what `run` did, for the QC report (settings, source videos and tracks, idle drops)."""
    (Path(dataset) / "meta" / "run.json").write_text(json.dumps(
        {"settings": settings, "videos": videos, "dropped": dropped, "caption": caption, "default_task": default_task},
        indent=1, ensure_ascii=False))
