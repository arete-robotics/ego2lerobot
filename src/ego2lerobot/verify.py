"""Check footage against what its spec sheet claims: `ego2lerobot verify`.

Measures each video (size, frame timing, codec, audio), each IMU file (rate, gaps), stereo pairs
(frame counts, clock offset), and looks for footage that appears twice: the same recording again,
or the same moment filmed by a second camera.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import av
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMU_EXT = {".csv", ".txt"}
SAMPLE_HZ = 5.0            # how often a frame is kept for the content signature
THUMB = 32                 # signature thumbnails are THUMB x THUMB grey
HASH_BITS = 8              # hamming distance at or below this counts as the same picture
MIN_OVERLAP_S = 20.0       # shorter overlaps are not compared
SAME_PICTURE = 0.8         # share of overlapping samples that must match for "same recording"
SAME_MOTION = 0.8          # motion correlation for "same moment, other camera"


def probe(path: Path) -> dict:
    """Facts about one video file, read from the container without decoding."""
    out: dict = {"file": str(path), "bytes": path.stat().st_size}
    if out["bytes"] == 0:
        return out | {"error": "empty file"}
    try:
        with av.open(str(path)) as c:
            v = next((s for s in c.streams if s.type == "video"), None)
            if v is None:
                return out | {"error": "no video stream"}
            tb = float(v.time_base)
            pts = sorted(p.pts for p in c.demux(v) if p.pts is not None and p.size)
            cc = v.codec_context
            out |= {
                "width": cc.width, "height": cc.height, "codec": cc.name,
                "b_frames": bool(getattr(cc, "has_b_frames", False)),
                "audio": any(s.type == "audio" for s in c.streams),
                "nominal_fps": float(v.average_rate) if v.average_rate else None,
                "frames": len(pts),
            }
    except av.error.FFmpegError as e:  # unreadable or truncated file
        return out | {"error": f"cannot read: {e}"}
    if len(pts) < 3:
        return out | {"error": "fewer than 3 frames"}
    t = np.asarray(pts, float) * tb
    d = np.diff(t)
    med = float(np.median(d))
    late = d > 1.5 * med
    out |= {
        "duration_s": round(float(t[-1] - t[0] + med), 3),
        "fps": round(1.0 / med, 3),
        "interval_ms_median": round(med * 1e3, 3),
        "interval_ms_p99": round(float(np.percentile(d, 99)) * 1e3, 3),
        "intervals_out_of_spec": int(late.sum() + (d < 0.5 * med).sum()),
        "frames_missing": int(np.round(d[late] / med - 1).sum()),
    }
    return out


def signature(path: Path, hz: float = SAMPLE_HZ) -> dict | None:
    """Small grey thumbnails at a steady rate: enough to recognise a recording, not to view it."""
    try:
        with av.open(str(path)) as c:
            v = next((s for s in c.streams if s.type == "video"), None)
            if v is None:
                return None
            v.thread_type = "AUTO"
            tb = float(v.time_base)
            thumbs, times, next_t = [], [], None
            for f in c.decode(v):
                if f.pts is None:
                    continue
                t = f.pts * tb
                if next_t is None:
                    next_t = t
                if t + 1e-9 < next_t:
                    continue
                thumbs.append(f.to_ndarray(width=THUMB, height=THUMB, format="gray"))
                times.append(t)
                next_t += 1.0 / hz
    except av.error.FFmpegError:
        return None
    if len(thumbs) < 4:
        return None
    x = np.stack(thumbs).astype(np.float32)
    small = x.reshape(len(x), 8, THUMB // 8, 8, THUMB // 8).mean((2, 4)).reshape(len(x), 64)
    return {
        "hz": hz,
        "hash": small > small.mean(1, keepdims=True),            # 64 bits per sample
        "motion": np.abs(np.diff(x, axis=0)).mean((1, 2)),      # how much the picture changed
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def _overlap(n: int, m: int, lag: int) -> tuple[slice, slice]:
    """Index ranges of a (length n) and b (length m) that line up when b starts `lag` samples after a."""
    lo, hi = max(0, lag), min(n, m + lag)
    return slice(lo, hi), slice(lo - lag, hi - lag)


def compare(a: dict, b: dict) -> dict | None:
    """How two signatures relate at their best time offset, or None when they do not overlap enough."""
    hz = a["hz"]
    need = int(MIN_OVERLAP_S * hz)
    ma, mb = a["motion"], b["motion"]
    if min(len(ma), len(mb)) < need:
        return None
    za, zb = (ma - ma.mean()) / (ma.std() + 1e-9), (mb - mb.mean()) / (mb.std() + 1e-9)
    full = np.correlate(za, zb, "full")                        # index k is lag k - (len(zb) - 1)
    lags = np.arange(len(full)) - (len(zb) - 1)
    count = np.minimum(len(za), len(zb) + lags) - np.maximum(0, lags)
    score = np.where(count >= need, full / np.maximum(count, 1), -np.inf)
    best = None
    for k in np.argsort(score)[-3:]:                           # exact check of the three best offsets
        if not np.isfinite(score[k]):
            continue
        sa, sb = _overlap(len(ma), len(mb), int(lags[k]))
        r = _pearson(ma[sa], mb[sb])
        if best is None or r > best["motion_corr"]:
            ha = a["hash"][sa.start: sa.stop + 1]
            hb = b["hash"][sb.start: sb.stop + 1]
            n = min(len(ha), len(hb))
            same = float(((ha[:n] != hb[:n]).sum(1) <= HASH_BITS).mean())
            best = {"motion_corr": round(r, 3), "same_picture": round(same, 3),
                    "offset_s": round(int(lags[k]) / hz, 2), "overlap_s": round((sa.stop - sa.start) / hz, 1)}
    if best is None:
        return None
    if best["same_picture"] >= SAME_PICTURE:
        best["verdict"] = "same recording"
    elif best["motion_corr"] >= SAME_MOTION:
        best["verdict"] = "same moment, other camera"
    else:
        best["verdict"] = "different"
    return best


def imu_facts(path: Path) -> dict:
    """Sample rate of an IMU text file. The headline rate is samples over duration, which stays right
    when the intervals alternate between two values."""
    out: dict = {"file": str(path), "bytes": path.stat().st_size}
    if out["bytes"] == 0:
        return out | {"error": "empty file"}
    with open(path, newline="", errors="ignore") as f:
        rows = list(csv.reader(f))
    if not rows:
        return out | {"error": "empty file"}
    col = 0
    try:
        float(rows[0][0])
        body = rows
    except (ValueError, IndexError):
        header = [h.strip().lower() for h in rows[0]]
        col = next((i for i, h in enumerate(header) if h in ("t", "time", "timestamp", "ts", "t_seconds")
                    or "time" in h), 0)
        body = rows[1:]
    t = []
    for r in body:
        try:
            t.append(float(r[col]))
        except (ValueError, IndexError):
            continue
    if len(t) < 3:
        return out | {"error": "no samples"}
    t = np.sort(np.asarray(t))
    span = float(t[-1] - t[0])
    if span <= 0:
        return out | {"error": "timestamps do not advance"}
    raw = (len(t) - 1) / span
    scale = next((s for s in (1.0, 1e3, 1e6, 1e9) if 10 <= raw * s <= 5000), None)  # s, ms, us or ns
    if scale is None:
        return out | {"error": "timestamp unit not recognised", "samples": len(t)}
    d = np.diff(t) / scale
    med = float(np.median(d))
    return out | {
        "samples": len(t), "duration_s": round(span / scale, 3), "hz": round(raw * scale, 2),
        "hz_from_median_interval": round(1.0 / med, 2) if med > 0 else None,
        "gaps": int((d > 3 * med).sum()) if med > 0 else 0,
    }


_SIDE = re.compile(r"(?i)(?<![a-z])(left|right|l|r)(?![a-z])")


def stereo_pairs(videos: list[Path]) -> list[tuple[Path, Path]]:
    """Files whose names differ only by a left/right word."""
    by_key: dict[tuple, dict[str, Path]] = {}
    for p in videos:
        m = [x for x in _SIDE.finditer(p.name)]
        if not m:
            continue
        side = m[-1].group(1).lower()[0]
        key = (str(p.parent), p.name[: m[-1].start()] + "*" + p.name[m[-1].end():])
        by_key.setdefault(key, {})[side] = p
    return [(v["l"], v["r"]) for v in by_key.values() if {"l", "r"} <= set(v)]


def pair_facts(left: Path, right: Path, rows: bool = True) -> dict:
    def stamps(p):
        with av.open(str(p)) as c:
            v = next(s for s in c.streams if s.type == "video")
            return np.sort(np.asarray([q.pts for q in c.demux(v) if q.pts is not None and q.size], float)) * float(v.time_base)
    a, b = stamps(left), stamps(right)
    n = min(len(a), len(b))
    off = np.abs((a[:n] - a[0]) - (b[:n] - b[0])) * 1e3
    out = {"left": str(left), "right": str(right), "frames_left": len(a), "frames_right": len(b),
           "clock_offset_ms_median": round(float(np.median(off)), 3), "clock_offset_ms_max": round(float(off.max()), 3)}
    if rows:
        out["rows"] = row_alignment(left, right)
    return out


def _frames_at(path: Path, shares: list[float]) -> list[np.ndarray]:
    """Grey frames at the given shares of the clip (0 = start, 1 = end)."""
    out = []
    with av.open(str(path)) as c:
        v = next(s for s in c.streams if s.type == "video")
        span = float(v.duration * v.time_base) if v.duration else float(c.duration or 0) / av.time_base
        for share in shares:
            c.seek(int(share * span / v.time_base), stream=v, any_frame=False, backward=True)
            frame = next(c.decode(v), None)
            if frame is not None:
                out.append(frame.to_ndarray(format="gray"))
    return out


def row_alignment(left: Path, right: Path, frames: int = 6) -> dict | None:
    """How well the rows of a stereo pair line up, from background features matched between the views.

    In a rectified pair a point sits on the same row in both views. Returns the median row offset
    (left minus right, pixels) and how it changes across the picture, or None without OpenCV."""
    try:
        import cv2
    except ImportError:
        return None
    shares = [(k + 0.5) / frames for k in range(frames)]
    a, b = _frames_at(left, shares), _frames_at(right, shares)
    finder = cv2.SIFT_create(3000) if hasattr(cv2, "SIFT_create") else cv2.ORB_create(3000)
    matcher = cv2.BFMatcher(cv2.NORM_L2 if hasattr(cv2, "SIFT_create") else cv2.NORM_HAMMING)
    rows = []
    for la, rb in zip(a, b):
        k1, d1 = finder.detectAndCompute(la, None)
        k2, d2 = finder.detectAndCompute(rb, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            continue
        for m in matcher.knnMatch(d1, d2, k=2):
            if len(m) == 2 and m[0].distance < 0.7 * m[1].distance:
                (x1, y1), (x2, y2) = k1[m[0].queryIdx].pt, k2[m[0].trainIdx].pt
                rows.append((x1 - la.shape[1] / 2, y1 - la.shape[0] / 2, y1 - y2, x1 - x2))
    if len(rows) < 30:
        return {"matches": len(rows), "error": "too few matched features"}
    r = np.asarray(rows)
    dy = r[:, 2]
    med = float(np.median(dy))
    keep = np.abs(dy - med) <= 3 * 1.4826 * float(np.median(np.abs(dy - med))) + 2   # drop wrong matches
    r, dy = r[keep], dy[keep]
    fit, *_ = np.linalg.lstsq(np.c_[np.ones(len(r)), r[:, 0] / 1000, r[:, 1] / 1000], dy, rcond=None)
    left_over = dy - (fit[0] + fit[1] * r[:, 0] / 1000 + fit[2] * r[:, 1] / 1000)
    q1, q3 = np.percentile(dy, [25, 75])
    out = {
        "matches": int(len(r)), "frames": len(a),
        "row_offset_px_median": round(float(np.median(dy)), 2),
        "row_offset_px_iqr": [round(float(q1), 2), round(float(q3), 2)],
        "row_offset_change_px_per_1000px": {"across": round(float(fit[1]), 2), "down": round(float(fit[2]), 2)},
        "row_scatter_px_after_fit": round(float(np.median(np.abs(left_over))), 2),
        "disparity_px_median": round(float(np.median(r[:, 3])), 1),
    }
    out["rows_aligned"] = bool(abs(out["row_offset_px_median"]) <= 2.0 and max(abs(fit[1]), abs(fit[2])) <= 4.0)
    return out


def check_claims(claims: dict, videos: list[dict], imus: list[dict]) -> list[dict]:
    """One row per claim: what the sheet says, what the files show, and whether they agree."""
    rows = []
    ok_videos = [v for v in videos if "error" not in v]

    def row(name, claimed, measured, ok):
        rows.append({"item": name, "claimed": claimed, "measured": measured, "matches": bool(ok)})

    def spread(values):
        u = sorted(set(values))
        return u[0] if len(u) == 1 else u

    if ok_videos:
        for key in ("width", "height"):
            if key in claims:
                got = [v[key] for v in ok_videos]
                row(key, claims[key], spread(got), all(g == int(claims[key]) for g in got))
        if "fps" in claims:
            got = [v["fps"] for v in ok_videos]
            row("fps", claims["fps"], spread([round(g, 1) for g in got]),
                all(abs(g - float(claims["fps"])) <= 0.02 * float(claims["fps"]) for g in got))
        if "audio" in claims:
            want = str(claims["audio"]).lower() in ("1", "true", "yes")
            got = [v["audio"] for v in ok_videos]
            row("audio", want, spread(got), all(g == want for g in got))
    if "imu_hz" in claims:
        got = [m["hz"] for m in imus if "hz" in m]
        row("imu_hz", claims["imu_hz"], spread([round(g) for g in got]) if got else "no readable IMU file",
            bool(got) and all(abs(g - float(claims["imu_hz"])) <= 0.05 * float(claims["imu_hz"]) for g in got))
    return rows


def verify(paths: list[Path], claims: dict | None = None, duplicates: bool = True, progress: bool = True) -> dict:
    files = []
    for p in paths:
        files += sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
    vids = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    imu_files = [p for p in files if p.suffix.lower() in IMU_EXT]
    videos = [probe(p) for p in vids]
    imus = [imu_facts(p) for p in imu_files]
    pairs = stereo_pairs([p for p, v in zip(vids, videos) if "error" not in v])
    report: dict = {
        "videos": videos, "imu": imus, "stereo_pairs": [pair_facts(l, r) for l, r in pairs],
        "claims": check_claims(claims or {}, videos, imus), "repeats": [],
    }
    if duplicates:
        paired = {frozenset((str(l), str(r))) for l, r in pairs}
        sigs = {}
        for p, v in zip(vids, videos):
            if "error" not in v:
                if progress:
                    print(f"  reading {p.name}", flush=True)
                s = signature(p)
                if s is not None:
                    sigs[str(p)] = s
        names = sorted(sigs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                c = compare(sigs[a], sigs[b])
                if c and c["verdict"] != "different":
                    report["repeats"].append({"a": a, "b": b, **c, "stereo_pair": frozenset((a, b)) in paired})
    good = [v for v in videos if "error" not in v]
    report["summary"] = {
        "videos": len(videos), "unreadable_videos": len(videos) - len(good),
        "hours": round(sum(v["duration_s"] for v in good) / 3600, 3),
        "videos_with_timing_faults": sum(v["intervals_out_of_spec"] > 0 for v in good),
        "imu_files": len(imus), "unreadable_imu_files": sum("error" in m for m in imus),
        "claims_checked": len(report["claims"]), "claims_not_matching": sum(not r["matches"] for r in report["claims"]),
        "repeats_outside_stereo_pairs": sum(not r["stereo_pair"] for r in report["repeats"]),
        "stereo_pairs_with_rows_not_aligned": sum((p.get("rows") or {}).get("rows_aligned") is False
                                                  for p in report["stereo_pairs"]),
    }
    return report


def to_markdown(report: dict) -> str:
    s = report["summary"]
    out = ["# Footage check", "",
           f"- Videos: {s['videos']} ({s['hours']} hours). Unreadable: {s['unreadable_videos']}.",
           f"- Videos with frame timing faults: {s['videos_with_timing_faults']}.",
           f"- IMU files: {s['imu_files']}. Empty or unreadable: {s['unreadable_imu_files']}.",
           f"- Spec claims checked: {s['claims_checked']}. Not matching: {s['claims_not_matching']}.",
           f"- Repeated footage outside stereo pairs: {s['repeats_outside_stereo_pairs']}.",
           f"- Stereo pairs whose rows do not line up: {s['stereo_pairs_with_rows_not_aligned']}.", ""]
    if report["claims"]:
        out += ["## Spec sheet against the files", "", "| Item | Sheet says | Files show | Match |", "|---|---|---|---|"]
        out += [f"| {r['item']} | {r['claimed']} | {r['measured']} | {'yes' if r['matches'] else '**no**'} |" for r in report["claims"]]
        out.append("")
    out += ["## Videos", "", "| File | Size | fps | Frames | Interval median / p99 (ms) | Faulty intervals | Missing frames | Codec | Audio |",
            "|---|---|---|---|---|---|---|---|---|"]
    for v in report["videos"]:
        name = Path(v["file"]).name
        if "error" in v:
            out.append(f"| {name} | **{v['error']}** | | | | | | | |")
        else:
            out.append(f"| {name} | {v['width']}x{v['height']} | {v['fps']} | {v['frames']} | {v['interval_ms_median']} / {v['interval_ms_p99']} | "
                       f"{v['intervals_out_of_spec']} | {v['frames_missing']} | {v['codec']}{' (B-frames)' if v['b_frames'] else ''} | {'yes' if v['audio'] else 'no'} |")
    if report["imu"]:
        out += ["", "## IMU files", "", "| File | Samples | Rate (Hz) | Rate from median interval | Gaps |", "|---|---|---|---|---|"]
        for m in report["imu"]:
            name = Path(m["file"]).name
            out.append(f"| {name} | **{m['error']}** | | | |" if "error" in m else
                       f"| {name} | {m['samples']} | {m['hz']} | {m['hz_from_median_interval']} | {m['gaps']} |")
    if report["stereo_pairs"]:
        out += ["", "## Stereo pairs", "",
                "| Left | Right | Frames | Clock offset median / max (ms) | Row offset median (px) | Offset change across / down (px per 1000 px) | Rows line up |",
                "|---|---|---|---|---|---|---|"]
        for p in report["stereo_pairs"]:
            r = p.get("rows") or {}
            if "row_offset_px_median" in r:
                c = r["row_offset_change_px_per_1000px"]
                rows = f"{r['row_offset_px_median']} | {c['across']} / {c['down']} | {'yes' if r['rows_aligned'] else '**no**'}"
            else:
                rows = f"{r.get('error', 'not checked (needs OpenCV)')} | | "
            out.append(f"| {Path(p['left']).name} | {Path(p['right']).name} | {p['frames_left']} / {p['frames_right']} | "
                       f"{p['clock_offset_ms_median']} / {p['clock_offset_ms_max']} | {rows} |")
    if report["repeats"]:
        out += ["", "## Repeated footage", "", "| A | B | Finding | Picture match | Motion match | Offset (s) | Overlap (s) |", "|---|---|---|---|---|---|---|"]
        out += [f"| {Path(r['a']).name} | {Path(r['b']).name} | {r['verdict']}{' (stereo pair)' if r['stereo_pair'] else ''} | "
                f"{r['same_picture']} | {r['motion_corr']} | {r['offset_s']} | {r['overlap_s']} |" for r in report["repeats"]]
    return "\n".join(out) + "\n"


def add_parser(sub):
    p = sub.add_parser("verify", help="check footage against its spec sheet and look for repeated footage")
    p.add_argument("paths", nargs="+", type=Path, help="video and IMU files, or folders of them")
    p.add_argument("--claim", action="append", default=[], metavar="KEY=VALUE",
                   help="what the spec sheet says: width, height, fps, imu_hz, audio (repeatable)")
    p.add_argument("--out", type=Path, help="folder for verify_report.json and verify_report.md (default: print)")
    p.add_argument("--no-repeats", action="store_true", help="skip the search for repeated footage (faster)")


def run(args) -> int:
    claims = dict(c.split("=", 1) for c in args.claim)
    report = verify(args.paths, claims, duplicates=not args.no_repeats)
    text = to_markdown(report)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "verify_report.json").write_text(json.dumps(report, indent=1))
        (args.out / "verify_report.md").write_text(text)
        print(f"wrote {args.out / 'verify_report.md'}")
    else:
        print(text)
    return 0
