"""Stereo section of the QC report (see qc.py): triangulation figures per hand and overall, a calibration summary,
and plain warnings when the numbers look implausible.

Inputs, all from the dataset: meta/triangulation_stats.json (per delivered frame and hand: reason code, epipolar and
reprojection error, and the checks used), meta/calibration.json, and the delivered 3D joints with their per-joint
status in data/*.parquet. Wrist jitter is measured on the triangulated 3D, only where the wrist was triangulated in
that frame (not filled across a gap).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .stereo import REASONS

LIMITS = {"min_baseline_m": 0.02, "max_baseline_m": 0.3, "max_rejected": 0.5, "max_epipolar_of_limit": 0.5,
          "max_aspect_difference": 0.01, "min_both_views_frames": 10}
DEPTH_RANGE = (0.15, 1.2)  # metres; as qc.DIST_RANGE
REJECTED = (1, 2, 3, 4, 5)


def _share(mask, of=None):
    mask = np.asarray(mask, bool)
    if of is not None:
        mask = mask[np.asarray(of, bool)]
    return round(float(mask.mean()), 3) if len(mask) else None


def _q(a, q, digits=2):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return round(float(np.percentile(a, q)), digits) if len(a) else None


def _array(values) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in values], float).reshape(-1, 2)


def load(dataset: Path) -> dict:
    """meta/triangulation_stats.json and meta/calibration.json (empty dicts when missing)."""
    meta = Path(dataset) / "meta"
    read = lambda name: json.loads((meta / name).read_text()) if (meta / name).exists() else {}  # noqa: E731
    stats = read("triangulation_stats.json")
    return {"stats": stats, "calibration": read("calibration.json"),
            "episodes": {e["episode_index"]: e for e in stats.get("episodes", [])}}


def measured_jitter(checks: dict, kp3d: np.ndarray, valid: np.ndarray, status: np.ndarray | None) -> None:
    """Replace the 3D jitter in qc.frame_checks output by jitter on measured wrists only: frames where the wrist was
    triangulated in that frame (status 1), not filled across a gap (a filled stretch is a straight line)."""
    from .qc import wrist_residual
    measured = (valid > 0) & ((status[:, :, 0] == 1) if status is not None else True)
    checks["jitter_m"] = np.stack([wrist_residual(kp3d[:, s, 0], measured[:, s]) for s in range(2)], 1)
    checks["jitter_z"] = np.stack([wrist_residual(kp3d[:, s, 0, 2:], measured[:, s]) for s in range(2)], 1)


def episode_arrays(tri: dict, index: int, frames: dict, checks: dict) -> dict | None:
    """Per delivered frame and hand (n, 2) for one episode, or None when the statistics do not match the frames."""
    per = tri["episodes"].get(index, {}).get("per_frame")
    n = len(frames["valid"])
    if not per or len(per["frame"]) != n:
        return None
    status = frames.get("status")
    return {"reason": np.array(per["reason"], int).reshape(-1, 2), "epipolar": _array(per["epipolar_px"]),
            "reprojection": _array(per["reprojection_px"]), "valid": frames["valid"] > 0,
            "depth": frames["kp3d"][:, :, 0, 2], "jitter_m": checks["jitter_m"], "jitter_z": checks["jitter_z"],
            "triangulated": (status == 1).sum(-1) if status is not None else np.full((n, 2), np.nan)}


def figures(a: dict, hands=(0, 1)) -> dict:
    """Triangulation figures over the frames in `a` (episode_arrays, possibly concatenated) for the given hands
    pooled: one hand gives its own figures, both hands their hand-frames together."""
    get = {k: np.concatenate([v[:, s] for s in hands]) for k, v in a.items()}
    reason, valid = get["reason"], get["valid"]
    seen = reason >= 0
    both = seen & (reason != 1)
    ok = reason == 0
    depth = get["depth"][valid]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        jm, jz = np.nanmedian(get["jitter_m"]), np.nanmedian(get["jitter_z"])
    joints = get["triangulated"][valid]
    return {
        "frames": int(len(reason)),
        "seen_any_view": _share(seen),
        "seen_both_views": _share(both),
        "both_views_frames": int(both.sum()),
        "passed_checks": _share(ok, both) if both.any() else None,
        "rejected": {REASONS[k]: int((reason == k).sum()) for k in REJECTED},
        "rejected_share": round(1 - float(ok[both].mean()), 3) if both.any() else None,
        "triangulated_of_seen": _share(ok, seen) if seen.any() else None,
        "in_3d": _share(valid),
        "joints_triangulated": round(float(np.nanmean(joints)) / 21, 3) if np.isfinite(joints).any() else None,
        "epipolar_px": {"median": _q(get["epipolar"][ok], 50), "p95": _q(get["epipolar"][ok], 95),
                        "median_all_matched": _q(get["epipolar"][both], 50)},
        "reprojection_px": {"median": _q(get["reprojection"][ok], 50), "p95": _q(get["reprojection"][ok], 95)},
        "wrist_depth_m": {"median": _q(depth, 50, 3), "min": _q(depth, 0, 3), "max": _q(depth, 100, 3),
                          "p5": _q(depth, 5, 3), "p95": _q(depth, 95, 3)},
        "jitter_mm": round(float(jm) * 1000, 1) if np.isfinite(jm) else None,
        "jitter_depth_mm": round(float(jz) * 1000, 1) if np.isfinite(jz) else None,
    }


def all_figures(a: dict) -> dict:
    return {"left": figures(a, (0,)), "right": figures(a, (1,)), "both": figures(a, (0, 1))}


def episode_flags(f: dict) -> list[str]:
    """Stereo warnings for one episode (same style as qc.episode_flags)."""
    out = []
    for name, side in (("L", "left"), ("R", "right")):
        h = f[side]
        if h["both_views_frames"] >= LIMITS["min_both_views_frames"] and h["rejected_share"] > LIMITS["max_rejected"]:
            out.append(f"{name} triangulation rejected {h['rejected_share']:.0%}")
    return out


def ranges(a: dict, fps: float, offset_s: float, min_s: float) -> list[dict]:
    """Stretches of at least `min_s` inside an episode without any hand in 3D, with the reason."""
    from .qc import runs
    out = []
    for lo, hi in runs(~a["valid"].any(1)):
        if (hi - lo) / fps < min_s:
            continue
        reason = a["reason"][lo:hi]
        seen = reason[reason >= 0]
        if len(seen) >= (hi - lo) / 2:
            common = max(REJECTED, key=lambda k: (seen == k).sum())
            text = f"hands seen but not triangulated (mostly: {REASONS[common]})"
        else:
            text = "no hand detected"
        out.append({"start_s": round(offset_s + lo / fps, 2), "end_s": round(offset_s + hi / fps, 2), "reason": text})
    return out


def calibration_summary(calibration: dict, checks: dict | None, video_size: tuple[int, int] | None = None) -> dict:
    """Baseline, image size, lens model, whether the input was already rectified, from meta/calibration.json."""
    rig = calibration.get("original") or {}
    if not rig:
        return {}
    R = np.asarray(rig["R"], float)
    angle = rig.get("rotation_deg")
    if angle is None:
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    rectified = rig.get("rectified_input")
    if rectified is None:
        rectified = bool(rig["left"] == rig["right"] and not any(rig["left"].get("distortion", []))
                         and np.allclose(R, np.eye(3)) and np.allclose(rig["T"][1:], 0))
    return {"model": rig.get("model"), "image_size": [rig["image_width"], rig["image_height"]],
            "calibrated_size": rig.get("calibrated_size"), "video_size": list(video_size) if video_size else None,
            "baseline_m": round(float(rig["baseline_m"]), 4),
            "focal_px": round((rig["left"]["fy"] + rig["right"]["fy"]) / 2, 1), "rotation_deg": round(angle, 2),
            "rectified_input": bool(rectified), "checks": checks}


def depth_error_mm(cal: dict, depth_m: float | None) -> float | None:
    """Depth change for one pixel of disparity error at `depth_m`: z² / (f · baseline)."""
    if not cal or not depth_m or not cal.get("baseline_m"):
        return None
    return round(depth_m ** 2 / (cal["focal_px"] * cal["baseline_m"]) * 1000, 1)


def plausibility(cal: dict, both: dict) -> list[str]:
    """Plain warnings when the calibration or the overall figures look implausible."""
    out = []
    b = cal.get("baseline_m")
    if b is not None and b < LIMITS["min_baseline_m"]:
        out.append(f"The baseline is {b * 100:.1f} cm, under {LIMITS['min_baseline_m'] * 100:g} cm. Depth from such a "
                   "short baseline is very sensitive to pixel errors; check the calibration.")
    if b is not None and b > LIMITS["max_baseline_m"]:
        out.append(f"The baseline is {b:.2f} m, more than {LIMITS['max_baseline_m']:g} m, which is unusual for a "
                   "head-mounted pair. Check that T is in metres.")
    made, video = cal.get("calibrated_size"), cal.get("image_size")
    if made and video and abs(made[0] / made[1] - video[0] / video[1]) > LIMITS["max_aspect_difference"] * video[0] / video[1]:
        out.append(f"The calibration was made for {made[0]}×{made[1]} images but the video is {video[0]}×{video[1]}, a "
                   "different shape. The calibration does not fit this video.")
    if both.get("both_views_frames", 0) == 0:
        out.append("No hand was seen in both views in the delivered frames.")
    elif both["rejected_share"] > LIMITS["max_rejected"]:
        worst = max(REJECTED[1:], key=lambda k: both["rejected"][REASONS[k]])
        out.append(f"{both['rejected_share']:.0%} of the hand frames seen in both views failed the triangulation checks "
                   f"(more than half; most often: {REASONS[worst]}). Check the calibration and that both videos start "
                   "on the same frame.")
    elif both.get("triangulated_of_seen") is not None and both["triangulated_of_seen"] < 1 - LIMITS["max_rejected"]:
        one = both["rejected"]["one view only"]
        failed = sum(both["rejected"][REASONS[k]] for k in REJECTED[1:])
        out.append(f"Only {both['triangulated_of_seen']:.0%} of the hand frames seen by a camera were triangulated "
                   f"(not triangulated: {one} seen by one camera only, {failed} failed the checks). A hand that only "
                   "one camera sees, for example near the image edge, has no 3D.")
    limit = (cal.get("checks") or {}).get("max_epipolar_px")
    epi = both.get("epipolar_px", {}).get("median_all_matched")
    if limit and epi is not None and epi > LIMITS["max_epipolar_of_limit"] * limit:
        out.append(f"The median epipolar error of matched hands is {epi:g} px, more than half the {limit:g} px limit. "
                   "The calibration may not fit this video.")
    depth = both.get("wrist_depth_m", {}).get("median")
    if depth is not None and not DEPTH_RANGE[0] <= depth <= DEPTH_RANGE[1]:
        out.append(f"The median wrist depth is {depth:g} m, outside {DEPTH_RANGE[0]}–{DEPTH_RANGE[1]} m, which is "
                   "implausible for a worn camera. Check the baseline and the focal length.")
    return out


def section(tri: dict, per_episode: list[dict]) -> dict:
    """The stereo part of the report: calibration, checks, figures per hand and overall, warnings."""
    checks = tri["stats"].get("checks")
    cal = calibration_summary(tri["calibration"], checks)
    if per_episode:
        pooled = {k: np.concatenate([a[k] for a in per_episode]) for k in per_episode[0]}
        overall = all_figures(pooled)
    else:
        overall = {}
    both = overall.get("both", {})
    return {"calibration": cal, "checks": checks, "overall": overall,
            "depth_error_mm_per_px": depth_error_mm(cal, both.get("wrist_depth_m", {}).get("median")),
            "warnings": plausibility(cal, both) if cal else ["No calibration found in meta/calibration.json."]}


# ---------- rendering ----------

def _pct(x):
    return "–" if x is None else f"{x:.0%}"


def _n(x, unit=""):
    return "–" if x is None else f"{x:g}{unit}"


def _checks_text(c: dict | None) -> str:
    if not c:
        return "The checks used were not recorded in this dataset."
    lo, hi = c["palm_size"]
    return (f"A hand is rejected when the median row difference of its joints between the two rectified views is above "
            f"{c['max_epipolar_px']:g} px (epipolar check), when it lies outside {c['min_depth']:g}–{c['max_depth']:g} m, "
            f"when its palm "
            f"is not {lo * 100:g}–{hi * 100:g} cm across, or when fewer than {c['min_joints']} joints remain; a single "
            f"joint is dropped when its reprojection error is above {c['max_reprojection_px']:g} px. Pixel limits are "
            f"for the {c['image_height']}-row video.")


def markdown(s: dict, episodes: list[dict]) -> list[str]:
    cal, o = s["calibration"], s["overall"]
    lines = ["## Stereo triangulation", ""]
    if cal:
        how = ("The two views were already rectified (one lens, baseline only)." if cal["rectified_input"] else
               f"The two views are rectified by this tool from R and T; the cameras are turned {cal['rotation_deg']:g}° "
               "relative to each other.")
        made = cal.get("calibrated_size")
        size = f"{cal['image_size'][0]}×{cal['image_size'][1]}"
        lines += [f"Calibration: {cal['model']} lenses, video {size}"
                  + (f" (calibrated at {made[0]}×{made[1]})" if made and list(made) != cal["image_size"] else "")
                  + f", baseline {cal['baseline_m'] * 100:.1f} cm, focal length {cal['focal_px']:g} px. {how}"]
        if s.get("depth_error_mm_per_px") is not None:
            lines[-1] += (f" At the median wrist depth ({o['both']['wrist_depth_m']['median']:g} m), one pixel of "
                          f"disparity error changes the depth by about {s['depth_error_mm_per_px']:g} mm.")
        lines.append("")
    lines += [_checks_text(s.get("checks")), ""]
    if o:
        L, R, B = o["left"], o["right"], o["both"]
        rej = lambda h: " / ".join(str(h["rejected"][REASONS[k]]) for k in REJECTED)  # noqa: E731
        depth = lambda h: (f"{_n(h['wrist_depth_m']['median'], ' m')} ({_n(h['wrist_depth_m']['min'])}–"  # noqa: E731
                           f"{_n(h['wrist_depth_m']['max'], ' m')})")
        pair = lambda d, unit: f"{_n(d['median'])} / {_n(d['p95'], unit)}"  # noqa: E731
        jit = lambda h: f"{_n(h['jitter_mm'], ' mm')} (depth {_n(h['jitter_depth_mm'], ' mm')})"  # noqa: E731
        rows = [("Seen in both views (share of delivered frames)", lambda h: _pct(h["seen_both_views"])),
                ("Passed the triangulation checks (share of frames seen in both views)", lambda h: _pct(h["passed_checks"])),
                ("Rejected, frames: one view only / epipolar / depth / hand size / few joints", rej),
                ("Hand in 3D (share of delivered frames; gaps up to 0.35 s filled)", lambda h: _pct(h["in_3d"])),
                ("Joints triangulated in their own frame (rest filled)", lambda h: _pct(h["joints_triangulated"])),
                ("Epipolar error, median / 95th percentile (accepted hands)", lambda h: pair(h["epipolar_px"], " px")),
                ("Reprojection error, median / 95th percentile (accepted hands)",
                 lambda h: pair(h["reprojection_px"], " px")),
                ("Wrist depth, median (range)", depth),
                ("Wrist jitter from measured 3D, median", jit)]
        lines += ["| | Left hand | Right hand | Both hands |", "|---|---|---|---|"]
        lines += [f"| {label} | {fn(L)} | {fn(R)} | {fn(B)} |" for label, fn in rows]
        lines.append("")
    if s["warnings"]:
        lines += ["**Stereo warnings:**", ""] + [f"- {w}" for w in s["warnings"]]
    else:
        lines.append("Stereo warnings: none.")
    rows = [e for e in episodes if e.get("stereo")]
    if rows:
        lines += ["", "### Stereo per episode", "",
                  "| # | Both views L / R | Passed L / R | In 3D L / R | Epipolar median | Depth median | Jitter L / R |",
                  "|---|---|---|---|---|---|---|"]
        for e in rows:
            L, R, B = e["stereo"]["left"], e["stereo"]["right"], e["stereo"]["both"]
            lines.append(f"| {e['episode_index']} | {_pct(L['seen_both_views'])} / {_pct(R['seen_both_views'])} | "
                         f"{_pct(L['passed_checks'])} / {_pct(R['passed_checks'])} | {_pct(L['in_3d'])} / "
                         f"{_pct(R['in_3d'])} | {_n(B['epipolar_px']['median'], ' px')} | "
                         f"{_n(B['wrist_depth_m']['median'], ' m')} | {_n(L['jitter_mm'])} / {_n(R['jitter_mm'], ' mm')} |")
    lines.append("")
    return lines


HOW_TO_READ = [
    "- *Hand in 3D* (stereo): the hand was triangulated from both views and passed the checks, or was filled "
    "across a gap of at most 0.35 s. Episode figures use the delivered frames.",
    "- *Seen in both views*: the hand detector found the hand in the left and in the right image of the same frame. "
    "*Passed the triangulation checks* is the share of those frames where the match was accepted.",
    "- *Epipolar error*: after rectification, a joint should lie on the same image row in both views; this is the "
    "median row difference over a hand's joints, in pixels of the original video. Large values everywhere point to a "
    "calibration that does not fit the video, or to videos that are not in sync.",
    "- *Reprojection error*: distance between a joint's detections and its triangulated 3D point projected back into "
    "both images (mean of the two), median over the hand's joints.",
    "- *Wrist depth* and *wrist distance*: measured by triangulation, along the left camera's viewing axis (depth) or "
    "straight from the camera (distance). The error grows with the square of the distance.",
    "- *Jitter* (stereo): as below, but in millimetres from the measured 3D after the tool's smoothing, and only "
    "where the wrist was triangulated in that frame and its neighbours (filled frames are left out).",
]
