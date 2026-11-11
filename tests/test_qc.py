import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ego2lerobot import HandTrack
from ego2lerobot.qc import (DISCLAIMER, audit_segments, border_cut, frame_checks, hand_summary, runs, video_ranges,
                            wrist_residual, write_report)


def track_from_presence(present: np.ndarray, fps: float = 10.0) -> HandTrack:
    n = len(present)
    score = np.zeros((n, 2), np.float32)
    score[present, 0] = 0.9
    return HandTrack(np.zeros((n, 2, 21, 2)), np.zeros((n, 2, 21, 3)), score, fps, 1920, 1080)


def test_border_cut_flags_keypoints_near_or_outside_edges():
    kp = np.full((3, 21, 2), 500.0)
    kp[1, 3] = (5.0, 500.0)      # within 1 % of 1000 px from the left edge
    kp[2, 7] = (500.0, 1200.0)   # below the image
    assert border_cut(kp, 1000, 800).tolist() == [False, True, True]


def test_wrist_residual_is_zero_for_linear_motion_and_catches_a_jump():
    t = np.arange(10, dtype=float)
    p = np.column_stack([0.01 * t, 0.02 * t, 0.4 + 0.0 * t])
    valid = np.ones(10, bool)
    r = wrist_residual(p, valid)
    assert np.isnan(r[0]) and np.isnan(r[-1]) and np.nanmax(r) < 1e-12
    p[5, 2] += 0.2  # depth jump
    r = wrist_residual(p, valid)
    assert np.isclose(r[5], 0.2) and np.isclose(r[4], 0.1)
    valid[5] = False
    assert np.isnan(wrist_residual(p, valid)[4:7]).all()


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)) == [(1, 3), (4, 5), (7, 8)]
    assert runs(np.zeros(4, bool)) == []


def test_segment_audit_gives_a_reason_for_every_dropped_stretch():
    present = np.zeros(700, bool)
    present[0:100] = True          # 10 s, exported
    present[150:170] = True        # 2 s: too short
    present[250:350:2] = True      # every other frame: hand coverage 0.5
    present[400:500] = True        # captioner said idle
    present[550:650] = True        # neither: e.g. --per-video limit
    track = track_from_presence(present)
    segs = audit_segments(track, {"min_coverage": 0.7}, exported={(0, 100)}, idle={(400, 500)})
    got = [(d["start"], d["kept"], d["kind"]) for d in segs]
    assert got == [(0, True, None), (150, False, "too short"), (250, False, "low hand coverage"),
                   (400, False, "caption said idle"), (550, False, "not selected")]
    ranges = video_ranges(track, segs)
    reasons = {(r["start_s"], r["reason"]) for r in ranges}
    assert (10.0, "no hand detected") in reasons and (15.0, "hands in view only 2.0 s (< 3 s)") in reasons
    assert all(not (r["start_s"] < 10.0) for r in ranges)  # the kept episode is not unusable


def synthetic_episode(n: int = 60):
    """Left hand moving smoothly at 0.4 m; right hand found in half the frames, partly at the border or too far."""
    t = np.arange(n)
    kp2d = np.zeros((n, 2, 21, 3))
    kp3d = np.zeros((n, 2, 21, 3))
    valid = np.zeros((n, 2))
    kp2d[:, 0, :, 0] = 200 + 2 * t[:, None]
    kp2d[:, 0, :, 1] = 180
    kp2d[:, 0, :, 2] = 0.9
    kp3d[:, 0, :, 0] = 0.001 * t[:, None]
    kp3d[:, 0, :, 2] = 0.4
    valid[:, 0] = 1
    right = t < 30
    valid[right, 1] = 1
    kp2d[right, 1, :, :2] = (400, 200)
    kp2d[:10, 1, 0, 0] = 639        # cut by the right border in 10 of its 30 frames
    kp3d[right, 1, :, 2] = 0.5
    kp3d[20:30, 1, :, 2] = 2.0      # implausibly far in 10 of its 30 frames
    return kp2d, kp3d, valid


def test_frame_checks_shares():
    kp2d, kp3d, valid = synthetic_episode()
    c = frame_checks(kp2d, kp3d, valid, 640, 360)
    left, right = hand_summary(c, 0), hand_summary(c, 1)
    assert left == {"detected": 1.0, "cut_by_border": 0.0, "distance_implausible": 0.0, "jitter_px": 0.0,
                    "jitter_mm": 0.0, "jitter_depth_mm": 0.0, "jumps": 0.0}
    assert right["detected"] == 0.5
    assert np.isclose(right["cut_by_border"], 10 / 30, atol=1e-3)
    assert np.isclose(right["distance_implausible"], 10 / 30, atol=1e-3)
    assert right["jumps"] > 0  # the 0.5 m → 2 m step
    z = c["jitter_z"][:, 1]
    assert np.isclose(np.nanmax(z), 0.75) and np.nanargmax(z) in (19, 20)  # the step is all depth


def test_write_report_end_to_end(tmp_path):
    dataset, cache = tmp_path / "ds", tmp_path / "ds_cache"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    cache.mkdir()
    (dataset / "meta" / "info.json").write_text(json.dumps(
        {"fps": 15, "features": {"observation.images.ego": {"shape": [360, 640, 3]}}}))
    (dataset / "meta" / "episodes_info.json").write_text(json.dumps([{
        "episode_index": 0, "task": "wipe the table", "frames": 60, "steps": ["pick up cloth", "wipe"],
        "source": {"video": "clip.mp4", "start_frame": 30, "end_frame": 150}}]))
    kp2d, kp3d, valid = synthetic_episode()
    table = pa.table({
        "episode_index": pa.array(np.zeros(60, np.int64)), "frame_index": pa.array(np.arange(60)),
        "observation.hands.kp2d": pa.array(kp2d.reshape(60, -1).tolist(), pa.list_(pa.float32(), 126)),
        "observation.hands.kp3d": pa.array(kp3d.reshape(60, -1).tolist(), pa.list_(pa.float32(), 126)),
        "observation.hands.valid": pa.array(valid.tolist(), pa.list_(pa.float32(), 2)),
    })
    pq.write_table(table, dataset / "data" / "chunk-000" / "file-000.parquet")
    present = np.zeros(300, bool)
    present[30:150] = True  # 30 fps source: the episode is 1–5 s, nothing else has hands
    track = track_from_presence(present, fps=30.0)
    track.save(cache / "clip.hands.npz")
    (dataset / "meta" / "run.json").write_text(json.dumps({
        "settings": {"min_coverage": 0.7}, "caption": True, "default_task": "manipulate objects with the hands",
        "videos": [{"name": "clip.mp4", "path": str(tmp_path / "clip.mp4"), "track": str(cache / "clip.hands.npz")}],
        "dropped": []}))

    report = write_report(dataset)

    o = report["overall"]
    assert o["episodes"] == 1 and o["frames"] == 60 and o["left"]["detected"] == 1.0 and o["right"]["detected"] == 0.5
    assert o["captions"] == {"enabled": True, "captioned": 1, "with_steps": 1, "episodes": 1, "idle_dropped": 0}
    assert report["videos"][0]["seconds"] == 10.0 and report["videos"][0]["any_hand"] == 0.4
    assert report["episodes"][0]["start_s"] == 1.0
    assert any("R implausible distance" in f for f in report["episodes"][0]["flags"])
    assert {"video": "clip.mp4", "start_s": 5.0, "end_s": 10.0, "reason": "no hand detected"} in report["unusable_ranges"]
    md = (dataset / "qc_report.md").read_text()
    assert DISCLAIMER in md and "wipe the table" in md
    assert "<table>" in (dataset / "qc_report.html").read_text()
    assert json.loads((dataset / "qc_report.json").read_text())["overall"]["episodes"] == 1
