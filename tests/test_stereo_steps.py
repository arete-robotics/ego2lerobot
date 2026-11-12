"""Stereo sub-steps, grasp events and the stereo QC report, on a synthetic two-camera scene with known 3D hands.

The scene (16 s at 30 fps): both hands rest, reach, pause and grasp, carry, pause and release, twice. The hands are
projected into a calibrated pair with lens distortion, a small rotation between the cameras and pixel noise, then
triangulated by label_stereo, as real detections would be.
"""
import json

import numpy as np
import pytest

from ego2lerobot.segments import find_steps
from ego2lerobot.signals import (CLOSE, OPEN, Motion, candidates, grasp_events, measured_motion, presence_cues,
                                 together_cues, wrist_speed)
from ego2lerobot.stereo import label_stereo
from test_stereo import HAND, RIG, _rotmat, views

FPS, N = 30.0, 480
MOVES = [(2.0, 3.5, [0.08, 0.05, 0.42], [-0.06, 0.07, 0.48]), (5.0, 6.5, [-0.06, 0.07, 0.48], [0.1, 0.09, 0.40]),
         (8.0, 9.5, [0.1, 0.09, 0.40], [0.0, 0.04, 0.50]), (11.0, 12.5, [0.0, 0.04, 0.50], [0.08, 0.05, 0.42])]
GRIPS = [(4.0, 1), (7.0, 0), (10.0, 1), (13.0, 0)]  # (time the grip starts to change, closed after)
LEFT_OFFSET = np.array([-0.24, 0.04, 0.02])  # the left hand does the same, to the left and a bit lower
STILL = [(0.0, 2.0), (3.5, 5.0), (6.5, 8.0), (9.5, 11.0), (12.5, 16.0)]
EVENT_TOLERANCE_S, CUT_TOLERANCE_S = 0.1, 0.3


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def curl(closed: float) -> np.ndarray:
    """HAND with the index and thumb chains turned about their knuckles (bone lengths kept) so that the tips meet:
    0 open (aperture about 1.05), 1 closed (tips 1.4 cm apart, aperture about 0.14)."""
    h = HAND.copy()
    h[6:9] = (h[6:9] - h[5]) @ _rot_z(1.94 * closed).T + h[5]
    h[2:5] = (h[2:5] - h[1]) @ _rot_z(0.48 * closed).T + h[1]
    return h


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x ** 3 * (10 - 15 * x + 6 * x ** 2)  # minimum jerk


def scene() -> np.ndarray:
    """(N, 2, 21, 3) true joints of both hands in the left camera frame."""
    t = np.arange(N) / FPS
    pos = np.tile(np.asarray(MOVES[0][2], float), (N, 1))
    for a, b, p, q in MOVES:
        pos = np.where((t >= a)[:, None], np.asarray(p) + _smooth((t - a) / (b - a))[:, None] * (np.subtract(q, p)), pos)
    closed, state = np.zeros(N), 0.0
    for tc, c in GRIPS:
        closed = np.where(t >= tc, state + (c - state) * _smooth((t - tc) / 0.25), closed)
        state = c
    R = _rotmat(np.array([0.3, -0.2, 0.1]))
    right = np.stack([curl(closed[i]) @ R.T + pos[i] for i in range(N)])
    return np.stack([right + LEFT_OFFSET, right], 1)


def true_events(truth: np.ndarray, side: int) -> list[tuple[float, str]]:
    """Times where the true aperture crosses CLOSE going down (grasp) or OPEN going up (release)."""
    a = np.linalg.norm(truth[:, side, 4] - truth[:, side, 8], axis=-1) / np.linalg.norm(truth[:, side, 9] - truth[:, side, 0], axis=-1)
    out = [(i / FPS, "grasp") for i in range(1, N) if a[i] < CLOSE <= a[i - 1]]
    return sorted(out + [(i / FPS, "release") for i in range(1, N) if a[i] > OPEN >= a[i - 1]])


def detections(truth, rig=RIG, noise=0.8, seed=0, hide_right_view=(), wrong_right_view=()):
    """Per view (kp (N, 2, 21, 2), conf (N, 2, 21), score (N, 2)), as label_stereo takes them.

    hide_right_view: (frames, hand slots) where the right camera does not see the hand;
    wrong_right_view: (frames, hand slots) where the right camera's detection is 40 % of the image height too low."""
    rng = np.random.default_rng(seed)
    per = [views(truth[i, 0], truth[i, 1], rig=rig, noise=noise, rng=rng) for i in range(N)]
    stacked = [[np.stack([f[v][k] for f in per]) for k in range(3)] for v in (0, 1)]
    kp, conf, score = stacked[1]
    for frames, hands in hide_right_view:
        for s in hands:
            kp[frames, s], conf[frames, s], score[frames, s] = 0, 0, 0
    for frames, hands in wrong_right_view:
        for s in hands:
            kp[frames, s, :, 1] += 0.4 * rig.height
    return [tuple(v) for v in stacked]


@pytest.fixture(scope="module")
def truth():
    t = scene()
    for v in (0, 1):  # every joint of both hands inside both images, away from the border
        uv = RIG.project(t.reshape(-1, 3), v)
        assert (uv > 30).all() and (uv[:, 0] < 1890).all() and (uv[:, 1] < 1050).all()
    return t


@pytest.fixture(scope="module")
def clean_labels(truth):
    return label_stereo(RIG, detections(truth), FPS)


GAP = slice(int(5.0 * FPS), int(6.4 * FPS))  # 1.4 s while both hands carry: the right camera sees neither hand


@pytest.fixture(scope="module")
def gap_labels(truth):
    return label_stereo(RIG, detections(truth, hide_right_view=[(GAP, (0, 1))]), FPS)


def test_grasp_and_release_events_from_the_measured_pinch(truth, clean_labels):
    events = grasp_events(clean_labels.step_track(RIG))
    for side, name in enumerate(("left", "right")):
        expected = true_events(truth, side)
        got = [(e["t"], e["event"]) for e in events[name]]
        assert [k for _, k in got] == [k for _, k in expected] == ["grasp", "release", "grasp", "release"]
        assert all(abs(a - b) <= EVENT_TOLERANCE_S for (a, _), (b, _) in zip(got, expected)), (got, expected)


def test_measured_wrist_speed_pauses_below_and_moves_above_the_threshold(clean_labels):
    speed = wrist_speed(clean_labels.step_track(RIG))  # m/s from the triangulated wrists
    t = np.arange(N) / FPS
    still = np.zeros(N, bool)
    for a, b in STILL:
        still[(t > a + 0.3) & (t < b - 0.3)] = True
    middle = np.zeros(N, bool)
    for a, b, *_ in MOVES:
        middle[(t > a + 0.5) & (t < b - 0.5)] = True
    assert np.nanpercentile(speed[still], 95) < 0.1 < np.nanmin(speed[middle])


def test_boundaries_without_a_judge_land_at_the_grasps_and_releases(truth, clean_labels):
    """Both hands grasp (or release) together, so each true boundary has two cues and is cut without a model."""
    track = clean_labels.step_track(RIG)
    steps, decisions = find_steps(track)
    cuts = [s.end / FPS for s in steps[:-1]]
    expected = [t for t, _ in true_events(truth, 1)]
    assert len(cuts) == len(expected) and all(abs(a - b) <= CUT_TOLERANCE_S for a, b in zip(cuts, expected)), cuts
    assert steps[0].start == 0 and steps[-1].end == N and all(s.hand_coverage == 1.0 for s in steps)
    assert all("two or more signals" in s.cuts["end"] for s in steps[:-1])


def test_a_triangulation_gap_is_not_a_boundary(truth, clean_labels, gap_labels):
    assert not gap_labels.valid[GAP.start + 12: GAP.stop - 12].any()  # the gap is real: no 3D for over a second
    assert (gap_labels.reason[GAP] == 1).all()  # "one view only"
    window = (GAP.start - 0.5 * FPS, GAP.stop + 0.5 * FPS)
    clean, gap = clean_labels.step_track(RIG), gap_labels.step_track(RIG)

    assert grasp_events(gap) == grasp_events(clean)
    assert not [c for c in candidates(gap) if window[0] <= c.frame <= window[1]]
    assert [(s.start, s.end) for s in find_steps(gap)[0]] == [(s.start, s.end) for s in find_steps(clean)[0]]
    yes = lambda *_: {"new_step": True}  # noqa: E731  a judge that confirms every candidate
    assert not [s.end for s in find_steps(gap, yes)[0] if window[0] <= s.end <= window[1]]

    # read as "hands out of view", the same gap would be cut: hands leave and enter, and the stretch splits
    naive = gap_labels.as_track()
    assert {name for _, name in presence_cues(naive)} >= {"left leaves", "right leaves", "left enters", "right enters"}
    assert [s.end for s in find_steps(naive)[0] if window[0] <= s.end <= window[1]]


def test_no_grasp_event_is_invented_across_a_gap(truth):
    """The right camera misses both hands while they close their grip: the grasp is lost, not placed after the gap."""
    gap = slice(int(9.6 * FPS), int(11.0 * FPS))
    events = grasp_events(label_stereo(RIG, detections(truth, hide_right_view=[(gap, (0, 1))]), FPS).step_track(RIG))
    for side in ("left", "right"):
        assert [e["event"] for e in events[side]] == ["grasp", "release", "release"]
        assert not [e for e in events[side] if gap.start - 3 <= e["frame"] <= gap.stop + 0.5 * FPS]


def test_hands_meeting_is_not_reported_across_a_long_gap_of_measured_wrists():
    n = 120
    wrist = np.zeros((n, 2, 3))
    wrist[:, 1, 0] = np.where(np.arange(n) < 60, 0.4, 0.1)  # 40 cm apart, then 10 cm after a 1 s gap
    valid = np.ones((n, 2), bool)
    valid[45:75] = False

    class Track:
        fps, motion = 30.0, measured_motion(wrist, valid)

    assert together_cues(Track()) == []
    Track.motion = Motion(wrist, valid, 1.0, "m", 0.1, 0.2, 0.3)  # max_gap_s None: the old rule, bridge any gap
    assert together_cues(Track()) == [(75, "hands meet")]


def test_step_track_presence_comes_from_the_detections(gap_labels):
    track = gap_labels.step_track(RIG)
    assert (track.score > 0).all()  # seen by the left camera throughout
    np.testing.assert_allclose(track.kp[GAP], gap_labels.kp2d[GAP, 0, :, :, :2])
    assert np.isnan(track.world[GAP.start + 12: GAP.stop - 12]).all()  # no measured grip in the gap


# ---------- the stereo command end to end, and the QC numbers ----------

W, H = 640, 360  # per view; the calibration file is for 1920×1080 and is scaled to the video
NOISE = 0.3  # pixels at 640×360
# both away from the grasps and releases (a hand without 3D at its release has no release event, so no cut there)
WRONG = slice(150, 180)  # right hand: the right view's detection is wrong (epipolar rejection) for 1 s
HIDDEN = slice(230, 260)  # left hand: not seen by the right camera for 1 s


@pytest.fixture(scope="module")
def stereo_run(tmp_path_factory, truth):
    av = pytest.importorskip("av")
    pytest.importorskip("lerobot")
    from ego2lerobot import HandTrack
    from ego2lerobot.cli import main

    root = tmp_path_factory.mktemp("stereo")
    video, cache, out = root / "pair.mp4", root / "ds_cache", root / "ds"
    cache.mkdir()
    with av.open(str(video), "w") as c:  # one side-by-side file
        s = c.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = 2 * W, H, "yuv420p"
        for i in range(N):
            for p in s.encode(av.VideoFrame.from_ndarray(np.full((H, 2 * W, 3), i % 200, np.uint8), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    calib = {"image_width": 1920, "image_height": 1080, "model": "pinhole",
             "left": {**{k: getattr(RIG.left, k) for k in ("fx", "fy", "cx", "cy")}, "distortion": list(RIG.left.distortion)},
             "right": {**{k: getattr(RIG.right, k) for k in ("fx", "fy", "cx", "cy")}, "distortion": list(RIG.right.distortion)},
             "R": RIG.R.tolist(), "T": RIG.T.tolist()}
    (root / "stereo.json").write_text(json.dumps(calib))
    rig = RIG.scaled(W, H)
    views_ = detections(truth, rig, NOISE, seed=5, wrong_right_view=[(WRONG, (1,))], hide_right_view=[(HIDDEN, (0,))])
    for name, (kp, conf, score) in zip(("left", "right"), views_):  # cached detections: MediaPipe is not run
        HandTrack(kp.astype(np.float32), np.zeros((N, 2, 21, 3), np.float32), score.astype(np.float32), FPS, W,
                  H).save(cache / f"pair.{name}.hands.npz")
    main(["stereo", "--sbs", str(video), "--calib", str(root / "stereo.json"), "--out", str(out), "--cache", str(cache),
          "--width", "64"])
    return out, json.loads((out / "qc_report.json").read_text())


def test_stereo_run_exports_steps_grasp_events_and_run_log(stereo_run, truth):
    out, report = stereo_run
    info = json.loads((out / "meta" / "episodes_info.json").read_text())
    expected = [t for t, _ in true_events(truth, 1)]
    starts = [e["source"]["start_frame"] / FPS for e in info[1:]]
    assert len(info) == 5 and all(abs(a - b) <= CUT_TOLERANCE_S for a, b in zip(starts, expected)), starts
    assert all({"cuts", "grasp_events", "hand_coverage", "seconds"} <= set(e) for e in info)
    assert info[0]["source"]["video"] == "pair.mp4" and info[0]["cuts"]["start"] == "hands come into view"
    assert [g["event"] for g in info[1]["grasp_events"]] == ["grasp", "grasp"]  # both hands, at the step start
    grasp = json.loads((out / "meta" / "grasp_events.json").read_text())
    assert grasp["source"] == "stereo triangulation"
    for side in (0, 1):
        got = [(e["t"], e["event"]) for e in grasp["videos"]["pair.mp4"][("left", "right")[side]]]
        assert [k for _, k in got] == [k for _, k in true_events(truth, side)]
        assert all(abs(a - b) <= EVENT_TOLERANCE_S for (a, _), (b, _) in zip(got, true_events(truth, side)))
    run = json.loads((out / "meta" / "run.json").read_text())
    assert run["settings"]["stereo"] and run["settings"]["segment"] == "action" and run["settings"]["min_seconds"] == 2.0
    assert run["videos"][0]["track"] == "pair.stereo.npz" and len(run["videos"][0]["steps"]) == 5
    assert run["videos"][0]["path"] == "../pair.mp4" and str(out.parent) not in (out / "meta" / "run.json").read_text()
    assert report["overall"]["episodes"] == 5


def test_stereo_qc_numbers_match_what_was_injected(stereo_run, truth):
    out, report = stereo_run
    stats = json.loads((out / "meta" / "triangulation_stats.json").read_text())
    delivered = np.concatenate([e["per_frame"]["frame"] for e in stats["episodes"]])
    n = len(delivered)
    wrong, hidden = np.isin(delivered, np.arange(N)[WRONG]).sum(), np.isin(delivered, np.arange(N)[HIDDEN]).sum()
    s = report["stereo"]
    L, R, B = s["overall"]["left"], s["overall"]["right"], s["overall"]["both"]
    assert L["frames"] == R["frames"] == n and wrong > 10 and hidden > 10

    assert L["seen_both_views"] == round((n - hidden) / n, 3) and L["rejected"]["one view only"] == hidden
    assert R["seen_both_views"] == 1.0 and R["rejected"]["epipolar"] == wrong
    assert R["passed_checks"] == round((n - wrong) / n, 3) and R["rejected_share"] == round(wrong / n, 3)
    assert B["rejected"] == {"one view only": hidden, "epipolar": wrong, "depth": 0, "hand size": 0, "few joints": 0}
    assert B["passed_checks"] == round((2 * n - hidden - wrong) / (2 * n - hidden), 3)

    # pixel noise σ per coordinate: the row difference of two views is N(0, 2σ²), so the median over a hand's joints
    # of its absolute value is about 0.674·√2·σ = 0.95σ; the reprojection error is half of it in each view
    assert abs(B["epipolar_px"]["median"] - 0.954 * NOISE) < 0.2 * 0.954 * NOISE, B["epipolar_px"]
    assert abs(B["reprojection_px"]["median"] - 0.477 * NOISE) < 0.25 * 0.477 * NOISE, B["reprojection_px"]
    true_depth = truth[delivered][:, :, 0, 2]
    assert abs(B["wrist_depth_m"]["median"] - np.median(true_depth)) < 0.005
    assert B["wrist_depth_m"]["min"] >= true_depth.min() - 0.01 and B["wrist_depth_m"]["max"] <= true_depth.max() + 0.01

    cal = s["calibration"]
    assert cal["image_size"] == [W, H] and cal["calibrated_size"] == [1920, 1080] and not cal["rectified_input"]
    assert abs(cal["baseline_m"] - RIG.baseline) < 1e-4 and s["checks"]["max_epipolar_px"] == 3.6
    assert s["warnings"] == []
    md = (out / "qc_report.md").read_text()
    assert "## Stereo triangulation" in md and "Stereo warnings: none." in md and "3.6 px" in md
    assert "### Stereo per episode" in md and "| Hand in 3D (episode frames) |" in md


def test_stereo_captions_use_the_left_view_and_fixed_segments_still_work(stereo_run, monkeypatch):
    """--caption with a scripted backend (no model): boundary checks and captions see the left view (16:9), not the
    side-by-side frame (32:9); captions are verified and stored as in `run`. --segment fixed keeps equal splits."""
    from ego2lerobot import caption
    from ego2lerobot.cli import main
    from test_steps import FOLD, FOLD2, ScriptedBackend
    out, _ = stereo_run
    root = out.parent
    backend = ScriptedBackend(*[FOLD, FOLD2] * 30)
    monkeypatch.setattr(caption, "make_backend", lambda *a, **k: backend)
    common = ["stereo", "--sbs", str(root / "pair.mp4"), "--calib", str(root / "stereo.json"), "--cache",
              str(root / "ds_cache"), "--width", "64"]
    main(common + ["--out", str(root / "ds_caption"), "--caption"])
    info = json.loads((root / "ds_caption" / "meta" / "episodes_info.json").read_text())
    tasks = {"fold the towel", "crease the towel in half"}  # the scripted answers alternate between steps
    assert info and all(e["task"] in tasks and e["verified"] is True for e in info)
    assert {"object", "check", "cuts", "grasp_events"} <= set(info[0])
    kinds = {kind for kind, _, _ in backend.calls}
    assert kinds == {"boundary", "caption"}
    assert all(sizes[0] == (448, 252) for _, sizes, _ in backend.calls)  # 640×360 left view scaled to 448 wide
    run = json.loads((root / "ds_caption" / "meta" / "run.json").read_text())
    assert run["caption"] and all("before" in d for d in run["videos"][0]["boundaries"])

    main(common + ["--out", str(root / "ds_fixed"), "--segment", "fixed"])
    fixed = json.loads((root / "ds_fixed" / "meta" / "episodes_info.json").read_text())
    assert [(e["source"]["start_frame"], e["source"]["end_frame"]) for e in fixed] == [(0, N // 2), (N // 2, N)]  # 16 s > 15 s
    assert "cuts" not in fixed[0] and sum(len(e["grasp_events"]) for e in fixed) == 8


def test_measured_jitter_matches_the_injected_noise_and_skips_filled_frames():
    from ego2lerobot.qc import frame_checks, hand_summary
    from ego2lerobot.stereo_qc import measured_jitter
    rng = np.random.default_rng(4)
    n, sigma = 4000, 0.002
    kp3d = np.zeros((n, 2, 21, 3))
    kp3d[:, :, 0] = np.linspace(0, 1, n)[:, None, None] * [0.3, 0.1, 0.2] + [0, 0, 0.45]
    kp3d[:, :, 0] += rng.normal(scale=sigma, size=(n, 2, 3))
    status = np.ones((n, 2, 21))
    status[n // 2:, 0, 0] = 2  # second half of the left wrist: filled (a straight line, no noise)
    kp3d[n // 2:, 0, 0] = np.linspace(kp3d[n // 2, 0, 0], kp3d[-1, 0, 0], n - n // 2)
    valid = np.ones((n, 2))
    checks = frame_checks(np.zeros((n, 2, 21, 3)), kp3d, valid, 640, 360)
    assert hand_summary(checks, 0)["jitter_mm"] < 0.6 * 1.884 * sigma * 1000  # filled frames pull the median down
    measured_jitter(checks, kp3d, valid, status)
    # residual p[t] − (p[t−1] + p[t+1]) / 2 of white noise: 1.5σ² per axis, median of its norm 1.538·√1.5·σ
    for s in (0, 1):
        assert abs(hand_summary(checks, s)["jitter_mm"] - 1.884 * sigma * 1000) < 0.1 * 1.884 * sigma * 1000


def test_plausibility_warnings():
    from ego2lerobot.stereo_qc import plausibility
    cal = {"baseline_m": 0.063, "image_size": [1920, 1080], "calibrated_size": [1920, 1080],
           "checks": {"max_epipolar_px": 10.8}}
    good = {"both_views_frames": 100, "rejected_share": 0.1, "triangulated_of_seen": 0.8,
            "rejected": {"one view only": 12, "epipolar": 8, "depth": 2, "hand size": 0, "few joints": 0},
            "epipolar_px": {"median_all_matched": 1.0}, "wrist_depth_m": {"median": 0.45}}
    assert plausibility(cal, good) == []
    one_view = plausibility(cal, {**good, "triangulated_of_seen": 0.4})
    assert len(one_view) == 1 and "Only 40% of the hand frames" in one_view[0] and "12 seen by one camera only" in one_view[0]
    warns = plausibility({**cal, "baseline_m": 0.012, "calibrated_size": [1280, 800]},
                         {**good, "rejected_share": 0.6, "epipolar_px": {"median_all_matched": 6.0},
                          "wrist_depth_m": {"median": 0.12}})
    assert len(warns) == 5
    assert "under 2 cm" in warns[0] and "different shape" in warns[1] and "60% of the hand frames" in warns[2]
    assert "most often: epipolar" in warns[2] and "more than half the 10.8 px limit" in warns[3] and "0.12 m" in warns[4]
    assert "check that t is in metres" in plausibility({**cal, "baseline_m": 63.0}, good)[0].lower()
