import json

import numpy as np
import pytest

from ego2lerobot.stereo import Checks, Lens, StereoRig, _rotmat, clean, label_stereo, match_frame, summarize

LEFT = Lens(1050, 1048, 962, 538, "pinhole", (-0.28, 0.09, 0.001, -0.0005, -0.01))
RIGHT = Lens(1046, 1045, 955, 541, "pinhole", (-0.27, 0.08, -0.0008, 0.0004, -0.012))
RIG = StereoRig(1920, 1080, LEFT, RIGHT, _rotmat(np.array([0.01, -0.02, 0.005])), (-0.063, 0.0012, -0.0008))
FISHEYE = Lens(1030.6, 1032.8, 966.7, 539.7, "fisheye", (-0.1166, -0.0236, 0.0694, -0.0463))

# A flat right hand in metres: wrist, thumb 1-4, then index, middle, ring and little finger (knuckle to tip).
HAND = np.array([[0, 0], [.025, -.02], [.045, -.04], [.06, -.06], [.07, -.08],
                 [.03, -.09], [.035, -.13], [.037, -.155], [.038, -.175],
                 [.01, -.095], [.011, -.14], [.012, -.167], [.013, -.19],
                 [-.01, -.09], [-.012, -.13], [-.013, -.155], [-.014, -.175],
                 [-.03, -.08], [-.034, -.11], [-.036, -.13], [-.038, -.145]])
HAND = np.column_stack([HAND, np.zeros(21)])


def hand_at(position, angles=(0.3, -0.2, 0.1)):
    return HAND @ _rotmat(np.array(angles)).T + position


def views(*hands, rig=RIG, noise=0.0, rng=None):
    """Per view (kp (2, 21, 2), conf (2, 21), score (2,)) with hands in slots 0 and 1 (None for no hand)."""
    out = []
    for v in (0, 1):
        kp, conf, score = np.zeros((2, 21, 2)), np.zeros((2, 21)), np.zeros(2)
        for s, X in enumerate(hands):
            if X is not None:
                kp[s] = rig.project(X, v) + (rng.normal(scale=noise, size=(21, 2)) if noise else 0)
                conf[s], score[s] = 0.9, 0.9
        out.append((kp, conf, score))
    return out


def test_lens_round_trip():
    xy = np.random.default_rng(0).uniform(-0.6, 0.6, (100, 2))
    for lens in (LEFT, FISHEYE):
        np.testing.assert_allclose(lens.rays(lens.pixels(xy)), xy, atol=1e-9)


def test_rectification_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    K = lambda lens: np.array([[lens.fx, 0, lens.cx], [0, lens.fy, lens.cy], [0, 0, 1.0]])  # noqa: E731
    R1, R2, P1, P2 = cv2.stereoRectify(K(LEFT), np.array([LEFT.distortion]), K(RIGHT), np.array([RIGHT.distortion]),
                                       (1920, 1080), RIG.R, RIG.T.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY)[:4]
    np.testing.assert_allclose(RIG.R1, R1, atol=1e-9)
    np.testing.assert_allclose(RIG.R2, R2, atol=1e-9)
    assert np.isclose(RIG.baseline, -P2[0, 3] / P2[0, 0])


@pytest.mark.parametrize("rig", [RIG, StereoRig(1920, 1080, FISHEYE, FISHEYE, np.eye(3), (-0.08, 0, 0))])
def test_triangulation_recovers_metric_points(rig):
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.uniform(-0.3, 0.3, (2000, 2)), rng.uniform(0.25, 0.8, 2000)])
    inside = np.all([((uv >= 0) & (uv < (1920, 1080))).all(1) for uv in (rig.project(X, 0), rig.project(X, 1))], 0)
    X = X[inside][:500]  # points both cameras see
    exact, epi, rep = rig.triangulate(rig.project(X, 0), rig.project(X, 1))
    np.testing.assert_allclose(exact, X, atol=1e-9)
    assert epi.max() < 1e-6 and rep.max() < 1e-6
    noisy, *_ = rig.triangulate(rig.project(X, 0) + rng.normal(size=(500, 2)), rig.project(X, 1) + rng.normal(size=(500, 2)))
    err = np.linalg.norm(noisy - X, axis=-1)
    # 1 px of noise at f ≈ 1000 px and a 6-8 cm baseline: a few millimetres at arm's length
    assert np.median(err) < 0.006 and np.percentile(err, 90) < 0.02


def test_hands_are_matched_across_views_even_when_handedness_differs():
    left_hand, right_hand = hand_at([-0.12, 0.15, 0.45]), hand_at([0.12, 0.12, 0.5], (0.2, 0.3, -0.1))
    lv, rv = views(left_hand, right_hand)
    rv = (rv[0][::-1], rv[1][::-1], rv[2][::-1])  # the right view calls the hands the other way round
    X, epi, rep, reason, order_l, order_r = match_frame(RIG, lv, rv, Checks())
    assert reason.tolist() == [0, 0] and order_l == [0, 1] and order_r == [1, 0]
    np.testing.assert_allclose(X[0], left_hand, atol=1e-6)
    np.testing.assert_allclose(X[1], right_hand, atol=1e-6)


def test_bad_matches_are_rejected():
    hand = hand_at([0.05, 0.12, 0.45])
    other = hand_at([0.0, 0.25, 0.45])  # a different hand lower in the image
    lv, _ = views(hand, None)
    _, rv = views(other, None)
    X, epi, rep, reason, *_ = match_frame(RIG, lv, rv, Checks())
    assert reason[0] == 2 and np.isnan(X).all() and epi[0] > 50  # rows disagree: wrong match

    lv, rv = views(hand, None)
    rv[0][0][...] = lv[0][0] + (100, 0)  # negative disparity: the rays cross behind the cameras
    assert match_frame(RIG, lv, rv, Checks())[3][0] == 3

    lv, rv = views(hand, None)
    rv[0][0][8] += (0, 25)  # one fingertip detected 25 px too low in the right view
    X, epi, rep, reason, *_ = match_frame(RIG, lv, rv, Checks())
    assert reason[0] == 0 and np.isnan(X[0, 8]).all()
    known = np.arange(21) != 8
    np.testing.assert_allclose(X[0, known], hand[known], atol=1e-6)

    lv, rv = views(hand * 3 - hand[0] * 2, None)  # a "hand" three times too big: implausible
    assert match_frame(RIG, lv, rv, Checks())[3][0] == 4


def test_clean_fills_short_gaps_smooths_and_drops_spikes():
    fps, n = 30.0, 120
    t = np.arange(n) / fps
    truth = np.zeros((n, 2, 21, 3))
    truth[:, 0] = HAND + np.stack([0.1 * np.sin(t), 0.05 * t, 0.4 + 0.05 * np.cos(t)], -1)[:, None]
    rng = np.random.default_rng(2)
    X = truth + rng.normal(scale=0.004, size=truth.shape)
    X[:, 1] = np.nan
    X[40:45, 0] = np.nan  # short gap: filled
    X[80:100, 0] = np.nan  # long gap: left empty
    X[60, 0, 8] += (0, 0, 0.2)  # depth spike
    out, status = clean(X, fps, bone_weight=0.0)
    assert (status[40:45, 0] == 2).all() and (status[80:100, 0] == 0).all() and (status[:, 1] == 0).all()
    assert status[60, 0, 8] == 2  # the spike was dropped, then filled from its neighbours
    keep = np.isfinite(out[:, 0, :, 0])
    raw_err = np.linalg.norm(X - truth, axis=-1)[:, 0][np.isfinite(X[:, 0, :, 0])]
    err = np.linalg.norm(out - truth, axis=-1)[:, 0][keep]
    assert np.median(err) < 0.6 * np.median(raw_err) and err.max() < 0.02


def test_label_stereo_sequence_and_stats():
    fps, n = 30.0, 90
    rng = np.random.default_rng(3)
    t = np.arange(n) / fps
    path = np.stack([0.1 * np.sin(2 * t), 0.15 + 0.03 * t, 0.45 + 0.05 * np.cos(t)], -1)
    truth = np.stack([hand_at(p) for p in path])
    per_frame = [views(truth[i], None, noise=0.8, rng=rng) for i in range(n)]
    for i in (20, 21, 22):  # the right view grabs something else for three frames
        per_frame[i][1][0][0] += (0, 80)
    stacked = [tuple(np.stack([f[v][k] for f in per_frame]) for k in range(3)) for v in (0, 1)]
    lab = label_stereo(RIG, stacked, fps)
    assert lab.valid[:, 0].all() and not lab.valid[:, 1].any()
    assert (lab.reason[20:23, 0] == 2).all() and (lab.status[20:23, 0] == 2).all()  # rejected, then filled
    err = np.linalg.norm(lab.kp3d[:, 0] - truth, axis=-1)
    assert np.median(err) < 0.006
    stats = summarize([(lab, np.arange(n), RIG)])
    assert stats["hand_frames_both_views"] == n and stats["hand_frames_triangulated"] == n - 3
    assert stats["valid_triangulation_rate"] == round((n - 3) / n, 3) and stats["rejected"]["epipolar"] == 3
    assert stats["reprojection_px_median"] < 1.5 and 0.35 < stats["wrist_depth_m"]["median"] < 0.55
    json.dumps(stats)


def test_calibration_json(tmp_path):
    full = {"image_width": 1920, "image_height": 1080, "model": "pinhole",
            "left": {"fx": 1050, "fy": 1048, "cx": 962, "cy": 538, "distortion": list(LEFT.distortion)},
            "right": {"fx": 1046, "fy": 1045, "cx": 955, "cy": 541, "distortion": list(RIGHT.distortion)},
            "R": RIG.R.tolist(), "T": RIG.T.tolist()}
    (tmp_path / "full.json").write_text(json.dumps(full))
    rig = StereoRig.from_json(tmp_path / "full.json", 960, 540)
    assert rig.left.fx == 525 and rig.right.cy == 270.5 and np.isclose(rig.baseline, RIG.baseline)
    (tmp_path / "rect.json").write_text(json.dumps({"image_width": 1280, "image_height": 720, "fx": 700, "fy": 700,
                                                    "cx": 640, "cy": 360, "baseline": 0.12}))
    rect = StereoRig.from_json(tmp_path / "rect.json")
    np.testing.assert_allclose(rect.R1, np.eye(3)) and np.testing.assert_allclose(rect.R2, np.eye(3))
    X, *_ = rect.triangulate(np.array([[740.0, 360.0]]), np.array([[600.0, 360.0]]))
    np.testing.assert_allclose(X, [[100 * 0.6 / 700, 0, 700 * 0.12 / 140]])
    full["T"] = [0.063, 0, 0]
    (tmp_path / "swapped.json").write_text(json.dumps(full))
    with pytest.raises(ValueError):
        StereoRig.from_json(tmp_path / "swapped.json")


def test_export_writes_both_views_calibration_stats_and_preview(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("lerobot")
    from ego2lerobot.segments import Segment
    from ego2lerobot.stereo_export import Imu, StereoEpisode, write_stereo_dataset
    from ego2lerobot.stereo_video import StereoVideo

    fps, n, w, h = 30, 24, 192, 128
    lens = Lens(150, 150, 96, 64)
    rig = StereoRig(w, h, lens, lens, np.eye(3), (-0.06, 0, 0))
    with av.open(str(tmp_path / "pair.mp4"), "w") as out:  # one side-by-side file: left camera in the left half
        stream = out.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 2 * w, h, "yuv420p"
        for i in range(n):
            image = np.full((h, 2 * w, 3), 40 + 4 * i, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    truth = np.stack([hand_at([0.02 * np.sin(i / 5), 0.1, 0.45]) for i in range(n)])
    frames = [views(truth[i], None, rig=rig) for i in range(n)]
    lab = label_stereo(rig, [tuple(np.stack([f[v][k] for f in frames]) for k in range(3)) for v in (0, 1)], fps)
    t = np.arange(0, 1, 0.01)
    np.savetxt(tmp_path / "imu.csv", np.column_stack([t, np.ones_like(t)]), delimiter=",", header="t,gz", comments="")
    episode = StereoEpisode(StereoVideo(tmp_path / "pair.mp4"), rig, lab, Segment(0, n, 1.0, 0.0), "test", Imu(tmp_path / "imu.csv"))

    summary = write_stereo_dataset([episode], tmp_path / "ds", "local/test", width=96, step=2, preview_dir=tmp_path / "prev")
    assert summary["frames"] == n // 2 and summary["valid_triangulation_rate"] == 1.0
    meta = tmp_path / "ds" / "meta"
    stats = json.loads((meta / "triangulation_stats.json").read_text())
    assert stats["summary"]["hand_frames_triangulated"] == n // 2 and len(stats["episodes"][0]["per_frame"]["frame"]) == n // 2
    calib = json.loads((meta / "calibration.json").read_text())
    assert calib["stored"]["left"]["fx"] == 75 and np.isclose(calib["stored"]["baseline_m"], 0.06)
    info = json.loads((meta / "info.json").read_text())["features"]
    assert {"observation.images.left", "observation.images.right", "observation.extrinsics", "observation.imu"} <= set(info)
    assert (meta / "imu" / "episode_000000.csv").exists() and (tmp_path / "prev" / "episode_000000.mp4").exists()


def test_imu_means_per_frame(tmp_path):
    from ego2lerobot.stereo_export import Imu
    t = np.arange(0, 1, 0.005)
    np.savetxt(tmp_path / "imu.csv", np.column_stack([t, t, -t]), delimiter=",", header="t,gx,gy", comments="")
    imu = Imu(tmp_path / "imu.csv")
    assert imu.names == ["gx", "gy"]
    np.testing.assert_allclose(imu.mean(0.1, 0.2), [0.1475, -0.1475], atol=1e-9)
    assert (imu.mean(5, 6) == 0).all()
