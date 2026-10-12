import numpy as np

from ego2lerobot import Camera, HandTrack, end_effector, find_segments, place_hand

FISHEYE = Camera(1920, 1080, 1030.6, 1032.8, 966.7, 539.7, "fisheye", (-0.1166, -0.0236, 0.0694, -0.0463))


def hand_shape():
    rng = np.random.default_rng(0)
    return rng.normal(scale=0.04, size=(21, 3))


def test_fisheye_undistort_inverts_project():
    pts = np.array([[0.1, -0.05, 0.4], [-0.3, 0.2, 0.5], [0.6, 0.3, 0.45]])
    uv = FISHEYE.project(pts)
    rays = FISHEYE.undistort(uv)
    np.testing.assert_allclose(rays, pts[:, :2] / pts[:, 2:], atol=1e-6)


def test_place_hand_recovers_translation():
    world = hand_shape()
    t = np.array([0.08, 0.12, 0.42])
    uv = FISHEYE.project(world + t)
    placed = place_hand(world, FISHEYE.undistort(uv))
    np.testing.assert_allclose(placed, world + t, atol=1e-6)


def test_place_hand_rejects_points_behind_camera():
    world = hand_shape()
    rays = np.zeros((21, 2))
    rays[:, 0] = world[:, 0] / (world[:, 2] - 0.3)
    rays[:, 1] = world[:, 1] / (world[:, 2] - 0.3)
    assert place_hand(world, rays) is None


def test_end_effector_palm_frame_is_orthonormal():
    p = hand_shape() + [0, 0, 0.4]
    ee = end_effector(p)
    x, y = ee[3:6], ee[6:9]
    assert abs(np.linalg.norm(x) - 1) < 1e-9 and abs(np.linalg.norm(y) - 1) < 1e-9 and abs(x @ y) < 1e-9
    assert np.isclose(ee[9], np.linalg.norm(p[4] - p[8]))


def test_segments_bridge_short_gaps_and_split_long_runs():
    fps = 30.0
    score = np.zeros((40 * 30, 2), np.float32)
    score[0:300, 0] = 0.9            # 10 s
    score[310:600, 0] = 0.9          # gap of 10 frames is bridged → one 20 s run
    score[900:950, 1] = 0.9          # 1.7 s: too short
    n = len(score)
    track = HandTrack(np.zeros((n, 2, 21, 2)), np.zeros((n, 2, 21, 3)), score, fps, 1920, 1080)
    segs = find_segments(track, min_seconds=3, max_seconds=15)
    assert [(s.start, s.end) for s in segs] == [(0, 300), (300, 600)]
