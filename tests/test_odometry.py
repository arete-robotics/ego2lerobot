import numpy as np
import pytest

from ego2lerobot.odometry import (LOST, SOLVED, HeadPose, Settings, accel_columns, compare, fit_distortion, hand_masks,
                                  level, track_frames, with_radial)
from ego2lerobot.stereo import Lens, StereoRig, _rotmat

W, H, F, BASELINE, DEPTH = 640, 480, 400.0, 0.1, 2.0


def rig():
    lens = Lens(F, F, W / 2, H / 2)
    return StereoRig(W, H, lens, lens, np.eye(3), (-BASELINE, 0.0, 0.0))


def view(texture, R, t, scale=400.0):
    """A camera at position t with rotation R (camera → world) looking at the textured plane z = DEPTH."""
    import cv2
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    plane_to_image = K @ np.column_stack([R.T[:, 0], R.T[:, 1], DEPTH * R.T[:, 2] - R.T @ t])
    texture_to_plane = np.array([[1 / scale, 0, -texture.shape[1] / scale / 2], [0, 1 / scale, -texture.shape[0] / scale / 2],
                                 [0, 0, 1]])
    return cv2.warpPerspective(texture, plane_to_image @ texture_to_plane, (W, H))


def scene_frames(poses, texture):
    for i, (R, t) in enumerate(poses):
        left = view(texture, R, t)
        right = view(texture, R, t + R @ np.array([BASELINE, 0.0, 0.0]))
        yield i, np.dstack([left] * 3), np.dstack([right] * 3)


def textured_plane():
    import cv2
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur((rng.random((1800, 2400)) * 255).astype(np.uint8), (0, 0), 2.0)


def moving_poses(n=24):
    return [(_rotmat(np.array([0.0, np.radians(0.25) * k, np.radians(0.1) * k])), np.array([0.008, 0.002, 0.004]) * k)
            for k in range(n)]


def test_track_frames_recovers_a_known_motion():
    pytest.importorskip("cv2")
    poses = moving_poses()
    pose = track_frames(scene_frames(poses, textured_plane()), rig())

    assert (pose.status == SOLVED).all() and pose.part.max() == 0
    R_true, t_true = poses[-1]
    assert np.linalg.norm(pose.t[-1] - t_true) < 0.006
    angle = np.degrees(np.arccos(np.clip((np.trace(pose.R[-1].T @ R_true) - 1) / 2, -1, 1)))
    assert angle < 0.25
    assert pose.summary()["tracked"] == 1.0 and pose.summary()["keyframes"] >= 1
    same = compare(pose, np.array([R for R, _ in poses]), np.array([t for _, t in poses]), seconds=(0.5,))
    assert same["0.5s"]["point_ahead_mm_median"] < 3 and same["0.5s"]["rotation_deg_median"] < 0.2


def blank_stretch(poses):
    frames = list(scene_frames(poses, textured_plane()))
    blank = np.full((H, W, 3), 128, np.uint8)
    for i in (5, 6):
        frames[i] = (i, blank, blank)
    return iter(frames)


def test_a_blank_stretch_is_marked_lost_and_starts_a_new_part():
    pytest.importorskip("cv2")
    pose = track_frames(blank_stretch(moving_poses(12)), rig(), settings=Settings(revisit=False))

    assert pose.status[4] == SOLVED and (pose.status[5:7] == LOST).all() and pose.status[7] == SOLVED
    assert pose.part[4] == 0 and pose.part[7] == 1
    assert np.allclose(pose.t[6], pose.t[4]) and np.allclose(pose.t[7], pose.t[4])  # held across the break
    assert pose.summary()["breaks"] == 1


def test_after_a_blank_stretch_the_pose_is_found_again_from_an_old_keyframe():
    pytest.importorskip("cv2")
    poses = moving_poses(12)
    pose = track_frames(blank_stretch(poses), rig())

    assert (pose.status[5:7] == LOST).all() and pose.status[7] == SOLVED
    assert pose.part.max() == 0 and pose.revisits >= 1  # no break: the frame is placed in the same world
    assert np.linalg.norm(pose.t[7] - poses[7][1]) < 0.006 and np.linalg.norm(pose.t[-1] - poses[-1][1]) < 0.006


def test_fit_distortion_finds_none_in_undistorted_views():
    pytest.importorskip("cv2")
    poses = moving_poses(14)
    texture = textured_plane()
    fit = fit_distortion(lambda: scene_frames(poses, texture), rig(), candidates=(-0.06, -0.03, 0.0, 0.03, 0.06))

    assert abs(fit["k1"]) < 0.015 and not fit["at_edge_of_search"] and len(fit["curve"]) == 5
    assert with_radial(rig(), -0.04).left.distortion[0] == -0.04


def test_hand_masks_cover_the_hand_and_the_forearm_below_it():
    kp = np.zeros((3, 2, 21, 3))
    kp[1, 0, :, 0] = np.linspace(300, 340, 21)
    kp[1, 0, :, 1] = np.linspace(200, 240, 21)
    kp[1, 0, :, 2] = 1.0
    mask_for = hand_masks(kp, 640, 480)
    mask = mask_for(0, 320, 240)  # the detection of frame 1 also masks frame 0, at half the size

    assert mask[110, 160] == 0 and mask[235, 160] == 0  # the hand, and straight below it to the bottom edge
    assert mask[50, 160] == 255 and mask[110, 20] == 255
    assert hand_masks(kp, 640, 480, spread=0)(0, 320, 240).min() == 255


def test_level_turns_the_world_so_that_y_is_gravity():
    fps, n = 30.0, 90
    start = _rotmat(np.array([np.radians(-60), 0.0, 0.0]))  # looking steeply down at the first frame
    up_world = np.array([0.0, -1.0, 0.0])
    R_true = np.array([_rotmat(np.array([0.0, np.radians(0.5) * k, 0.0])) @ start for k in range(n)])
    first = R_true[0]
    pose = HeadPose(fps, np.einsum("ij,njk->nik", first.T, R_true), np.zeros((n, 3)), np.ones(n, np.int8),
                    np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n))
    t = np.arange(n * 4) / (fps * 4)
    accel = np.array([9.81 * R_true[min(n - 1, int(round(s * fps)))].T @ up_world for s in t])

    levelled, check = level(pose, t, accel)

    assert check["tilt_deg_median"] < 0.05 and abs(check["first_camera_pitch_deg"] + 60) < 0.5
    for i in (0, 40, 89):
        up_now = levelled.R[i] @ (R_true[i].T @ up_world)
        assert np.allclose(up_now, [0, -1, 0], atol=1e-3)
    assert abs(levelled.R[0][0, 2]) < 1e-6  # z of the world is the first viewing direction, levelled
    assert "gravity" in levelled.world


def test_level_needs_accelerometer_samples_near_gravity():
    pose = HeadPose(30.0, np.tile(np.eye(3), (60, 1, 1)), np.zeros((60, 3)), np.ones(60, np.int8), np.zeros(60, np.int32),
                    np.zeros(60, np.int32), np.zeros(60, np.int32), np.zeros(60))
    with pytest.raises(ValueError):
        level(pose, np.arange(240) / 120, np.zeros((240, 3)))


def test_compare_reports_scale_and_ignores_the_world_frame():
    n, fps = 120, 30.0
    R = np.array([_rotmat(np.array([0.0, 0.004 * k, 0.0])) for k in range(n)])
    t = np.array([[0.004 * k, 0.0, 0.002 * k] for k in range(n)])
    pose = HeadPose(fps, R, t * 1.1, np.ones(n, np.int8), np.zeros(n, np.int32), np.zeros(n, np.int32),
                    np.zeros(n, np.int32), np.zeros(n))
    turn = _rotmat(np.array([0.3, -0.2, 0.1]))  # the reference lives in another world frame
    out = compare(pose, np.einsum("ij,njk->nik", turn, R), t @ turn.T + 5.0, seconds=(1.0,))

    assert out["1s"]["rotation_deg_median"] < 1e-4
    assert abs(out["1s"]["scale"] - 1.1) < 0.01


def test_head_pose_round_trip_and_vector(tmp_path):
    n = 5
    pose = HeadPose(30.0, np.tile(_rotmat(np.array([0.0, 0.1, 0.0])), (n, 1, 1)), np.arange(n * 3.0).reshape(n, 3),
                    np.array([1, 1, 0, 1, 1], np.int8), np.array([0, 0, 0, 1, 1], np.int32), np.zeros(n, np.int32),
                    np.full(n, 50, np.int32), np.full(n, 0.5), settings={"width": 960})
    pose.save(tmp_path / "head.npz")
    back = HeadPose.load(tmp_path / "head.npz")

    assert np.array_equal(back.status, pose.status) and back.settings == {"width": 960} and back.world == pose.world
    assert np.allclose(back.vector(1), [3, 4, 5, *pose.R[1][:, 0], *pose.R[1][:, 1]])
    assert np.allclose(back.to_world(np.array([0.0, 0.0, 1.0]), 1), pose.R[1][:, 2] + pose.t[1])


def test_accel_columns():
    assert accel_columns(["gx", "gy", "gz", "ax", "ay", "az"]) == [3, 4, 5]
    assert accel_columns(["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]) == [0, 1, 2]
    assert accel_columns(["gx", "gy", "gz"]) is None


def test_export_puts_the_hands_in_the_world_frame():
    from ego2lerobot.stereo_export import features, head_labels

    class Labels:
        valid = np.array([[1, 0], [1, 1]], bool)
        kp3d = np.zeros((2, 2, 21, 3))
    rng = np.random.default_rng(1)
    Labels.kp3d[:] = rng.normal(0.0, 0.05, (2, 2, 21, 3)) + [0.0, 0.0, 0.5]
    R = _rotmat(np.array([0.0, np.pi / 2, 0.0]))
    pose = HeadPose(30.0, np.array([R, R]), np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), np.array([1, 0], np.int8),
                    np.zeros(2, np.int32), np.zeros(2, np.int32), np.zeros(2, np.int32), np.zeros(2))

    row = head_labels(pose, Labels, 0)
    world = row["observation.hands.kp3d_world"].reshape(2, 21, 3)
    assert np.allclose(world[0], Labels.kp3d[0, 0] @ R.T + [1, 2, 3], atol=1e-5)
    assert not world[1].any()  # a hand that is not in 3D stays zero
    assert row["observation.head.valid"][0] == 1 and np.allclose(row["observation.head.pose"][:3], [1, 2, 3])
    assert np.allclose(row["observation.hands.ee_world"].reshape(2, 10)[0, :3], world[0, 0], atol=1e-6)

    held = head_labels(pose, Labels, 1)  # the pose of frame 1 is held: no world-frame hands
    assert held["observation.head.valid"][0] == 0 and not held["observation.hands.kp3d_world"].any()
    assert np.allclose(held["observation.head.pose"][:3], [1, 2, 3])

    f = features(640, 400, head=True)
    assert f["observation.head.pose"]["shape"] == (9,) and f["observation.hands.kp3d_world"]["shape"] == (126,)
    assert "observation.head.pose" not in features(640, 400)
