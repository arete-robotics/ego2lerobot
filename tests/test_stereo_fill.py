import numpy as np

from ego2lerobot.stereo import Lens, StereoRig, _rotmat, label_stereo
from ego2lerobot.stereo_fill import PLACED, FillSettings, fill_hands, fit_rigid, smooth_placed

W, H = 1280, 720


def rig():
    lens = Lens(700.0, 700.0, W / 2, H / 2)
    return StereoRig(W, H, lens, lens, np.eye(3), (-0.1, 0.0, 0.0))


def hand(seed=0):
    """A hand-sized cloud of 21 points about 0.5 m in front of the left camera."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.07, 0.07, (21, 3)) * [1.0, 1.0, 0.3] + [0.02, 0.01, 0.5]


def moved(P, k):
    c = P.mean(0)
    return (P - c) @ _rotmat(np.array([0.0, 0.0, 0.03 * k])).T + c + np.array([0.004, 0.002, 0.003]) * k


def views_for(hands, hide=()):
    """Per camera (kp (N, 2, 21, 2), conf (N, 2, 21), score (N, 2)) for the left hand at `hands` (N, 21, 3).
    `hide`: (frame, view) pairs where the detector found nothing."""
    r, n = rig(), len(hands)
    out = []
    for view in range(2):
        kp, conf, score = np.zeros((n, 2, 21, 2)), np.zeros((n, 2, 21)), np.zeros((n, 2))
        for i in range(n):
            if (i, view) not in hide:
                kp[i, 0], conf[i, 0], score[i, 0] = r.project(hands[i], view), 1.0, 1.0
        out.append((kp, conf, score))
    return out


def test_fit_rigid_moves_the_shape_onto_the_keypoints_of_one_view():
    r, P0 = rig(), hand()
    target = moved(P0, 3)
    obs = [(0, r.project(target, 0), np.ones(21, bool))]
    P, agree, used, error = fit_rigid(r, P0, obs, FillSettings())

    assert agree == 21 and used == 21 and error < 0.5
    assert np.abs(P - target).max() < 0.004  # one view: the known size of the hand fixes its distance


def test_fit_rigid_rejects_keypoints_that_do_not_fit_the_shape():
    r, P0 = rig(), hand()
    other = hand(seed=5) + [0.0, 0.0, 0.02]
    assert fit_rigid(r, P0, [(0, r.project(other, 0), np.ones(21, bool))], FillSettings()) == "keypoints disagree"
    far = [(0, r.project(P0 + [0.3, 0.0, 0.0], 0), np.ones(21, bool))]
    assert fit_rigid(r, P0, far, FillSettings()) == "moved too far"


def test_fill_hands_places_a_hand_seen_in_one_view_and_leaves_measured_frames_alone():
    r, n = rig(), 12
    hands = np.array([moved(hand(), k) for k in range(n)])
    X = np.full((n, 2, 21, 3), np.nan)
    valid = np.zeros((n, 2), bool)
    measured = [0, 1, 2, 3, 8, 9, 10, 11]
    X[measured, 0], valid[measured, 0] = hands[measured], True
    views = views_for(hands, hide={(4, 1), (5, 1), (6, 1), (7, 1)})  # frames 4 to 7: the right view lost the hand
    counts = {}
    out, placed = fill_hands(r, X, valid, views, fps=30.0, counts=counts)

    assert placed[:, 0].tolist() == [i not in measured for i in range(n)] and not placed[:, 1].any()
    assert np.array_equal(out[measured], X[measured], equal_nan=True)
    assert np.abs(out[4:8, 0] - hands[4:8]).max() < 0.006
    assert counts["placed"] >= 4

    smooth = smooth_placed(out, placed, fps=30.0)
    assert np.array_equal(smooth[measured, 0], X[measured, 0]) and np.abs(smooth[4:8, 0] - hands[4:8]).max() < 0.008


def test_fill_hands_does_not_reach_beyond_its_limit_or_without_keypoints():
    r, n = rig(), 12
    hands = np.array([moved(hand(), k) for k in range(n)])
    X = np.full((n, 2, 21, 3), np.nan)
    valid = np.zeros((n, 2), bool)
    X[[0, 1], 0], valid[[0, 1], 0] = hands[[0, 1]], True
    views = views_for(hands, hide={(i, v) for i in range(8, 12) for v in range(2)})  # nothing detected from frame 8
    _, placed = fill_hands(r, X, valid, views, fps=30.0, settings=FillSettings(reach=0.1))  # three frames

    assert placed[:, 0].tolist() == [False, False, True, True, True] + [False] * 7
    _, placed = fill_hands(r, X, valid, views, fps=30.0)
    assert placed[2:8, 0].all() and not placed[8:, 0].any()


def test_label_stereo_marks_placed_joints_with_status_3():
    r, n = rig(), 40
    hands = np.array([moved(hand(), k * 0.3) for k in range(n)])
    hide = {(i, 1) for i in range(15, 30)}
    labels = label_stereo(r, views_for(hands, hide), fps=30.0)
    plain = label_stereo(r, views_for(hands, hide), fps=30.0, fill=False)

    assert not plain.valid[17:28, 0].any() and labels.valid[17:28, 0].all()
    assert (labels.status[17:28, 0] == PLACED).all() and (plain.status[17:28, 0] == 0).all()
    assert (labels.status[:10, 0] == 1).all() and np.isfinite(labels.kp3d[17:28, 0]).all()
    assert np.abs(labels.kp3d[17:28, 0] - hands[17:28]).max() < 0.01
