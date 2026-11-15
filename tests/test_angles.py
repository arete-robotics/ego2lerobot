"""Finger joint angles from 21 joints."""
import numpy as np

from ego2lerobot.angles import ANGLE_NAMES, joint_angles


def hand(bend=0.0, spread=0.0):
    """A flat hand in the x-y plane, fingers along +y; every finger joint bent by `bend` towards -z."""
    kp = np.zeros((21, 3))
    for f, base in enumerate((1, 5, 9, 13, 17)):
        turn = (f - 2) * spread
        root = np.array([0.02 * (f - 2), 0.09, 0.0])
        kp[base] = root
        direction, angle = np.array([np.sin(turn), np.cos(turn), 0.0]), 0.0
        for k in range(1, 4):
            angle += bend
            step = 0.03 * (direction * np.cos(angle) + np.array([0, 0, -1.0]) * np.sin(angle))
            kp[base + k] = kp[base + k - 1] + step
    return kp


def test_straight_fingers_have_no_bend_and_bent_ones_report_the_angle():
    flat, bent = joint_angles(hand()), joint_angles(hand(bend=np.pi / 3))
    i = ANGLE_NAMES.index("index_middle")
    assert abs(flat[i]) < 1e-6 and abs(bent[i] - np.pi / 3) < 1e-6
    assert abs(bent[ANGLE_NAMES.index("ring_end")] - np.pi / 3) < 1e-6


def test_closure_goes_from_open_to_fist_and_spread_is_measured():
    assert joint_angles(hand())[-1] < 0.2
    assert joint_angles(hand(bend=np.pi / 2))[-1] > 0.95
    s = joint_angles(hand(spread=0.2))
    assert abs(s[ANGLE_NAMES.index("spread_index_middle")] - 0.2) < 1e-6


def test_missing_joints_give_nan_only_where_they_are_used():
    kp = np.stack([hand(), hand()])
    kp[1, 8] = np.nan                                  # index fingertip of the second hand
    out = joint_angles(kp)
    assert out.shape == (2, 20) and np.isfinite(out[0]).all()
    assert np.isnan(out[1, ANGLE_NAMES.index("index_end")]) and np.isfinite(out[1, ANGLE_NAMES.index("middle_end")])
