import math

import numpy as np
import pytest

from ego2lerobot import Camera
from ego2lerobot.camera import PRESETS
from ego2lerobot.rectify import PinholeView


def test_fisheye_fov_is_equidistant_and_reaches_the_frame_edge():
    cam = Camera.from_fisheye_fov(3840, 2160, 124.0)
    theta = math.radians(62.0)
    edge = cam.project(np.array([[math.tan(theta), 0.0, 1.0]]))
    assert np.allclose(edge, [[3840, 1080]])
    half = cam.project(np.array([[math.tan(theta / 2), 0.0, 1.0]]))
    assert np.allclose(half[0, 0] - 1920, 960)  # half the angle → half the radius


def test_action_preset_is_nominal_fisheye_sized_to_the_video():
    assert PRESETS["action-ultrawide"] == ("fisheye", 124.0)
    a, b = Camera.preset("action-ultrawide", 3840, 2160), Camera.preset("action-ultrawide", 1920, 1080)
    assert a.model == "fisheye" and np.isclose(a.fx, 2 * b.fx)
    with pytest.raises(ValueError):
        Camera.preset("nope", 640, 480)


def test_pinhole_view_maps_keypoints_back_to_the_fisheye_frame():
    cam = Camera(1920, 1080, 1030.6, 1032.8, 966.7, 539.7, "fisheye", (-0.1166, -0.0236, 0.0694, -0.0463))
    view = PinholeView(cam, hfov=100, width=960)
    assert view.size == (960, 540)
    # a point in the pinhole view lands where the remap table samples it, in full-resolution pixels
    uv = np.array([[100.0, 400.0], [480.0, 270.0], [900.0, 50.0]])
    src = view.to_source(uv)
    sampled = np.column_stack([view.map_x[uv[:, 1].astype(int), uv[:, 0].astype(int)],
                               view.map_y[uv[:, 1].astype(int), uv[:, 0].astype(int)]]) * 2  # 960 → 1920
    assert np.allclose(src, sampled, atol=1.5)
    # the rays agree: undistorting the mapped-back pixels gives the pinhole view's rays
    assert np.allclose(cam.undistort(src), view.pinhole.undistort(uv), atol=1e-6)


def test_pinhole_view_moves_a_bright_spot_consistently():
    cam = Camera.from_fisheye_fov(960, 540, 124.0)
    view = PinholeView(cam, hfov=100, width=960)
    image = np.zeros((540, 960, 3), np.uint8)
    target = np.array([[700.0, 400.0]])  # off-centre, where the fisheye bends things most
    rays = cam.undistort(target)
    spot_view = view.pinhole.project(np.column_stack([rays, [1.0]]))[0]
    u, v = np.round(target[0]).astype(int)
    image[v - 3:v + 4, u - 3:u + 4] = 255
    out = view.image(image)
    ys, xs = np.nonzero(out[..., 0] > 128)
    assert np.allclose([xs.mean(), ys.mean()], spot_view, atol=1.5)
    assert np.allclose(view.to_source(np.array([[xs.mean(), ys.mean()]])), target, atol=2.0)


def test_rectify_needs_a_fisheye_camera():
    with pytest.raises(ValueError):
        PinholeView(Camera.from_fov(1920, 1080, 100), 100)
