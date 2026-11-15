"""Footage checks: measured facts against spec claims, and footage that appears twice."""
from fractions import Fraction

import av
import numpy as np

from ego2lerobot.verify import check_claims, compare, imu_facts, probe, signature, stereo_pairs, verify


def scene(n, seed, size=(160, 120)):
    """A bright square on a random walk over a fixed textured background: n RGB frames."""
    rng = np.random.default_rng(seed)
    w, h = size
    back = rng.integers(0, 90, (h, w, 1), dtype=np.uint8).repeat(3, 2)
    pos = np.cumsum(rng.normal(0, 6, (n, 2)), 0)
    frames = []
    for x, y in pos:
        img = back.copy()
        cx, cy = int(w / 2 + x) % (w - 30), int(h / 2 + y) % (h - 30)
        img[cy:cy + 30, cx:cx + 30] = 250
        frames.append(img)
    return frames


def write(path, frames, rate=30, drop=(), audio=False):
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=rate)
        s.height, s.width = frames[0].shape[:2]
        s.pix_fmt = "yuv420p"
        s.bit_rate = 2_000_000
        s.codec_context.time_base = Fraction(1, rate)
        if audio:
            a = c.add_stream("aac", rate=16000)
        for i, img in enumerate(frames):
            if i in drop:
                continue
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts, f.time_base = i, Fraction(1, rate)
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        if audio:
            f = av.AudioFrame.from_ndarray(np.zeros((1, 16000), np.float32), format="fltp", layout="mono")
            f.sample_rate = 16000
            for p in a.encode(f):
                c.mux(p)
            for p in a.encode():
                c.mux(p)


def test_probe_reports_size_rate_and_missing_frames(tmp_path):
    frames = scene(120, 0)
    write(tmp_path / "ok.mp4", frames)
    write(tmp_path / "gaps.mp4", frames, drop={40, 41, 90})
    ok, gaps = probe(tmp_path / "ok.mp4"), probe(tmp_path / "gaps.mp4")
    assert (ok["width"], ok["height"], ok["frames"]) == (160, 120, 120)
    assert abs(ok["fps"] - 30) < 0.1 and ok["intervals_out_of_spec"] == 0 and not ok["audio"]
    assert gaps["frames_missing"] == 3 and gaps["intervals_out_of_spec"] == 2


def test_empty_and_broken_files_are_reported_not_raised(tmp_path):
    (tmp_path / "empty.mp4").write_bytes(b"")
    (tmp_path / "junk.mp4").write_bytes(b"not a video" * 50)
    assert probe(tmp_path / "empty.mp4")["error"] == "empty file"
    assert "error" in probe(tmp_path / "junk.mp4")


def test_same_recording_found_after_trim_and_reencode(tmp_path):
    frames = scene(900, 1)
    write(tmp_path / "a.mp4", frames)
    write(tmp_path / "b.mp4", frames[60:], rate=30)          # starts 2 s later
    c = compare(signature(tmp_path / "a.mp4"), signature(tmp_path / "b.mp4"))
    assert c["verdict"] == "same recording" and abs(c["offset_s"] - 2.0) <= 0.4


def test_second_camera_on_the_same_moment_is_told_apart_from_a_copy(tmp_path):
    frames = scene(900, 2)
    other = [np.roll(255 - f, 40, axis=1) for f in frames]   # same motion, different picture
    write(tmp_path / "a.mp4", frames)
    write(tmp_path / "b.mp4", other)
    c = compare(signature(tmp_path / "a.mp4"), signature(tmp_path / "b.mp4"))
    assert c["verdict"] == "same moment, other camera"


def test_unrelated_recordings_are_not_flagged(tmp_path):
    write(tmp_path / "a.mp4", scene(900, 3))
    write(tmp_path / "b.mp4", scene(900, 4))
    assert compare(signature(tmp_path / "a.mp4"), signature(tmp_path / "b.mp4"))["verdict"] == "different"


def test_imu_rate_survives_alternating_intervals_and_unit_guess(tmp_path):
    t, out = 0, ["timestamp_us,ax,ay,az"]
    for i in range(2000):                                    # 4 ms and 6 ms alternate: 200 Hz on average
        out.append(f"{t},0,0,9.8")
        t += 4000 if i % 2 else 6000
    (tmp_path / "imu.csv").write_text("\n".join(out))
    (tmp_path / "none.csv").write_text("")
    m = imu_facts(tmp_path / "imu.csv")
    assert abs(m["hz"] - 200) < 1 and m["gaps"] == 0
    assert imu_facts(tmp_path / "none.csv")["error"] == "empty file"


def test_claims_table_marks_what_does_not_match():
    videos = [{"width": 1920, "height": 1080, "fps": 30.0, "audio": False}]
    rows = {r["item"]: r for r in check_claims({"width": "1534", "height": "1080", "fps": "60", "imu_hz": "200"},
                                               videos, [{"hz": 200.3}])}
    assert not rows["width"]["matches"] and rows["height"]["matches"]
    assert not rows["fps"]["matches"] and rows["imu_hz"]["matches"]


def test_stereo_pair_is_recognised_and_not_counted_as_a_repeat(tmp_path):
    frames = scene(700, 5)
    write(tmp_path / "clip_left.mp4", frames)
    write(tmp_path / "clip_right.mp4", [np.roll(f, 6, axis=1) for f in frames])
    assert len(stereo_pairs(sorted(tmp_path.glob("*.mp4")))) == 1
    r = verify([tmp_path], {"fps": "30"}, progress=False)
    assert r["stereo_pairs"][0]["frames_left"] == r["stereo_pairs"][0]["frames_right"] == 700
    assert r["summary"]["repeats_outside_stereo_pairs"] == 0 and r["summary"]["claims_not_matching"] == 0
