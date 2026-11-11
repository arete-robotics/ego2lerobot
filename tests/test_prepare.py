"""Proxy videos: frame-rate halving, downscaling and clean timestamps for sources with phone-style time bases."""
import json
from fractions import Fraction

import av
import numpy as np

from ego2lerobot.prepare import prepare


def write_source(path, frames=60, rate=60, size=(320, 180)):
    """A 60 fps clip whose stream time base is 1/600 s, as phones write it."""
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=rate)
        s.width, s.height, s.pix_fmt = *size, "yuv420p"
        s.codec_context.time_base = Fraction(1, 600)
        for i in range(frames):
            img = np.full((size[1], size[0], 3), (i * 4) % 256, np.uint8)
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts, f.time_base = i * 10, Fraction(1, 600)
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_prepare_halves_rate_and_keeps_timestamps_increasing(tmp_path):
    src = tmp_path / "src.mp4"
    write_source(src)
    facts = prepare(src, tmp_path / "proxy" / "src.mp4", width=160, fps=30)
    assert facts["fps"] == 60 and facts["frames"] == 60
    assert facts["proxy"]["width"] == 160 and facts["proxy"]["height"] == 90 and facts["proxy"]["frames"] == 30
    assert json.loads((tmp_path / "proxy" / "src.source.json").read_text())["width"] == 320
    with av.open(str(tmp_path / "proxy" / "src.mp4")) as c:
        s = c.streams.video[0]
        t = np.array([float(f.pts * s.time_base) for f in c.decode(s)])
        assert (s.codec_context.width, s.codec_context.height) == (160, 90)
    assert len(t) == 30 and np.all(np.diff(t) > 0)
    assert abs(np.median(np.diff(t)) - 1 / 30) < 1e-3
