"""Proxy videos for large action-camera files: smaller frames and at most `fps` frames per second.

4K at 60 fps is eight times the pixels per second of 1080p at 30 fps, and every later step (hand detection, caption
keyframes, export) decodes the video again. Detection already works on 960-pixel-wide frames and the dataset images
are 640 wide, so a 1920-wide, 30 fps proxy loses nothing downstream and makes each pass several times faster.

Writes OUT.mp4 (H.264, no audio, no metadata) and OUT.source.json with facts about the original for the QC report.
Intrinsics need no change: --intrinsics JSON is rescaled to the video size, and FOV-based cameras are size-free.
"""
from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import av


def prepare(src: str | Path, out: str | Path, width: int = 1920, fps: float = 30.0, crf: int = 20) -> dict:
    """Write a proxy of `src` to `out` (.mp4) and its facts to `out` with suffix .source.json; return the facts."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    container = av.open(str(src))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    src_w, src_h = stream.codec_context.width, stream.codec_context.height
    rate = stream.average_rate
    every = max(1, round(float(rate) / fps))  # keep every k-th frame: 59.94 → 29.97, 60 → 30, 50 → 25
    ow = min(width, src_w) // 2 * 2
    oh = round(src_h * ow / src_w / 2) * 2
    out_rate = Fraction(rate / every).limit_denominator(1001)  # phones write rates like 7388400/123263
    writer = av.open(str(out), "w")
    enc = writer.add_stream("h264", rate=out_rate)
    enc.width, enc.height, enc.pix_fmt = ow, oh, "yuv420p"
    enc.options = {"crf": str(crf), "preset": "fast"}
    decoded = kept = 0
    for frame in container.decode(stream):
        if decoded % every == 0:
            small = frame.reformat(width=ow, height=oh, format="yuv420p")
            small.pts = None
            for packet in enc.encode(small):
                writer.mux(packet)
            kept += 1
        decoded += 1
    for packet in enc.encode():
        writer.mux(packet)
    writer.close()
    facts = {
        "source": Path(str(src)).name, "width": src_w, "height": src_h, "fps": round(float(rate), 3),
        "frames": decoded, "seconds": round(decoded / float(rate), 2), "codec": stream.codec_context.name,
        "pix_fmt": stream.codec_context.pix_fmt,
        "proxy": {"width": ow, "height": oh, "fps": round(float(out_rate), 3), "frames": kept, "crf": crf},
        "prepare_seconds": round(time.time() - t0, 1),
    }
    container.close()
    out.with_suffix(".source.json").write_text(json.dumps(facts, indent=1))
    return facts
