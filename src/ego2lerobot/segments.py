"""Cut a long video into episodes where hands are at work."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .hands import HandTrack


@dataclass
class Segment:
    start: int  # first frame
    end: int  # one past the last frame
    hand_coverage: float  # share of frames with at least one hand
    two_hands: float  # share of frames with both hands

    def seconds(self, fps: float) -> float:
        return (self.end - self.start) / fps

    def to_dict(self) -> dict:
        return asdict(self)


def find_segments(track: HandTrack, min_seconds: float = 3.0, max_seconds: float = 15.0, gap_seconds: float = 1.0,
                  min_coverage: float = 0.0) -> list[Segment]:
    """Stretches with a hand in view; gaps shorter than `gap_seconds` are bridged, and stretches longer than
    `max_seconds` are split into equal parts."""
    fps = track.fps
    present = track.score.max(1) > 0
    both = track.score.min(1) > 0
    runs, start, last = [], None, None
    for i in np.flatnonzero(present):
        if start is None:
            start = i
        elif i - last > gap_seconds * fps:
            runs.append((start, last + 1))
            start = i
        last = i
    if start is not None:
        runs.append((start, last + 1))
    segments = []
    for a, b in runs:
        length = (b - a) / fps
        if length < min_seconds:
            continue
        edges = np.linspace(a, b, int(np.ceil(length / max_seconds)) + 1).round().astype(int)
        for s, e in zip(edges[:-1], edges[1:]):
            if (e - s) / fps < min_seconds:
                continue
            seg = Segment(int(s), int(e), round(float(present[s:e].mean()), 3), round(float(both[s:e].mean()), 3))
            if seg.hand_coverage >= min_coverage:
                segments.append(seg)
    return segments


def spread(segments: list[Segment], k: int) -> list[Segment]:
    """At most k segments, evenly spread through the video."""
    if len(segments) <= k:
        return list(segments)
    idx = np.linspace(0, len(segments) - 1, k).round().astype(int)
    return [segments[i] for i in idx]
