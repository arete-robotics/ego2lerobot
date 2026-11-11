"""Cut a long video into episodes where hands are at work.

Two ways to cut:
  find_steps     (default, `--segment action`) cuts at action boundaries: candidates from the hand signals
                 (signals.candidates), each confirmed or rejected by a vision-language check of the frames around it.
  find_segments  (`--segment fixed`, the behaviour before sub-steps) splits long stretches into equal parts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .hands import HandTrack
from .signals import Candidate, candidates as signal_candidates, wrist_speed


@dataclass
class Segment:
    start: int  # first frame
    end: int  # one past the last frame
    hand_coverage: float  # share of frames with at least one hand
    two_hands: float  # share of frames with both hands
    cuts: dict = field(default_factory=dict)  # how the start and end were chosen (find_steps only)

    def seconds(self, fps: float) -> float:
        return (self.end - self.start) / fps

    def to_dict(self) -> dict:
        return asdict(self)


def hand_runs(track: HandTrack, gap_seconds: float = 1.0) -> list[tuple[int, int]]:
    """(start, end) frame ranges with a hand in view, gaps shorter than `gap_seconds` bridged."""
    runs, start, last = [], None, None
    for i in np.flatnonzero(track.score.max(1) > 0):
        if start is None:
            start = i
        elif i - last > gap_seconds * track.fps:
            runs.append((int(start), int(last) + 1))
            start = i
        last = i
    if start is not None:
        runs.append((int(start), int(last) + 1))
    return runs


def find_segments(track: HandTrack, min_seconds: float = 3.0, max_seconds: float = 15.0, gap_seconds: float = 1.0,
                  min_coverage: float = 0.0) -> list[Segment]:
    """Stretches with a hand in view; gaps shorter than `gap_seconds` are bridged, and stretches longer than
    `max_seconds` are split into equal parts."""
    fps = track.fps
    present = track.score.max(1) > 0
    both = track.score.min(1) > 0
    segments = []
    for a, b in hand_runs(track, gap_seconds):
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


Judge = Callable[[int, int, int], dict]  # (candidate frame, stretch start, stretch end) → {"new_step": bool, ...}


def find_steps(track: HandTrack, judge: Judge | None = None, min_seconds: float = 2.0, max_seconds: float = 15.0,
               gap_seconds: float = 1.0, events: dict | None = None, fill_seconds: float = 4.0,
               ) -> tuple[list[Segment], list[dict]]:
    """Steps that follow the work, and the boundary decisions behind them.

    Inside each stretch with a hand in view (gaps under `gap_seconds` bridged), candidate boundaries come from the
    hand signals; a stretch longer than `fill_seconds` without a candidate gets one at its slowest wrist moment.
    `judge` confirms or rejects each candidate (caption.StepCaptioner.boundary compares the frames before and after
    it); without a judge a candidate counts when at least two cues agree (score ≥ 2). Confirmed candidates are taken
    strongest first and never closer than `min_seconds` to another cut. A step still longer than `max_seconds` is split
    at its strongest unconfirmed candidate, or into equal parts when there is none."""
    fps = track.fps
    present, both = track.score.max(1) > 0, track.score.min(1) > 0
    min_f, max_f = min_seconds * fps, max_seconds * fps
    cands = signal_candidates(track, events)
    speed = wrist_speed(track)
    steps, decisions = [], []
    for a, b in hand_runs(track, gap_seconds):
        if b - a < min_f:
            continue
        inner = _fill([c for c in cands if a + min_f <= c.frame <= b - min_f], a, b, speed, fill_seconds * fps, min_f)
        confirmed = []
        for c in inner:
            verdict = judge(c.frame, a, b) if judge else {"new_step": c.score >= 2.0}
            decisions.append({**c.to_dict(fps), **verdict})
            if verdict.get("new_step"):
                confirmed.append(c)
        cuts = {a: "hands come into view", b: "hands leave the view"}  # frame → why the cut is there
        for c in sorted(confirmed, key=lambda c: -c.score):
            if all(abs(c.frame - f) >= min_f for f in cuts):
                cuts[c.frame] = ("new step confirmed" if judge else "two or more signals") + f" ({', '.join(c.cues)})"
        rest = [c for c in inner if c.frame not in cuts]
        while True:
            edges = sorted(cuts)
            long = [(s, e) for s, e in zip(edges[:-1], edges[1:]) if e - s > max_f]
            if not long:
                break
            s, e = long[0]
            options = [c for c in rest if s + min_f <= c.frame <= e - min_f]
            if options:
                best = max(options, key=lambda c: (c.score, -abs(c.frame - (s + e) / 2)))
                cuts[best.frame] = f"longer than {max_seconds:g} s: split at the strongest other candidate"
                rest.remove(best)
            else:
                parts = int(np.ceil((e - s) / max_f))
                for f in np.linspace(s, e, parts + 1)[1:-1].round().astype(int):
                    cuts[int(f)] = f"longer than {max_seconds:g} s: split into equal parts"
        edges = sorted(cuts)
        for s, e in zip(edges[:-1], edges[1:]):
            steps.append(Segment(int(s), int(e), round(float(present[s:e].mean()), 3), round(float(both[s:e].mean()), 3),
                                 {"start": cuts[s], "end": cuts[e]}))
    return steps, decisions


def _fill(inner: list[Candidate], a: int, b: int, speed: np.ndarray, gap: float, margin: float) -> list[Candidate]:
    """Add a candidate at the slowest wrist moment (in the middle half) of every stretch longer than `gap` frames
    without one; ties go to the middle."""
    out = sorted(inner, key=lambda c: c.frame)
    while True:
        edges = [a] + [c.frame for c in out] + [b]
        for s, e in zip(edges[:-1], edges[1:]):
            lo, hi = int(max(s + (e - s) / 4, a + margin)), int(min(e - (e - s) / 4, b - margin))
            if e - s > gap and hi > lo:
                window = np.nan_to_num(speed[lo:hi + 1], nan=np.inf)
                i = lo + int(np.argmin(window + 1e-6 * np.abs(np.arange(len(window)) - len(window) / 2)))
                out = sorted(out + [Candidate(i, 0.25, ["slowest moment"])], key=lambda c: c.frame)
                break
        else:
            return out


def merge_agreeing(steps: list[Segment], labels: list, agree: Callable, track: HandTrack, max_seconds: float = 15.0,
                   ) -> tuple[list[Segment], list, list[bool]]:
    """Merge touching neighbours whose labels agree (shortest merged pair first) while the result is at most
    `max_seconds`. Returns the steps, a label for each (the longer part's), and whether each was merged."""
    steps, labels, merged = list(steps), list(labels), [False] * len(steps)
    present, both = track.score.max(1) > 0, track.score.min(1) > 0
    while True:
        pairs = [(steps[k + 1].end - steps[k].start, k) for k in range(len(steps) - 1)
                 if steps[k].end == steps[k + 1].start and steps[k + 1].end - steps[k].start <= max_seconds * track.fps
                 and agree(labels[k], labels[k + 1])]
        if not pairs:
            return steps, labels, merged
        _, k = min(pairs)
        a, b = steps[k], steps[k + 1]
        steps[k:k + 2] = [Segment(a.start, b.end, round(float(present[a.start:b.end].mean()), 3),
                                  round(float(both[a.start:b.end].mean()), 3), {"start": a.cuts.get("start", ""),
                                  "end": b.cuts.get("end", ""), "merged_at": a.cuts.get("merged_at", []) + [a.end]
                                  + b.cuts.get("merged_at", [])})]
        labels[k:k + 2] = [labels[k] if a.end - a.start >= b.end - b.start else labels[k + 1]]
        merged[k:k + 2] = [True]
