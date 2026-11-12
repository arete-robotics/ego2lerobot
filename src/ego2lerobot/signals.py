"""Hand signals that need no model and no video: per-hand grasp and release events, and candidate step boundaries.

Grip aperture is the thumb-tip to index-tip distance divided by the palm length (wrist to middle-finger knuckle), both
from MediaPipe's metric hand shape, so it does not change with distance to the camera. A hand counts as closed below
CLOSE and open above OPEN (hysteresis); a state has to last MIN_STATE_S to count, and a hand lost for longer than
MAX_GAP_S makes the state unknown, so no event is invented while the hand is out of view.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .hands import SIDES, HandTrack

CLOSE, OPEN = 0.35, 0.6  # aperture thresholds (thumb–index distance / palm length)
MIN_STATE_S = 0.3
MAX_GAP_S = 0.3
PAUSE = 0.1  # image widths per second: the wrists are nearly still below this
TOGETHER, APART = 0.2, 0.3  # wrist-to-wrist distance in image widths (hysteresis)
# The same cues from measured 3D wrists (stereo), in metres. One image width is about 1 m at half a metre from a
# 90–100° lens, so the numbers stay the same.
PAUSE_M = 0.1  # m/s
TOGETHER_M, APART_M = 0.2, 0.3  # m
WEIGHTS = {"grasp": 1.0, "release": 1.0, "enters": 1.0, "leaves": 1.0, "pause": 0.5, "hands meet": 0.5,
           "hands part": 0.5}


@dataclass(frozen=True)
class Motion:
    """The wrist positions behind the motion cues (wrist_speed, pause_cues, together_cues) and their thresholds.

    By default (image_motion) these are the wrists in the image, in frame widths. A track can carry its own Motion in
    a `motion` attribute, for example the measured 3D wrists of a stereo clip in metres
    (stereo.StereoLabels.step_track); frames where `valid` is False are gaps, never motion."""

    wrist: np.ndarray  # (N, 2, D) positions
    valid: np.ndarray  # (N, 2) True where the position is known
    scale: float  # positions / scale are in `unit`
    unit: str
    pause: float  # unit per second
    together: float  # unit
    apart: float  # unit
    max_gap_s: float | None = None  # hands meeting or parting: no cue across a gap longer than this (None: any gap)


def image_motion(track: HandTrack) -> Motion:
    """Wrists in the image, in frame widths (the monocular path)."""
    return Motion(track.kp[:, :, 0], track.score > 0, track.width, "frame widths", PAUSE, TOGETHER, APART)


def measured_motion(wrist: np.ndarray, valid: np.ndarray) -> Motion:
    """Measured 3D wrists (N, 2, 3) in metres, NaN or `valid` False where unknown."""
    valid = np.asarray(valid, bool) & np.isfinite(wrist).all(-1)
    return Motion(np.asarray(wrist, np.float64), valid, 1.0, "m", PAUSE_M, TOGETHER_M, APART_M, MAX_GAP_S)


def wrist_motion(track) -> Motion:
    """The track's own Motion if it carries one, else its image wrists."""
    own = getattr(track, "motion", None)
    return own if own is not None else image_motion(track)


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Median of the valid values in a centred window; NaN where x is NaN."""
    half = max(0, window // 2)
    padded = np.concatenate([np.full(half, np.nan), x, np.full(half, np.nan)])
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(view, axis=1)
    return np.where(np.isnan(x), np.nan, out)


def aperture(track: HandTrack) -> np.ndarray:
    """(N, 2) grip aperture per hand, NaN where the hand was not found."""
    w = track.world.astype(np.float64)
    palm = np.linalg.norm(w[:, :, 9] - w[:, :, 0], axis=-1)
    gap = np.linalg.norm(w[:, :, 4] - w[:, :, 8], axis=-1)
    ok = (track.score > 0) & (palm > 1e-6)
    return np.where(ok, gap / np.where(ok, palm, 1.0), np.nan)


def runs_of(values: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end, value) for each run of equal values."""
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(values)]])
    return [(int(s), int(e), int(values[s])) for s, e in zip(starts, ends)]


def grip_states(a: np.ndarray, fps: float, close: float = CLOSE, open_: float = OPEN, min_state_s: float = MIN_STATE_S,
                max_gap_s: float = MAX_GAP_S) -> np.ndarray:
    """(N,) 1 closed, 0 open, -1 unknown, from one hand's aperture."""
    a = rolling_median(a, max(1, round(0.2 * fps)) | 1)
    state = np.full(len(a), -1, np.int8)
    current, last_seen = -1, -np.inf
    for i, x in enumerate(a):
        if np.isnan(x):
            if i - last_seen > max_gap_s * fps:
                current = -1
        else:
            last_seen = i
            current = 1 if x < close else 0 if x > open_ else current
        state[i] = current
    shortest = max(1, round(min_state_s * fps))
    previous = -1
    for s, e, v in runs_of(state):  # a closed or open spell shorter than min_state_s keeps the state before it
        if v >= 0 and e - s < shortest:
            state[s:e] = previous
        else:
            previous = v
    return state


def grasp_events(track: HandTrack, **kw) -> dict[str, list[dict]]:
    """Per hand, the frames where it closes (grasp) and opens (release); {"left": [...], "right": [...]}.

    Events are only emitted between two known states, never across a stretch where the hand was out of view."""
    a = aperture(track)
    out = {}
    for s, side in enumerate(SIDES):
        state = grip_states(a[:, s], track.fps, **kw)
        events, known = [], -1
        for start, _, v in runs_of(state):
            if v < 0:
                known = -1
                continue
            if known >= 0 and v != known:
                events.append({"frame": start, "t": round(start / track.fps, 3), "event": "grasp" if v == 1 else "release"})
            known = v
        out[side] = events
    return out


def wrist_speed(track: HandTrack, smooth_s: float = 0.3) -> np.ndarray:
    """(N,) the faster wrist's speed (median-smoothed); NaN with no hand. In frame widths per second for image
    wrists, in m/s for a track with measured 3D wrists (see Motion)."""
    m = wrist_motion(track)
    kp = m.wrist.astype(np.float64)
    ok = m.valid
    v = np.full(ok.shape, np.nan)
    v[1:] = np.linalg.norm(kp[1:] - kp[:-1], axis=-1) / m.scale * track.fps
    v[1:][~(ok[1:] & ok[:-1])] = np.nan
    v[0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmax(v, axis=1)
    return rolling_median(speed, max(1, round(smooth_s * track.fps)) | 1)


def presence_cues(track: HandTrack, min_s: float = 0.5, gap_s: float = MAX_GAP_S) -> list[tuple[int, str]]:
    """A hand entering or leaving the view (stays of at least min_s, gaps under gap_s bridged)."""
    cues = []
    for s, side in enumerate(SIDES):
        seen = track.score[:, s] > 0
        for a, b, v in runs_of(seen.astype(np.int8)):  # bridge short gaps
            if not v and b - a < gap_s * track.fps and a > 0 and b < len(seen):
                seen[a:b] = True
        for a, b, v in runs_of(seen.astype(np.int8)):
            if v and b - a >= min_s * track.fps:
                if a > 0:
                    cues.append((a, f"{side} enters"))
                if b < len(seen):
                    cues.append((b, f"{side} leaves"))
    return cues


def pause_cues(speed: np.ndarray, fps: float, threshold: float = PAUSE) -> list[tuple[int, str]]:
    """Local minima of wrist speed below `threshold`, at least a second apart."""
    cues, half = [], max(1, round(0.5 * fps))
    for i in np.flatnonzero(speed < threshold):
        window = speed[max(0, i - half): i + half + 1]
        if speed[i] <= np.nanmin(window) and (not cues or i - cues[-1][0] >= 2 * half):
            cues.append((int(i), "pause"))
    return cues


def together_cues(track: HandTrack, together: float | None = None, apart: float | None = None) -> list[tuple[int, str]]:
    """Both hands in view and their wrists coming together or moving apart (hysteresis). Thresholds default to the
    track's Motion (TOGETHER and APART image widths, or TOGETHER_M and APART_M metres for measured 3D wrists)."""
    m = wrist_motion(track)
    together = m.together if together is None else together
    apart = m.apart if apart is None else apart
    both = m.valid.all(1)
    d = np.linalg.norm(m.wrist[:, 0] - m.wrist[:, 1], axis=-1) / m.scale
    d = rolling_median(np.where(both, d, np.nan), max(1, round(0.2 * track.fps)) | 1)
    cues, state, last = [], None, -np.inf
    for i, x in enumerate(d):
        if np.isnan(x):
            if m.max_gap_s is not None and i - last > m.max_gap_s * track.fps:
                state = None  # what happened during a long gap is unknown
            continue
        last = i
        new = "together" if x < together else "apart" if x > apart else state
        if state is not None and new != state:
            cues.append((i, "hands meet" if new == "together" else "hands part"))
        state = new
    return cues


def cue_weight(name: str) -> float:
    """"left grasp" → WEIGHTS["grasp"], "hands meet" → WEIGHTS["hands meet"]."""
    return WEIGHTS.get(name, WEIGHTS.get(name.split(" ", 1)[-1], 0.5))


@dataclass
class Candidate:
    frame: int
    score: float
    cues: list[str] = field(default_factory=list)

    def to_dict(self, fps: float) -> dict:
        return {"frame": self.frame, "t": round(self.frame / fps, 2), "score": round(self.score, 2), "cues": self.cues}


def candidates(track: HandTrack, events: dict | None = None, window_s: float = 0.5, min_gap_s: float = 1.5,
               min_score: float = 1.0) -> list[Candidate]:
    """Candidate step boundaries: grasp/release, a hand entering or leaving, a pause, hands meeting or parting.

    Cues within `window_s` are pooled into one candidate scored by the sum of their weights (each cue counted once);
    then the best candidates at least `min_gap_s` apart and scoring at least `min_score` are kept, in time order.
    Pauses and hands meeting or parting come from the track's Motion (image wrists, or measured 3D wrists)."""
    events = grasp_events(track) if events is None else events
    cues = [(e["frame"], f"{side} {e['event']}") for side in SIDES for e in events.get(side, [])]
    pauses = pause_cues(wrist_speed(track), track.fps, wrist_motion(track).pause)
    cues += presence_cues(track) + pauses + together_cues(track)
    cues.sort()
    pooled: list[Candidate] = []
    group: list[tuple[int, str]] = []
    for cue in cues + [(10**12, "")]:
        if group and cue[0] - group[0][0] > window_s * track.fps:
            names = sorted({name for _, name in group})
            best = max(group, key=lambda c: cue_weight(c[1]))
            pooled.append(Candidate(best[0], sum(cue_weight(n) for n in names), names))
            group = []
        group.append(cue)
    kept: list[Candidate] = []
    for c in sorted(pooled, key=lambda c: -c.score):
        if c.score >= min_score and all(abs(c.frame - k.frame) >= min_gap_s * track.fps for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: c.frame)
