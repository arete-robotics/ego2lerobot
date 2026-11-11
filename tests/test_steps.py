"""Sub-steps: grasp events, boundary candidates, action segmentation, step captions with verification, and `run`."""
import json
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from ego2lerobot import HandTrack
from ego2lerobot.caption import (BOUNDARY_PROMPT, STEP_PROMPT, VERIFY_PROMPT, StepCaptioner, hand_box, imperative,
                                 names_object)
from ego2lerobot.segments import find_steps
from ego2lerobot.signals import candidates, grasp_events


def track_with(apertures: list[float | None], fps: float = 30.0, width: int = 1000, height: int = 500,
               centre=(500.0, 300.0), spread: float = 40.0) -> HandTrack:
    """A left hand that stands still at `centre`, with the given grip aperture per frame (None: hand not found)."""
    n = len(apertures)
    kp, world, score = np.zeros((n, 2, 21, 2)), np.zeros((n, 2, 21, 3)), np.zeros((n, 2))
    for i, a in enumerate(apertures):
        if a is None:
            continue
        world[i, 0, 9] = (0, -0.09, 0)          # palm length 9 cm
        world[i, 0, 4] = (0, -0.1, 0)           # thumb tip
        world[i, 0, 8] = (a * 0.09, -0.1, 0)    # index tip: aperture = distance / palm length
        kp[i, 0] = np.array(centre) + np.random.default_rng(i).uniform(-spread, spread, (21, 2))
        kp[i, 0, 0] = centre
        score[i, 0] = 0.9
    return HandTrack(kp, world, score, fps, width, height)


def test_grasp_events_use_hysteresis_minimum_duration_and_no_events_across_gaps():
    a = [0.8] * 30 + [0.2] * 30 + [0.8] * 3 + [0.2] * 27 + [0.5] * 30 + [0.8] * 30 + [None] * 30 + [0.2] * 30
    #    open       closed at 30  3-frame blip   still closed  between: keep  release at 120  gap 1 s     closed again
    events = grasp_events(track_with(a))
    assert [(e["frame"], e["event"]) for e in events["left"]] == [(30, "grasp"), (120, "release")]
    assert events["left"][0]["t"] == 1.0 and events["right"] == []


def test_candidates_pool_cues_and_keep_strong_ones():
    a = [0.8] * 150 + [0.2] * 150   # a still hand (pauses everywhere) that grasps at 5 s
    found = candidates(track_with(a))
    assert [c.frame for c in found] == [150]
    assert found[0].score == 1.5 and "left grasp" in found[0].cues and "pause" in found[0].cues


def test_find_steps_cuts_where_the_judge_confirms_and_respects_min_and_max():
    track = track_with([0.8] * 300 + [0.2] * 300)  # 20 s, grasp at 10 s
    calls = []

    def judge(frame, lo, hi):
        calls.append(frame)
        return {"new_step": abs(frame - 300) < 15, "before": "a", "after": "b"}

    steps, decisions = find_steps(track, judge, min_seconds=2.0, max_seconds=15.0)
    assert [(s.start, s.end) for s in steps] == [(0, 300), (300, 600)]
    assert steps[0].cuts["end"].startswith("new step confirmed") and steps[0].cuts["start"] == "hands come into view"
    assert calls == [d["frame"] for d in decisions] and any(d["new_step"] for d in decisions)

    steps, _ = find_steps(track, lambda *_: {"new_step": False}, min_seconds=2.0, max_seconds=8.0)
    lengths = [(s.end - s.start) / 30 for s in steps]
    assert len(steps) >= 3 and max(lengths) <= 8.0 and min(lengths) >= 2.0
    assert steps[0].start == 0 and steps[-1].end == 600
    assert all(a.end == b.start for a, b in zip(steps, steps[1:]))

    steps, decisions = find_steps(track, max_seconds=30.0)  # no judge: a cut needs two cues (score 2); grasp + pause is 1.5
    assert [(s.start, s.end) for s in steps] == [(0, 600)] and not any(d["new_step"] for d in decisions)
    steps, _ = find_steps(track)  # ... but a 20 s step is split at the strongest candidate
    assert [(s.start, s.end) for s in steps] == [(0, 300), (300, 600)] and "longer than 15 s" in steps[0].cuts["end"]


@pytest.mark.parametrize("text, expected, ok", [
    ("Assembling cardboard box", "assemble cardboard box", True),
    ("Places the box on the shelf.", "place the box on the shelf", True),
    ("pick up the cable", "pick up the cable", True),
    ("The worker folds the flap", "The worker folds the flap", False),
    ("cardboard box folding", "cardboard box folding", False),
    ("", "", False),
])
def test_imperative(text, expected, ok):
    instruction, problems = imperative(text)
    assert instruction == expected and (not problems) == ok


def test_names_object():
    assert names_object("insert the cables into the loops", "cable")
    assert names_object("fold the flap of the cardboard sheet", "cardboard box")
    assert not names_object("place it on the shelf", "box")


def test_hand_box_is_padded_4_by_3_and_inside_the_frame():
    track = track_with([0.5], width=1920, height=1080, centre=(1850.0, 100.0))
    left, top, right, bottom = hand_box(track, 0)
    assert np.isclose((right - left) / (bottom - top), 4 / 3) and bottom - top >= 0.4 * 1080
    assert 0 <= left and right <= 1920 and 0 <= top and bottom <= 1080
    assert left <= 1810 and right >= 1890  # the hand's keypoints are inside
    assert hand_box(track_with([None]), 0) is None


@pytest.fixture(scope="module")
def wide_video(tmp_path_factory) -> Path:
    """96×48 at 30 fps, 60 frames: a 2:1 frame, so full views and 4:3 close-ups have different shapes."""
    path = tmp_path_factory.mktemp("v") / "wide.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = 96, 48, "yuv420p"
        for i in range(60):
            for p in s.encode(av.VideoFrame.from_ndarray(np.full((48, 96, 3), i * 4, np.uint8), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


class ScriptedBackend:
    """Answers each of the three prompts with a fixed reply and records the images it was shown."""

    def __init__(self, caption='{"object": "towel", "instruction": "Folding the towel", "objects": ["towel"]}',
                 verify='{"in_hands": "towel", "motion": "folding", "object_matches": "yes", "action_matches": "yes"}',
                 boundary='{"before": "fold towel", "after": "place towel", "new_step": true}'):
        self.replies = {"caption": caption, "verify": verify, "boundary": boundary}
        self.calls = []

    def generate(self, images, prompt):
        kind = "boundary" if "moment T" in prompt else "verify" if "Proposed instruction" in prompt else "caption"
        self.calls.append((kind, [Image.open(p).size for p in images], prompt))
        return self.replies[kind]


def step_track(n=60):
    return track_with([0.5] * n, width=96, height=48, centre=(70.0, 30.0), spread=4.0)


def test_step_captioner_prompts_images_and_schema(wide_video):
    backend = ScriptedBackend()
    out = StepCaptioner(backend=backend)(wide_video, step_track(), 0, 60)
    assert out == {"task": "fold the towel", "object": "towel", "objects": ["towel"], "verified": True, "confidence": 1.0,
                   "check": {"in_hands": "towel", "motion": "folding", "object_matches": True, "action_matches": True},
                   "caption_problems": []}
    (kind, sizes, prompt), (kind2, sizes2, prompt2) = backend.calls
    assert kind == "caption" and prompt == STEP_PROMPT.format(n=3, n1=4, m=6)
    assert sizes[:3] == [(448, 224)] * 3  # full views (2:1), then 4:3 close-ups around the hands
    assert len(sizes) == 6 and all(w == 448 and abs(h - 336) <= 1 for w, h in sizes[3:])
    assert kind2 == "verify" and prompt2 == VERIFY_PROMPT.format(n=4, instruction="fold the towel") and len(sizes2) == 4


@pytest.mark.parametrize("caption, verify, problems", [
    ('{"object": "towel", "instruction": "fold the towel"}',
     '{"in_hands": "box", "object_matches": "no", "action_matches": "yes"}', []),
    ('{"object": "towel", "instruction": "towel folding"}',
     '{"object_matches": "yes", "action_matches": "yes"}', ["not an imperative: starts with 'towel'"]),
    ('{"object": "towel", "instruction": "fold it"}',
     '{"object_matches": "yes", "action_matches": "yes"}', ["the instruction does not name the object"]),
])
def test_step_captions_that_fail_the_check_are_marked_not_verified(wide_video, caption, verify, problems):
    out = StepCaptioner(backend=ScriptedBackend(caption, verify))(wide_video, step_track(), 0, 60)
    assert out["verified"] is False and out["caption_problems"] == problems and out["confidence"] < 1


def test_idle_step_skips_verification(wide_video):
    backend = ScriptedBackend(caption='{"object": "", "instruction": "idle"}')
    assert StepCaptioner(backend=backend)(wide_video, step_track(), 0, 60)["task"] == "idle"
    assert [c[0] for c in backend.calls] == ["caption"]


def test_boundary_check_sees_frames_on_both_sides(wide_video):
    backend = ScriptedBackend()
    out = StepCaptioner(backend=backend).boundary(wide_video, step_track(), 30, 0, 60)
    assert out == {"new_step": True, "before": "fold towel", "after": "place towel"}
    kind, sizes, prompt = backend.calls[0]
    assert kind == "boundary" and len(sizes) == 4 and prompt == BOUNDARY_PROMPT.format(n=4, k=2, k1=3)


# ---------- `run` end to end (no model: action segmentation from hand signals only) ----------

@pytest.fixture()
def clip(tmp_path) -> tuple[Path, Path]:
    """source/clip.mp4 (160×80, 30 fps, 4 s) and a hand track for it in the cache."""
    source, cache = tmp_path / "source", tmp_path / "work" / "ds_cache"
    source.mkdir()
    cache.mkdir(parents=True)
    video = source / "clip.mp4"
    with av.open(str(video), "w") as c:
        s = c.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = 160, 80, "yuv420p"
        for i in range(120):
            for p in s.encode(av.VideoFrame.from_ndarray(np.full((80, 160, 3), i * 2, np.uint8), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    track_with([0.8] * 45 + [0.2] * 75, width=160, height=80, centre=(80.0, 40.0), spread=5.0).save(cache / "clip.hands.npz")
    return video, cache


@pytest.mark.parametrize("segment", ["action", "fixed"])
def test_run_writes_portable_paths_grasp_events_and_no_empty_image_folder(tmp_path, clip, segment):
    pytest.importorskip("lerobot")
    from ego2lerobot.cli import main
    video, cache = clip
    out = tmp_path / "work" / "ds"
    main(["run", str(video), "--out", str(out), "--cache", str(cache), "--hfov", "90", "--width", "64",
          "--segment", segment])

    assert not (out / "images").exists()
    run_text = (out / "meta" / "run.json").read_text()
    assert str(tmp_path) not in run_text and "/Users/" not in run_text
    run = json.loads(run_text)
    entry = run["videos"][0]
    assert entry["path"] == "clip.mp4" and entry["track"] == "clip.hands.npz"  # source/ is two levels up: name only
    assert run["settings"]["segment"] == segment and run["settings"]["min_seconds"] == (2.0 if segment == "action" else 3.0)
    grasp = json.loads((out / "meta" / "grasp_events.json").read_text())
    assert grasp["videos"]["clip.mp4"]["left"] == [{"frame": 45, "t": 1.5, "event": "grasp"}]
    info = json.loads((out / "meta" / "episodes_info.json").read_text())
    assert info and info[0]["grasp_events"] == [{"hand": "left", "event": "grasp", "t": 1.5}]
    assert ("cuts" in info[0]) == (segment == "action")
    report = json.loads((out / "qc_report.json").read_text())
    assert report["videos"][0]["seconds"] == 4.0 and report["overall"]["episodes"] == len(info)


def test_portable_keeps_paths_next_to_the_dataset_and_hides_the_rest(tmp_path):
    from ego2lerobot.cli import portable
    assert portable(tmp_path / "source" / "a.mp4", tmp_path / "dataset") == "../source/a.mp4"
    assert portable(tmp_path / "dataset" / "a.mp4", tmp_path / "dataset") == "a.mp4"
    assert portable(Path("/somewhere/else/a.mp4"), tmp_path / "x" / "dataset") == "a.mp4"
