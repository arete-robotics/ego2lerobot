"""Step types (--step-vocab): list files, picks, rotated orders, agreement, fallback, the auto list, and `run`."""
import json
import re
from pathlib import Path

import av
import numpy as np
import pytest

from ego2lerobot.caption import StepCaptioner
from ego2lerobot.vocab import (TYPE_PROMPT, VOCAB_PROMPT, StepTyper, build_vocab, filter_types, load_vocab, options,
                               orders, parse_pick, parse_types, save_vocab)

from test_steps import FOLD, FOLD2, step_track, track_with

TYPES = ["fold the side flap", "press the box open into shape", "place the box on the shelf",
         "take a flat blank from the stack"]


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
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


def shown(prompt: str) -> dict[str, int]:
    """Type text → the number it has in this prompt's list."""
    return {m.group(2): int(m.group(1)) for m in re.finditer(r"^(\d+)\. (.+)$", prompt, re.M) if m.group(1) != "0"}


class Picker:
    """A stub model. Type questions: answers `answers` in turn, each either a type text (replied with the number that
    type has in the prompt's list) or a raw reply string starting with '!'. Free captions: `captions` in turn."""

    def __init__(self, answers=(), captions=(FOLD, FOLD2), boundary='{"new_step": false}', groups=None):
        self.answers, self.captions, self.boundary, self.groups = list(answers), list(captions), boundary, groups
        self.calls = []

    def generate(self, images, prompt, max_tokens=None):
        kind = ("boundary" if "moment T" in prompt else "type" if "step types listed below" in prompt
                else "group" if "Group the descriptions" in prompt else "caption")
        self.calls.append((kind, len(images), prompt, max_tokens))
        if kind == "boundary":
            return self.boundary
        if kind == "group":
            return self.groups
        if kind == "caption":
            return self.captions.pop(0)
        answer = self.answers.pop(0)
        if answer.startswith("!"):
            return answer[1:]
        return json.dumps({"hands": "fold a flap", "type": shown(prompt).get(answer, 0)})


def test_load_vocab_skips_comments_blanks_and_duplicates(tmp_path):
    path = tmp_path / "types.txt"
    path.write_text("# box line 3\n\nfold the side flap.\n  press the box   open into shape  # 4 steps\n"
                    "Fold the side flap\nplace the box on the shelf\n")
    assert load_vocab(path) == ["fold the side flap", "press the box open into shape", "place the box on the shelf"]
    save_vocab(tmp_path / "again.txt", TYPES, ["made by a test"])
    assert load_vocab(tmp_path / "again.txt") == TYPES
    assert (tmp_path / "again.txt").read_text().startswith("# made by a test\n")


def test_second_order_moves_every_type_to_another_number():
    for k in range(2, 13):
        first, second = orders(k)
        assert first == list(range(k)) and sorted(second) == first
        assert all(a != b for a, b in zip(first, second))
    assert orders(1) == ([0], [0])
    assert orders(4)[1] == [2, 3, 0, 1]
    assert options(TYPES, [2, 3, 0, 1]).splitlines()[0] == "1. place the box on the shelf"


@pytest.mark.parametrize("reply, expected", [
    ('{"hands": "folding", "type": 2}', 2),
    ('```json\n{"hands": "x", "type": "3"}\n```', 3),
    ('{"type": 0}', 0),
    ("4", 4),
    ("2. press the box open into shape", 2),
    ('{"type": "place the box on the shelf"}', 3),
    ('{"hands": "x", "type": 7}', None),     # not in the list
    ('{"hands": "x", "type": true}', None),
    ('{"hands": "x", "type": 3', 3),         # cut off
    ("I cannot tell", None),
    ("", None),
])
def test_parse_pick(reply, expected):
    assert parse_pick(reply, TYPES, list(range(4))) == expected


def test_parse_pick_maps_shown_number_back_to_the_list():
    order = orders(4)[1]  # shown: 1 place, 2 take, 3 fold, 4 press
    assert parse_pick('{"type": 1}', TYPES, order) == 3
    assert parse_pick('{"type": 3}', TYPES, order) == 1
    assert parse_pick('{"type": "fold the side flap"}', TYPES, order) == 1
    assert parse_pick('{"type": 0}', TYPES, order) == 0


def test_two_agreeing_picks_give_the_type(video):
    backend = Picker(["press the box open into shape"] * 2)
    out = StepTyper(StepCaptioner(backend=backend), TYPES)(video, step_track(), 0, 60)
    assert out["task"] == "press the box open into shape" and out["step_type"] == 2
    assert out["verified"] is True and out["confidence"] == 1.0
    assert out["check"]["picks"] == [2, 2] and out["check"]["same_type"] is True
    assert [c[0] for c in backend.calls] == ["type", "type"]  # no free caption needed
    (_, n1, p1, _), (_, n2, p2, _) = backend.calls
    assert n1 == n2 == 6  # three full views and three close-ups, as for a caption
    assert p1 == TYPE_PROMPT.format(n=3, n1=4, m=6, options=options(TYPES, [0, 1, 2, 3]))
    assert p2 == TYPE_PROMPT.format(n=3, n1=4, m=6, options=options(TYPES, [2, 3, 0, 1]))
    assert shown(p1)["press the box open into shape"] == 2 and shown(p2)["press the box open into shape"] == 4


def test_a_model_that_always_says_one_does_not_agree_with_itself(video):
    backend = Picker(["!1", "!1"])  # position bias: always the first number shown
    out = StepTyper(StepCaptioner(backend=backend), TYPES)(video, step_track(), 0, 60)
    assert out["check"]["picks"] == [1, 3] and out["check"]["fallback"] == "the two picks differ"
    assert out["step_type"] == 0 and out["verified"] is False
    assert out["task"] == "fold the towel" and out["check"]["second_instruction"] == "crease the towel in half"
    assert [c[0] for c in backend.calls] == ["type", "type", "caption", "caption"]


@pytest.mark.parametrize("answers, picks, why", [
    (["!{\"type\": 0}"], [0], "first pick: none of these"),
    (["!no idea"], [None], "first pick could not be read"),
    (["fold the side flap", "!{\"type\": 0}"], [1, 0], "second pick: none of these"),
    (["fold the side flap", "place the box on the shelf"], [1, 3], "the two picks differ"),
])
def test_fallback_to_the_free_caption_is_never_verified(video, answers, picks, why):
    backend = Picker(answers, captions=[FOLD, FOLD2])  # the free captions would verify on their own
    out = StepTyper(StepCaptioner(backend=backend), TYPES)(video, step_track(), 0, 60)
    assert out["check"]["picks"] == picks and out["check"]["fallback"] == why
    assert out["verified"] is False and out["step_type"] == 0 and out["task"] == "fold the towel"
    assert out["check"]["same_action"] is True  # the free caption's own check is kept


def test_fallback_reuses_a_first_caption_and_keeps_idle(video):
    backend = Picker(["!{\"type\": 0}"], captions=[FOLD2])
    first = {"task": "fold the towel", "object": "towel", "objects": [], "caption_problems": []}
    out = StepTyper(StepCaptioner(backend=backend), TYPES)(video, step_track(), 0, 60, first)
    assert [c[0] for c in backend.calls] == ["type", "caption"]  # only the second free caption is asked
    assert out["task"] == "fold the towel" and out["verified"] is False
    backend = Picker(["!{\"type\": 0}"], captions=['{"object": "", "instruction": "idle"}'])
    assert StepTyper(StepCaptioner(backend=backend), TYPES)(video, step_track(), 0, 60)["task"] == "idle"


def captions_of(*tasks):
    return [{"task": t, "object": o, "objects": [o]} for t, o in tasks]


def test_filter_types_drops_weak_and_invented_types():
    caps = captions_of(("fold the box flap", "box flap"), ("fold flap of box", "box flap"),
                       ("press down pizza box lid", "pizza box lid"), ("press down box lid", "box lid"),
                       ("place box on shelf", "box"), ("put the box on the shelf", "box"),
                       ("open the box", "box"))
    proposed = [
        {"type": "Fold the box flap", "object": "box flap", "steps": [1, 2]},
        {"type": "press down the pizza box lid", "object": "pizza box lid", "steps": [3, 4]},  # 'pizza' once
        {"type": "press down the box lid", "object": "box lid", "steps": [3, 4]},
        {"type": "place the box on the shelf", "object": "box", "steps": [5, 6, 99]},
        {"type": "open the box", "object": "box", "steps": [7]},                    # one caption
        {"type": "tape the box", "object": "tape", "steps": [1, 2]},                # tape: no caption
        {"type": "box handling", "object": "box", "steps": [1, 2, 5]},              # not an imperative
        {"type": "fold the box flap", "object": "flap", "steps": [1, 2, 7]},       # duplicate
    ]
    kept, report = filter_types(proposed, caps)
    assert kept == ["fold the box flap", "press down the box lid", "place the box on the shelf"]
    why = {r["type"]: r.get("dropped") for r in report}
    assert why["press down the pizza box lid"] == "names 'pizza' (1 caption)"
    assert why["open the box"] == "supported by 1 caption"
    assert why["tape the box"].startswith("names 'tape' (0 captions)")
    assert why["box handling"].startswith("not an imperative")
    assert report[3]["support"] == [5, 6]  # step 99 does not exist
    assert report[0]["support"] == [1, 2, 7] and report[-1]["dropped"] == "duplicate"
    kept, report = filter_types(proposed, caps, max_types=2)
    assert kept == ["fold the box flap", "press down the box lid"]
    assert any(r.get("dropped") == "more than 2 types" for r in report)


def test_parse_types_reads_a_cut_off_list():
    text = ('{"types": [{"type": "fold the flap", "object": "flap", "steps": [1, 2]}, '
            '{"type": "press the box", "object": "box", "steps": ["3", 4]}, {"type": "place the bo')
    assert parse_types(text) == [{"type": "fold the flap", "object": "flap", "steps": [1, 2]},
                                 {"type": "press the box", "object": "box", "steps": [3, 4]}]
    assert parse_types('{"types": ["open the case", 3]}') == [{"type": "open the case", "object": "", "steps": []}]


def test_build_vocab_asks_text_only_and_leaves_idle_out():
    groups = json.dumps({"types": [{"type": "fold the box flap", "object": "box flap", "steps": [1, 3]},
                                   {"type": "place the box on the shelf", "object": "box", "steps": [2]}]})
    backend = Picker(groups=groups)
    firsts = captions_of(("fold the box flap", "box flap"), ("place box on shelf", "box")) + \
        [{"task": "idle", "object": "", "objects": []}] + captions_of(("fold flap of box", "flap"))
    built = build_vocab(StepCaptioner(backend=backend), firsts)
    assert built["types"] == ["fold the box flap"] and built["captions"][2] == "fold flap of box"
    kind, n_images, prompt, max_tokens = backend.calls[0]
    assert kind == "group" and n_images == 0 and max_tokens > 160
    assert prompt == VOCAB_PROMPT.format(n=3, captions="1. fold the box flap\n2. place box on shelf\n3. fold flap of box",
                                         lo=4, hi=12)
    assert build_vocab(StepCaptioner(backend=Picker()), firsts[:1])["types"] == []  # one caption: nothing to group


# ---------- `run` ----------

@pytest.fixture()
def clip(tmp_path) -> tuple[Path, Path]:
    """source/clip.mp4 (160×80, 30 fps, 4 s), a hand track that grasps at 1.5 s, and the cache holding it."""
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


def run_with(tmp_path, clip, monkeypatch, backend, *extra):
    pytest.importorskip("lerobot")
    from ego2lerobot import caption
    from ego2lerobot.cli import main
    monkeypatch.setattr(caption, "make_backend", lambda *a, **k: backend)
    video, cache = clip
    out = tmp_path / "work" / "ds"
    main(["run", str(video), "--out", str(out), "--cache", str(cache), "--hfov", "90", "--width", "64", "--caption",
          *extra])
    meta = out / "meta"
    vocab = json.loads((meta / "step_vocab.json").read_text()) if (meta / "step_vocab.json").exists() else None
    return out, json.loads((meta / "episodes_info.json").read_text()), json.loads((meta / "run.json").read_text()), vocab


def test_run_with_a_vocab_file_stores_type_and_list(tmp_path, clip, monkeypatch):
    path = tmp_path / "types.txt"
    path.write_text("\n".join(TYPES) + "\n")
    backend = Picker(["take a flat blank from the stack"] * 2)
    out, info, run, vocab = run_with(tmp_path, clip, monkeypatch, backend, "--step-vocab", str(path))
    assert len(info) == 1 and info[0]["task"] == "take a flat blank from the stack"
    assert info[0]["step_type"] == 4 and info[0]["verified"] is True
    assert vocab["source"] == "file" and vocab["types"] == TYPES and vocab["episodes_per_type"]["4"] == 1
    assert vocab["types"][info[0]["step_type"] - 1] == info[0]["task"]
    assert run["settings"]["step_vocab"] == "types.txt" and str(tmp_path) not in json.dumps(run)
    assert "caption" not in [c[0] for c in backend.calls]  # agreeing picks: no free caption asked


CUT = '{"before": "fold", "after": "fold", "new_step": true}'


def test_run_with_auto_vocab_saves_an_editable_list_next_to_the_dataset(tmp_path, clip, monkeypatch):
    # two steps (cut at the grasp); the only proposed type lists one of them, so it is dropped: free captions are used
    groups = json.dumps({"types": [{"type": "fold the towel", "object": "towel", "steps": [1]}]})
    backend = Picker(captions=[FOLD, FOLD, FOLD2, FOLD2], boundary=CUT, groups=groups)
    out, info, run, vocab = run_with(tmp_path, clip, monkeypatch, backend, "--step-vocab", "auto", "--min-seconds", "1")
    assert vocab["source"] == "auto" and vocab["types"] == [] and vocab["proposed"][0]["dropped"] == "supported by 1 caption"
    assert [e["task"] for e in info] == ["fold the towel"] * 2 and all(e["verified"] for e in info)
    assert all("step_type" not in e for e in info)
    saved = out.with_name("ds_step_vocab.txt")
    assert saved.exists() and load_vocab(saved) == [] and run["settings"]["step_vocab"] == "auto"


def test_run_with_auto_vocab_classifies_against_the_built_list(tmp_path, clip, monkeypatch):
    from ego2lerobot import cli
    groups = json.dumps({"types": [{"type": "fold the towel", "object": "towel", "steps": [1, 2]}]})
    backend = Picker(["fold the towel"] * 4, captions=[FOLD, FOLD], boundary=CUT, groups=groups)
    out, info, run, vocab = run_with(tmp_path, clip, monkeypatch, backend, "--step-vocab", "auto", "--min-seconds", "1")
    assert vocab["types"] == ["fold the towel"] and vocab["proposed"][0]["support"] == [1, 2]
    assert [(e["step_type"], e["verified"]) for e in info] == [(1, True), (1, True)]
    assert vocab["episodes_per_type"] == {"0": 0, "1": 2}
    assert [c[0] for c in backend.calls if c[0] != "boundary"] == ["caption"] * 2 + ["group"] + ["type"] * 4
    assert load_vocab(cli.vocab_path(out)) == ["fold the towel"]


def test_run_without_the_flag_has_no_step_types(tmp_path, clip, monkeypatch):
    out, info, run, vocab = run_with(tmp_path, clip, monkeypatch, Picker())
    assert vocab is None and "step_type" not in info[0] and "step_vocab" not in run["settings"]
    assert not out.with_name("ds_step_vocab.txt").exists()


def test_step_vocab_needs_captions(tmp_path, clip):
    from ego2lerobot.cli import main
    video, cache = clip
    with pytest.raises(SystemExit):
        main(["run", str(video), "--out", str(tmp_path / "ds"), "--cache", str(cache), "--step-vocab", "auto"])
