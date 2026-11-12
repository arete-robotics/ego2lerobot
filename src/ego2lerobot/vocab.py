"""Step types: label each sub-step with one entry of a closed, per-workstation list instead of a free caption.

A workstation repeats a small set of steps. With `--step-vocab FILE` the operator gives that set, one imperative step
type per line ("fold the bottom flaps inward"); with `--step-vocab auto` it is built from the video (build_vocab).

For each step the model sees the same images as for a free caption (three full frames and three close-ups around the
hands) and the numbered list, and answers with one number, or 0 for "none of these". A second pick, from the other
moments of the step that the caption check uses and with the list in a rotated order, must give the same type. Then
the step's task is that type's text and `step_type` its number in the list (1-based). If either pick is 0, cannot be
read, or the two differ, the step gets the usual free caption (with its own two-caption check) and is marked
verified: false, step_type: 0.

build_vocab (auto) asks the model, text only, to group the free first captions of all steps into 4 to 12 step types
and to say which captions each type covers. A type is dropped when fewer than two captions support it, or when it
names an object that appears in fewer than two captions (so that an object one caption made up does not become part
of the list).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .caption import CAPTION_AT, VERIFY_AT, base_verb, imperative, parse, same_action

TYPE_PROMPT = """Images 1 to {n} are full views from a camera on a worker's head during one short step of hand work, in time order. Images {n1} to {m} are close-ups around the hands at the same moments.
At this workstation the worker repeats the step types listed below. Which one is the main thing the hands do in this step? Only objects the hands hold or touch count. If several things happen, choose the one that takes most of the time.
{options}
0. none of these
Reply with JSON only: {{"hands": "<what the hands do, at most 10 words>", "type": <number>}}"""

VOCAB_PROMPT = """A worker at one workstation repeats a small set of steps. A video of the work was cut into {n} short steps, and a vision model described each step on its own. The descriptions can be vague or wrong, and can name objects that are not there.
{captions}
Group the descriptions into the distinct step types this workstation repeats: between {lo} and {hi} types. Write each type as a short imperative instruction for a robot (at most 10 words) that names the object and says exactly what is done to it, so that the types can be told apart. Use only objects that appear in the descriptions. Descriptions that say the same thing in different words belong to one type.
Reply with JSON only:
{{"types": [{{"type": "<imperative instruction>", "object": "<the object it names, 1 to 4 words>", "steps": [<numbers of the descriptions of this type>]}}]}}"""

MIN_TYPES, MAX_TYPES = 4, 12
VOCAB_TOKENS = 1024  # the grouped list is longer than one caption
IGNORE = frozenset("the and with into onto from for its their this that then each one both left right hand hands "
                   "worker".split())


def load_vocab(path: str | Path) -> list[str]:
    """Step types from a text file: one per line; blank lines and anything after '#' are ignored; duplicates dropped."""
    types = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = re.sub(r"\s+", " ", line.split("#", 1)[0]).strip().rstrip(".")
        if text and text.lower() not in (t.lower() for t in types):
            types.append(text)
    return types


def save_vocab(path: str | Path, types: list[str], notes: list[str] = ()) -> None:
    """Write `types` one per line, with `notes` as '#' comments at the top (load_vocab reads it back)."""
    lines = [f"# {n}" for n in notes] + list(types)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def orders(k: int) -> tuple[list[int], list[int]]:
    """The order of the list (0-based canonical indices, as shown) for the first and the second pick.

    The second is the first rotated by half its length, so every type moves to another number (for k >= 2) and a
    model that prefers a position, rather than a type, gives two different types."""
    first = list(range(k))
    shift = k // 2 if k > 1 else 0
    return first, first[shift:] + first[:shift]


def options(types: list[str], order: list[int]) -> str:
    return "\n".join(f"{n}. {types[i]}" for n, i in enumerate(order, 1))


def parse_pick(text: str, types: list[str], order: list[int]) -> int | None:
    """The canonical (1-based) type a reply picks, 0 for "none of these", None if it cannot be read.

    Accepts {"type": 3}, {"type": "3"}, a bare number, "3. fold the flap", or the text of one of the types."""
    reply = parse(text) if text else {}
    value = None
    if isinstance(reply, dict):
        for key in ("type", "choice", "number", "answer", "step_type"):
            if key in reply:
                value = reply[key]
                break
    if value is None:
        m = re.search(r'"type"\s*:\s*"?(\d+)', text or "")
        value = m.group(1) if m else text
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        shown = int(value)
    else:
        value = str(value).strip()
        named = [n for n, i in enumerate(order, 1) if value.lower().strip(" .") == types[i].lower()]
        m = re.match(r"\D{0,12}?(\d+)\b", value)
        if named:
            shown = named[0]
        elif m:
            shown = int(m.group(1))
        else:
            return None
    if shown == 0:
        return 0
    return order[shown - 1] + 1 if 1 <= shown <= len(order) else None


class StepTyper:
    """typer(video, track, start, end, first=None) → caption dict with step_type; see the module docstring.

    `captioner` is a caption.StepCaptioner (its frames, close-ups and backend are reused); `first` is the step's free
    first caption when it has already been made (auto mode), so the fallback does not ask for it again."""

    def __init__(self, captioner, types: list[str]):
        if not types:
            raise ValueError("the step vocabulary is empty")
        self.captioner, self.types = captioner, list(types)
        self.orders = orders(len(self.types))

    def pick(self, video, track, start: int, end: int, at_share, order: list[int]) -> tuple[int | None, dict]:
        at = [start + int((end - start) * f) for f in at_share]
        images = self.captioner.views(self.captioner.frames(video, track, at), track, at)
        n = len(images) // 2
        text = self.captioner.reply(images, TYPE_PROMPT.format(n=n, n1=n + 1, m=2 * n,
                                                               options=options(self.types, order)))
        reply = parse(text)
        hands = str(reply.get("hands", "")).strip() if isinstance(reply, dict) else ""
        return parse_pick(text, self.types, order), {"hands": hands, "reply": text.strip()[:200]}

    def __call__(self, video, track, start: int, end: int, first: dict | None = None) -> dict:
        p1, r1 = self.pick(video, track, start, end, CAPTION_AT, self.orders[0])
        picks, replies = [p1], [r1]
        if p1:  # a second pick only when the first chose a type
            p2, r2 = self.pick(video, track, start, end, VERIFY_AT, self.orders[1])
            picks.append(p2)
            replies.append(r2)
        check = {"picks": picks, "second_order": [i + 1 for i in self.orders[1]], "hands": [r["hands"] for r in replies]}
        if p1 and len(picks) == 2 and picks[1] == p1:
            text = self.types[p1 - 1]
            return {"task": text, "step_type": p1, "verified": True, "confidence": 1.0,
                    "check": {"second_instruction": text, **check, "same_type": True}, "caption_problems": []}
        captioner = self.captioner
        first = first if first is not None else captioner.caption(video, track, start, end)
        out = first if first["task"] == "idle" else captioner.verify(video, track, start, end, first)
        why = ("first pick: none of these" if p1 == 0 else "first pick could not be read" if p1 is None
               else "second pick could not be read" if picks[1] is None
               else "second pick: none of these" if picks[1] == 0 else "the two picks differ")
        return {**out, "step_type": 0, "verified": False,
                "check": {**out.get("check", {}), **check, "same_type": False, "fallback": why}}


# ---------- --step-vocab auto ----------

def _stem(word: str) -> str:
    if len(word) > 4 and re.search(r"(x|s|ch|sh)es$", word):
        return word[:-2]
    return word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word


def words(text: str) -> set[str]:
    """Content words of `text`, singular, without generic words (caption.GENERIC) or pronouns."""
    from .caption import GENERIC, NOT_IMPERATIVE
    out = {_stem(w) for w in re.findall(r"[a-z]+", text.lower())}
    return {w for w in out if len(w) >= 3 and w not in GENERIC and w not in NOT_IMPERATIVE and w not in IGNORE}


def caption_text(c: dict) -> str:
    return " ".join([c.get("task", ""), c.get("object", "")] + list(c.get("objects", [])))


def parse_types(text: str) -> list[dict]:
    """[{"type", "object", "steps"}] from the grouping reply; tolerates a cut-off list."""
    reply = parse(text)
    items = reply.get("types") if isinstance(reply, dict) else None
    if not isinstance(items, list):
        items = []
        for m in re.finditer(r"\{[^{}]*\}", text):  # whole entries of a list that was cut off
            try:
                items.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                pass
    out = []
    for item in items:
        if isinstance(item, str):
            item = {"type": item}
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            continue
        steps = item.get("steps") if isinstance(item.get("steps"), list) else []
        out.append({"type": item["type"].strip(), "object": str(item.get("object") or "").strip(),
                    "steps": [int(s) for s in steps if isinstance(s, (int, float, str)) and str(s).strip().isdigit()]})
    return out


def filter_types(proposed: list[dict], captions: list[dict], max_types: int = MAX_TYPES) -> tuple[list[str], list[dict]]:
    """(kept types, every proposal with its support and why it was dropped).

    `captions` are the free captions the model grouped ({"task", "object", "objects"}, numbered from 1). A proposal's
    support is the captions it lists that share an action kind or a content word with it. It is dropped when:
    it is not an imperative; fewer than two captions support it; or an object it names (a word of its "object", or a
    word of its text that is an object in some caption) appears in fewer than two captions. Duplicates are merged;
    at most `max_types` are kept, best supported first."""
    per_step = [words(caption_text(c)) for c in captions]
    freq = Counter(w for ws in per_step for w in ws)
    object_words = set().union(*(words(" ".join([c.get("object", "")] + list(c.get("objects", [])))) for c in captions)) \
        if captions else set()
    report, kept = [], {}
    for p in proposed:
        text, problems = imperative(p["type"])
        text = text[:1].lower() + text[1:]
        entry = {"type": text, "object": p.get("object", ""), "steps": sorted(set(p.get("steps", [])))}
        listed = [s for s in entry["steps"] if 1 <= s <= len(captions)]
        mine = words(text + " " + entry["object"])
        support = [s for s in listed if mine & per_step[s - 1]
                   or same_action(text, captions[s - 1].get("task", ""))]
        named = (words(text.split(" ", 1)[1] if " " in text else "") & object_words) | words(entry["object"])
        named = {w for w in named if base_verb(w) in (None, w)}  # not "folded", "opening"
        rare = sorted(w for w in named if freq[w] < 2)
        entry["support"] = support
        if problems:
            entry["dropped"] = problems[0]
        elif len(support) < 2:
            entry["dropped"] = f"supported by {len(support)} caption{'s' if len(support) != 1 else ''}"
        elif rare:
            entry["dropped"] = "names " + ", ".join(f"'{w}' ({freq[w]} caption{'s' if freq[w] != 1 else ''})"
                                                    for w in rare)
        elif text.lower() in kept:
            entry["dropped"] = "duplicate"
            kept[text.lower()]["support"] = sorted(set(kept[text.lower()]["support"]) | set(support))
        else:
            kept[text.lower()] = entry
        report.append(entry)
    best = sorted(kept.values(), key=lambda e: -len(e["support"]))
    for e in best[max_types:]:
        e["dropped"] = f"more than {max_types} types"
    return [e["type"] for e in best[:max_types]], report


def build_vocab(captioner, firsts: list[dict], lo: int = MIN_TYPES, hi: int = MAX_TYPES) -> dict:
    """{"types", "proposed", "reply", "captions"}: step types from the free first captions of all steps (idle ones
    left out), grouped by the model (text only) and checked by filter_types. "types" is empty when nothing passes."""
    captions = [c for c in firsts if c and c.get("task") and c["task"] != "idle"]
    if len(captions) < 2:
        return {"types": [], "proposed": [], "reply": "", "captions": [c.get("task", "") for c in captions]}
    listing = "\n".join(f"{k}. {c['task']}" for k, c in enumerate(captions, 1))
    text = captioner.reply([], VOCAB_PROMPT.format(n=len(captions), captions=listing, lo=lo, hi=hi),
                           max_tokens=VOCAB_TOKENS)
    types, report = filter_types(parse_types(text), captions, hi)
    return {"types": types, "proposed": report, "reply": text, "captions": [c["task"] for c in captions]}
