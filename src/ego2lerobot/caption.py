"""Language instructions from a vision-language model (Qwen3-VL, Apache 2.0), with three backends:

  mlx           Apple silicon, through mlx-vlm.          pip install "ego2lerobot[caption]"
  transformers  Linux + NVIDIA GPU (or any torch device). pip install "ego2lerobot[caption-cuda]"
  openai        Any OpenAI-compatible server, e.g. vLLM:  vllm serve Qwen/Qwen3-VL-8B-Instruct  (stdlib only)

Every backend gets the same images (JPEG, 448 px wide), the same prompts, greedy decoding and at most MAX_TOKENS new
tokens, and every reply goes through the same parser, so the output schema does not depend on the backend.

StepCaptioner (default) labels sub-steps with two prompts:
  boundary   frames just before and just after a candidate cut: does a new step start here?   (used by find_steps)
  caption    three full frames and three close-ups around the hands: an imperative instruction naming the object,
             asked twice for every step, from two different sets of moments in the step
and returns {"task", "object", "objects", "verified", "confidence", "check", "caption_problems"}. The step is
verified when the second, independent answer agrees with the first on the action (same verb or a verb of the same
kind, see VERB_GROUPS) and on the object, and the first is an imperative naming its object. A caption that fails is
kept but marked verified: false.

Captioner is the earlier one-question labeller ({"task", "steps", "objects"} from four frames), kept so that datasets
made with `--segment fixed` can be reproduced. Without --caption, episodes get the `--default-task` text.
"""
from __future__ import annotations

import base64
import json
import os
import platform
import re
import sys
import tempfile
import urllib.request
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np
from PIL import Image

DEFAULT_MODELS = {
    "mlx": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    "transformers": "Qwen/Qwen3-VL-8B-Instruct",
    "openai": "Qwen/Qwen3-VL-8B-Instruct",  # the name the server was started with
}
DEFAULT_MODEL = DEFAULT_MODELS["mlx"]
MAX_TOKENS = 160

PROMPT = """These {n} frames are in time order from a head- or chest-mounted camera worn by a worker. Describe the hand manipulation they show.
Reply with JSON only, in English:
{{"task": "<one imperative instruction a robot could follow, at most 12 words, naming the actual objects you see>",
 "steps": ["<2 to 4 short sub-steps in order>"],
 "objects": ["<objects the hands touch>"]}}
Describe only what is visible in these frames; be specific about the objects and the motion (pick, place, push, turn, sort, inspect...).
If the hands are not doing anything clear, use "task": "idle"."""


def keyframes(video: str | Path, start: int, end: int, fps: float, n: int, folder: Path, width: int = 448) -> list[str]:
    """n frames evenly inside [start, end), saved as JPEGs."""
    wanted = {int(start + (end - start) * (k + 0.5) / n): k for k in range(n)}
    paths = [""] * n
    container = av.open(str(video))
    stream = container.streams.video[0]
    container.seek(int(max(0, start / fps - 1) / stream.time_base), stream=stream)
    for frame in container.decode(stream):
        i = round(float(frame.pts * stream.time_base) * fps)
        if i in wanted:
            k = wanted.pop(i)
            image = frame.to_image()
            image = image.resize((width, round(image.height * width / image.width)))
            paths[k] = str(folder / f"{k}.jpg")
            image.save(paths[k], quality=90)
        if not wanted or i >= end:
            break
    container.close()
    return [p for p in paths if p]


def parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    task = re.search(r'"task"\s*:\s*"([^"]+)"', text)  # broken or cut-off JSON
    if task:
        return {"task": task.group(1)}
    first = text.strip().splitlines()[0] if text.strip() else ""
    return {"task": first[:120]}


def normalize(reply: dict) -> dict:
    """Any parsed reply → {"task": str, "steps": [str], "objects": [str]}."""
    reply = reply if isinstance(reply, dict) else {}

    def strings(value):
        value = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        return [str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip()]

    task = reply.get("task")
    return {"task": task.strip() if isinstance(task, str) else "", "steps": strings(reply.get("steps")),
            "objects": strings(reply.get("objects"))}


class MLXBackend:
    """Qwen3-VL through mlx-vlm on Apple silicon (the original backend)."""

    def __init__(self, model: str | None = None):
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        self._generate, self._template = generate, apply_chat_template
        self.model, self.processor = load(model or DEFAULT_MODELS["mlx"])

    def generate(self, images: list[str], prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        chat = self._template(self.processor, self.model.config, prompt, num_images=len(images))
        result = self._generate(self.model, self.processor, chat, image=images or None, max_tokens=max_tokens,
                                temperature=0.0, verbose=False)
        return result.text if hasattr(result, "text") else str(result)


def plain_linear_where_not_quantized(model) -> int:
    """Pre-quantized bitsandbytes checkpoints can store some layers (the vision tower) unquantized, while the loader
    still builds 4-bit layers for them; those layers then fail on the first forward pass. Give them back a plain
    linear layer with the stored weights. Returns the number of layers replaced (0 without bitsandbytes)."""
    try:
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        return 0
    from torch import nn
    swaps = [(name, m) for name, m in model.named_modules()
             if isinstance(m, Linear4bit) and m.weight.shape[1] != 1 and getattr(m.weight, "quant_state", None) is None]
    for name, old in swaps:
        new = nn.Linear(old.in_features, old.out_features, bias=old.bias is not None,
                        device=old.weight.device, dtype=old.weight.dtype)
        new.weight = nn.Parameter(old.weight.data, requires_grad=False)
        if old.bias is not None:
            new.bias = nn.Parameter(old.bias.data, requires_grad=False)
        parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
        setattr(parent, name.rpartition(".")[2], new)
    return len(swaps)


class TransformersBackend:
    """Qwen3-VL through Hugging Face transformers; bf16 on a CUDA GPU (device_map="auto")."""

    def __init__(self, model: str | None = None, device_map: str = "auto"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        name = model or DEFAULT_MODELS["transformers"]
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype="auto", device_map=device_map)
        plain_linear_where_not_quantized(self.model)

    def generate(self, images: list[str], prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        from PIL import Image
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in images] + [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                    add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class OpenAIBackend:
    """An OpenAI-compatible /chat/completions endpoint (vLLM, SGLang, ...). Images go inline as base64 JPEG."""

    def __init__(self, model: str | None = None, url: str = "http://localhost:8000/v1", api_key: str | None = None,
                 timeout: float = 300.0):
        self.model = model or DEFAULT_MODELS["openai"]
        self.url = url.rstrip("/") + "/chat/completions"
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def generate(self, images: list[str], prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()}}
                   for p in images] + [{"type": "text", "text": prompt}]
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0.0}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)["choices"][0]["message"]["content"] or ""


BACKENDS = {"mlx": MLXBackend, "transformers": TransformersBackend, "openai": OpenAIBackend}


def default_backend(url: str | None = None) -> str:
    """openai if a server URL is given, mlx on Apple silicon, transformers elsewhere."""
    if url:
        return "openai"
    return "mlx" if sys.platform == "darwin" and platform.machine() == "arm64" else "transformers"


def make_backend(name: str = "auto", model: str | None = None, url: str | None = None):
    name = default_backend(url) if name == "auto" else name
    if name not in BACKENDS:
        raise ValueError(f"unknown caption backend {name!r}; choose from auto, {', '.join(BACKENDS)}")
    if name == "openai":
        return OpenAIBackend(model, url or "http://localhost:8000/v1")
    if url:
        raise ValueError("--caption-url only applies to the openai backend")
    return BACKENDS[name](model)


class Captioner:
    """captioner(video, start, end, fps) → {"task", "steps", "objects"} for one episode."""

    def __init__(self, model: str | None = None, frames: int = 4, backend="auto", url: str | None = None):
        self.backend = make_backend(backend, model, url) if isinstance(backend, str) else backend
        self.frames = frames

    def __call__(self, video: str | Path, start: int, end: int, fps: float) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            images = keyframes(video, start, end, fps, self.frames, Path(tmp))
            text = self.backend.generate(images, PROMPT.format(n=len(images)))
        return normalize(parse(text))


# ---------- sub-steps: boundary check, caption with hand close-ups, verification ----------

BOUNDARY_PROMPT = """These {n} frames come from a camera on a worker's head, in time order. Frames 1 to {k} are from the second just before a moment T, frames {k1} to {n} from the second just after it.
Did the worker's hands start a new step at T? A new step means a different action or a different object: for example the hands stop folding and start placing, put an object down, pick up a new object, or open or close something. Continuing or repeating the same motion on the same object is the same step, even if the hands or the camera move.
Reply with JSON only:
{{"before": "<what the hands do before T: verb and object>", "after": "<what the hands do after T: verb and object>", "new_step": true or false}}"""

STEP_PROMPT = """Images 1 to {n} are full views from a camera on a worker's head during one short step of hand work, in time order. Images {n1} to {m} are close-ups around the hands at the same moments.
Say what the hands do in this step. Only objects the hands hold or touch count: finished products, machines and stacks in the background are not part of the step unless the hands touch them. If several things happen, describe the main one.
Reply with JSON only, in English:
{{"object": "<the object the hands manipulate, 1 to 4 words>",
 "instruction": "<one imperative instruction for a robot, at most 12 words, starting with a verb (pick up, place, fold, insert, press, open, turn ...) and naming the object>",
 "objects": ["<everything the hands touch>"]}}
If the hands are not manipulating anything, use "instruction": "idle"."""

CAPTION_AT = (0.2, 0.5, 0.8)  # where in the step the caption frames are taken (share of its length)
VERIFY_AT = (0.08, 0.35, 0.92)  # the second, independent caption: other moments, so the two do not share images
BOUNDARY_AT = (-1.2, -0.4, 0.4, 1.2)  # seconds around a candidate cut

VERBS = frozenset("""
adjust align apply arrange assemble attach bag bend bring brush buckle build bundle button carry check clamp clean
clear clip close coil collect comb connect count cover crease crimp cut detach dip disassemble discard disconnect
dispose drill drop dry empty examine fasten feed fetch file fill fit flatten flip fold form gather glue grab grasp
grip hammer hand hang hold hook insert inspect install join knot label lay lift load lock loosen lower mark measure
mix mount move nail open organise organize pack pass peel pick pinch place plug polish position pour prepare press
pull punch push put raise reach release remove replace return rinse roll rotate route rub sand scan scoop scrape
screw seal secure separate set shake shape shift slide smooth solder sort spray spread squeeze stack staple stick
stir store straighten stretch strip swap sweep take tap tape tear thread throw tie tighten touch toss transfer trim
tuck turn twist type uncap uncoil uncover unfasten unfold unhook unlock unpack unplug unroll unscrew unstack untie
unwind unwrap unzip use weigh wind wipe wrap zip
""".split())
NOT_IMPERATIVE = frozenset("a an the this that it its he she they we i you someone person worker workers hand hands "
                           "both left right".split())
VERB_GROUPS = tuple(frozenset(g.split()) for g in (  # verbs that describe the same kind of action
    "place put set lay insert drop load position stack store pack fit move transfer carry bring",
    "pick take grab grasp lift remove fetch collect pull unload unpack raise",
    "open unfold unwrap unroll spread tear peel unzip untie unpack",
    "close seal tape cover",
    "fold crease bend tuck press flatten smooth form shape",
    "attach connect plug mount install assemble join screw clip fasten thread insert fit tie wrap",
    "turn rotate flip twist",
    "inspect check examine scan count measure",
    "wipe clean rub polish brush sweep",
    "cut trim strip",
    "sort arrange organise organize align adjust straighten position stack",
    "apply glue stick spray solder tape",
    "push slide press",
))
GENERIC = frozenset("object item thing part piece component small large big long short black white blue green red brown "
                    "grey gray yellow orange pink clear plastic metal the and with into onto from top bottom side".split())


def base_verb(word: str) -> str | None:
    """The base form of a verb written as -ing, -s/-es or -ed, if it is in VERBS."""
    w = word.lower()
    if w in VERBS:
        return w
    for suffix in ("ing", "es", "s", "ed", "d"):
        if w.endswith(suffix) and len(w) > len(suffix) + 1:
            stem = w[: -len(suffix)]
            for guess in (stem, stem + "e", stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else ""):
                if guess in VERBS:
                    return guess
    return None


def imperative(text: str) -> tuple[str, list[str]]:
    """(instruction, problems): the text as an imperative starting with a known verb, lower case, no full stop.

    "Assembling cardboard box" becomes "assemble cardboard box"; text that does not start with a verb is returned
    as it is, with a problem noted."""
    words = re.sub(r"\s+", " ", text.strip().strip(".").strip()).split(" ")
    if not words or not words[0]:
        return "", ["empty instruction"]
    first = words[0].lower()
    verb = base_verb(first)
    if verb is None or first in NOT_IMPERATIVE:
        return " ".join(words), [f"not an imperative: starts with '{words[0]}'"]
    return " ".join([verb] + words[1:]), []


def names_object(instruction: str, obj: str) -> bool:
    """True if a content word of `obj` (or its singular) appears in `instruction`."""
    said = re.findall(r"[a-z]+", instruction.lower())
    for word in re.findall(r"[a-z]+", obj.lower()):
        if len(word) < 3 or word in NOT_IMPERATIVE:
            continue
        stem = word[:-1] if word.endswith("s") and len(word) > 3 else word
        if any(w.startswith(stem) for w in said):
            return True
    return False


def same_action(a: str, b: str) -> bool:
    """True if two instructions start with the same verb, or verbs of the same kind (VERB_GROUPS)."""
    va, vb = (base_verb(t.split(" ", 1)[0]) if t else None for t in (a, b))
    return bool(va and vb) and (va == vb or any(va in g and vb in g for g in VERB_GROUPS))


def content_words(text: str) -> set[str]:
    words = (w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z]+", text.lower()))
    return {w for w in words if len(w) >= 3 and w not in GENERIC and w not in NOT_IMPERATIVE}


def same_object(a_object: str, a_text: str, b_object: str, b_text: str) -> bool:
    """True if either answer's object shares a content word with the other answer's object or instruction."""
    return bool(content_words(a_object) & content_words(b_object + " " + b_text)
                or content_words(b_object) & content_words(a_object + " " + a_text))


def yes(value) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in ("yes", "true", "y"))


class FrameReader:
    """Frames of one video by index, decoded in order (seeking only for jumps), scaled to `width` pixels."""

    def __init__(self, video: str | Path, fps: float, width: int = 1280, keep: int = 48):
        self.video, self.fps, self.width, self.keep = Path(video), fps, width, keep
        self.cache: OrderedDict[int, Image.Image] = OrderedDict()
        self.container = self.frames = None
        self.position = -1

    def __call__(self, indices) -> dict[int, Image.Image]:
        out = {}
        for i in sorted(set(int(i) for i in indices)):
            if i not in self.cache:
                self._decode_to(i)
            if i in self.cache:
                out[i] = self.cache[i]
        return out

    def _decode_to(self, i: int) -> None:
        if self.container is None:
            self.container = av.open(str(self.video))
            self.stream = self.container.streams.video[0]
            self.stream.thread_type = "AUTO"
            start = self.stream.start_time
            self.origin = float(start * self.stream.time_base) if start is not None else 0.0
        if self.frames is None or i <= self.position or i - self.position > 2 * self.fps:
            self.container.seek(int(max(0.0, self.origin + i / self.fps - 1.0) / self.stream.time_base), stream=self.stream)
            self.frames, self.position = self.container.decode(self.stream), -1
        for frame in self.frames:
            j = round((float(frame.pts * self.stream.time_base) - self.origin) * self.fps)
            self.position = j
            if j >= i:
                height = round(frame.height * self.width / frame.width)
                self.cache[i] = Image.fromarray(frame.to_ndarray(format="rgb24", width=self.width, height=height))
                if len(self.cache) > self.keep:
                    self.cache.popitem(last=False)
                return

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
        self.container = self.frames = None


def hand_box(track, i: int, pad: float = 0.6, min_side: float = 0.4, aspect: float = 4 / 3):
    """(left, top, right, bottom) around the hands at frame i, in the track's pixels, or None without a hand.

    Uses the nearest frame within half a second that has a hand. The box is padded by `pad` times the hands' size so
    that the object in the hands is included, is at least `min_side` of the frame height, and has a 4:3 shape."""
    n, reach = len(track.score), round(0.5 * track.fps)
    for d in sorted(range(-reach, reach + 1), key=abs):
        j = i + d
        if 0 <= j < n and (track.score[j] > 0).any():
            pts = track.kp[j][track.score[j] > 0].reshape(-1, 2)
            break
    else:
        return None
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    size = max(x1 - x0, y1 - y0)
    h = max(y1 - y0 + 2 * pad * size, min_side * track.height)
    w = max(x1 - x0 + 2 * pad * size, h * aspect)
    w, h = min(w, track.width), min(max(h, w / aspect), track.height)
    left = float(np.clip((x0 + x1 - w) / 2, 0, track.width - w))
    top = float(np.clip((y0 + y1 - h) / 2, 0, track.height - h))
    return left, top, left + w, top + h


class StepCaptioner:
    """Boundary checks, instructions and verification for sub-steps; see the module docstring."""

    def __init__(self, model: str | None = None, backend="auto", url: str | None = None, width: int = 448):
        self.backend = make_backend(backend, model, url) if isinstance(backend, str) else backend
        self.width = width
        self.reader: FrameReader | None = None

    def frames(self, video, track, indices) -> dict[int, Image.Image]:
        if self.reader is None or self.reader.video != Path(video):
            if self.reader is not None:
                self.reader.close()
            self.reader = FrameReader(video, track.fps)
        return self.reader([min(max(int(i), 0), len(track.score) - 1) for i in indices])

    def closeup(self, image: Image.Image, track, i: int) -> Image.Image:
        box = hand_box(track, i)
        if box is None:
            return image
        scale = image.width / track.width
        return image.crop(tuple(round(v * scale) for v in box))

    def reply(self, images: list[Image.Image], prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        """The model's raw answer to `prompt` about `images` (scaled to `width` pixels, saved as JPEG)."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for k, image in enumerate(images):
                path = Path(tmp) / f"{k}.jpg"
                image.resize((self.width, round(image.height * self.width / image.width))).save(path, quality=90)
                paths.append(str(path))
            if max_tokens == MAX_TOKENS:
                return self.backend.generate(paths, prompt)
            return self.backend.generate(paths, prompt, max_tokens=max_tokens)

    def ask(self, images: list[Image.Image], prompt: str) -> dict:
        reply = parse(self.reply(images, prompt))
        return reply if isinstance(reply, dict) else {}

    def boundary(self, video, track, at: int, lo: int, hi: int) -> dict:
        """{"new_step": bool, "before": str, "after": str} for a candidate cut at frame `at` inside [lo, hi)."""
        idx = [min(max(round(at + s * track.fps), lo), hi - 1) for s in BOUNDARY_AT]
        got = self.frames(video, track, idx)
        images = [got[i] for i in idx if i in got]
        reply = self.ask(images, BOUNDARY_PROMPT.format(n=len(images), k=len(images) // 2, k1=len(images) // 2 + 1))
        return {"new_step": yes(reply.get("new_step")), "before": str(reply.get("before", "")),
                "after": str(reply.get("after", ""))}

    def views(self, got: dict, track, at: list[int]) -> list[Image.Image]:
        """Full frames at the frames `at` (those that could be read), then a close-up around the hands for each."""
        at = [i for i in at if i in got]
        return [got[i] for i in at] + [self.closeup(got[i], track, i) for i in at]

    def describe(self, got: dict, track, at: list[int]) -> dict:
        """One caption from full frames and hand close-ups at the frames `at`: {"instruction", "object", ...}."""
        images = self.views(got, track, at)
        n = len(images) // 2
        return self.ask(images, STEP_PROMPT.format(n=n, n1=n + 1, m=2 * n))

    def caption(self, video, track, start: int, end: int) -> dict:
        """The first answer for frames [start, end): {"task", "object", "objects", "caption_problems"}."""
        at = [start + int((end - start) * f) for f in CAPTION_AT]
        reply = self.describe(self.frames(video, track, at), track, at)
        obj = str(reply.get("object") or "").strip()
        objects = normalize({"objects": reply.get("objects")})["objects"]
        raw = str(reply.get("instruction") or reply.get("task") or "").strip()
        if raw.lower().strip(". ") == "idle":
            return {"task": "idle", "object": obj, "objects": objects}
        task, problems = imperative(raw)
        if not names_object(task, obj):
            problems.append("the instruction does not name the object" if obj else "no object given")
        return {"task": task, "object": obj, "objects": objects, "caption_problems": problems}

    def verify(self, video, track, start: int, end: int, first: dict) -> dict:
        """`first` plus verified, confidence and check, from a second caption at other moments of the step."""
        again = [start + int((end - start) * f) for f in VERIFY_AT]
        second = self.describe(self.frames(video, track, again), track, again)
        second_task = imperative(str(second.get("instruction") or "").strip())[0]
        second_obj = str(second.get("object") or "").strip()
        action_ok = same_action(first["task"], second_task)
        object_ok = same_object(first["object"], first["task"], second_obj, second_task)
        format_ok = not first["caption_problems"]
        return {**first, "verified": action_ok and object_ok and format_ok,
                "confidence": round((action_ok + object_ok + format_ok) / 3, 2),
                "check": {"second_instruction": second_task, "second_object": second_obj, "same_action": action_ok,
                          "same_object": object_ok}}

    def __call__(self, video, track, start: int, end: int) -> dict:
        first = self.caption(video, track, start, end)
        return first if first["task"] == "idle" else self.verify(video, track, start, end, first)


def captions_agree(a: dict, b: dict) -> bool:
    """Two first answers describe the same step: same kind of action on the same object (and neither is idle)."""
    if "idle" in (a.get("task"), b.get("task")):
        return False
    return same_action(a["task"], b["task"]) and same_object(a["object"], a["task"], b["object"], b["task"])
