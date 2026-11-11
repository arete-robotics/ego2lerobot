"""Language instructions for episodes from a vision-language model (Qwen3-VL, Apache 2.0), with three backends:

  mlx           Apple silicon, through mlx-vlm.          pip install "ego2lerobot[caption]"
  transformers  Linux + NVIDIA GPU (or any torch device). pip install "ego2lerobot[caption-cuda]"
  openai        Any OpenAI-compatible server, e.g. vLLM:  vllm serve Qwen/Qwen3-VL-8B-Instruct  (stdlib only)

Every backend gets the same keyframes (JPEG, 448 px wide), the same prompt, greedy decoding and at most MAX_TOKENS
new tokens, and the reply goes through the same parser, so the output schema is identical:
{"task": str, "steps": [str], "objects": [str]}. Without --caption, episodes get the `--default-task` text.
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
from pathlib import Path

import av

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

    def generate(self, images: list[str], prompt: str) -> str:
        chat = self._template(self.processor, self.model.config, prompt, num_images=len(images))
        result = self._generate(self.model, self.processor, chat, image=images, max_tokens=MAX_TOKENS, temperature=0.0, verbose=False)
        return result.text if hasattr(result, "text") else str(result)


class TransformersBackend:
    """Qwen3-VL through Hugging Face transformers; bf16 on a CUDA GPU (device_map="auto")."""

    def __init__(self, model: str | None = None, device_map: str = "auto"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        name = model or DEFAULT_MODELS["transformers"]
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype="auto", device_map=device_map)

    def generate(self, images: list[str], prompt: str) -> str:
        from PIL import Image
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in images] + [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                    add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class OpenAIBackend:
    """An OpenAI-compatible /chat/completions endpoint (vLLM, SGLang, ...). Images go inline as base64 JPEG."""

    def __init__(self, model: str | None = None, url: str = "http://localhost:8000/v1", api_key: str | None = None,
                 timeout: float = 300.0):
        self.model = model or DEFAULT_MODELS["openai"]
        self.url = url.rstrip("/") + "/chat/completions"
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def generate(self, images: list[str], prompt: str) -> str:
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()}}
                   for p in images] + [{"type": "text", "text": prompt}]
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": MAX_TOKENS, "temperature": 0.0}
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
