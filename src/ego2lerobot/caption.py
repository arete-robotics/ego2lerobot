"""Language instructions for episodes from a local vision-language model (Qwen3-VL via MLX on Apple silicon).

Install with `pip install "ego2lerobot[caption]"`. Without it, episodes get the `--default-task` text.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import av

DEFAULT_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"  # Qwen3-VL is Apache 2.0

PROMPT = """These {n} frames are in time order from a head-mounted camera worn by a worker. Describe the hand manipulation they show.
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
            task = re.search(r'"task"\s*:\s*"([^"]+)"', match.group(0))
            if task:
                return {"task": task.group(1)}
    first = text.strip().splitlines()[0] if text.strip() else ""
    return {"task": first[:120]}


class Captioner:
    def __init__(self, model: str = DEFAULT_MODEL, frames: int = 4):
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        self._generate, self._template = generate, apply_chat_template
        self.model, self.processor = load(model)
        self.frames = frames

    def __call__(self, video: str | Path, start: int, end: int, fps: float) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            images = keyframes(video, start, end, fps, self.frames, Path(tmp))
            prompt = self._template(self.processor, self.model.config, PROMPT.format(n=len(images)), num_images=len(images))
            result = self._generate(self.model, self.processor, prompt, image=images, max_tokens=160, temperature=0.0, verbose=False)
        return parse(result.text if hasattr(result, "text") else str(result))
