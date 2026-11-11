import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import av
import numpy as np
import pytest

from ego2lerobot import caption
from ego2lerobot.caption import PROMPT, Captioner, OpenAIBackend, default_backend, make_backend, normalize, parse

REPLY = '```json\n{"task": "wipe the table", "steps": ["pick up cloth", "wipe"], "objects": ["cloth", "table"]}\n```'
SCHEMA = {"task": "wipe the table", "steps": ["pick up cloth", "wipe"], "objects": ["cloth", "table"]}


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("v") / "clip.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(60):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 4, np.uint8), format="rgb24")
            for p in s.encode(frame):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


class FakeBackend:
    def __init__(self, reply=REPLY):
        self.reply, self.calls = reply, []

    def generate(self, images, prompt):
        assert all(Path(p).exists() for p in images)  # frames exist while the backend runs
        self.calls.append((list(images), prompt))
        return self.reply


def test_captioner_uses_backend_with_shared_prompt_and_schema(video):
    fake = FakeBackend()
    out = Captioner(backend=fake)(video, 0, 60, 30.0)
    assert out == SCHEMA
    images, prompt = fake.calls[0]
    assert len(images) == 4 and prompt == PROMPT.format(n=4)


@pytest.mark.parametrize("text, expected", [
    (REPLY, SCHEMA),
    ('{"task": "idle"}', {"task": "idle", "steps": [], "objects": []}),
    ('{"task": "fold towel", "steps": "fold in half", "objects": [null, 3]}',
     {"task": "fold towel", "steps": ["fold in half"], "objects": ["3"]}),
    ('{"task": "open the drawer", "steps": [broken', {"task": "open the drawer", "steps": [], "objects": []}),
    ("Pick up the sponge.\nMore text", {"task": "Pick up the sponge.", "steps": [], "objects": []}),
    ("", {"task": "", "steps": [], "objects": []}),
])
def test_parse_and_normalize_always_give_the_same_schema(text, expected):
    assert normalize(parse(text)) == expected


def test_default_backend_selection(monkeypatch):
    monkeypatch.setattr(caption.sys, "platform", "darwin")
    monkeypatch.setattr(caption.platform, "machine", lambda: "arm64")
    assert default_backend() == "mlx"
    assert default_backend("http://localhost:8000/v1") == "openai"
    monkeypatch.setattr(caption.sys, "platform", "linux")
    monkeypatch.setattr(caption.platform, "machine", lambda: "x86_64")
    assert default_backend() == "transformers"
    assert isinstance(make_backend("auto", url="http://gpu:8000/v1"), OpenAIBackend)
    with pytest.raises(ValueError):
        make_backend("tensorrt")
    with pytest.raises(ValueError):
        make_backend("mlx", url="http://gpu:8000/v1")


def test_openai_backend_request_and_reply(video):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"choices": [{"message": {"role": "assistant", "content": REPLY}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        backend = OpenAIBackend("Qwen/Qwen3-VL-8B-Instruct", f"http://127.0.0.1:{server.server_port}/v1", api_key="k")
        out = Captioner(backend=backend)(video, 0, 60, 30.0)
    finally:
        server.shutdown()
    assert out == SCHEMA
    body = seen["body"]
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer k"
    assert body["model"] == "Qwen/Qwen3-VL-8B-Instruct" and body["temperature"] == 0.0 and body["max_tokens"] == caption.MAX_TOKENS
    content = body["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image_url"] * 4 + ["text"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[-1]["text"] == PROMPT.format(n=4)
