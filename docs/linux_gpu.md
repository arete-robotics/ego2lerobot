# Running on Linux with an NVIDIA GPU

Hand detection, 3D lifting, export and the QC report are plain Python and run on any Linux machine (MediaPipe runs on
the CPU). Only the caption step needs a GPU. It has three backends that send the same four keyframes and the same
prompt, decode greedily, and parse the reply into the same `{"task", "steps", "objects"}` schema:

| Backend | Where | Install | Select with |
|---|---|---|---|
| `mlx` | Apple silicon | `pip install "ego2lerobot[caption]"` | default on Apple silicon |
| `transformers` | Linux + NVIDIA GPU | `pip install "ego2lerobot[caption-cuda]"` | default elsewhere, or `--caption-backend transformers` |
| `openai` | any OpenAI-compatible server (vLLM, SGLang) | nothing extra | `--caption-url http://HOST:8000/v1` |

All three default to Qwen3-VL-8B-Instruct (Apache 2.0): `Qwen/Qwen3-VL-8B-Instruct` in bf16 for `transformers` and
vLLM, `mlx-community/Qwen3-VL-8B-Instruct-4bit` for MLX. Override with `--caption-model`.

**Status (2026-11-11).** The `transformers` backend has been run end to end on an RTX 4090 (Ubuntu 22.04, driver
580.105, torch 2.15.0, transformers 5.22.2, mediapipe 1.2.2, lerobot 0.7.0): all tests pass, and hand detection
matches the Mac tracks (median keypoint difference 0.01 to 0.1 px on two clips). The `openai` backend against a vLLM
server (steps 4 and 5) has still only been checked against a fake server; it has not been run on a GPU.

| Caption model | Peak GPU memory | Run time per minute of video* |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct`, bf16 | 17.7 GB | about 2.0 min |
| `unsloth/Qwen3-VL-32B-Instruct-bnb-4bit` (needs `pip install bitsandbytes`) | 20.9 GB | about 2.7 min |

*Two 90 to 100 s factory clips, action segmentation with the two-caption check, hand tracks already cached, model
load included. Detection, export and previews without captions took about 1.8 min per minute of video on the same
machine.

On the same two clips the bf16 8B model captions about as well as the 4-bit MLX build (so quantization is not what
limits it); the 32B model makes fewer wrong captions and its second-caption check catches more of them, but it words
steps more generically. Numbers are in the evaluation notes, on 26 to 34 steps per model.

## 1. Set up (one machine, about 30 GB free disk)

```bash
apt-get install -y libegl1 libgles2   # MediaPipe needs libEGL.so.1 even on a headless machine
git clone https://github.com/arete-robotics/ego2lerobot && cd ego2lerobot
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[caption-cuda,test]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q
```
Pass: `True` and a GPU name; all tests pass. Qwen3-VL-8B in bf16 needs about 17 GB of GPU memory for the weights, so
use a 24 GB card or larger. On smaller cards use `--caption-model Qwen/Qwen3-VL-4B-Instruct` (also Apache 2.0).

## 2. A short test clip

Any first-person clip with hands works. To make one from a longer video:
```bash
ffmpeg -ss 60 -t 60 -i long.mp4 -an -c:v libx264 -crf 20 clip.mp4
```

## 3. transformers backend

```bash
ego2lerobot run clip.mp4 --out ds_hf --hfov 100 --caption --caption-backend transformers --preview prev_hf
nvidia-smi --query-gpu=name,memory.used --format=csv   # in a second shell while it runs
```
Pass: no errors; the log shows `QC report: ...`; in `ds_hf/qc_report.md` the Captions row reads `N/N episodes`
with sub-steps; `ds_hf/meta/episodes_info.json` has a sensible English `task` per episode.

## 4. vLLM server and the openai backend

vLLM (Apache 2.0) is best kept in its own environment:
```bash
python3 -m venv ~/vllm && ~/vllm/bin/pip install vllm
~/vllm/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 8192 --limit-mm-per-prompt '{"image": 4}'
# older vLLM versions spell the last flag: --limit-mm-per-prompt image=4
```
When it prints that the server is up, in the ego2lerobot environment:
```bash
curl -s localhost:8000/v1/models
ego2lerobot run clip.mp4 --out ds_vllm --hfov 100 --caption --caption-url http://localhost:8000/v1 --cache ds_hf_cache
```
`--cache ds_hf_cache` reuses the hand detections from step 3, so only captions differ.

Pass: same checks as step 3.

## 5. Compare the two backends

```bash
python - <<'EOF'
import json
a = json.load(open("ds_hf/meta/episodes_info.json"))
b = json.load(open("ds_vllm/meta/episodes_info.json"))
print(len(a), len(b))
for x, y in zip(a, b):
    print(x["source"]["start_frame"], "|", x["task"], "|", y["task"])
EOF
```
Pass: the same episodes; tasks agree in meaning (exact wording can differ between bf16 runtimes).

## 6. What to record

GPU model, driver and CUDA version (`nvidia-smi`), `torch`, `transformers` and `vllm` versions, seconds per episode
for each backend, and peak GPU memory. For the vLLM path, then update the status at the top of this page.

## Throughput note for large jobs

Detection is the slow CPU step: about 10 ms per frame at 960 px wide on one process, so one hour of 30 fps video
takes roughly 20 minutes. Run `ego2lerobot prepare` first on 4K or 60 fps footage, and run one `ego2lerobot run`
process per video file in parallel up to the number of CPU cores. Caption time on one GPU is in the table at the top; with vLLM, several `run` processes can share one server.
