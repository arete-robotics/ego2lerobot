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

**Status: the CUDA path has not yet been run on a GPU.** What has been checked on a Mac: backend selection, the
request sent to an OpenAI-compatible server and the reply parsing (unit tests with a fake server), and the
`transformers` and `mlx` code paths end to end on a random-weight tiny Qwen3-VL. The steps below verify it on a GPU.

## 1. Set up (one machine, about 30 GB free disk)

```bash
git clone https://github.com/ChizkiyahuOhayon/ego2lerobot && cd ego2lerobot
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
for each backend, and peak GPU memory. Then remove "has not yet been run on a GPU" from this page.

## Throughput note for large jobs

Detection is the slow CPU step: about 10 ms per frame at 960 px wide on one process, so one hour of 30 fps video
takes roughly 20 minutes. Run `ego2lerobot prepare` first on 4K or 60 fps footage, and run one `ego2lerobot run`
process per video file in parallel up to the number of CPU cores. Caption time per episode on a GPU is not measured
yet (step 6); with vLLM, several `run` processes can share one server.
