#!/usr/bin/env python3
"""Rebuild a sample like arete-robotics/egocentric-10k-actions-sample from Egocentric-10K.

Egocentric-10K is gated: accept its terms on Hugging Face and run `hf auth login` first.
For each of N factories, the first video of one worker's first shard is streamed (no 1 GB download), then
`ego2lerobot run` labels up to 8 well-spread episodes per video.

  python examples/egocentric10k.py work/ --factories 30
  ego2lerobot run work/videos/*.mp4 --out dataset ... (printed at the end; one command per camera)
"""
import argparse
import random
import tarfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers

REPO = "builddotai/Egocentric-10K"


def first_video(shard: str, folder: Path) -> Path | None:
    """Stream the shard and stop after its first complete MP4."""
    with requests.get(hf_hub_url(REPO, shard, repo_type="dataset"), headers=build_hf_headers(), stream=True, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".mp4"):
                    path = folder / Path(member.name).name
                    path.write_bytes(tar.extractfile(member).read())
                    return path
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--factories", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    videos = args.work / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    shards = {}
    for f in HfApi().list_repo_files(REPO, repo_type="dataset"):
        if f.endswith("_part00.tar"):
            factory, _, worker = f.split("/")[:3]
            shards.setdefault(factory, []).append((worker, f))
    rng = random.Random(args.seed)
    factories = sorted(shards)
    rng.shuffle(factories)
    for factory in factories[: args.factories]:
        worker, shard = rng.choice(sorted(shards[factory]))
        intr = hf_hub_download(REPO, f"{factory}/workers/{worker}/intrinsics.json", repo_type="dataset", local_dir=args.work)
        video = first_video(shard, videos)
        if video:
            print(f"ego2lerobot run {video} --out {args.work / 'datasets' / video.stem} --intrinsics {intr} --per-video 8 --caption")


if __name__ == "__main__":
    main()
