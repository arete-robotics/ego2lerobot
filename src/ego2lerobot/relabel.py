"""Apply reviewed step names to a dataset written by ego2lerobot: `ego2lerobot relabel DATASET REVIEW.json`.

REVIEW.json: {"relabel": {"EPISODE_INDEX": "new step name", ...}, "source": "human review"}.
Rewrites meta/tasks.parquet, the per-frame task_index, the `tasks` column of meta/episodes, total_tasks in
meta/info.json and meta/episodes_info.json (task, the earlier name as model_task, label_source, verified).
The files it changes are first copied to *.orig once.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def _keep(path: Path) -> None:
    orig = path.with_name(path.name + ".orig")
    if not orig.exists():
        shutil.copy2(path, orig)


def relabel(dataset: Path, names: dict[int, str], source: str = "human review") -> dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    info_path = dataset / "meta" / "episodes_info.json"
    episodes = json.loads(info_path.read_text())
    unknown = sorted(set(names) - {e["episode_index"] for e in episodes})
    if unknown:
        raise SystemExit(f"no such episodes: {unknown}")
    _keep(info_path)
    changed = 0
    for e in episodes:
        new = names.get(e["episode_index"])
        if new is None:
            continue
        new = new.strip()
        if new != e["task"]:
            e.setdefault("model_task", e["task"])
            e["task"] = new
            changed += 1
        e["label_source"], e["verified"] = source, True
    tasks = sorted({e["task"] for e in episodes})
    index = {t: k for k, t in enumerate(tasks)}
    tp = dataset / "meta" / "tasks.parquet"
    _keep(tp)
    pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task")).to_parquet(tp)
    by_episode = {e["episode_index"]: e["task"] for e in episodes}

    def rewrite(path: Path, column: str, value):
        t = pq.read_table(path)
        i = t.schema.get_field_index(column)
        new = pa.array([value(e) for e in t.column("episode_index").to_pylist()], type=t.schema.field(column).type)
        pq.write_table(t.set_column(i, t.schema.field(column), new), path)

    for f in sorted(dataset.glob("data/*/*.parquet")):
        rewrite(f, "task_index", lambda e: index[by_episode[e]])
    for f in sorted(dataset.glob("meta/episodes/*/*.parquet")):
        rewrite(f, "tasks", lambda e: [by_episode[e]])
    ip = dataset / "meta" / "info.json"
    info = json.loads(ip.read_text())
    info["total_tasks"] = len(tasks)
    ip.write_text(json.dumps(info, indent=4))
    info_path.write_text(json.dumps(episodes, indent=1, ensure_ascii=False))
    return {"episodes": len(episodes), "reviewed": len(names), "renamed": changed, "tasks": len(tasks)}


def add_parser(sub):
    p = sub.add_parser("relabel", help="apply reviewed step names to a dataset")
    p.add_argument("dataset", type=Path)
    p.add_argument("review", type=Path, help='JSON: {"relabel": {"0": "name", ...}, "source": "human review"}')


def run(args) -> int:
    review = json.loads(args.review.read_text())
    out = relabel(args.dataset, {int(k): v for k, v in review["relabel"].items()}, review.get("source", "human review"))
    print(f"{out['reviewed']} episodes reviewed, {out['renamed']} renamed; {out['tasks']} distinct step names")
    return 0
