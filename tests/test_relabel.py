"""Reviewed step names are written to every place a LeRobot v3 dataset keeps them."""
import json

import pandas as pd
import pytest

from ego2lerobot.relabel import relabel


def tiny_dataset(root):
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["household task"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    pd.DataFrame({"episode_index": [0, 0, 1, 1, 2], "task_index": [0] * 5, "frame_index": [0, 1, 0, 1, 0]}).to_parquet(
        root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"episode_index": [0, 1, 2], "tasks": [["household task"]] * 3}).to_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({"total_tasks": 1}))
    (root / "meta" / "episodes_info.json").write_text(json.dumps(
        [{"episode_index": i, "task": "household task"} for i in range(3)]))


def test_names_reach_tasks_frames_and_episode_tables(tmp_path):
    tiny_dataset(tmp_path)
    out = relabel(tmp_path, {0: "slice the carrot", 2: "slice the carrot", 1: "household task"})
    assert out == {"episodes": 3, "reviewed": 3, "renamed": 2, "tasks": 2}
    tasks = pd.read_parquet(tmp_path / "meta" / "tasks.parquet")
    frames = pd.read_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    eps = pd.read_parquet(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    carrot = int(tasks.loc["slice the carrot", "task_index"])
    assert frames.task_index.tolist() == [carrot, carrot, 1 - carrot, 1 - carrot, carrot]
    assert [list(t) for t in eps.tasks] == [["slice the carrot"], ["household task"], ["slice the carrot"]]
    info = json.loads((tmp_path / "meta" / "episodes_info.json").read_text())
    assert info[0]["model_task"] == "household task" and info[0]["verified"] and "model_task" not in info[1]
    assert json.loads((tmp_path / "meta" / "info.json").read_text())["total_tasks"] == 2
    assert (tmp_path / "meta" / "tasks.parquet.orig").exists()


def test_unknown_episode_is_refused(tmp_path):
    tiny_dataset(tmp_path)
    with pytest.raises(SystemExit):
        relabel(tmp_path, {7: "x"})
