"""Validated JSONL loading for PMOPD's three task streams."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

TASKS = ("math", "code", "reason")


def _load_jsonl(path: Path, task: str) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {task} data file: {path}")
    rows: List[dict] = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object in {path}:{line_number}")
            if "id" not in row or not str(row["id"]).strip():
                raise ValueError(f"missing non-empty id in {path}:{line_number}")
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"missing non-empty prompt in {path}:{line_number}")
            sample_id = str(row["id"])
            if sample_id in seen:
                raise ValueError(f"duplicate id {sample_id!r} in {path}")
            seen.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"no examples found in {path}")
    return rows


def load_task_rows(data_dir: str, tasks: Iterable[str] = TASKS) -> Dict[str, List[dict]]:
    """Load ``<data_dir>/<task>.jsonl`` for every requested task.

    Training only consumes ``id`` and ``prompt``. Extra fields are preserved so the
    same records can also carry evaluation metadata when desired.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"data directory does not exist: {root}")
    requested = tuple(tasks)
    unknown = sorted(set(requested) - set(TASKS))
    if unknown:
        raise ValueError(f"unsupported tasks: {unknown}; expected a subset of {TASKS}")
    rows = {task: _load_jsonl(root / f"{task}.jsonl", task) for task in requested}
    print("loaded task data: " + ", ".join(f"{task}={len(items)}" for task, items in rows.items()), flush=True)
    return rows
