#!/usr/bin/env python3
"""Select PMOPD's cycle count from same-task update-subspace similarity.

Each candidate run must have been trained with ``--save_initial --save_all_stages``.
For every target matrix and task visit, this script forms the stage displacement
Delta W, extracts its top-K right singular subspace, and compares consecutive
visits of the same task using ||V_t^T V_{t-1}||_F^2 / K.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import torch
from safetensors import safe_open

from pmopd.diagnostics import choose_cycle_count, subspace_similarity


STAGE_RE = re.compile(r"cycle_(\d+)_stage_(.+)$")


def parse_run(value: str) -> tuple[int, Path]:
    try:
        cycles, path = value.split("=", 1)
        return int(cycles), Path(path).expanduser().resolve()
    except Exception as exc:
        raise argparse.ArgumentTypeError("run must be CYCLES=/path/to/output") from exc


def tensor_keys(path: Path) -> list[str]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return list(f.keys())


def get_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def right_subspace(delta: torch.Tensor, rank: int, device: str, seed: int) -> torch.Tensor:
    x = delta.to(device=device, dtype=torch.float32)
    q = min(max(rank + 8, 24), min(x.shape))
    if q < 1:
        raise ValueError(f"cannot extract a subspace from shape {tuple(x.shape)}")
    devices = [] if not str(device).startswith("cuda") else [torch.device(device)]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        _, _, v = torch.svd_lowrank(x, q=q, niter=4)
    return v[:, :min(rank, v.shape[1])].cpu()


def ordered_snapshots(root: Path, expected_cycles: int) -> tuple[list[str], list[tuple[int, str, Path]]]:
    config_path = root / "training_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    task_order = list(cfg["task_order"])
    configured = int(cfg["cycles"])
    epochs = int(cfg.get("epochs", 1))
    total_cycles = configured * epochs
    if total_cycles != expected_cycles:
        raise ValueError(f"{root}: label says {expected_cycles} cycles but config has {total_cycles}")
    stages = []
    for cycle in range(1, total_cycles + 1):
        for task in task_order:
            directory = root / f"cycle_{cycle:02d}_stage_{task}"
            path = directory / "model.safetensors"
            if not path.is_file():
                raise FileNotFoundError(f"missing {path}; train with --save_all_stages")
            stages.append((cycle, task, path))
    return task_order, stages


def analyze_run(cycles: int, root: Path, rank: int, device: str, include: str | None) -> dict:
    initial = root / "initial" / "model.safetensors"
    if not initial.is_file():
        raise FileNotFoundError(f"missing {initial}; train with --save_initial")
    task_order, stages = ordered_snapshots(root, cycles)
    paths = [initial] + [p for _, _, p in stages]
    common = set(tensor_keys(paths[0]))
    for path in paths[1:]:
        common.intersection_update(tensor_keys(path))
    names = sorted(n for n in common if include is None or re.search(include, n))
    if not names:
        raise ValueError(f"{root}: no common matrices match --include")

    sums = defaultdict(float)
    counts = defaultdict(int)
    event_sums = defaultdict(float)
    event_counts = defaultdict(int)
    for matrix_index, name in enumerate(names):
        previous_weight = get_tensor(initial, name)
        previous_by_task = {}
        for visit_index, (cycle, task, path) in enumerate(stages):
            weight = get_tensor(path, name)
            delta = weight.float() - previous_weight.float()
            digest = hashlib.sha256(f"{name}|{visit_index}".encode()).digest()
            seed = int.from_bytes(digest[:4], "little")
            subspace = right_subspace(delta, rank, device, seed)
            if task in previous_by_task:
                sim = subspace_similarity(previous_by_task[task], subspace)
                sums[task] += sim
                counts[task] += 1
                event_sums[(task, cycle)] += sim
                event_counts[(task, cycle)] += 1
            previous_by_task[task] = subspace
            previous_weight = weight
        print(f"[{cycles} cycles] matrices {matrix_index + 1}/{len(names)}", flush=True)

    per_task = {task: sums[task] / counts[task] for task in task_order if counts[task]}
    mean = sum(per_task.values()) / len(per_task) if per_task else None
    events = [
        {"task": task, "cycle": cycle, "mean": event_sums[(task, cycle)] / event_counts[(task, cycle)],
         "matrices": event_counts[(task, cycle)]}
        for task in task_order for cycle in range(2, cycles + 1)
        if event_counts[(task, cycle)]
    ]
    return {
        "cycles": cycles,
        "run": str(root),
        "rank": rank,
        "matrix_count": len(names),
        "task_order": task_order,
        "per_task": per_task,
        "mean": mean,
        "revisit_events": events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, type=parse_run,
                    metavar="CYCLES=OUTPUT_DIR")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="SVD device, e.g. cpu or cuda:0")
    ap.add_argument("--include", default=None,
                    help="optional regex selecting snapshot keys; default: every common target matrix")
    ap.add_argument("--output", default="cycle_selection.json")
    args = ap.parse_args()
    if args.rank < 1:
        ap.error("--rank must be >= 1")
    runs = dict(args.runs)
    if len(runs) != len(args.runs):
        ap.error("duplicate cycle candidate")

    results = {cycles: analyze_run(cycles, root, args.rank, args.device, args.include)
               for cycles, root in sorted(runs.items())}
    recommended = choose_cycle_count(results)
    output = {
        "similarity_definition": "Sim_K(V,V') = ||V^T V'||_F^2 / K",
        "aggregation": "mean over matrices and adjacent same-task revisits, then macro-average over tasks",
        "recommended_cycles": recommended,
        "candidates": {str(k): v for k, v in results.items()},
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for cycles, result in results.items():
        print(f"cycles={cycles}: mean={result['mean']} per_task={result['per_task']}")
    print("recommended cycles:", recommended)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
