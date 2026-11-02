"""Schedule-selection diagnostics used by PMOPD.

This module contains model-independent aggregation helpers for the task-conflict
probe and the subspace-similarity criterion used to select the cycle count.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math

import torch


def subspace_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return ||A^T B||_F^2 / k for two orthonormal column subspaces.

    The inputs may have different ranks; in that case the shared effective rank
    ``k=min(rank(A), rank(B))`` is used. Columns are orthonormalized here so the
    function is also safe for numerically approximate SVD outputs.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("subspaces must be rank-2 tensors")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"ambient dimensions differ: {a.shape[0]} != {b.shape[0]}")
    k = min(a.shape[1], b.shape[1])
    if k < 1:
        raise ValueError("subspaces must contain at least one direction")
    qa = torch.linalg.qr(a.float(), mode="reduced").Q[:, :k]
    qb = torch.linalg.qr(b.float(), mode="reduced").Q[:, :k]
    value = (qa.T @ qb).square().sum() / k
    return float(value.clamp(0.0, 1.0).item())


def summarize_directional_conflicts(
    records: Iterable[Mapping], metric: str = "grad_removed_mean"
) -> dict:
    """Aggregate directional records into undirected pairs and a task order.

    Each record needs ``anchor_task``, ``trained_task``, ``fold``, and ``metric``.
    The directional value for A->B is the gradient fraction removed from B by
    the update subspace learned on A. Pair conflict is the mean of A->B and B->A;
    task conflict is the mean of all undirected pairs incident to that task.
    """
    rows = list(records)
    if not rows:
        raise ValueError("no directional conflict records")
    lookup: dict[tuple[object, str, str], float] = {}
    tasks = set()
    for row in rows:
        a, b, fold = str(row["anchor_task"]), str(row["trained_task"]), row["fold"]
        if a == b:
            raise ValueError(f"self-direction is invalid: {a}->{b}")
        key = (fold, a, b)
        if key in lookup:
            raise ValueError(f"duplicate directional record: fold={fold} {a}->{b}")
        value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError(f"non-finite conflict value for fold={fold} {a}->{b}")
        lookup[key] = value
        tasks.update((a, b))

    folds = sorted({key[0] for key in lookup}, key=str)
    ordered_tasks = sorted(tasks)
    fold_summaries = []
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    task_values: dict[str, list[float]] = defaultdict(list)
    for fold in folds:
        pairs = {}
        for i, a in enumerate(ordered_tasks):
            for b in ordered_tasks[i + 1:]:
                try:
                    value = 0.5 * (lookup[(fold, a, b)] + lookup[(fold, b, a)])
                except KeyError as exc:
                    raise ValueError(f"missing reverse direction in fold={fold}: {a}<->{b}") from exc
                pairs[f"{a}|{b}"] = value
                pair_values[(a, b)].append(value)
        scores = {
            task: sum(value for pair, value in pairs.items() if task in pair.split("|"))
            / (len(ordered_tasks) - 1)
            for task in ordered_tasks
        }
        for task, value in scores.items():
            task_values[task].append(value)
        fold_summaries.append({
            "fold": fold,
            "pair_scores": pairs,
            "task_scores": scores,
            "task_order": sorted(ordered_tasks, key=lambda t: (scores[t], t)),
        })

    mean = lambda xs: sum(xs) / len(xs)
    pair_scores = {f"{a}|{b}": mean(values) for (a, b), values in pair_values.items()}
    task_scores = {task: mean(values) for task, values in task_values.items()}
    return {
        "metric": metric,
        "num_folds": len(folds),
        "directional_definition": "r_{a->b} = ||G_b M_a M_a^T||_F / ||G_b||_F",
        "undirected_definition": "c(a,b) = (r_{a->b} + r_{b->a}) / 2",
        "task_definition": "c(a) = mean_{b != a} c(a,b)",
        "pair_scores": pair_scores,
        "task_scores": task_scores,
        "recommended_task_order": sorted(ordered_tasks, key=lambda t: (task_scores[t], t)),
        "folds": fold_summaries,
    }


def choose_cycle_count(results: Mapping[int, Mapping[str, float]]) -> int:
    """Choose the cycle candidate with the largest finite ``mean`` similarity."""
    valid = {int(c): float(v["mean"]) for c, v in results.items()
             if v.get("mean") is not None and math.isfinite(float(v["mean"]))}
    if not valid:
        raise ValueError("no cycle candidate has a finite mean similarity")
    return max(valid, key=lambda c: (valid[c], -c))
