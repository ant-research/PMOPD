#!/usr/bin/env python3
"""Convert directional PMOPD conflict-probe JSONL records into a task order."""
import argparse
import json
from pathlib import Path

from pmopd.diagnostics import summarize_directional_conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="directional JSONL files (for example anchor_*.jsonl)")
    ap.add_argument("--metric", default="grad_removed_mean")
    ap.add_argument("--output", default="task_conflict_summary.json")
    ap.add_argument("--percent", action="store_true", help="also print scores as percentages")
    args = ap.parse_args()
    records = []
    for name in args.inputs:
        with Path(name).open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    result = summarize_directional_conflicts(records, args.metric)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scale = 100 if args.percent else 1
    suffix = "%" if args.percent else ""
    print("recommended task order:", " ".join(result["recommended_task_order"]))
    for task in result["recommended_task_order"]:
        print(f"  {task}: {scale * result['task_scores'][task]:.4f}{suffix}")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
