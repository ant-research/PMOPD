#!/usr/bin/env python3
"""Resumable multiple-choice reasoning evaluation."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from common import basic_summary, generate_batch, load_completed, load_jsonl, load_model


def extract_answer(response: str):
    segment = response.split("####")[-1] if "####" in response else response
    match = re.search(r"(?i)(?:answer|option|choice)\s*[:：]?\s*(?:is\s*)?\b([A-E])\b", segment)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*([A-E])(?:\b|[.)])", segment, re.I)
    if match:
        return match.group(1).upper()
    matches = re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", segment, re.I)
    return matches[-1].upper() if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--override", help="PMOPD snapshot or LoRA adapter directory")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt", type=int, default=768)
    parser.add_argument("--max-new", type=int, default=768)
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    for index, row in enumerate(rows):
        row.setdefault("id", str(index))
        if "prompt" not in row or not isinstance(row.get("gold", {}).get("answer"), str):
            raise ValueError("each row must contain prompt and gold.answer")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = load_completed(output, "id")
    todo = [row for row in rows if str(row["id"]) not in done]
    model, tokenizer, device = load_model(args.model, args.gpu, args.override)

    started = time.time()
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            generations = generate_batch(
                model, tokenizer, device, [row["prompt"] for row in batch], args.max_prompt, args.max_new
            )
            for row, (response, raw, audit) in zip(batch, generations):
                gold = row["gold"]["answer"].upper()
                record = {
                    "id": str(row["id"]), "response": response, "raw_response": raw,
                    "gold": row["gold"], "predicted_answer": extract_answer(response),
                    "correct": extract_answer(response) == gold, "generation": audit,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[str(row["id"])] = record
            print(f"completed={len(done)}/{len(rows)} elapsed_min={(time.time()-started)/60:.1f}", flush=True)

    ordered = [done[str(row["id"])] for row in rows if str(row["id"]) in done]
    summary = basic_summary(ordered, len(rows))
    summary.update({"data": args.data, "output": args.output, "model": args.model, "override": args.override})
    Path(str(output) + ".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
