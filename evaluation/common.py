"""Shared generation and reporting helpers for task evaluators."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pmopd.model_io import apply_model_override_

STOP_MARKERS = ("<|im_end|>", "<|im_start|>", "\nHuman:", "Human:", "\nUser:", "User:")


def clean_response(text: str) -> str:
    end = len(text)
    for marker in STOP_MARKERS:
        position = text.find(marker)
        if position >= 0:
            end = min(end, position)
    return text[:end].strip()


def load_jsonl(path: str) -> List[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_completed(path: Path, key: str) -> Dict[str, dict]:
    completed = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
                completed[str(row[key])] = row
            except (KeyError, json.JSONDecodeError):
                continue
    return completed


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> List[float]:
    if not total:
        return [0.0, 1.0]
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def load_model(model_name: str, gpu: int, override: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{gpu}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    if override:
        print(apply_model_override_(model, override), flush=True)
    return model, tokenizer, device


def generate_batch(model, tokenizer, device: str, prompts: List[str], max_prompt: int, max_new: int):
    prompt_lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_prompt, add_special_tokens=False,
    ).to(device)
    width = encoded["input_ids"].shape[1]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(
            **encoded, do_sample=False, max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    results = []
    for index in range(len(prompts)):
        ids = generated[index, width:]
        hits = (ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
        ended = bool(len(hits))
        count = int(hits[0]) + 1 if ended else ids.numel()
        raw = tokenizer.decode(ids[:count], skip_special_tokens=False)
        response = clean_response(raw)
        if tokenizer.eos_token:
            response = response.replace(tokenizer.eos_token, "").strip()
        results.append((response, raw, {
            "tokens": int(count), "ended_by_eos": ended, "hit_max": count >= max_new,
            "prompt_tokens": prompt_lengths[index], "prompt_truncated": prompt_lengths[index] > max_prompt,
        }))
    return results


def basic_summary(rows: Iterable[dict], expected: int) -> dict:
    values = list(rows)
    total = len(values)
    correct = sum(bool(row.get("correct")) for row in values)
    generations = [row["generation"] for row in values]
    return {
        "n": total, "correct": correct, "accuracy": correct / total if total else None,
        "wilson95": wilson_interval(correct, total), "complete": total == expected,
        "mean_tokens": sum(x["tokens"] for x in generations) / total if total else None,
        "hit_max_rate": sum(x["hit_max"] for x in generations) / total if total else None,
        "prompt_trunc_rate": sum(x["prompt_truncated"] for x in generations) / total if total else None,
    }
