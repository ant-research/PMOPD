#!/usr/bin/env python3
"""Measure directional task conflicts used to choose PMOPD's task order.

Run once per anchor task. For each fold, the script trains the anchor from the
base model, extracts its update subspace, then branches from that state and
trains every other task while measuring the fraction of gradient removed by
projection onto the anchor memory. Use ``analyze_task_conflict.py`` to combine
all directions into an undirected task ranking.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from pmopd.data import load_task_rows
from pmopd.optimizer import Adafactor
from pmopd.train import (
    TASKS, append_ortho, build_student_sequences, build_teacher_sequences,
    generalized_jsd, is_target, left_tokenize, load_teacher_override,
    project_grads, response_logits, right_subspace,
)


def split_folds(rows, task, seed, n_folds, rows_per_fold):
    needed = n_folds * rows_per_fold
    if len(rows) < needed:
        raise ValueError(f"task {task} has {len(rows)} rows; need at least {needed}")
    offset = int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], "little")
    order = list(range(len(rows)))
    random.Random(seed + offset).shuffle(order)
    return [[rows[i] for i in order[f * rows_per_fold:(f + 1) * rows_per_fold]]
            for f in range(n_folds)]


def state_dict(trainable, dtype=torch.float32):
    return {n: p.detach().to("cpu", dtype=dtype).contiguous().clone() for n, p in trainable}


def restore(trainable, state):
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(state[name].to(parameter.device, dtype=parameter.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, choices=TASKS)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data_dir", required=True)
    for task in TASKS:
        ap.add_argument(f"--teacher_{task}", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--student_gpu", type=int, default=0)
    ap.add_argument("--teacher_gpu", type=int, default=1)
    ap.add_argument("--svd_device", default="cpu")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--rows_per_fold", type=int, default=60)
    ap.add_argument("--split_seed", type=int, default=20260819)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--max_prompt", type=int, default=768)
    ap.add_argument("--max_prompt_code", type=int, default=2048)
    ap.add_argument("--max_new", type=int, default=768)
    ap.add_argument("--top_k_loss", type=int, default=256)
    args = ap.parse_args()
    if min(args.n_folds, args.rows_per_fold, args.rank, args.batch, args.grad_accum) < 1:
        ap.error("fold, rank, and batch arguments must be positive")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"anchor_{args.anchor}.jsonl"
    done = set()
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); done.add((row["fold"], row["trained_task"]))
    (out / f"anchor_{args.anchor}_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    student_device = torch.device(f"cuda:{args.student_gpu}")
    teacher_device = torch.device(f"cuda:{args.teacher_gpu}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.padding_side = "left"; tokenizer.pad_token_id = tokenizer.eos_token_id
    student = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(student_device)
    for name, parameter in student.named_parameters():
        if is_target(name):
            parameter.data = parameter.data.float(); parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(False)
    student.gradient_checkpointing_enable(); student.config.use_cache = False
    trainable = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    base_state = state_dict(trainable, torch.bfloat16)

    teacher_paths = {task: getattr(args, f"teacher_{task}") for task in TASKS}
    teachers = {}
    for task in TASKS:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(teacher_device).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model, _ = load_teacher_override(model, teacher_paths[task])
        teachers[task] = model

    rows = load_task_rows(args.data_dir, TASKS)
    folds = {task: split_folds(rows[task], task, args.split_seed,
                               args.n_folds, args.rows_per_fold) for task in TASKS}
    manifest = {"split_seed": args.split_seed, "n_folds": args.n_folds,
                "rows_per_fold": args.rows_per_fold,
                "fold_ids": {task: [[str(r["id"]) for r in fold] for fold in fs]
                             for task, fs in folds.items()}}
    manifest_path = out / "split_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing split_manifest.json does not match this run")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    generation = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                  "max_new_tokens": args.max_new, "pad_token_id": tokenizer.pad_token_id,
                  "eos_token_id": tokenizer.eos_token_id}

    def loss_for(task, batch_rows):
        limit = args.max_prompt_code if task == "code" else args.max_prompt
        prompts = left_tokenize(tokenizer, [r["prompt"] for r in batch_rows], limit, student_device)
        student.gradient_checkpointing_disable(); student.config.use_cache = True; student.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = student.generate(**prompts, **generation)
        sids, sattn, response, mask, splens = build_student_sequences(
            tokenizer, prompts, generated, prompts["input_ids"].shape[1], student_device)
        tids, tattn, tplens = build_teacher_sequences(
            tokenizer, batch_rows, response.to(teacher_device), mask.to(teacher_device),
            limit, teacher_device)
        student.gradient_checkpointing_enable(); student.config.use_cache = False; student.train()
        slogits = response_logits(student, sids, sattn, splens, mask, grad=True)
        tlogits = response_logits(teachers[task], tids, tattn, tplens,
                                  mask.to(teacher_device), grad=False).to(student_device)
        return generalized_jsd(slogits, tlogits, mask, 0.0, 1.0, args.top_k_loss, None)

    def train(task, task_rows, memory, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        optimizer = Adafactor([p for _, p in trainable], lr=args.lr, relative_step=False,
                              scale_parameter=False, warmup_init=False, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        ratios, losses, accum, steps = [], [], 0, 0
        for pos in range(0, len(task_rows), args.batch):
            loss = loss_for(task, task_rows[pos:pos + args.batch]); losses.append(float(loss.item()))
            (loss / args.grad_accum).backward(); accum += 1
            if accum == args.grad_accum or pos + args.batch >= len(task_rows):
                if accum < args.grad_accum:
                    for _, p in trainable:
                        if p.grad is not None: p.grad.mul_(args.grad_accum / accum)
                if memory is not None:
                    diag = project_grads(trainable, memory)
                    ratios.append(float(diag["pmopd_grad_removed_ratio"]))
                    optimizer.set_update_projectors({p: memory[n] for n, p in trainable})
                optimizer.step(); optimizer.zero_grad(set_to_none=True); accum = 0; steps += 1
        return {"optimizer_steps": steps, "loss_mean": float(np.mean(losses)),
                "grad_removed_mean": float(np.mean(ratios)) if ratios else None,
                "grad_removed_sd": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0 if ratios else None}

    for fold in range(args.n_folds):
        pending = [task for task in TASKS if task != args.anchor and (fold, task) not in done]
        if not pending: continue
        restore(trainable, base_state)
        before = state_dict(trainable)
        anchor_seed = args.seed + fold * 1009 + 11
        anchor_stats = train(args.anchor, folds[args.anchor][fold], None, anchor_seed)
        memory = {}
        with torch.no_grad():
            for name, parameter in trainable:
                delta = parameter.detach() - before[name].to(student_device)
                memory[name] = append_ortho(None, right_subspace(delta, args.rank, args.svd_device).to(student_device))
        branch_state = state_dict(trainable)
        for index, task in enumerate(pending):
            restore(trainable, branch_state)
            stats = train(task, folds[task][fold], memory, args.seed + fold * 1009 + 101 + index)
            record = {"anchor_task": args.anchor, "trained_task": task, "fold": fold,
                      "rows_anchor": args.rows_per_fold, "rows_trained": args.rows_per_fold,
                      "rank": args.rank, "anchor_optimizer_steps": anchor_stats["optimizer_steps"], **stats}
            with result_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"fold={fold} {args.anchor}->{task} conflict={stats['grad_removed_mean']:.6f}", flush=True)
        del before, branch_state, memory
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
