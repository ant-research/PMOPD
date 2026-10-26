#!/usr/bin/env python
"""Full-data epoch multi-task PMOPD for decoder-only language models.

Both student and task-specific RL teacher use the exact same plain-causal prompt.  The
student first samples an on-policy response; student and teacher then score those same
response tokens.  No ChatML and no hidden tests/gold references are exposed to teachers.

Modes:
  mixed: balanced round-robin task batches, no gradient projection.
  sequential: configurable task order, no gradient projection.
  pmopd: same sequential curriculum/budget as ``sequential``, with PMOPD projection only.
"""
import argparse
import contextlib
import json
import os
import random
import time

from .data import load_task_rows

import torch
import torch.nn.functional as F

TASKS = ("math", "code", "reason")
LIN = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
       "gate_proj.weight", "up_proj.weight", "down_proj.weight")


def is_target(name):
    return name.endswith(LIN) and ".layers." in name


def log(msg, f=None):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    if f: f.write(s + "\n"); f.flush()


def snapshot(model, out):
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    save_file({n: p.detach().float().cpu().contiguous() for n, p in model.named_parameters() if is_target(n)},
              os.path.join(out, "model.safetensors"))


def load_teacher_override(model, path):
    """Load either a legacy full linear snapshot or a PEFT LoRA teacher adapter."""
    path = os.path.realpath(path)
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload().eval()
        return model, f"LoRA adapter merged from {path}"

    from safetensors.torch import load_file
    snapshot_path = os.path.join(path, "model.safetensors")
    if not os.path.isfile(snapshot_path):
        raise RuntimeError(f"teacher override is neither PEFT adapter nor linear snapshot: {path}")
    sd = load_file(snapshot_path)
    named = dict(model.named_parameters()); count = 0
    with torch.no_grad():
        for n, v in sd.items():
            if n in named and is_target(n):
                named[n].copy_(v.to(named[n].device, dtype=named[n].dtype)); count += 1
    return model, f"full linear snapshot matrices={count} from {path}"


class Sampler:
    def __init__(self, rows, seed):
        self.rows = rows; self.rng = random.Random(seed)
        self.order = list(range(len(rows))); self.rng.shuffle(self.order); self.pos = 0

    def next(self, n):
        out = []
        while len(out) < n:
            if self.pos == len(self.order):
                self.rng.shuffle(self.order); self.pos = 0
            out.append(self.rows[self.order[self.pos]]); self.pos += 1
        return out


def left_tokenize(tok, texts, max_len, dev):
    old = tok.padding_side; tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len, add_special_tokens=False)
    tok.padding_side = old
    return {k: v.to(dev) for k, v in enc.items()}


def build_student_sequences(tok, prompts, generated, prompt_width, dev):
    """Remove left prompt padding, append generated response, then right-pad full student sequences."""
    seqs, response_ids = [], []
    eos = tok.eos_token_id
    for i in range(len(prompts["input_ids"])):
        p = prompts["input_ids"][i][prompts["attention_mask"][i].bool()]
        r = generated[i, prompt_width:]
        # Strip generation padding after first EOS, retaining EOS itself.
        hit = (r == eos).nonzero(as_tuple=False)
        if len(hit): r = r[:int(hit[0]) + 1]
        if r.numel() == 0: r = torch.tensor([eos], device=dev)
        seqs.append(torch.cat([p, r])); response_ids.append(r)
    max_full, max_resp = max(x.numel() for x in seqs), max(x.numel() for x in response_ids)
    ids = torch.full((len(seqs), max_full), tok.pad_token_id, dtype=torch.long, device=dev)
    attn = torch.zeros_like(ids); mask = torch.zeros((len(seqs), max_resp), dtype=torch.bool, device=dev)
    resp = torch.full((len(seqs), max_resp), tok.pad_token_id, dtype=torch.long, device=dev)
    prompt_lens = []
    for i, (s, r) in enumerate(zip(seqs, response_ids)):
        ids[i, :s.numel()] = s; attn[i, :s.numel()] = 1
        resp[i, :r.numel()] = r; mask[i, :r.numel()] = True
        prompt_lens.append(s.numel() - r.numel())
    return ids, attn, resp, mask, prompt_lens


def build_teacher_sequences(tok, rows, response_ids, response_mask, max_prompt, dev):
    # Same plain-causal prompt as the student.  In particular, code hidden tests are
    # never included in model context; they are used only by the external verifier.
    encoded = [tok(r["prompt"], add_special_tokens=False, truncation=True,
                   max_length=max_prompt)["input_ids"] for r in rows]
    seqs, lens = [], []
    for i, p in enumerate(encoded):
        r = response_ids[i][response_mask[i]].tolist()
        seqs.append(torch.tensor(p + r, dtype=torch.long, device=dev)); lens.append(len(p))
    m = max(x.numel() for x in seqs)
    ids = torch.full((len(seqs), m), tok.pad_token_id, dtype=torch.long, device=dev)
    attn = torch.zeros_like(ids)
    for i, seq in enumerate(seqs): ids[i, :seq.numel()] = seq; attn[i, :seq.numel()] = 1
    return ids, attn, lens


def response_logits(model, ids, attn, prompt_lens, response_mask, grad):
    ctx = contextlib.nullcontext() if grad else torch.inference_mode()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
    b, rlen = response_mask.shape; vocab = logits.shape[-1]
    out = torch.empty((b, rlen, vocab), dtype=logits.dtype, device=logits.device)
    for i, plen in enumerate(prompt_lens):
        n = int(response_mask[i].sum())
        # logits at plen-1 predicts first response token.
        out[i, :n] = logits[i, plen - 1:plen - 1 + n]
        if n < rlen: out[i, n:] = 0
    return out


def generalized_jsd(student_logits, teacher_logits, mask, beta, temperature, top_k, token_clip):
    s = student_logits / temperature; t = teacher_logits / temperature
    if top_k and top_k > 0:
        idx = torch.topk(t, k=top_k, dim=-1).indices
        s, t = torch.gather(s, -1, idx), torch.gather(t, -1, idx)
    slp, tlp = F.log_softmax(s.float(), -1), F.log_softmax(t.float(), -1)
    if beta == 0:
        tok = F.kl_div(slp, tlp, reduction="none", log_target=True).sum(-1)
    elif beta == 1:
        tok = F.kl_div(tlp, slp, reduction="none", log_target=True).sum(-1)
    else:
        b = torch.tensor(beta, device=slp.device, dtype=slp.dtype)
        mix = torch.logsumexp(torch.stack([slp + torch.log1p(-b), tlp + torch.log(b)]), 0)
        kt = F.kl_div(mix, tlp, reduction="none", log_target=True).sum(-1)
        ks = F.kl_div(mix, slp, reduction="none", log_target=True).sum(-1)
        tok = b * kt + (1 - b) * ks
    if token_clip is not None: tok = tok.clamp(max=token_clip)
    return tok[mask].mean()


def right_subspace(dw, k, svd_device="cpu"):
    # SVD one matrix at a time off the training GPU to avoid a multi-GB randomized-SVD workspace.
    x = dw.float().to(svd_device)
    q = min(max(k + 8, 24), min(x.shape))
    _, _, v = torch.svd_lowrank(x, q=q, niter=4)
    return v[:, :min(k, v.shape[1])].contiguous()


def append_ortho(old, v):
    cat = v if old is None else torch.cat([old, v], dim=1)
    q, r = torch.linalg.qr(cat, mode="reduced")
    return q[:, r.diagonal().abs() > 1e-6].contiguous()


def project_grads(tp, memories):
    """Project gradients and return aggregate diagnostics (squared-norm weighted)."""
    pre2 = post2 = removed2 = residual2 = 0.0
    with torch.no_grad():
        for n, p in tp:
            m = memories.get(n)
            if p.grad is None or m is None:
                continue
            g = p.grad
            pre2 += float(g.float().square().sum())
            component = (g @ m) @ m.T
            removed2 += float(component.float().square().sum())
            g.sub_(component)
            post2 += float(g.float().square().sum())
            # ||g_projected M|| measures numerical non-orthogonality after projection.
            residual2 += float((g @ m).float().square().sum())
    pre = pre2 ** 0.5; post = post2 ** 0.5
    return {"pmopd_grad_pre_norm": pre, "pmopd_grad_post_norm": post,
            "pmopd_grad_removed_ratio": (removed2 ** 0.5) / max(pre, 1e-12),
            "pmopd_grad_orth_residual": residual2 ** 0.5}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mixed", "sequential", "pmopd"], default="pmopd",
                    help="training strategy; pmopd enables subspace protection")
    ap.add_argument("--task_order", nargs=3, choices=TASKS, default=["code", "reason", "math"],
                    metavar=("TASK1", "TASK2", "TASK3"),
                    help="sequential/PMOPD task order; the released recipe uses code reason math")
    ap.add_argument("--mixed_schedule",
                    choices=["sample_shuffle", "grouped_fixed", "grouped_random",
                             "within_batch_fixed", "within_batch_random"],
                    default="sample_shuffle",
                    help="full-data mixed scheduling: per-sample shuffle, homogeneous groups, "
                         "fixed one-per-task updates, or randomly mixed logical batches")
    ap.add_argument("--mixed_group_order", nargs=3, choices=TASKS,
                    default=["code", "reason", "math"],
                    metavar=("TASK1", "TASK2", "TASK3"),
                    help="task cycle for --mixed_schedule grouped_fixed; must contain each task exactly once")
    ap.add_argument("--output_dir", required=True,
                    help="directory for checkpoints, metrics, samples, and the training log")
    ap.add_argument("--base", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--teacher_math", required=True)
    ap.add_argument("--teacher_code", required=True)
    ap.add_argument("--teacher_reason", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--teacher_gpus", nargs=3, type=int, default=[1, 2, 3],
                    metavar=("MATH_GPU", "CODE_GPU", "REASON_GPU"))
    ap.add_argument("--steps", type=int, default=600, help="legacy fixed-budget optimizer steps; ignored with --full_data_epoch")
    ap.add_argument("--epochs", type=int, default=1, help="number of exact full-data epochs")
    ap.add_argument("--full_data_epoch", action="store_true", help="consume every unique distill row exactly once per epoch")
    ap.add_argument("--cycles", type=int, default=1,
                    help="repeat task_order this many times; PMOPD memory resets at each cycle boundary")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--subspace_rank", type=int, default=16,
                    help="number of right-singular directions retained after each task stage")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="generalized-JSD beta: 0=KL(teacher||student), 1=KL(student||teacher)")
    ap.add_argument("--kl_direction", choices=["forward", "reverse", "jsd"], default=None,
                    help="explicit auditable KL direction; forward forces beta=0, reverse forces beta=1")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k_loss", type=int, default=256)
    ap.add_argument("--token_clip", type=float, default=None)
    ap.add_argument("--max_prompt", type=int, default=768, help="math/reason prompt limit")
    ap.add_argument("--max_prompt_code", type=int, default=2048)
    ap.add_argument("--max_new", type=int, default=768)
    ap.add_argument("--save_every", type=int, default=100,
                    help="save every N optimizer steps; 0 disables step checkpoints")
    ap.add_argument("--save_every_samples", type=int, default=0,
                    help="save after every N ordinary training samples; batch=1 gives exact sample boundaries")
    ap.add_argument("--sample_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--no_stage_save", action="store_true",
                    help="skip intermediate stage snapshots while still saving checkpoints/final")
    ap.add_argument("--save_all_stages", action="store_true",
                    help="save a cycle-specific snapshot after every task stage, including the last task")
    ap.add_argument("--save_cycle_ends", action="store_true",
                    help="save one snapshot at the end of every full-data cycle (after the last task)")
    ap.add_argument("--pmopd_gradient_only", action="store_true",
                    help="PMOPD ablation: project raw gradients but disable Adafactor update projection")
    ap.add_argument("--keep_optimizer_state_across_cycles", action="store_true",
                    help="do not clear Adafactor state at a cycle boundary; task-internal switches still reset it")
    ap.add_argument("--svd_device", default="cpu", help="CPU is safest; a spare CUDA device is faster")
    args = ap.parse_args()
    if len(set(args.task_order)) != len(TASKS) or set(args.task_order) != set(TASKS):
        ap.error("--task_order must contain math, code, reason exactly once")
    if len(set(args.mixed_group_order)) != len(TASKS) or set(args.mixed_group_order) != set(TASKS):
        ap.error("--mixed_group_order must contain math, code, reason exactly once")
    if args.mixed_schedule != "sample_shuffle":
        if not args.full_data_epoch or args.mode != "mixed":
            ap.error("grouped mixed schedules require --mode mixed --full_data_epoch")
        if args.batch != 1:
            ap.error("grouped mixed schedules currently require --batch 1")
        if args.grad_accum < 1:
            ap.error("grouped mixed schedules require --grad_accum >= 1")
        if args.mixed_schedule == "within_batch_fixed" and args.grad_accum != len(TASKS):
            ap.error("--mixed_schedule within_batch_fixed requires --grad_accum 3")
        if args.mixed_schedule == "within_batch_random" and args.grad_accum < len(TASKS):
            ap.error("--mixed_schedule within_batch_random requires --grad_accum >= 3")
    if args.cycles < 1:
        ap.error("--cycles must be >= 1")
    if args.pmopd_gradient_only and args.mode != "pmopd":
        ap.error("--pmopd_gradient_only requires --mode pmopd")
    if args.epochs < 1:
        ap.error("--epochs must be >= 1")
    if args.full_data_epoch and args.mode == "mixed" and args.cycles != 1:
        ap.error("--cycles > 1 is only defined for sequential/PMOPD full-data curricula")
    if args.save_every_samples < 0:
        ap.error("--save_every_samples must be >= 0")
    if args.kl_direction == "forward":
        args.beta = 0.0
    elif args.kl_direction == "reverse":
        args.beta = 1.0
    elif args.kl_direction == "jsd" and not (0.0 < args.beta < 1.0):
        ap.error("--kl_direction jsd requires 0 < --beta < 1")
    if not (0.0 <= args.beta <= 1.0):
        ap.error("--beta must be in [0, 1]")
    effective_kl_direction = ("forward" if args.beta == 0 else "reverse" if args.beta == 1 else "jsd")
    if not args.full_data_epoch and args.mode != "mixed" and args.steps % (args.cycles * len(TASKS)) != 0:
        ap.error("--steps must be divisible by cycles * 3 for equal task segments")

    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sdev = f"cuda:{args.gpu}"
    tdevs = {t: f"cuda:{g}" for t, g in zip(TASKS, args.teacher_gpus)}
    out = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out, exist_ok=True)
    lf = open(os.path.join(out, "training.log"), "w", encoding="utf-8")
    json.dump({**vars(args), "effective_kl_direction": effective_kl_direction,
               "kl_formula": "KL(teacher||student)" if effective_kl_direction == "forward" else
                             "KL(student||teacher)" if effective_kl_direction == "reverse" else "generalized_jsd"},
              open(os.path.join(out, "training_config.json"), "w"), ensure_ascii=False, indent=2)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base); tok.padding_side = "left"
    tok.pad_token_id = tok.eos_token_id
    student = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").to(sdev)
    for n, p in student.named_parameters():
        if is_target(n): p.data = p.data.float(); p.requires_grad_(True)
        else: p.requires_grad_(False)
    student.gradient_checkpointing_enable(); student.config.use_cache = False
    tp = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    from .optimizer import Adafactor
    opt = Adafactor([p for _, p in tp], lr=args.lr, relative_step=False,
                    scale_parameter=False, warmup_init=False, weight_decay=0.0)

    teacher_paths = {"math": args.teacher_math, "code": args.teacher_code, "reason": args.teacher_reason}
    teachers = {}
    for task in TASKS:
        teacher = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa").to(tdevs[task]).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        teacher, load_detail = load_teacher_override(teacher, teacher_paths[task])
        for p in teacher.parameters(): p.requires_grad_(False)
        teachers[task] = teacher
        log(f"teacher={task} gpu={tdevs[task]} loaded {load_detail}", lf)

    rows = load_task_rows(args.data_dir, TASKS)
    for t in TASKS:
        ids = [str(r["id"]) for r in rows[t]]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate sample ids in {t}: rows={len(ids)} unique={len(set(ids))}")
    smp = {t: Sampler(rows[t], args.seed + i) for i, t in enumerate(TASKS)}
    coverage = {t: set() for t in TASKS}
    coverage_calls = {t: 0 for t in TASKS}
    epoch_orders = {}
    if args.full_data_epoch:
        for epoch in range(1, args.epochs + 1):
            epoch_orders[epoch] = {}
            for i, t in enumerate(TASKS):
                order = list(range(len(rows[t])))
                random.Random(args.seed + epoch * 1009 + i * 100003).shuffle(order)
                epoch_orders[epoch][t] = order
    log(f"mode={args.mode} steps={args.steps} batch={args.batch} accum={args.grad_accum} lr={args.lr} "
        f"beta={args.beta} kl_direction={effective_kl_direction} top_k={args.top_k_loss} subspace_rank={args.subspace_rank} max_new={args.max_new} "
        f"max_prompt={args.max_prompt} max_prompt_code={args.max_prompt_code} "
        f"task_order={','.join(args.task_order)} cycles={args.cycles} epochs={args.epochs} full_data_epoch={args.full_data_epoch} "
        f"mixed_schedule={args.mixed_schedule} mixed_group_order={','.join(args.mixed_group_order)} "
        f"memory_reset=cycle_boundary pmopd_gradient_only={args.pmopd_gradient_only} "
        f"keep_optimizer_state_across_cycles={args.keep_optimizer_state_across_cycles} "
        f"save_cycle_ends={args.save_cycle_ends} eos={tok.eos_token_id}", lf)
    metrics_f = open(os.path.join(out, "metrics.jsonl"), "w")
    samples_f = open(os.path.join(out, "rollout_samples.jsonl"), "w")
    rollout_lengths, rollout_eos, rollout_hit_max = [], [], []
    prompt_lengths, prompt_truncated = [], []

    gen_cfg = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                   max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id,
                   eos_token_id=tok.eos_token_id)

    def loss_for(task, batch_rows):
        for r in batch_rows:
            coverage[task].add(str(r["id"]))
            coverage_calls[task] += 1
        teacher, tdev = teachers[task], tdevs[task]
        prompt_limit = args.max_prompt_code if task == "code" else args.max_prompt
        raw_prompt_lens = [len(tok(r["prompt"], add_special_tokens=False)["input_ids"]) for r in batch_rows]
        prompt_lengths.extend(raw_prompt_lens)
        prompt_truncated.extend(x > prompt_limit for x in raw_prompt_lens)
        prompts = left_tokenize(tok, [r["prompt"] for r in batch_rows], prompt_limit, sdev)
        student.gradient_checkpointing_disable(); student.config.use_cache = True; student.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = student.generate(**prompts, **gen_cfg)
        sids, sattn, resp, rmask, splens = build_student_sequences(tok, prompts, gen, prompts["input_ids"].shape[1], sdev)
        tids, tattn, tplens = build_teacher_sequences(tok, batch_rows, resp.to(tdev), rmask.to(tdev), prompt_limit, tdev)
        lens = rmask.sum(1).tolist()
        ended = [bool((resp[i, :int(n)] == tok.eos_token_id).any()) for i, n in enumerate(lens)]
        hits = [int(n) >= args.max_new for n in lens]
        rollout_lengths.extend(int(n) for n in lens); rollout_eos.extend(ended); rollout_hit_max.extend(hits)
        if len(rollout_lengths) <= 3 * args.batch or (micro > 0 and micro % (args.sample_every * args.grad_accum) == 0):
            for i, n in enumerate(lens):
                text = tok.decode(resp[i, :int(n)], skip_special_tokens=True).strip()
                samples_f.write(json.dumps({"micro": micro, "task": task, "id": batch_rows[i]["id"],
                    "tokens": int(n), "ended_by_eos": ended[i], "hit_max": hits[i],
                    "prompt_tokens": raw_prompt_lens[i], "prompt_truncated": raw_prompt_lens[i] > prompt_limit,
                    "rollout": text}, ensure_ascii=False) + "\n")
            samples_f.flush()
        student.gradient_checkpointing_enable(); student.config.use_cache = False; student.train()
        slog = response_logits(student, sids, sattn, splens, rmask, grad=True)
        tlog = response_logits(teacher, tids, tattn, tplens, rmask.to(tdev), grad=False).to(sdev)
        loss = generalized_jsd(slog, tlog, rmask, args.beta, args.temperature,
                               args.top_k_loss, args.token_clip)
        del slog, tlog, prompts, gen, sids, sattn, tids, tattn, resp
        return loss

    memories = {n: None for n, _ in tp}
    t0 = time.time(); opt.zero_grad(set_to_none=True); micro = 0; global_step = 0
    current_cycle = 0

    saved_sample_marks = set()

    def record_step(task, loss, gn, pmopd_diag=None):
        recent_n = max(1, args.batch * args.grad_accum)
        rec = {"step": global_step, "micro": micro, "cycle": current_cycle,
               "task": task, "loss": float(loss), "gnorm": float(gn),
               "rollout_len_mean_recent": sum(rollout_lengths[-recent_n:]) / len(rollout_lengths[-recent_n:]),
               "rollout_len_max_recent": max(rollout_lengths[-recent_n:]),
               "eos_rate_recent": sum(rollout_eos[-recent_n:]) / len(rollout_eos[-recent_n:]),
               "hit_max_rate_recent": sum(rollout_hit_max[-recent_n:]) / len(rollout_hit_max[-recent_n:]),
               "prompt_trunc_rate_all": sum(prompt_truncated) / len(prompt_truncated)}
        if pmopd_diag:
            rec.update(pmopd_diag)
        metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
        if global_step <= 3 or global_step % 10 == 0: log(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), lf)
        if args.save_every and global_step % args.save_every == 0:
            snapshot(student, os.path.join(out, f"step_{global_step}"))
        ordinary_samples = sum(coverage_calls.values())
        if args.save_every_samples and ordinary_samples % args.save_every_samples == 0 \
                and ordinary_samples not in saved_sample_marks:
            sample_dir = os.path.join(out, f"sample_{ordinary_samples:06d}")
            snapshot(student, sample_dir)
            json.dump({"ordinary_samples": ordinary_samples, "optimizer_step": global_step,
                       "cycle": current_cycle, "task": task,
                       "kl_direction": effective_kl_direction, "beta": args.beta},
                      open(os.path.join(sample_dir, "checkpoint_meta.json"), "w"), indent=2)
            saved_sample_marks.add(ordinary_samples)
            log(f"sample_checkpoint samples={ordinary_samples} step={global_step} path={sample_dir}", lf)

    def train_segment(task, steps, protect, remember, exact_rows=None, epoch=None,
                      reset_optimizer_state=True):
        nonlocal micro, global_step
        start = {n: p.detach().cpu().clone() for n, p in tp} if remember else None
        if global_step > 0 and reset_optimizer_state:
            opt.state.clear()
            opt.zero_grad(set_to_none=True)
            log(f"optimizer_state_reset before task={task} step={global_step}", lf)
        elif global_step > 0:
            # This branch is used only at a new full-data cycle when the explicit
            # ablation keeps Adafactor's state continuous across cycles.
            opt.zero_grad(set_to_none=True)
            log(f"optimizer_state_kept across_cycle before task={task} step={global_step}", lf)
        local_micro = 0
        accum_count = 0
        pos = 0
        while (pos < len(exact_rows)) if exact_rows is not None else (global_step < steps[1]):
            if exact_rows is not None:
                batch_rows = exact_rows[pos:pos + args.batch]
                pos += len(batch_rows)
            else:
                batch_rows = smp[task].next(args.batch)
            loss = loss_for(task, batch_rows)
            (loss / args.grad_accum).backward(); micro += 1; local_micro += 1; accum_count += 1
            boundary = accum_count == args.grad_accum
            final_partial = exact_rows is not None and pos == len(exact_rows) and not boundary
            ordinary_samples = sum(coverage_calls.values())
            sample_boundary = bool(args.save_every_samples and
                                   ordinary_samples % args.save_every_samples == 0 and not boundary)
            if boundary or final_partial or sample_boundary:
                # Finish a short accumulation at task ends or exact sample-checkpoint boundaries.
                if accum_count < args.grad_accum:
                    scale = args.grad_accum / accum_count
                    for _, p in tp:
                        if p.grad is not None: p.grad.mul_(scale)
                gdiag = project_grads(tp, memories) if protect else None
                gn = torch.nn.utils.clip_grad_norm_([p for _, p in tp], 1.0)
                # Full PMOPD projects both raw gradients and optimizer updates.  The
                # gradient-only ablation retains the former but explicitly clears every
                # Adafactor update projector before the step.
                if protect and not args.pmopd_gradient_only:
                    opt.set_update_projectors({p: memories[n] for n, p in tp if memories[n] is not None})
                else:
                    opt.set_update_projectors({})
                opt.step()
                if protect and not args.pmopd_gradient_only and opt.last_projection_stats:
                    gdiag.update(opt.last_projection_stats)
                opt.zero_grad(set_to_none=True); global_step += 1; accum_count = 0
                record_step(task, loss.item(), gn, gdiag)
        if exact_rows is not None:
            log(f"full_data_stage epoch={epoch} task={task} calls={len(exact_rows)} unique_total={len(coverage[task])}/{len(rows[task])} optimizer_steps={global_step}", lf)
        stage_dir = os.path.join(
            out,
            f"cycle_{current_cycle:02d}_stage_{task}" if args.save_all_stages
            else (f"epoch_{epoch:02d}_stage_{task}" if epoch else f"cycle_{current_cycle:02d}_stage_{task}"),
        )
        if (args.save_all_stages or remember) and not args.no_save and not args.no_stage_save:
            snapshot(student, stage_dir)
            json.dump({"cycle": current_cycle, "epoch": epoch, "task": task,
                       "optimizer_step": global_step,
                       "ordinary_samples": sum(coverage_calls.values())},
                      open(os.path.join(stage_dir, "checkpoint_meta.json"), "w"), indent=2)
            log(f"stage_checkpoint cycle={current_cycle} task={task} step={global_step} path={stage_dir}", lf)
        if remember:
            log(f"extracting PMOPD memory after {task}", lf)
            with torch.no_grad():
                for n, p in tp:
                    dw = p.detach() - start[n].to(sdev)
                    v = right_subspace(dw, args.subspace_rank, args.svd_device).to(sdev)
                    memories[n] = append_ortho(memories[n], v)
            dims = [memories[n].shape[1] for n, _ in tp if memories[n] is not None]
            log(f"memory_dim_min={min(dims)} max={max(dims)} stage_snapshot={stage_dir}", lf)

    if args.full_data_epoch:
        if args.mode == "mixed":
            for epoch in range(1, args.epochs + 1):
                current_cycle = epoch
                if args.mixed_schedule == "sample_shuffle":
                    stream = [(t, idx) for t in TASKS for idx in epoch_orders[epoch][t]]
                    random.Random(args.seed + epoch * 1000003).shuffle(stream)
                elif args.mixed_schedule == "within_batch_fixed":
                    # A logical/effective batch is routed as three batch=1 microbatches,
                    # one per task, followed by exactly one optimizer update. This avoids
                    # holding three 7B-model activation graphs simultaneously while giving
                    # the same equally weighted task loss at the update boundary.
                    lengths = {t: len(epoch_orders[epoch][t]) for t in TASKS}
                    if len(set(lengths.values())) != 1:
                        raise RuntimeError(f"within-batch fixed mixing requires equal task sizes: {lengths}")
                    if args.grad_accum != len(args.mixed_group_order):
                        raise RuntimeError(
                            f"within-batch fixed mixing requires grad_accum={len(args.mixed_group_order)}")
                    stream = []
                    schedule_audit = []
                    for step_in_epoch in range(1, next(iter(lengths.values())) + 1):
                        group = []
                        for t in args.mixed_group_order:
                            idx = epoch_orders[epoch][t][step_in_epoch - 1]
                            stream.append((t, idx))
                            group.append({"task": t, "sample_id": str(rows[t][idx]["id"])})
                        schedule_audit.append({
                            "optimizer_step_in_epoch": step_in_epoch,
                            "task_sequence": list(args.mixed_group_order),
                            "samples": group,
                        })
                    schedule_path = os.path.join(out, f"mixed_schedule_epoch_{epoch:02d}.json")
                    json.dump({
                        "mixed_schedule": args.mixed_schedule,
                        "mixed_group_order": args.mixed_group_order,
                        "physical_microbatch_size": args.batch,
                        "gradient_accumulation_steps": args.grad_accum,
                        "logical_batch_size": args.batch * args.grad_accum,
                        "optimizer_steps": len(schedule_audit),
                        "samples_per_task": lengths,
                        "groups": schedule_audit,
                    }, open(schedule_path, "w"), ensure_ascii=False, indent=2)
                    log(f"mixed_schedule_built epoch={epoch} type={args.mixed_schedule} "
                        f"groups={len(schedule_audit)} logical_batch="
                        f"{','.join(args.mixed_group_order)} audit={schedule_path}", lf)
                elif args.mixed_schedule == "within_batch_random":
                    # Build random logical batches while keeping physical batch=1 so that
                    # samples can be routed to different task teachers without retaining
                    # multiple 7B activation graphs. Each full logical batch contains
                    # grad_accum samples; the final batch contains the exact-data remainder.
                    # We explicitly require every logical batch (including the remainder)
                    # to contain all tasks, randomize sample order inside each batch, and
                    # independently randomize batch order.
                    base_stream = [(t, idx) for t in TASKS for idx in epoch_orders[epoch][t]]
                    capacities = [args.grad_accum] * (len(base_stream) // args.grad_accum)
                    if len(base_stream) % args.grad_accum:
                        capacities.append(len(base_stream) % args.grad_accum)
                    if capacities and min(capacities) < len(TASKS):
                        raise RuntimeError(
                            f"within-batch random mixing cannot put all tasks in final batch: "
                            f"total={len(base_stream)} logical_batch={args.grad_accum} "
                            f"final_batch={capacities[-1]}")

                    groups = None
                    shuffle_attempt = None
                    for attempt in range(10000):
                        rng = random.Random(args.seed + epoch * 1000003 + attempt * 9176)
                        candidate = list(base_stream)
                        rng.shuffle(candidate)
                        built = []
                        p = 0
                        for cap in capacities:
                            group = candidate[p:p + cap]
                            p += cap
                            # Explicit second shuffle documents/randomizes within-batch order.
                            rng.shuffle(group)
                            built.append(group)
                        if all(set(t for t, _ in group) == set(TASKS) for group in built):
                            groups = built
                            shuffle_attempt = attempt
                            break
                    if groups is None:
                        raise RuntimeError(
                            "failed to construct random logical batches containing all tasks")

                    # Batch-to-batch order is independently randomized after construction.
                    # Keep the standard short remainder batch last so accumulation boundaries
                    # remain aligned with logical-batch boundaries.
                    batch_rng = random.Random(args.seed + epoch * 1000003 + 700000001)
                    if groups and len(groups[-1]) < args.grad_accum:
                        partial_group = groups[-1]
                        full_groups = groups[:-1]
                        batch_rng.shuffle(full_groups)
                        groups = full_groups + [partial_group]
                    else:
                        batch_rng.shuffle(groups)
                    stream = [item for group in groups for item in group]
                    schedule_audit = []
                    for step_in_epoch, group in enumerate(groups, 1):
                        counts = {t: sum(gt == t for gt, _ in group) for t in TASKS}
                        schedule_audit.append({
                            "optimizer_step_in_epoch": step_in_epoch,
                            "logical_batch_size": len(group),
                            "task_counts": counts,
                            "task_sequence": [t for t, _ in group],
                            "samples": [
                                {"task": t, "sample_id": str(rows[t][idx]["id"])}
                                for t, idx in group
                            ],
                        })
                    schedule_path = os.path.join(out, f"mixed_schedule_epoch_{epoch:02d}.json")
                    json.dump({
                        "mixed_schedule": args.mixed_schedule,
                        "physical_microbatch_size": args.batch,
                        "gradient_accumulation_steps": args.grad_accum,
                        "requested_logical_batch_size": args.batch * args.grad_accum,
                        "optimizer_steps": len(schedule_audit),
                        "full_logical_batches": sum(len(g) == args.grad_accum for g in groups),
                        "final_partial_batch_size": (len(base_stream) % args.grad_accum),
                        "shuffle_attempt": shuffle_attempt,
                        "within_batch_order": "random",
                        "between_batch_order": "random_full_batches_then_final_remainder",
                        "batch_order_seed": args.seed + epoch * 1000003 + 700000001,
                        "samples_per_task": {t: len(epoch_orders[epoch][t]) for t in TASKS},
                        "groups": schedule_audit,
                    }, open(schedule_path, "w"), ensure_ascii=False, indent=2)
                    log(f"mixed_schedule_built epoch={epoch} type={args.mixed_schedule} "
                        f"groups={len(schedule_audit)} requested_logical_batch={args.grad_accum} "
                        f"final_batch={len(groups[-1]) if groups else 0} "
                        f"shuffle_attempt={shuffle_attempt} audit={schedule_path}", lf)
                else:
                    # One optimizer update contains grad_accum physical batch=1 microbatches.
                    # Keep all microbatches in an update task-homogeneous, while preserving
                    # the independently shuffled per-task sample order and exact coverage.
                    chunks = {}
                    for t in TASKS:
                        order = epoch_orders[epoch][t]
                        if len(order) % args.grad_accum:
                            raise RuntimeError(
                                f"grouped mixed schedule requires task rows divisible by grad_accum: "
                                f"task={t} rows={len(order)} grad_accum={args.grad_accum}")
                        chunks[t] = [order[i:i + args.grad_accum]
                                     for i in range(0, len(order), args.grad_accum)]
                    if args.mixed_schedule == "grouped_fixed":
                        task_groups = []
                        max_groups = max(len(chunks[t]) for t in TASKS)
                        for group_idx in range(max_groups):
                            for t in args.mixed_group_order:
                                if group_idx < len(chunks[t]):
                                    task_groups.append(t)
                    else:
                        task_groups = [t for t in TASKS for _ in chunks[t]]
                        random.Random(args.seed + epoch * 1000003).shuffle(task_groups)
                    used_groups = {t: 0 for t in TASKS}
                    stream = []
                    schedule_audit = []
                    for step_in_epoch, t in enumerate(task_groups, 1):
                        idxs = chunks[t][used_groups[t]]
                        used_groups[t] += 1
                        stream.extend((t, idx) for idx in idxs)
                        schedule_audit.append({
                            "optimizer_step_in_epoch": step_in_epoch,
                            "task": t,
                            "sample_ids": [str(rows[t][idx]["id"]) for idx in idxs],
                        })
                    assert all(used_groups[t] == len(chunks[t]) for t in TASKS)
                    schedule_path = os.path.join(out, f"mixed_schedule_epoch_{epoch:02d}.json")
                    json.dump({
                        "mixed_schedule": args.mixed_schedule,
                        "mixed_group_order": args.mixed_group_order,
                        "physical_batch_size": args.batch,
                        "gradient_accumulation_steps": args.grad_accum,
                        "samples_per_optimizer_step": args.batch * args.grad_accum,
                        "task_group_counts": {t: len(chunks[t]) for t in TASKS},
                        "groups": schedule_audit,
                    }, open(schedule_path, "w"), ensure_ascii=False, indent=2)
                    log(f"mixed_schedule_built epoch={epoch} type={args.mixed_schedule} "
                        f"groups={len(task_groups)} group_size={args.grad_accum} "
                        f"counts=" + json.dumps({t: len(chunks[t]) for t in TASKS}) +
                        f" audit={schedule_path}", lf)
                local_micro = 0
                update_losses = []
                update_tasks = []
                pos = 0
                while pos < len(stream):
                    # batch=1 is the formal comparable setting; grouped same-task batches are also supported.
                    task, idx = stream[pos]
                    batch_rows = [rows[task][idx]]
                    pos += 1
                    while len(batch_rows) < args.batch and pos < len(stream) and stream[pos][0] == task:
                        _, idx = stream[pos]; batch_rows.append(rows[task][idx]); pos += 1
                    loss = loss_for(task, batch_rows)
                    update_losses.append(float(loss.item()))
                    update_tasks.append(task)
                    (loss / args.grad_accum).backward(); micro += 1; local_micro += 1
                    boundary = local_micro % args.grad_accum == 0
                    final_partial = pos == len(stream) and not boundary
                    if boundary or final_partial:
                        if final_partial:
                            scale = args.grad_accum / (local_micro % args.grad_accum)
                            for _, p in tp:
                                if p.grad is not None: p.grad.mul_(scale)
                        gn = torch.nn.utils.clip_grad_norm_([p for _, p in tp], 1.0)
                        opt.step(); opt.zero_grad(set_to_none=True); global_step += 1
                        if args.mixed_schedule == "within_batch_fixed":
                            step_task = "+".join(update_tasks)
                        elif args.mixed_schedule == "within_batch_random":
                            step_task = "mixed_random"
                        else:
                            step_task = task
                        step_loss = sum(update_losses) / len(update_losses)
                        record_step(step_task, step_loss, gn)
                        update_losses.clear(); update_tasks.clear()
                log(f"full_data_epoch_complete epoch={epoch} calls={sum(coverage_calls.values())} coverage=" + json.dumps({t:f"{len(coverage[t])}/{len(rows[t])}" for t in TASKS}), lf)
        else:
            use_pmopd = args.mode == "pmopd"
            for epoch in range(1, args.epochs + 1):
                # Split each task's shuffled full-data order into disjoint, near-equal cycle shards.
                # Remainders go to earlier cycles, e.g. Math 3737 -> 1869 + 1868 for two cycles.
                cycle_shards = {t: [] for t in TASKS}
                for task in TASKS:
                    order = epoch_orders[epoch][task]
                    q, r = divmod(len(order), args.cycles)
                    start_pos = 0
                    for c in range(args.cycles):
                        size = q + (1 if c < r else 0)
                        cycle_shards[task].append(order[start_pos:start_pos + size])
                        start_pos += size
                    assert start_pos == len(order)
                for cycle_idx in range(args.cycles):
                    current_cycle = (epoch - 1) * args.cycles + cycle_idx + 1
                    if use_pmopd:
                        for n in memories: memories[n] = None
                        log(f"pmopd_memory_reset epoch={epoch}/{args.epochs} cycle={cycle_idx+1}/{args.cycles} step={global_step}", lf)
                    sizes = {t: len(cycle_shards[t][cycle_idx]) for t in TASKS}
                    log(f"full_data_cycle_start epoch={epoch} cycle={cycle_idx+1}/{args.cycles} shard_sizes=" + json.dumps(sizes), lf)
                    for i, task in enumerate(args.task_order):
                        exact = [rows[task][j] for j in cycle_shards[task][cycle_idx]]
                        cycle_boundary = cycle_idx > 0 and i == 0
                        reset_opt = not (args.keep_optimizer_state_across_cycles and cycle_boundary)
                        train_segment(task, (global_step, global_step),
                                      protect=use_pmopd and i > 0,
                                      remember=use_pmopd and i < len(args.task_order)-1,
                                      exact_rows=exact, epoch=epoch,
                                      reset_optimizer_state=reset_opt)
                    if args.save_cycle_ends and not args.no_save:
                        cycle_dir = os.path.join(out, f"cycle_{current_cycle:02d}_end")
                        snapshot(student, cycle_dir)
                        json.dump({"cycle": current_cycle, "epoch": epoch,
                                   "task": args.task_order[-1], "optimizer_step": global_step,
                                   "ordinary_samples": sum(coverage_calls.values()),
                                   "checkpoint_kind": "cycle_end",
                                   "mode": args.mode,
                                   "pmopd_gradient_only": args.pmopd_gradient_only},
                                  open(os.path.join(cycle_dir, "checkpoint_meta.json"), "w"), indent=2)
                        log(f"cycle_end_checkpoint cycle={current_cycle} step={global_step} path={cycle_dir}", lf)
                    log(f"full_data_cycle_complete epoch={epoch} cycle={cycle_idx+1}/{args.cycles} calls={sum(coverage_calls.values())} coverage=" + json.dumps({t:f"{len(coverage[t])}/{len(rows[t])}" for t in TASKS}), lf)
                log(f"full_data_epoch_complete epoch={epoch} cycles={args.cycles} calls={sum(coverage_calls.values())} coverage=" + json.dumps({t:f"{len(coverage[t])}/{len(rows[t])}" for t in TASKS}), lf)
    elif args.mode == "mixed":
        while global_step < args.steps:
            task = TASKS[micro % len(TASKS)]
            loss = loss_for(task, smp[task].next(args.batch))
            (loss / args.grad_accum).backward(); micro += 1
            if micro % args.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_([p for _, p in tp], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); global_step += 1
                record_step(task, loss.item(), gn)
    else:
        segment_steps = args.steps // (args.cycles * len(TASKS))
        use_pmopd = args.mode == "pmopd"
        for cycle_idx in range(args.cycles):
            current_cycle = cycle_idx + 1
            if use_pmopd:
                for n in memories: memories[n] = None
                log(f"pmopd_memory_reset cycle={current_cycle}/{args.cycles} step={global_step}", lf)
            for i, task in enumerate(args.task_order):
                end_step = global_step + segment_steps
                cycle_boundary = cycle_idx > 0 and i == 0
                reset_opt = not (args.keep_optimizer_state_across_cycles and cycle_boundary)
                train_segment(task, (global_step, end_step), protect=use_pmopd and i > 0,
                              remember=use_pmopd and i < len(args.task_order)-1,
                              reset_optimizer_state=reset_opt)

    import numpy as np
    q = np.percentile(rollout_lengths, [50, 90, 95, 99]).tolist()
    summary = {"n": len(rollout_lengths), "mean_len": float(np.mean(rollout_lengths)),
               "p50": q[0], "p90": q[1], "p95": q[2], "p99": q[3], "max_len": max(rollout_lengths),
               "eos_rate": float(np.mean(rollout_eos)), "hit_max_rate": float(np.mean(rollout_hit_max)),
               "prompt_mean_len": float(np.mean(prompt_lengths)), "prompt_max_len": max(prompt_lengths),
               "prompt_trunc_rate": float(np.mean(prompt_truncated))}
    coverage_summary = {"full_data_epoch": args.full_data_epoch, "epochs": args.epochs,
        "cycles_per_epoch": args.cycles,
        "expected_per_epoch": {t: len(rows[t]) for t in TASKS},
        "ordinary_calls": coverage_calls, "unique_ids": {t: len(coverage[t]) for t in TASKS},
        "coverage": {t: len(coverage[t]) / len(rows[t]) for t in TASKS},
        "total_expected_per_epoch": sum(len(rows[t]) for t in TASKS),
        "total_calls": sum(coverage_calls.values()), "total_unique": sum(len(coverage[t]) for t in TASKS)}
    if args.full_data_epoch:
        expected_calls = {t: len(rows[t]) * args.epochs for t in TASKS}
        if coverage_calls != expected_calls or any(len(coverage[t]) != len(rows[t]) for t in TASKS):
            raise RuntimeError(f"full-data coverage validation failed: {coverage_summary}")
    json.dump(coverage_summary, open(os.path.join(out, "coverage_summary.json"), "w"), indent=2)
    log("coverage_summary=" + json.dumps(coverage_summary), lf)
    json.dump(summary, open(os.path.join(out, "rollout_summary.json"), "w"), indent=2)
    log("rollout_summary=" + json.dumps(summary), lf)
    if not args.no_save:
        snapshot(student, f"{out}/final")
    log(f"done in {(time.time()-t0)/60:.1f} min steps={global_step}", lf)


if __name__ == "__main__":
    main()
