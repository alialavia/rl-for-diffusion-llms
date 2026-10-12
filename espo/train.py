"""ESPO training entry point.

Single node, 8 GPUs:
    torchrun --standalone --nproc_per_node=8 -m espo.train --config configs/sudoku_llada.yaml

Override any config value with dotted keys:
    ... --set espo.beta=0.02 train.max_steps=2000 model.gradient_checkpointing=true
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time

import torch
import yaml

from .model import HFPolicy, ModelArgs, load_model_and_tokenizer
from .trainer import ESPOConfig, ESPOTrainer


@dataclasses.dataclass
class OptimArgs:
    learning_rate: float = 3e-6
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    warmup_ratio: float = 0.001   # constant schedule with linear warmup


@dataclasses.dataclass
class TrainArgs:
    task: str = "sudoku"
    output_dir: str = "outputs/run"
    max_steps: int = 10000        # optimizer steps (= rollouts * mu)
    save_steps: int = 500
    log_completions_every: int = 100
    max_prompt_length: int = 400
    n_synthetic: int = 20000
    data_seed: int = 0
    train_file: str | None = None  # optional dataset file (sudoku csv / countdown jsonl)
    countdown_source: str = "hf"   # "hf" (Jiayi-Pan/Countdown-Tasks-3to4) or "synthetic"
    resume_from: str | None = None
    wandb_project: str | None = None
    tf32: bool = True


def _coerce(value: str):
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def load_config(path: str | None, overrides: list[str]):
    raw = {}
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        section, _, name = key.partition(".")
        raw.setdefault(section, {})[name] = _coerce(val)
    unknown = set(raw) - {"model", "espo", "optim", "train"}
    if unknown:
        raise ValueError(f"unknown config sections {unknown}")
    return (ModelArgs(**raw.get("model", {})), ESPOConfig(**raw.get("espo", {})),
            OptimArgs(**raw.get("optim", {})), TrainArgs(**raw.get("train", {})))


def setup_distributed():
    if "RANK" not in os.environ:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return 0, 1, 0, torch.device(dev)
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    from datetime import timedelta

    # long timeout: rank 0 downloads the model/datasets while the others wait at a barrier
    dist.init_process_group(backend, timeout=timedelta(hours=2))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return dist.get_rank(), dist.get_world_size(), local_rank, device


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. espo.beta=0.02")
    a = ap.parse_args(argv)
    margs, ecfg, oargs, targs = load_config(a.config, a.set)

    rank, world, local_rank, device = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(ecfg.seed + rank)
    if targs.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(targs.output_dir, exist_ok=True)
    if is_main:
        with open(os.path.join(targs.output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump({k: dataclasses.asdict(v) for k, v in
                            zip(("model", "espo", "optim", "train"), (margs, ecfg, oargs, targs))}, f)

    # ---------------------------------------------------------------- data
    from .data import load_rows_from_file, load_task_rows
    from .tasks import TextTask

    def barrier():
        if world > 1:
            import torch.distributed as dist

            dist.barrier()

    # rank 0 downloads / builds caches first, the other ranks then read them
    if rank != 0:
        barrier()
    if targs.train_file:
        rows = load_rows_from_file(targs.task, targs.train_file)
    else:
        rows = load_task_rows(targs.task, "train", seed=targs.data_seed, n_synthetic=targs.n_synthetic,
                              countdown_source=targs.countdown_source)

    # ---------------------------------------------------------------- model
    if targs.resume_from:
        margs.adapter_path = targs.resume_from
    model, tok, family, mask_id, ref_model = load_model_and_tokenizer(
        margs, device, trainable=True, need_ref=ecfg.beta > 0)
    if rank == 0:
        barrier()
    ddp = None
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        ddp = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                  find_unused_parameters=False)
    policy = HFPolicy(model, family, mask_id, ddp_model=ddp, ref_model=ref_model)
    task = TextTask(targs.task, rows, tok, max_prompt_length=targs.max_prompt_length,
                    seed=targs.data_seed, rank=rank, world_size=world)

    params = policy.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=oargs.learning_rate, weight_decay=oargs.weight_decay,
                            betas=(oargs.adam_beta1, oargs.adam_beta2))
    warmup = max(1, math.ceil(oargs.warmup_ratio * targs.max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))

    all_reduce = None
    if world > 1:
        import torch.distributed as dist

        def all_reduce(t):
            dist.all_reduce(t)
            return t

    wandb_run = None
    if is_main and targs.wandb_project:
        import wandb

        wandb_run = wandb.init(project=targs.wandb_project, name=os.path.basename(targs.output_dir),
                               config={"model": dataclasses.asdict(margs), "espo": dataclasses.asdict(ecfg),
                                       "optim": dataclasses.asdict(oargs), "train": dataclasses.asdict(targs)})
    metrics_path = os.path.join(targs.output_dir, "metrics.jsonl")
    t_start = time.time()

    def log_fn(stats):
        if not is_main:
            return
        stats = dict(stats, elapsed_h=(time.time() - t_start) / 3600)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in stats.items()}),
              flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        if wandb_run is not None:
            wandb_run.log(stats, step=stats["step"])

    trainer = ESPOTrainer(policy, task, opt, ecfg, scheduler=sched, device=device, rank=rank,
                          world_size=world, all_reduce=all_reduce, log_fn=log_fn)

    if targs.resume_from:
        state = torch.load(os.path.join(targs.resume_from, "trainer_state.pt"), map_location="cpu",
                           weights_only=False)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        trainer.global_step = state["global_step"]
        # round down to the start of a rollout phase so masks/old ELBOs are consistent
        trainer.global_step -= trainer.global_step % ecfg.num_iterations
        task.load_state_dict(state["task"][rank] if rank < len(state["task"]) else state["task"][0])

    orig_make_rollout = trainer.make_rollout

    def make_rollout_and_log():
        ro = orig_make_rollout()
        if is_main and ro.texts and trainer.num_rollouts % max(1, targs.log_completions_every // ecfg.num_iterations) == 1:
            with open(os.path.join(targs.output_dir, "completions.jsonl"), "a") as f:
                for t, r in zip(ro.texts[: ecfg.num_generations], ro.rewards.tolist()):
                    f.write(json.dumps({"step": trainer.global_step, "reward": r, "completion": t}) + "\n")
        return ro

    trainer.make_rollout = make_rollout_and_log

    def save(step):
        task_states = [task.state_dict()]
        if world > 1:
            import torch.distributed as dist

            gathered = [None] * world
            dist.all_gather_object(gathered, task.state_dict())
            task_states = gathered
        if not is_main:
            return
        out = os.path.join(targs.output_dir, f"checkpoint-{step}")
        model.save_pretrained(out)
        tok.save_pretrained(out)
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "global_step": step, "task": task_states}, os.path.join(out, "trainer_state.pt"))
        print(f"saved {out}", flush=True)

    def on_step(tr, stats):
        if stats["step"] % targs.save_steps == 0 or stats["step"] == targs.max_steps:
            save(stats["step"])

    trainer.train(targs.max_steps, on_step=on_step)
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
