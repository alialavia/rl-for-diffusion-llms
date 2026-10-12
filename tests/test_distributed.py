"""Distributed property: DDP keeps replicas identical (2 gloo ranks on CPU)."""
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from espo.toy import TinyDiffusionLM, ToyPolicy, ToyTask
        from espo.trainer import ESPOConfig, ESPOTrainer

        torch.manual_seed(0)  # identical init on every rank (DDP also broadcasts it)
        task = ToyTask(vocab_size=5, prompt_length=2, seed=100 + rank)  # different data per rank
        model = TinyDiffusionLM(task.vocab_size, max_len=8, d_model=16, n_layers=1, n_heads=2)
        policy = ToyPolicy(model, task.mask_id, ddp=True)
        cfg = ESPOConfig(num_generations=4, prompts_per_rank=2, micro_batch_size=3, num_iterations=2,
                         num_mc=2, beta=0.01, gen_length=6, block_length=3, diffusion_steps=6,
                         temperature=1.0, generation_batch_size=8, seed=0)
        opt = torch.optim.AdamW(policy.trainable_parameters(), lr=1e-2)

        def all_reduce(t):
            dist.all_reduce(t)
            return t

        tr = ESPOTrainer(policy, task, opt, cfg, rank=rank, world_size=world, all_reduce=all_reduce)
        logs = tr.train(num_steps=4)
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered = [torch.zeros_like(flat) for _ in range(world)]
        dist.all_gather(gathered, flat)
        q.put((rank, all(torch.equal(gathered[0], g) for g in gathered), logs[-1]["reward"]))
    finally:
        dist.destroy_process_group()


def test_DDP_replicas_stay_identical():
    """DDP. With different prompts/rollouts per rank, trainable parameters remain bitwise
    identical across ranks after several ESPO steps (gradient all-reduce with several
    forwards per backward and no_sync accumulation).  Oracle: rank-0 parameters.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = [q.get(timeout=300) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert all(same for _, same, _ in results)
