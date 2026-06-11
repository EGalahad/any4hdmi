from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from any4hdmi.limmt.common import DEFAULT_OUTPUT_ROOT, resolve_dataset_root, write_json
from any4hdmi.limmt.hme import (
    CachedHmeWindowDataset,
    HME_FEATURE_TYPE,
    PeriodicAutoencoder,
    build_hme_feature_cache,
    wait_for_hme_feature_cache,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LIMMT HME/PeriodicAutoencoder on an any4hdmi dataset.")
    parser.add_argument("--input-root", default=str(DEFAULT_OUTPUT_ROOT / "amass_limmt_pass"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "hme"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--win-sec", type=float, default=4.0)
    parser.add_argument("--phase-dim", type=int, default=8)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--downsample-rate", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cache-dir", default=None, help="Feature cache directory. Defaults to <output-dir>/feature_cache.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild cached joint_pos/joint_vel/root_pose/root_vel features before training.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("none", "cosine", "onecycle"), default="none")
    parser.add_argument("--max-lr", type=float, default=None, help="Peak LR for onecycle. Defaults to --lr.")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Final LR for cosine.")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.15)
    parser.add_argument("--onecycle-div-factor", type=float, default=10.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=100.0)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _ddp_state() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        dist.init_process_group(backend="gloo")
    return True, int(os.environ["RANK"]), local_rank, dist.get_world_size()


def _build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer, *, steps_per_epoch: int):
    total_steps = max(1, int(args.epochs) * max(1, int(steps_per_epoch)))
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(args.min_lr),
        )
    if args.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(args.max_lr if args.max_lr is not None else args.lr),
            total_steps=total_steps,
            pct_start=float(args.onecycle_pct_start),
            div_factor=float(args.onecycle_div_factor),
            final_div_factor=float(args.onecycle_final_div_factor),
        )
    raise ValueError(f"Unsupported scheduler {args.scheduler!r}")


def main() -> None:
    args = _parse_args()
    is_ddp, rank, local_rank, world_size = _ddp_state()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    input_root = resolve_dataset_root(args.input_root)
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else output_dir / "feature_cache"
    if rank == 0:
        if args.rebuild_cache and cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)
        if args.rebuild_cache or not (cache_dir / "index.json").is_file():
            build_hme_feature_cache(
                dataset_root=input_root,
                cache_dir=cache_dir,
                win_sec=args.win_sec,
                downsample_rate=args.downsample_rate,
                stride=args.stride,
            )
    if is_ddp:
        dist.barrier()
    else:
        wait_for_hme_feature_cache(cache_dir)
    if rank != 0:
        wait_for_hme_feature_cache(cache_dir)
    dataset = CachedHmeWindowDataset(cache_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    sample = dataset[0]
    model = PeriodicAutoencoder(
        inp_ch=int(sample.shape[-1]),
        latent_ch=args.phase_dim,
        win_len=dataset.win_len,
        hidden_dims=tuple(int(dim) for dim in args.hidden_dims),
        win_sec=args.win_sec,
    ).to(device)
    train_model = DistributedDataParallel(model, device_ids=[local_rank]) if is_ddp and device.type == "cuda" else model
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(args, optimizer, steps_per_epoch=len(loader))
    losses: list[dict[str, float | int]] = []

    if rank == 0:
        write_json(
            output_dir / "train_config.json",
            {
                **vars(args),
                "input_root": str(input_root),
                "cache_dir": str(cache_dir),
                "state_dim": int(sample.shape[-1]),
                "feature_type": HME_FEATURE_TYPE,
                "win_len": int(dataset.win_len),
                "num_windows": len(dataset),
                "world_size": world_size,
                "normalization": {
                    "type": "empirical_mean_std",
                    "mean": dataset.feature_mean.tolist(),
                    "std": dataset.feature_std.tolist(),
                },
            },
        )

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_model.train()
        total = 0.0
        count = 0
        for batch in loader:
            inp = batch.permute(0, 2, 1).to(device=device, dtype=torch.float32, non_blocking=True)
            out = train_model(inp)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 10.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach().item())
            count += 1
        if is_ddp:
            loss_stats = torch.tensor([total, float(count)], device=device, dtype=torch.float64)
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
            avg_loss = float((loss_stats[0] / loss_stats[1].clamp_min(1.0)).item())
        else:
            avg_loss = total / max(count, 1)
        if rank == 0:
            lr_now = float(optimizer.param_groups[0]["lr"])
            losses.append({"epoch": epoch, "loss": avg_loss, "lr": lr_now})
            print(f"epoch={epoch} loss={avg_loss:.6f} lr={lr_now:.8g}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": int(sample.shape[-1]),
                    "feature_type": HME_FEATURE_TYPE,
                    "phase_dim": args.phase_dim,
                    "hidden_dims": [int(dim) for dim in args.hidden_dims],
                    "win_len": dataset.win_len,
                    "win_sec": args.win_sec,
                    "downsample_rate": args.downsample_rate,
                    "feature_normalization": {
                        "type": "empirical_mean_std",
                        "mean": dataset.feature_mean.tolist(),
                        "std": dataset.feature_std.tolist(),
                    },
                },
                output_dir / "hme.pt",
            )
            (output_dir / "loss.json").write_text(json.dumps(losses, indent=2) + "\n", encoding="utf-8")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
