from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

import torch
import tyro
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from any4hdmi.limmt.common import project_hme_root, project_pass_root, resolve_project_root, write_json
from any4hdmi.limmt.hme.features import HME_FEATURE_TYPE
from any4hdmi.limmt.hme.loading import (
    CachedHmeWindowDataset,
    build_hme_feature_cache,
    hme_feature_cache_is_current,
    wait_for_hme_feature_cache,
)
from any4hdmi.limmt.hme.model import PeriodicAutoencoder


@dataclass(frozen=True)
class TrainHmeArgs:
    """Train LIMMT HME/PeriodicAutoencoder on an any4hdmi dataset."""

    project_path: str
    pass_dataset_name: str = "passed"
    hme_folder: str = "hme"
    batch_size: int = 256
    epochs: int = 30
    win_sec: float = 4.0
    phase_dim: int = 8
    hidden_dims: tuple[int, ...] = (64, 64)
    downsample_rate: int = 5
    stride: int = 1
    rebuild_cache: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: Literal["none", "cosine", "onecycle"] = "none"
    max_lr: float | None = None
    min_lr: float = 1e-5
    onecycle_pct_start: float = 0.15
    onecycle_div_factor: float = 10.0
    onecycle_final_div_factor: float = 100.0
    num_workers: int = max(1, (os.cpu_count() or 2) // 2)
    device: str | None = None


def _ddp_state() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return True, rank, local_rank, world_size


def _init_ddp(local_rank: int) -> None:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        dist.init_process_group(backend="gloo")


def _build_scheduler(args: TrainHmeArgs, optimizer: torch.optim.Optimizer, *, steps_per_epoch: int):
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
    args = tyro.cli(TrainHmeArgs)
    is_ddp, rank, local_rank, world_size = _ddp_state()
    project_root = resolve_project_root(args.project_path)
    input_root = project_pass_root(project_root, args.pass_dataset_name)
    output_dir = project_hme_root(project_root, args.hme_folder)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / "feature_cache"
    if rank == 0:
        if args.rebuild_cache and cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)
        if args.rebuild_cache or not hme_feature_cache_is_current(
            cache_dir,
            dataset_root=input_root,
            win_sec=args.win_sec,
            downsample_rate=args.downsample_rate,
            stride=args.stride,
        ):
            build_hme_feature_cache(
                dataset_root=input_root,
                cache_dir=cache_dir,
                win_sec=args.win_sec,
                downsample_rate=args.downsample_rate,
                stride=args.stride,
            )
    if rank != 0:
        wait_for_hme_feature_cache(cache_dir)
    else:
        wait_for_hme_feature_cache(cache_dir)

    if is_ddp:
        _init_ddp(local_rank)

    device = torch.device(args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
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
    sample_window = dataset[0]
    model = PeriodicAutoencoder(
        inp_ch=int(sample_window.shape[-1]),
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
                "project_root": str(project_root),
                "input_root": str(input_root),
                "output_dir": str(output_dir),
                "cache_dir": str(cache_dir),
                "state_dim": int(sample_window.shape[-1]),
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
        loss_total = 0.0
        batch_count = 0
        for batch in loader:
            input_windows = batch.permute(0, 2, 1).to(device=device, dtype=torch.float32, non_blocking=True)
            model_output = train_model(input_windows)
            loss = model_output["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 10.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_total += float(loss.detach().item())
            batch_count += 1
        if is_ddp:
            loss_stats = torch.tensor([loss_total, float(batch_count)], device=device, dtype=torch.float64)
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
            avg_loss = float((loss_stats[0] / loss_stats[1].clamp_min(1.0)).item())
        else:
            avg_loss = loss_total / max(batch_count, 1)
        if rank == 0:
            lr_now = float(optimizer.param_groups[0]["lr"])
            losses.append({"epoch": epoch, "loss": avg_loss, "lr": lr_now})
            print(f"epoch={epoch} loss={avg_loss:.6f} lr={lr_now:.8g}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": int(sample_window.shape[-1]),
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
