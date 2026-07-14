from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from any4hdmi.dataset.base import BaseDataset
from any4hdmi.dataset.distributed_shard import shard_cache_entry
from any4hdmi.dataset.fk_cache import FKCache, FKCacheEntry
from any4hdmi.dataset.fp16_backing import prepare_fp16_cache_entry
from any4hdmi.dataset.full import FullMotionDataset
from any4hdmi.dataset.loading import (
    resolve_any4hdmi_dataset_context,
    resolve_input_paths,
    resolve_motion_filenames,
)
from any4hdmi.dataset.windowed import WindowedMotionDataset


def prepare_cache_entry(
    *,
    root_path: str | Path | list[str] | list[Path],
    target_fps: int,
    base_dir: Path,
    filenames: Sequence[str] | None = None,
    filenames_path: str | Path | None = None,
    body_names: list[str] | None = None,
    joint_names: list[str] | None = None,
) -> FKCacheEntry:
    """Resolve, cache, prune, and expose one shared FP16 backing entry."""
    input_paths = resolve_input_paths(base_dir, root_path)
    filenames_root = base_dir
    if filenames_path is not None and filenames is None:
        filenames_root, _ = resolve_any4hdmi_dataset_context(input_paths)
    motion_filenames = resolve_motion_filenames(
        filenames_root,
        filenames=filenames,
        filenames_path=filenames_path,
    )
    entry = FKCache.from_inputs(
        input_paths=input_paths,
        target_fps=target_fps,
        base_dir=base_dir,
        motion_filenames=motion_filenames,
    ).get_or_build()
    return prepare_fp16_cache_entry(
        entry,
        body_names=body_names,
        joint_names=joint_names,
    )


def partition_cache_entry(
    entry: FKCacheEntry,
    *,
    shard: bool,
    rank: int,
    world_size: int,
) -> FKCacheEntry:
    """Apply only the requested motion-aligned visibility partition."""
    if not shard:
        print(
            f"[any4hdmi][partition] shard=0 motions={len(entry.ends)}"
            f" frames={entry.ends[-1]}"
        )
        return entry
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    partitioned = shard_cache_entry(entry, rank=rank, world_size=world_size)
    print(
        f"[any4hdmi][partition] shard=1 rank={rank}/{world_size}"
        f" motions={len(partitioned.ends)} frames={partitioned.ends[-1]}"
    )
    return partitioned


def build_runtime_dataset(
    entry: FKCacheEntry,
    *,
    full_motion: bool,
    num_envs: int,
    windowed_next_window_device: str | torch.device | None = "current",
    windowed_pin_window_load: bool = True,
) -> BaseDataset:
    """Build the requested runtime strategy without inspecting partition state."""
    if full_motion:
        dataset: BaseDataset = FullMotionDataset.from_cache_entry(
            entry,
            num_envs=num_envs,
            output_float_dtype=torch.float32,
        )
    else:
        dataset = WindowedMotionDataset.from_cache_entry(
            entry,
            num_envs=num_envs,
            next_window_device=windowed_next_window_device,
            pin_window_load=windowed_pin_window_load,
        )
    print(
        f"[any4hdmi][runtime] full_motion={int(full_motion)}"
        f" dataset={type(dataset).__name__} motions={len(entry.ends)}"
        f" frames={entry.ends[-1]}"
        f" storage_dtype={entry.storage_fields['body_pos_w'].dtype}"
        " output_dtype=torch.float32"
    )
    return dataset


def load_any4hdmi_dataset(
    *,
    root_path: str | Path | list[str] | list[Path],
    target_fps: int,
    base_dir: Path,
    num_envs: int,
    full_motion: bool,
    shard: bool = False,
    rank: int = 0,
    world_size: int = 1,
    filenames: Sequence[str] | None = None,
    filenames_path: str | Path | None = None,
    body_names: list[str] | None = None,
    joint_names: list[str] | None = None,
    windowed_next_window_device: str | torch.device | None = "current",
    windowed_pin_window_load: bool = True,
) -> BaseDataset:
    entry = prepare_cache_entry(
        root_path=root_path,
        target_fps=target_fps,
        base_dir=base_dir,
        filenames=filenames,
        filenames_path=filenames_path,
        body_names=body_names,
        joint_names=joint_names,
    )
    entry = partition_cache_entry(
        entry,
        shard=shard,
        rank=rank,
        world_size=world_size,
    )
    return build_runtime_dataset(
        entry,
        full_motion=full_motion,
        num_envs=num_envs,
        windowed_next_window_device=windowed_next_window_device,
        windowed_pin_window_load=windowed_pin_window_load,
    )


__all__ = [
    "FKCache",
    "FKCacheEntry",
    "FullMotionDataset",
    "WindowedMotionDataset",
    "build_runtime_dataset",
    "load_any4hdmi_dataset",
    "partition_cache_entry",
    "prepare_cache_entry",
]
