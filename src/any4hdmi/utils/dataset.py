from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from any4hdmi.core.format import load_motion

DEFAULT_MOTION_LOADER_NUM_WORKERS = min(4, max(0, (os.cpu_count() or 1) - 1))
DEFAULT_MOTION_LOADER_PREFETCH_FACTOR = 2


class MotionTensorDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        input_root: Path | None,
        motion_paths: list[Path],
        fps: float | None = None,
    ) -> None:
        self.input_root = input_root
        self.motion_paths = motion_paths
        self.fps = None if fps is None else float(fps)

    def __len__(self) -> int:
        return len(self.motion_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        motion_path = self.motion_paths[index]
        item: dict[str, Any] = {
            "motion_path": motion_path,
            "qpos": torch.from_numpy(load_motion(motion_path)).contiguous(),
        }
        if self.input_root is not None:
            item["rel_motion"] = motion_path.relative_to(self.input_root)
        if self.fps is not None:
            item["fps"] = self.fps
        return item


def unwrap_single_motion_item(items: list[dict[str, Any]]) -> dict[str, Any]:
    return items[0]


def build_motion_loader(
    *,
    input_root: Path | None,
    motion_paths: list[Path],
    fps: float | None,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader[dict[str, Any]]:
    loader_kwargs: dict[str, Any] = {
        "dataset": MotionTensorDataset(input_root=input_root, motion_paths=motion_paths, fps=fps),
        "batch_size": 1,
        "shuffle": False,
        "num_workers": max(0, int(num_workers)),
        "collate_fn": unwrap_single_motion_item,
        "pin_memory": pin_memory,
        "persistent_workers": int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**loader_kwargs)
