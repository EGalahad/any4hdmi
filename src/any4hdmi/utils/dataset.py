from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import mujoco
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from any4hdmi.core.format import load_motion
from any4hdmi.core.model import load_model

DEFAULT_MOTION_LOADER_NUM_WORKERS = min(16, max(0, (os.cpu_count() or 1) - 1))
DEFAULT_MOTION_LOADER_PREFETCH_FACTOR = 4


def compute_motion_qvel(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos to be rank 2, got shape {qpos.shape}")
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Motion qpos width {qpos.shape[1]} does not match model.nq={model.nq}")

    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float32)
    if qpos.shape[0] <= 1:
        return qvel

    dt = 1.0 / float(fps)
    work = np.zeros(model.nv, dtype=np.float64)
    for frame_idx in range(qpos.shape[0] - 1):
        mujoco.mj_differentiatePos(model, work, dt, qpos[frame_idx], qpos[frame_idx + 1])
        qvel[frame_idx] = np.asarray(work, dtype=np.float32)
    qvel[-1] = qvel[-2]
    return qvel


class MotionTensorDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        input_root: Path | None,
        motion_paths: list[Path],
        mjcf_path: Path | None = None,
        fps: float | None = None,
    ) -> None:
        self.input_root = input_root
        self.motion_paths = motion_paths
        self.mjcf_path = None if mjcf_path is None else Path(mjcf_path).expanduser().absolute()
        self.fps = None if fps is None else float(fps)
        self._model: mujoco.MjModel | None = None

    def _get_model(self) -> mujoco.MjModel:
        if self.mjcf_path is None:
            raise RuntimeError("mjcf_path is required to compute qvel in MotionTensorDataset")
        if self._model is None:
            self._model = load_model(self.mjcf_path)
        return self._model

    def __len__(self) -> int:
        return len(self.motion_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        motion_path = self.motion_paths[index]
        qpos_np = load_motion(motion_path)
        item: dict[str, Any] = {
            "motion_path": motion_path,
            "qpos": torch.from_numpy(qpos_np).contiguous(),
        }
        if self.fps is not None and self.mjcf_path is not None:
            item["qvel"] = torch.from_numpy(
                compute_motion_qvel(self._get_model(), qpos_np, self.fps)
            ).contiguous()
        if self.input_root is not None:
            item["rel_motion"] = motion_path.relative_to(self.input_root)
        if self.fps is not None:
            item["fps"] = self.fps
        return item


def unwrap_single_motion_item(items: list[dict[str, Any]]) -> dict[str, Any]:
    return items[0]


def move_motion_item_to_device(
    item: dict[str, Any],
    *,
    tensor_device: torch.device | str | None,
) -> dict[str, Any]:
    if tensor_device is None:
        return item
    moved = dict(item)
    for key in ("qpos", "qvel"):
        tensor = moved.get(key)
        if isinstance(tensor, torch.Tensor):
            moved[key] = tensor.to(
                device=tensor_device,
                dtype=torch.float32,
                non_blocking=True,
            ).contiguous()
    return moved


class MotionLoaderView(Iterable[dict[str, Any]]):
    def __init__(
        self,
        loader: DataLoader[dict[str, Any]],
        *,
        tensor_device: torch.device | str | None,
    ) -> None:
        self._loader = loader
        self._tensor_device = tensor_device

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for item in self._loader:
            yield move_motion_item_to_device(item, tensor_device=self._tensor_device)

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


def build_motion_loader(
    *,
    input_root: Path | None,
    motion_paths: list[Path],
    mjcf_path: Path | None,
    fps: float | None,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    multiprocessing_context: str | None = None,
    tensor_device: torch.device | str | None = None,
) -> Iterable[dict[str, Any]]:
    loader_kwargs: dict[str, Any] = {
        "dataset": MotionTensorDataset(
            input_root=input_root,
            motion_paths=motion_paths,
            mjcf_path=mjcf_path,
            fps=fps,
        ),
        "batch_size": 1,
        "shuffle": False,
        "num_workers": max(0, int(num_workers)),
        "collate_fn": unwrap_single_motion_item,
        "pin_memory": pin_memory,
        "persistent_workers": int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        if multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context
    return MotionLoaderView(
        DataLoader(**loader_kwargs),
        tensor_device=tensor_device,
    )
