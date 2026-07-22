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
mujoco_api: Any = mujoco


def compute_motion_qvel(
    model: Any, qpos: np.ndarray, fps: float
) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos to be rank 2, got shape {qpos.shape}")
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"Motion qpos width {qpos.shape[1]} does not match model.nq={model.nq}"
        )

    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float32)
    if qpos.shape[0] <= 1:
        return qvel

    inv_dt = float(fps)
    joint_types = np.asarray(model.jnt_type)
    qpos_addrs = np.asarray(model.jnt_qposadr)
    dof_addrs = np.asarray(model.jnt_dofadr)

    scalar_joints = np.flatnonzero(
        (joint_types == mujoco_api.mjtJoint.mjJNT_HINGE)
        | (joint_types == mujoco_api.mjtJoint.mjJNT_SLIDE)
    )
    scalar_qpos_addrs = qpos_addrs[scalar_joints]
    scalar_dof_addrs = dof_addrs[scalar_joints]
    qvel[:-1, scalar_dof_addrs] = (
        qpos[1:, scalar_qpos_addrs] - qpos[:-1, scalar_qpos_addrs]
    ) * inv_dt

    quaternion_joints = np.flatnonzero(
        (joint_types == mujoco_api.mjtJoint.mjJNT_FREE)
        | (joint_types == mujoco_api.mjtJoint.mjJNT_BALL)
    )
    for joint_idx in quaternion_joints:
        joint_type = int(joint_types[joint_idx])
        qpos_addr = int(qpos_addrs[joint_idx])
        dof_addr = int(dof_addrs[joint_idx])
        if joint_type == mujoco_api.mjtJoint.mjJNT_FREE:
            qvel[:-1, dof_addr : dof_addr + 3] = (
                qpos[1:, qpos_addr : qpos_addr + 3]
                - qpos[:-1, qpos_addr : qpos_addr + 3]
            ) * inv_dt
            qpos_addr += 3
            dof_addr += 3

        previous = qpos[:-1, qpos_addr : qpos_addr + 4]
        current = qpos[1:, qpos_addr : qpos_addr + 4]
        previous_conjugate = previous.copy()
        previous_conjugate[:, 1:] *= -1.0
        aw, ax, ay, az = previous_conjugate.T
        bw, bx, by, bz = current.T
        difference = np.column_stack(
            (
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            )
        )
        axis = difference[:, 1:].copy()
        sin_half_angle = np.linalg.norm(axis, axis=1)
        regular = sin_half_angle >= mujoco_api.mjMINVAL
        axis[regular] /= sin_half_angle[regular, None]
        axis[~regular] = (1.0, 0.0, 0.0)
        speed = 2.0 * np.arctan2(sin_half_angle, difference[:, 0])
        speed = np.where(speed > np.pi, speed - 2.0 * np.pi, speed)
        qvel[:-1, dof_addr : dof_addr + 3] = axis * (speed * inv_dt)[:, None]

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
        prevalidated_paths: bool = False,
    ) -> None:
        self.input_root = input_root
        self.motion_paths = motion_paths
        self.mjcf_path = (
            None if mjcf_path is None else Path(mjcf_path).expanduser().absolute()
        )
        self.fps = None if fps is None else float(fps)
        self.prevalidated_paths = bool(prevalidated_paths)
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self.mjcf_path is None:
            raise RuntimeError(
                "mjcf_path is required to compute qvel in MotionTensorDataset"
            )
        if self._model is None:
            self._model = load_model(self.mjcf_path)
        return self._model

    def __len__(self) -> int:
        return len(self.motion_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        motion_path = self.motion_paths[index]
        if self.prevalidated_paths:
            with np.load(motion_path, allow_pickle=False) as payload:
                if "qpos" in payload:
                    qpos_np = np.asarray(payload["qpos"], dtype=np.float32)
                else:
                    qpos_np = load_motion(motion_path)
            if qpos_np.ndim == 1:
                qpos_np = qpos_np[None, :]
            if qpos_np.ndim != 2:
                raise ValueError(
                    f"Expected qpos to be rank 2, got shape {qpos_np.shape}"
                )
        else:
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


def pack_motion_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Expected at least one motion item")
    lengths = torch.as_tensor(
        [int(item["qpos"].shape[0]) for item in items],
        dtype=torch.long,
    )
    packed: dict[str, Any] = {
        "qpos": torch.cat([item["qpos"] for item in items], dim=0),
        "lengths": lengths,
    }
    qvel_items = [item.get("qvel") for item in items]
    qvel_tensors = [qvel for qvel in qvel_items if isinstance(qvel, torch.Tensor)]
    if len(qvel_tensors) == len(qvel_items):
        packed["qvel"] = torch.cat(qvel_tensors, dim=0)
    elif any(qvel is not None for qvel in qvel_items):
        raise ValueError(
            "Expected every packed motion item to either have qvel or omit it"
        )

    for key in ("motion_path", "rel_motion", "fps"):
        values = [item[key] for item in items if key in item]
        if values and len(values) != len(items):
            raise ValueError(
                f"Expected every packed motion item to either have {key} or omit it"
            )
        if values:
            packed[key] = values
    return packed


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
    batch_size: int = 1,
    prevalidated_paths: bool = False,
    multiprocessing_context: str | None = None,
    tensor_device: torch.device | str | None = None,
) -> Iterable[dict[str, Any]]:
    loader_batch_size = max(1, int(batch_size))
    loader_kwargs: dict[str, Any] = {
        "dataset": MotionTensorDataset(
            input_root=input_root,
            motion_paths=motion_paths,
            mjcf_path=mjcf_path,
            fps=fps,
            prevalidated_paths=prevalidated_paths,
        ),
        "batch_size": loader_batch_size,
        "shuffle": False,
        "num_workers": max(0, int(num_workers)),
        "collate_fn": (
            unwrap_single_motion_item if loader_batch_size == 1 else pack_motion_items
        ),
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
