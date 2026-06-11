from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
try:
    from tensordict import TensorClass
except ImportError:
    TensorClass = None


class _MotionDataFallback:
    def __init__(
        self,
        *,
        motion_id: torch.Tensor,
        step: torch.Tensor,
        body_pos_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        device=None,
        batch_size=None,
    ):
        self.motion_id = motion_id
        self.step = step
        self.body_pos_w = body_pos_w
        self.body_lin_vel_w = body_lin_vel_w
        self.body_quat_w = body_quat_w
        self.body_ang_vel_w = body_ang_vel_w
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel

    @property
    def device(self) -> torch.device:
        return self.motion_id.device

    def to(self, device: torch.device | str):
        return type(self)(
            motion_id=self.motion_id.to(device),
            step=self.step.to(device),
            body_pos_w=self.body_pos_w.to(device),
            body_lin_vel_w=self.body_lin_vel_w.to(device),
            body_quat_w=self.body_quat_w.to(device),
            body_ang_vel_w=self.body_ang_vel_w.to(device),
            joint_pos=self.joint_pos.to(device),
            joint_vel=self.joint_vel.to(device),
        )


class MotionData(TensorClass if TensorClass is not None else _MotionDataFallback):
    motion_id: torch.Tensor
    step: torch.Tensor
    body_pos_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


@dataclass
class DatasetIndex:
    motion_id: torch.Tensor
    step: torch.Tensor

    def __len__(self) -> int:
        return int(self.motion_id.shape[0])

    @property
    def device(self) -> torch.device:
        return self.motion_id.device

    def to(self, device: torch.device | str) -> DatasetIndex:
        return DatasetIndex(
            motion_id=self.motion_id.to(device),
            step=self.step.to(device),
        )


@dataclass
class MotionSample:
    motion_id: torch.Tensor
    motion_len: torch.Tensor
    start_t: torch.Tensor

    def __len__(self) -> int:
        return int(self.motion_id.shape[0])

    @property
    def device(self) -> torch.device:
        return self.motion_id.device

    def to(self, device: torch.device | str) -> MotionSample:
        return MotionSample(
            motion_id=self.motion_id.to(device),
            motion_len=self.motion_len.to(device),
            start_t=self.start_t.to(device),
        )


class BaseDataset(ABC):
    body_names: list[str]
    joint_names: list[str]
    motion_paths: list[Path]
    starts: torch.Tensor
    ends: torch.Tensor
    lengths: torch.Tensor
    data: DatasetIndex
    device: torch.device

    @property
    def num_motions(self) -> int:
        return int(self.starts.shape[0])

    @property
    def num_steps(self) -> int:
        return len(self.data)

    @abstractmethod
    def to(self, device: torch.device | str) -> BaseDataset:
        raise NotImplementedError

    @abstractmethod
    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
        *,
        profile_name: str | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def sample_motion(
        self,
        env_ids: torch.Tensor,
        *,
        terminated_t: torch.Tensor,
        rewind_mask: torch.Tensor,
        rewind_steps: torch.Tensor,
    ) -> MotionSample:
        raise NotImplementedError
