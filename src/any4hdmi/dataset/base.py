from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorClass


class MotionData(TensorClass):
    motion_id: torch.Tensor
    step: torch.Tensor
    body_pos_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


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
    device: torch.device

    @property
    def num_motions(self) -> int:
        return int(self.starts.shape[0])

    @property
    @abstractmethod
    def num_steps(self) -> int:
        raise NotImplementedError

    @property
    def sample_id_span(self) -> int:
        """Number of IDs that sample_motion may return for this runtime."""
        return self.num_motions

    @abstractmethod
    def to(self, device: torch.device | str) -> BaseDataset:
        raise NotImplementedError

    @abstractmethod
    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
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
