from __future__ import annotations

from pathlib import Path

import torch

from any4hdmi.dataset.base import BaseDataset, MotionData, MotionSample
from any4hdmi.dataset.fk_cache import FKCacheEntry


class FullMotionDataset(BaseDataset):
    def __init__(
        self,
        *,
        body_names: list[str],
        joint_names: list[str],
        motion_paths: list[Path],
        starts: list[int],
        ends: list[int],
        data: MotionData,
        num_envs: int,
        device: torch.device | str | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.body_names = list(body_names)
        self.joint_names = list(joint_names)
        self.motion_paths = list(motion_paths)
        self.data = data
        self.device = torch.device(data.device)
        self.starts = torch.as_tensor(starts, device=self.device, dtype=torch.long)
        self.ends = torch.as_tensor(ends, device=self.device, dtype=torch.long)
        self.lengths = self.ends - self.starts
        self._num_envs = int(num_envs)
        self._env_motion_id = torch.full((self._num_envs,), -1, device=self.device, dtype=torch.long)
        self._env_motion_len = torch.zeros((self._num_envs,), device=self.device, dtype=torch.long)
        if device is not None:
            self.to(device)

    @classmethod
    def from_cache_entry(
        cls,
        entry: FKCacheEntry,
        *,
        num_envs: int,
        device: torch.device | str | None = None,
    ) -> FullMotionDataset:
        return cls(
            body_names=entry.body_names,
            joint_names=entry.joint_names,
            motion_paths=entry.motion_paths,
            starts=entry.starts,
            ends=entry.ends,
            data=entry.as_motion_data(),
            num_envs=num_envs,
            device=device,
        )

    def to(self, device: torch.device | str) -> FullMotionDataset:
        target_device = torch.device(device)
        self.data = self.data.to(target_device)
        self.starts = self.starts.to(target_device)
        self.ends = self.ends.to(target_device)
        self.lengths = self.lengths.to(target_device)
        self._env_motion_id = self._env_motion_id.to(target_device)
        self._env_motion_len = self._env_motion_len.to(target_device)
        self.device = target_device
        return self

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
        *,
        profile_name: str | None = None,
    ) -> MotionData:
        del profile_name
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        starts = starts.to(device=self.device, dtype=torch.long)
        steps = steps.to(device=self.device, dtype=torch.long)
        idx = (self.starts[motion_ids] + starts).unsqueeze(1) + steps.unsqueeze(0)
        idx.clamp_max_(self.ends[motion_ids].unsqueeze(1) - 1)
        idx.clamp_min_(self.starts[motion_ids].unsqueeze(1))
        return self.data[idx]

    def sample_motion(
        self,
        env_ids: torch.Tensor,
        *,
        terminated_t: torch.Tensor,
        rewind_mask: torch.Tensor,
        rewind_steps: torch.Tensor,
    ) -> MotionSample:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        terminated_t = terminated_t.to(device=self.device, dtype=torch.long)
        rewind_mask = rewind_mask.to(device=self.device, dtype=torch.bool)
        rewind_steps = rewind_steps.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return MotionSample(motion_id=empty, motion_len=empty, start_t=empty)

        sampled_frame_ids = torch.randint(
            0,
            self.num_steps,
            size=(env_ids.numel(),),
            device=self.device,
        )
        sampled_motion_ids = self.data.motion_id[sampled_frame_ids].long()
        sampled_start_t = self.data.step[sampled_frame_ids].long()
        sampled_motion_len = self.lengths[sampled_motion_ids].long()

        if bool(torch.any(rewind_mask).item()):
            rewind_motion_ids = self._env_motion_id.index_select(0, env_ids)
            rewind_t = torch.clamp(terminated_t - rewind_steps, min=0)
            sampled_motion_ids = torch.where(rewind_mask, rewind_motion_ids, sampled_motion_ids)
            sampled_motion_len = torch.where(
                rewind_mask,
                self.lengths[rewind_motion_ids].long(),
                sampled_motion_len,
            )
            sampled_start_t = torch.where(rewind_mask, rewind_t, sampled_start_t)

        self._env_motion_id.index_copy_(0, env_ids, sampled_motion_ids)
        self._env_motion_len.index_copy_(0, env_ids, sampled_motion_len)
        return MotionSample(
            motion_id=sampled_motion_ids,
            motion_len=sampled_motion_len,
            start_t=sampled_start_t,
        )
