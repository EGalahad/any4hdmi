from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MotionData:
    motion_id: torch.Tensor
    step: torch.Tensor
    body_pos_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor

    def __len__(self) -> int:
        return int(self.motion_id.shape[0])

    @property
    def device(self) -> torch.device:
        return self.motion_id.device

    def to(self, device: torch.device | str) -> MotionData:
        return MotionData(
            motion_id=self.motion_id.to(device),
            step=self.step.to(device),
            body_pos_w=self.body_pos_w.to(device),
            body_lin_vel_w=self.body_lin_vel_w.to(device),
            body_quat_w=self.body_quat_w.to(device),
            body_ang_vel_w=self.body_ang_vel_w.to(device),
            joint_pos=self.joint_pos.to(device),
            joint_vel=self.joint_vel.to(device),
        )

    def __getitem__(self, item) -> MotionData:
        return MotionData(
            motion_id=self.motion_id[item],
            step=self.step[item],
            body_pos_w=self.body_pos_w[item],
            body_lin_vel_w=self.body_lin_vel_w[item],
            body_quat_w=self.body_quat_w[item],
            body_ang_vel_w=self.body_ang_vel_w[item],
            joint_pos=self.joint_pos[item],
            joint_vel=self.joint_vel[item],
        )
