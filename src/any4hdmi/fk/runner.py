from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

if "WARP_CACHE_PATH" not in os.environ:
    _warp_cache_dir = Path(tempfile.gettempdir()) / "any4hdmi-warp-cache"
    _warp_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(_warp_cache_dir)

def compute_root_qvel_torch(qpos: torch.Tensor, fps: float) -> torch.Tensor:
    if not isinstance(qpos, torch.Tensor):
        raise TypeError(f"Expected qpos to be a torch.Tensor, got {type(qpos)!r}")
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos to be rank 2, got shape {tuple(qpos.shape)}")

    qpos = qpos.to(dtype=torch.float32)
    root_qvel = torch.zeros((qpos.shape[0], 6), dtype=qpos.dtype, device=qpos.device)
    if qpos.shape[0] <= 1:
        return root_qvel

    dt = float(1.0 / fps)
    root_qvel[:-1, :3] = (qpos[1:, :3] - qpos[:-1, :3]) / dt

    q1 = qpos[:-1, 3:7].clone()
    q2 = qpos[1:, 3:7]
    q1[:, 1:] *= -1.0

    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    q_delta = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )

    negative_w = q_delta[:, 0] < 0.0
    q_delta[negative_w] *= -1.0

    vec = q_delta[:, 1:]
    sin_half = torch.linalg.norm(vec, dim=-1, keepdim=True)
    cos_half = torch.clamp(q_delta[:, 0:1], -1.0, 1.0)
    angle = 2.0 * torch.atan2(sin_half, cos_half)
    safe_sin_half = torch.where(sin_half > 1.0e-12, sin_half, torch.ones_like(sin_half))
    axis = vec / safe_sin_half
    axis = torch.where(sin_half > 1.0e-12, axis, torch.zeros_like(axis))
    root_qvel[:-1, 3:] = axis * (angle / dt)
    root_qvel[-1] = root_qvel[-2]
    return root_qvel


def compute_root_qvel_many_torch(
    qpos_list: Sequence[torch.Tensor],
    fps_list: Sequence[float],
) -> list[torch.Tensor]:
    return [
        compute_root_qvel_torch(qpos, fps)
        for qpos, fps in zip(qpos_list, fps_list, strict=True)
    ]


class FKRunner:
    def __init__(
        self,
        mjcf_path: Path,
        batch_size: int,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.batch_size = max(1, int(batch_size))
        self.backend = "mujoco"
        self.device = torch.device(device) if device is not None else self._default_device()
        self._wp = None
        self._mjw = None
        self._warp_model = None
        self._warp_data = None
        self._warned_about_warp_fallback = False

        if self.device.type == "cuda":
            self._wp = wp
            self._mjw = mjw
            self._warp_model = mjw.put_model(self.model)
            self._warp_data = mjw.make_data(self.model, nworld=self.batch_size)
            self.device = torch.device(str(self._warp_data.qpos.device))
            self.backend = "mujoco_warp"
        else:
            self.device = torch.device("cpu")

    @staticmethod
    def _default_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_device(self, qpos: torch.Tensor) -> torch.Tensor:
        if not isinstance(qpos, torch.Tensor):
            raise TypeError(f"Expected qpos to be a torch.Tensor, got {type(qpos)!r}")
        if qpos.ndim != 2:
            raise ValueError(f"Expected qpos to be rank 2, got shape {tuple(qpos.shape)}")
        if qpos.shape[1] != self.model.nq:
            raise ValueError(f"Motion qpos width {qpos.shape[1]} does not match model.nq={self.model.nq}")
        return qpos.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()

    def _forward_positions_cpu(self, qpos: torch.Tensor) -> torch.Tensor:
        qpos_cpu = qpos.to(device="cpu", dtype=torch.float32).contiguous()
        qpos_np = qpos_cpu.numpy()
        xpos = torch.empty((qpos_cpu.shape[0], self.model.nbody, 3), dtype=torch.float32)
        for frame_idx, qpos_frame in enumerate(qpos_np):
            self.data.qpos[:] = qpos_frame
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            xpos[frame_idx] = torch.as_tensor(np.asarray(self.data.xpos, dtype=np.float32))
        return xpos

    def _forward_kinematics_cpu(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        *,
        joint_qpos_addrs: torch.Tensor,
        joint_dof_addrs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        qpos_cpu = qpos.to(device="cpu", dtype=torch.float32).contiguous()
        qvel_cpu = qvel.to(device="cpu", dtype=torch.float32).contiguous()
        joint_qpos_addrs_cpu = joint_qpos_addrs.to(device="cpu", dtype=torch.long)
        joint_dof_addrs_cpu = joint_dof_addrs.to(device="cpu", dtype=torch.long)

        frames = int(qpos_cpu.shape[0])
        body_pos_w = torch.empty((frames, self.model.nbody, 3), dtype=torch.float32)
        body_lin_vel_w = torch.empty((frames, self.model.nbody, 3), dtype=torch.float32)
        body_quat_w = torch.empty((frames, self.model.nbody, 4), dtype=torch.float32)
        body_ang_vel_w = torch.empty((frames, self.model.nbody, 3), dtype=torch.float32)
        joint_pos = torch.empty((frames, int(joint_qpos_addrs_cpu.shape[0])), dtype=torch.float32)
        joint_vel = torch.empty((frames, int(joint_dof_addrs_cpu.shape[0])), dtype=torch.float32)

        qpos_np = qpos_cpu.numpy()
        qvel_np = qvel_cpu.numpy()
        joint_qpos_idx = joint_qpos_addrs_cpu.numpy()
        joint_dof_idx = joint_dof_addrs_cpu.numpy()

        for frame_idx, (qpos_frame, qvel_frame) in enumerate(zip(qpos_np, qvel_np, strict=True)):
            self.data.qpos[:] = qpos_frame
            self.data.qvel[:] = qvel_frame
            mujoco.mj_forward(self.model, self.data)
            body_pos_w[frame_idx] = torch.as_tensor(np.asarray(self.data.xpos, dtype=np.float32))
            body_lin_vel_w[frame_idx] = torch.as_tensor(np.asarray(self.data.cvel[:, 3:6], dtype=np.float32))
            body_quat_w[frame_idx] = torch.as_tensor(np.asarray(self.data.xquat, dtype=np.float32))
            body_ang_vel_w[frame_idx] = torch.as_tensor(np.asarray(self.data.cvel[:, 0:3], dtype=np.float32))
            joint_pos[frame_idx] = torch.as_tensor(np.asarray(self.data.qpos[joint_qpos_idx], dtype=np.float32))
            joint_vel[frame_idx] = torch.as_tensor(np.asarray(self.data.qvel[joint_dof_idx], dtype=np.float32))

        return {
            "body_pos_w": body_pos_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_quat_w": body_quat_w,
            "body_ang_vel_w": body_ang_vel_w,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

    def _forward_kinematics_warp(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        *,
        joint_qpos_addrs: torch.Tensor,
        joint_dof_addrs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert self._mjw is not None
        assert self._wp is not None
        assert self._warp_model is not None
        assert self._warp_data is not None

        qpos = self.to_device(qpos)
        qvel = qvel.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()
        joint_qpos_addrs = joint_qpos_addrs.to(device=self.device, dtype=torch.long)
        joint_dof_addrs = joint_dof_addrs.to(device=self.device, dtype=torch.long)

        nworld = int(qpos.shape[0])
        if nworld > self.batch_size:
            raise ValueError(f"Chunk size {nworld} exceeds batch_size {self.batch_size}")

        qpos_work = qpos
        qvel_work = qvel
        if nworld < self.batch_size:
            padded_qpos = torch.zeros((self.batch_size, self.model.nq), dtype=qpos.dtype, device=self.device)
            padded_qvel = torch.zeros((self.batch_size, self.model.nv), dtype=qvel.dtype, device=self.device)
            padded_qpos[:nworld] = qpos
            padded_qvel[:nworld] = qvel
            qpos_work = padded_qpos
            qvel_work = padded_qvel

        self._wp.copy(self._warp_data.qpos, self._wp.from_torch(qpos_work))
        self._wp.copy(self._warp_data.qvel, self._wp.from_torch(qvel_work))
        self._mjw.fwd_position(self._warp_model, self._warp_data)
        self._mjw.fwd_velocity(self._warp_model, self._warp_data)
        if hasattr(self._wp, "synchronize"):
            self._wp.synchronize()

        cvel = self._wp.to_torch(self._warp_data.cvel)[:nworld].clone()
        return {
            "body_pos_w": self._wp.to_torch(self._warp_data.xpos)[:nworld].clone(),
            "body_lin_vel_w": cvel[..., 3:6].clone(),
            "body_quat_w": self._wp.to_torch(self._warp_data.xquat)[:nworld].clone(),
            "body_ang_vel_w": cvel[..., 0:3].clone(),
            "joint_pos": qpos_work[:nworld].index_select(1, joint_qpos_addrs).clone(),
            "joint_vel": qvel_work[:nworld].index_select(1, joint_dof_addrs).clone(),
        }

    def _forward_positions_warp(self, qpos: torch.Tensor) -> torch.Tensor:
        assert self._mjw is not None
        assert self._wp is not None
        assert self._warp_model is not None
        assert self._warp_data is not None

        qpos = self.to_device(qpos)
        nworld = int(qpos.shape[0])
        if nworld > self.batch_size:
            raise ValueError(f"Chunk size {nworld} exceeds batch_size {self.batch_size}")

        if nworld < self.batch_size:
            padded_qpos = torch.zeros(
                (self.batch_size, self.model.nq),
                dtype=qpos.dtype,
                device=self.device,
            )
            padded_qpos[:nworld] = qpos
            warp_qpos = self._wp.from_torch(padded_qpos)
        else:
            warp_qpos = self._wp.from_torch(qpos)

        self._wp.copy(self._warp_data.qpos, warp_qpos)
        self._mjw.kinematics(self._warp_model, self._warp_data)
        if hasattr(self._wp, "synchronize"):
            self._wp.synchronize()
        return self._wp.to_torch(self._warp_data.xpos)[:nworld].clone()

    def forward_positions_many(self, qpos_list: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if not qpos_list:
            return []

        normalized_qpos = [self.to_device(qpos) for qpos in qpos_list]
        lengths = [int(qpos.shape[0]) for qpos in normalized_qpos]
        merged_qpos = torch.cat(normalized_qpos, dim=0)
        chunks: list[torch.Tensor] = []

        for start in range(0, merged_qpos.shape[0], self.batch_size):
            stop = min(merged_qpos.shape[0], start + self.batch_size)
            qpos_chunk = merged_qpos[start:stop]
            if self.backend == "mujoco_warp":
                chunks.append(self._forward_positions_warp(qpos_chunk))
                continue
            chunks.append(self._forward_positions_cpu(qpos_chunk))

        merged_xpos = (
            torch.cat(chunks, dim=0)
            if chunks
            else torch.zeros((0, self.model.nbody, 3), dtype=torch.float32, device=self.device)
        )
        outputs: list[torch.Tensor] = []
        start = 0
        for length in lengths:
            stop = start + length
            outputs.append(merged_xpos[start:stop])
            start = stop
        return outputs

    def forward_kinematics_many(
        self,
        qpos_list: Sequence[torch.Tensor],
        qvel_list: Sequence[torch.Tensor],
        *,
        joint_qpos_addrs: torch.Tensor,
        joint_dof_addrs: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        if len(qpos_list) != len(qvel_list):
            raise ValueError("qpos_list and qvel_list must have the same length")
        if not qpos_list:
            return []
        if self.backend != "mujoco_warp":
            return [
                self._forward_kinematics_cpu(
                    qpos,
                    qvel,
                    joint_qpos_addrs=joint_qpos_addrs,
                    joint_dof_addrs=joint_dof_addrs,
                )
                for qpos, qvel in zip(qpos_list, qvel_list, strict=True)
            ]

        normalized_qpos = [self.to_device(qpos) for qpos in qpos_list]
        normalized_qvel = [
            qvel.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()
            for qvel in qvel_list
        ]
        lengths = [int(qpos.shape[0]) for qpos in normalized_qpos]
        merged_qpos = torch.cat(normalized_qpos, dim=0)
        merged_qvel = torch.cat(normalized_qvel, dim=0)
        chunk_outputs: list[dict[str, torch.Tensor]] = []

        for start in range(0, merged_qpos.shape[0], self.batch_size):
            stop = min(merged_qpos.shape[0], start + self.batch_size)
            chunk_outputs.append(
                self._forward_kinematics_warp(
                    merged_qpos[start:stop],
                    merged_qvel[start:stop],
                    joint_qpos_addrs=joint_qpos_addrs,
                    joint_dof_addrs=joint_dof_addrs,
                )
            )

        merged_outputs = {
            key: torch.cat([chunk[key] for chunk in chunk_outputs], dim=0)
            for key in chunk_outputs[0]
        }
        outputs: list[dict[str, torch.Tensor]] = []
        start = 0
        for length in lengths:
            stop = start + length
            outputs.append({key: value[start:stop] for key, value in merged_outputs.items()})
            start = stop
        return outputs
