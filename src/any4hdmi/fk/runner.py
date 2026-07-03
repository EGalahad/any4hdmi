from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from any4hdmi.core.model import body_names_from_model, hinge_joint_info


class FKRunner:
    def __init__(
        self,
        mjcf_path: Path,
        batch_size: int,
        device: str | torch.device | None = None,
        *,
        contact_nconmax: int | None = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.body_names = body_names_from_model(self.model)
        self.joint_names, joint_qpos_addrs, joint_dof_addrs = hinge_joint_info(self.model)
        self.batch_size = max(1, int(batch_size))
        self.contact_nconmax = int(contact_nconmax) if contact_nconmax is not None else None
        self.backend = "mujoco"
        self.device = torch.device("cpu")
        self._wp = None
        self._mjw = None
        self._warp_model = None
        self._warp_data = None
        self._padded_qpos: torch.Tensor | None = None
        self._padded_qvel: torch.Tensor | None = None
        self._warned_about_warp_fallback = False
        self._fake_zero_outputs = os.environ.get("ANY4HDMI_FAKE_FK_ZERO_OUTPUTS", "0") == "1"
        self._constant_kinematics_outputs_cache: dict[int, dict[str, torch.Tensor]] = {}
        self.joint_qpos_addrs = torch.as_tensor(joint_qpos_addrs, dtype=torch.long)
        self.joint_dof_addrs = torch.as_tensor(joint_dof_addrs, dtype=torch.long)
        requested_device = torch.device(device) if device is not None else self._default_device()
        self.to(requested_device)
        if self._fake_zero_outputs:
            print(
                "[any4hdmi][fk_runner] ANY4HDMI_FAKE_FK_ZERO_OUTPUTS=1,"
                " reusing one canonical FK result for all frames"
            )

    @staticmethod
    def _default_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _load_mujoco_warp() -> tuple[Any, Any] | None:
        try:
            import warp as wp
            import mujoco_warp as mjw
        except ImportError:
            return None
        return wp, mjw

    def _warn_mujoco_warp_fallback(self) -> None:
        if self._warned_about_warp_fallback:
            return
        warnings.warn(
            "mujoco_warp/warp is unavailable; falling back to CPU MuJoCo FK.",
            RuntimeWarning,
            stacklevel=3,
        )
        self._warned_about_warp_fallback = True

    def _use_mujoco_backend(self) -> None:
        self.device = torch.device("cpu")
        self.backend = "mujoco"
        self._wp = None
        self._mjw = None
        self._warp_model = None
        self._warp_data = None
        self._padded_qpos = None
        self._padded_qvel = None

    def to(self, device: torch.device | str) -> FKRunner:
        requested_device = torch.device(device)
        if requested_device.type == "cuda":
            warp_modules = self._load_mujoco_warp()
            if warp_modules is None:
                self._warn_mujoco_warp_fallback()
                self._use_mujoco_backend()
                self.joint_qpos_addrs = self.joint_qpos_addrs.to(device=self.device)
                self.joint_dof_addrs = self.joint_dof_addrs.to(device=self.device)
                return self

            wp, mjw = warp_modules
            self._wp = wp
            self._mjw = mjw
            self._warp_model = mjw.put_model(self.model)
            self._warp_data = mjw.make_data(self.model, nworld=self.batch_size, nconmax=self.contact_nconmax)
            self.device = torch.device(str(self._warp_data.qpos.device))
            self._padded_qpos = torch.zeros(
                (self.batch_size, self.model.nq),
                dtype=torch.float32,
                device=self.device,
            )
            self._padded_qvel = torch.zeros(
                (self.batch_size, self.model.nv),
                dtype=torch.float32,
                device=self.device,
            )
            self.backend = "mujoco_warp"
        else:
            self._use_mujoco_backend()

        self.joint_qpos_addrs = self.joint_qpos_addrs.to(device=self.device)
        self.joint_dof_addrs = self.joint_dof_addrs.to(device=self.device)
        return self

    def to_device(self, tensor: torch.Tensor, *, name: str = "tensor") -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(tensor)!r}")
        if tensor.ndim != 2:
            raise ValueError(f"Expected {name} to be rank 2, got shape {tuple(tensor.shape)}")
        return tensor.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()

    def _constant_kinematics_outputs(self, batch_size: int) -> dict[str, torch.Tensor]:
        cached = self._constant_kinematics_outputs_cache.get(batch_size)
        if cached is not None:
            return cached

        canonical_qpos = torch.zeros((1, self.model.nq), dtype=torch.float32, device=self.device)
        canonical_qvel = torch.zeros((1, self.model.nv), dtype=torch.float32, device=self.device)
        if self.model.nq >= 7:
            # Canonical free-root pose: xyz=(0,0,1.0), quat=(1,0,0,0), joints=0.
            canonical_qpos[0, 2] = 1.0
            canonical_qpos[0, 3] = 1.0

        if self.backend == "mujoco_warp":
            single = self._forward_kinematics_warp(canonical_qpos, canonical_qvel, synchronize=False)
        else:
            single = self._forward_kinematics_cpu(canonical_qpos, canonical_qvel)

        repeated = {
            key: value.expand(batch_size, *value.shape[1:]).clone()
            for key, value in single.items()
        }
        self._constant_kinematics_outputs_cache[batch_size] = repeated
        return repeated

    @property
    def uses_constant_outputs(self) -> bool:
        return self._fake_zero_outputs

    def constant_kinematics_outputs(self, batch_size: int) -> dict[str, torch.Tensor]:
        return self._constant_kinematics_outputs(batch_size)

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
    ) -> dict[str, torch.Tensor]:
        orig_device = qpos.device
        qpos_cpu = qpos.to(device="cpu", dtype=torch.float32).contiguous()
        qvel_cpu = qvel.to(device="cpu", dtype=torch.float32).contiguous()
        joint_qpos_addrs_cpu = self.joint_qpos_addrs.to(device="cpu", dtype=torch.long)
        joint_dof_addrs_cpu = self.joint_dof_addrs.to(device="cpu", dtype=torch.long)

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
            "body_pos_w": body_pos_w.to(device=orig_device),
            "body_lin_vel_w": body_lin_vel_w.to(device=orig_device),
            "body_quat_w": body_quat_w.to(device=orig_device),
            "body_ang_vel_w": body_ang_vel_w.to(device=orig_device),
            "joint_pos": joint_pos.to(device=orig_device),
            "joint_vel": joint_vel.to(device=orig_device),
        }

    def _forward_kinematics_warp(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        *,
        synchronize: bool,
    ) -> dict[str, torch.Tensor]:
        assert self._mjw is not None
        assert self._wp is not None
        assert self._warp_model is not None
        assert self._warp_data is not None

        qpos = self.to_device(qpos, name="qpos")
        qvel = self.to_device(qvel, name="qvel")

        nworld = int(qpos.shape[0])
        if nworld > self.batch_size:
            raise ValueError(f"Chunk size {nworld} exceeds batch_size {self.batch_size}")

        qpos_work = qpos
        qvel_work = qvel
        if nworld < self.batch_size:
            assert self._padded_qpos is not None
            assert self._padded_qvel is not None
            self._padded_qpos[:nworld] = qpos
            self._padded_qvel[:nworld] = qvel
            self._padded_qpos[nworld:].zero_()
            self._padded_qvel[nworld:].zero_()
            qpos_work = self._padded_qpos
            qvel_work = self._padded_qvel

        self._wp.copy(self._warp_data.qpos, self._wp.from_torch(qpos_work))
        self._wp.copy(self._warp_data.qvel, self._wp.from_torch(qvel_work))
        self._mjw.fwd_position(self._warp_model, self._warp_data)
        self._mjw.fwd_velocity(self._warp_model, self._warp_data)
        if synchronize and hasattr(self._wp, "synchronize"):
            self._wp.synchronize()

        cvel = self._wp.to_torch(self._warp_data.cvel)[:nworld].clone()
        return {
            "body_pos_w": self._wp.to_torch(self._warp_data.xpos)[:nworld].clone(),
            "body_lin_vel_w": cvel[..., 3:6].clone(),
            "body_quat_w": self._wp.to_torch(self._warp_data.xquat)[:nworld].clone(),
            "body_ang_vel_w": cvel[..., 0:3].clone(),
            "joint_pos": qpos_work[:nworld].index_select(1, self.joint_qpos_addrs).clone(),
            "joint_vel": qvel_work[:nworld].index_select(1, self.joint_dof_addrs).clone(),
        }

    def _forward_positions_warp(self, qpos: torch.Tensor) -> torch.Tensor:
        assert self._mjw is not None
        assert self._wp is not None
        assert self._warp_model is not None
        assert self._warp_data is not None

        qpos = self.to_device(qpos, name="qpos")
        nworld = int(qpos.shape[0])
        if nworld > self.batch_size:
            raise ValueError(f"Chunk size {nworld} exceeds batch_size {self.batch_size}")

        if nworld < self.batch_size:
            assert self._padded_qpos is not None
            self._padded_qpos[:nworld] = qpos
            self._padded_qpos[nworld:].zero_()
            warp_qpos = self._wp.from_torch(self._padded_qpos)
        else:
            warp_qpos = self._wp.from_torch(qpos)

        self._wp.copy(self._warp_data.qpos, warp_qpos)
        self._mjw.kinematics(self._warp_model, self._warp_data)
        if hasattr(self._wp, "synchronize"):
            self._wp.synchronize()
        return self._wp.to_torch(self._warp_data.xpos)[:nworld].clone()

    def _empty_contact_summary(self, frames: int, *, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        target = self.device if device is None else device
        return {
            "floor_min_dist": torch.full((frames,), 100.0, dtype=torch.float32, device=target),
            "non_floor_contact_count": torch.zeros((frames,), dtype=torch.float32, device=target),
            "contact_buffer_saturated": torch.zeros((frames,), dtype=torch.bool, device=target),
        }

    def _forward_contact_summary_cpu(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        *,
        floor_geom_id: int,
    ) -> dict[str, torch.Tensor]:
        orig_device = qpos.device
        qpos_cpu = qpos.to(device="cpu", dtype=torch.float32).contiguous()
        qvel_cpu = qvel.to(device="cpu", dtype=torch.float32).contiguous()
        qpos_np = qpos_cpu.numpy()
        qvel_np = qvel_cpu.numpy()
        frames = int(qpos_cpu.shape[0])

        floor_min_dist = torch.full((frames,), 100.0, dtype=torch.float32)
        non_floor_contact_count = torch.zeros((frames,), dtype=torch.float32)
        contact_buffer_saturated = torch.zeros((frames,), dtype=torch.bool)
        model_nconmax = int(getattr(self.model, "nconmax", 0) or 0)

        for frame_idx, (qpos_frame, qvel_frame) in enumerate(zip(qpos_np, qvel_np, strict=True)):
            self.data.qpos[:] = qpos_frame
            self.data.qvel[:] = qvel_frame
            mujoco.mj_forward(self.model, self.data)

            if model_nconmax > 0 and int(self.data.ncon) >= model_nconmax:
                contact_buffer_saturated[frame_idx] = True

            min_dist = 100.0
            non_floor_count = 0
            for contact_idx in range(int(self.data.ncon)):
                contact = self.data.contact[contact_idx]
                geom0 = int(contact.geom[0])
                geom1 = int(contact.geom[1])
                has_floor = geom0 == floor_geom_id or geom1 == floor_geom_id
                if has_floor:
                    min_dist = min(min_dist, float(contact.dist))
                elif float(contact.dist) < 0.0:
                    non_floor_count += 1
            floor_min_dist[frame_idx] = min_dist
            non_floor_contact_count[frame_idx] = float(non_floor_count)

        return {
            "floor_min_dist": floor_min_dist.to(device=orig_device),
            "non_floor_contact_count": non_floor_contact_count.to(device=orig_device),
            "contact_buffer_saturated": contact_buffer_saturated.to(device=orig_device),
        }

    def _clear_warp_contacts(self) -> None:
        assert self._wp is not None
        assert self._warp_data is not None
        contact = self._warp_data.contact
        self._wp.to_torch(contact.geom).fill_(-1)
        self._wp.to_torch(contact.worldid).fill_(-1)
        self._wp.to_torch(contact.dim).zero_()
        self._wp.to_torch(contact.dist).fill_(100.0)

    def _forward_contact_summary_warp(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        *,
        floor_geom_id: int,
        synchronize: bool,
    ) -> dict[str, torch.Tensor]:
        assert self._mjw is not None
        assert self._wp is not None
        assert self._warp_model is not None
        assert self._warp_data is not None

        qpos = self.to_device(qpos, name="qpos")
        qvel = self.to_device(qvel, name="qvel")
        nworld = int(qpos.shape[0])
        if nworld > self.batch_size:
            raise ValueError(f"Chunk size {nworld} exceeds batch_size {self.batch_size}")

        qpos_work = qpos
        qvel_work = qvel
        if nworld < self.batch_size:
            assert self._padded_qpos is not None
            assert self._padded_qvel is not None
            self._padded_qpos[:nworld] = qpos
            self._padded_qvel[:nworld] = qvel
            self._padded_qpos[nworld:].zero_()
            self._padded_qvel[nworld:].zero_()
            qpos_work = self._padded_qpos
            qvel_work = self._padded_qvel

        self._wp.copy(self._warp_data.qpos, self._wp.from_torch(qpos_work))
        self._wp.copy(self._warp_data.qvel, self._wp.from_torch(qvel_work))
        self._mjw.fwd_position(self._warp_model, self._warp_data)
        self._clear_warp_contacts()
        self._mjw.collision(self._warp_model, self._warp_data)
        if synchronize and hasattr(self._wp, "synchronize"):
            self._wp.synchronize()

        contact = self._warp_data.contact
        geom = self._wp.to_torch(contact.geom)
        dist = self._wp.to_torch(contact.dist).to(dtype=torch.float32)
        worldid = self._wp.to_torch(contact.worldid).to(dtype=torch.long)
        dim = self._wp.to_torch(contact.dim)

        active = (
            (worldid >= 0)
            & (worldid < nworld)
            & (dim > 0)
            & (geom[:, 0] >= 0)
            & (geom[:, 1] >= 0)
        )
        has_floor = (geom[:, 0] == int(floor_geom_id)) | (geom[:, 1] == int(floor_geom_id))
        floor_active = active & has_floor
        non_floor_active = active & (~has_floor) & (dist < 0.0)

        summary = self._empty_contact_summary(nworld)
        if floor_active.any():
            summary["floor_min_dist"].scatter_reduce_(
                0,
                worldid[floor_active],
                dist[floor_active],
                reduce="amin",
                include_self=True,
            )
        if non_floor_active.any():
            summary["non_floor_contact_count"].scatter_add_(
                0,
                worldid[non_floor_active],
                torch.ones_like(dist[non_floor_active], dtype=torch.float32),
            )
        if self.contact_nconmax is not None:
            active_count = torch.zeros((nworld,), dtype=torch.float32, device=self.device)
            if active.any():
                active_count.scatter_add_(
                    0,
                    worldid[active],
                    torch.ones_like(dist[active], dtype=torch.float32),
                )
            summary["contact_buffer_saturated"] = active_count >= float(self.contact_nconmax)
        return summary

    def forward_positions(self, qpos: torch.Tensor) -> torch.Tensor:
        normalized_qpos = self.to_device(qpos, name="qpos")
        if normalized_qpos.shape[0] == 0:
            return torch.zeros((0, self.model.nbody, 3), dtype=torch.float32, device=self.device)
        chunks: list[torch.Tensor] = []

        for start in range(0, normalized_qpos.shape[0], self.batch_size):
            stop = min(normalized_qpos.shape[0], start + self.batch_size)
            qpos_chunk = normalized_qpos[start:stop]
            if self.backend == "mujoco_warp":
                chunks.append(self._forward_positions_warp(qpos_chunk))
                continue
            chunks.append(self._forward_positions_cpu(qpos_chunk))

        return (
            torch.cat(chunks, dim=0)
            if chunks
            else torch.zeros((0, self.model.nbody, 3), dtype=torch.float32, device=self.device)
        )

    def forward_kinematics(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        *,
        synchronize: bool = True,
    ) -> dict[str, torch.Tensor]:
        normalized_qpos = self.to_device(qpos, name="qpos")
        normalized_qvel = self.to_device(qvel, name="qvel")
        if normalized_qpos.shape != (normalized_qpos.shape[0], self.model.nq):
            raise ValueError(f"Expected qpos shape [batch, {self.model.nq}], got {tuple(normalized_qpos.shape)}")
        if normalized_qvel.shape != (normalized_qvel.shape[0], self.model.nv):
            raise ValueError(f"Expected qvel shape [batch, {self.model.nv}], got {tuple(normalized_qvel.shape)}")
        if normalized_qpos.shape[0] != normalized_qvel.shape[0]:
            raise ValueError(
                f"Expected qpos/qvel to share batch dimension, got {normalized_qpos.shape[0]} and {normalized_qvel.shape[0]}"
            )
        if normalized_qpos.shape[0] == 0:
            empty = torch.zeros((0,), dtype=torch.float32, device=self.device)
            return {
                "body_pos_w": empty.view(0, self.model.nbody, 3),
                "body_lin_vel_w": empty.view(0, self.model.nbody, 3),
                "body_quat_w": empty.view(0, self.model.nbody, 4),
                "body_ang_vel_w": empty.view(0, self.model.nbody, 3),
                "joint_pos": empty.view(0, self.joint_qpos_addrs.shape[0]),
                "joint_vel": empty.view(0, self.joint_dof_addrs.shape[0]),
            }
        if self._fake_zero_outputs:
            return self._constant_kinematics_outputs(int(normalized_qpos.shape[0]))
        # self.backend = "cpu"
        if self.backend != "mujoco_warp":
            return self._forward_kinematics_cpu(normalized_qpos, normalized_qvel)

        chunk_outputs: list[dict[str, torch.Tensor]] = []
        for start in range(0, normalized_qpos.shape[0], self.batch_size):
            stop = min(normalized_qpos.shape[0], start + self.batch_size)
            chunk_outputs.append(
                self._forward_kinematics_warp(
                    normalized_qpos[start:stop],
                    normalized_qvel[start:stop],
                    synchronize=synchronize and stop >= normalized_qpos.shape[0],
                )
            )

        return {
            key: torch.cat([chunk[key] for chunk in chunk_outputs], dim=0)
            for key in chunk_outputs[0]
        }

    def forward_contact_summary(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        *,
        floor_geom_id: int,
        synchronize: bool = True,
    ) -> dict[str, torch.Tensor]:
        normalized_qpos = self.to_device(qpos, name="qpos")
        normalized_qvel = self.to_device(qvel, name="qvel")
        if normalized_qpos.shape != (normalized_qpos.shape[0], self.model.nq):
            raise ValueError(f"Expected qpos shape [batch, {self.model.nq}], got {tuple(normalized_qpos.shape)}")
        if normalized_qvel.shape != (normalized_qvel.shape[0], self.model.nv):
            raise ValueError(f"Expected qvel shape [batch, {self.model.nv}], got {tuple(normalized_qvel.shape)}")
        if normalized_qpos.shape[0] != normalized_qvel.shape[0]:
            raise ValueError(
                f"Expected qpos/qvel to share batch dimension, got {normalized_qpos.shape[0]} and {normalized_qvel.shape[0]}"
            )
        if normalized_qpos.shape[0] == 0:
            return self._empty_contact_summary(0)
        if self._fake_zero_outputs:
            return self._empty_contact_summary(int(normalized_qpos.shape[0]))
        if self.backend != "mujoco_warp":
            return self._forward_contact_summary_cpu(
                normalized_qpos,
                normalized_qvel,
                floor_geom_id=int(floor_geom_id),
            )

        chunk_outputs: list[dict[str, torch.Tensor]] = []
        for start in range(0, normalized_qpos.shape[0], self.batch_size):
            stop = min(normalized_qpos.shape[0], start + self.batch_size)
            chunk_outputs.append(
                self._forward_contact_summary_warp(
                    normalized_qpos[start:stop],
                    normalized_qvel[start:stop],
                    floor_geom_id=int(floor_geom_id),
                    synchronize=synchronize and stop >= normalized_qpos.shape[0],
                )
            )

        return {
            key: torch.cat([chunk[key] for chunk in chunk_outputs], dim=0)
            for key in chunk_outputs[0]
        }

    def forward_positions_many(self, qpos_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if not qpos_list:
            return []
        lengths = [int(qpos.shape[0]) for qpos in qpos_list]
        packed = torch.cat(qpos_list, dim=0)
        packed_positions = self.forward_positions(packed)
        return list(packed_positions.split(lengths, dim=0))

    def forward_kinematics_many(
        self,
        qpos_list: list[torch.Tensor],
        qvel_list: list[torch.Tensor],
    ) -> list[dict[str, torch.Tensor]]:
        if len(qpos_list) != len(qvel_list):
            raise ValueError(f"Expected qpos/qvel list lengths to match, got {len(qpos_list)} and {len(qvel_list)}")
        if not qpos_list:
            return []
        lengths = [int(qpos.shape[0]) for qpos in qpos_list]
        packed_qpos = torch.cat(qpos_list, dim=0)
        packed_qvel = torch.cat(qvel_list, dim=0)
        packed_outputs = self.forward_kinematics(packed_qpos, packed_qvel)
        split_outputs = {
            key: value.split(lengths, dim=0)
            for key, value in packed_outputs.items()
        }
        return [
            {key: split_outputs[key][idx] for key in split_outputs}
            for idx in range(len(lengths))
        ]
