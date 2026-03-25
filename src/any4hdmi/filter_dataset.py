from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from tqdm import tqdm

from any4hdmi.format import ensure_dir, load_manifest, load_motion, write_manifest

try:
    import warp as wp
except Exception:
    wp = None


REPORT_NAME = "filter_report.json"
DEFAULT_MAX_ROOT_QVEL = 10.0
DEFAULT_MIN_FRAMES = 250
DEFAULT_ALL_OFF_GROUND_Z = 0.2
DEFAULT_MAX_ALL_OFF_GROUND_SECONDS = 1.0
DEFAULT_MIN_MAX_BODY_Z = 0.2
DEFAULT_BATCH_SIZE = 2048


if wp is not None:

    @wp.kernel
    def _root_qvel_kernel(
        qpos_in: wp.array2d(dtype=float),
        frame_clip_ids: wp.array(dtype=int),
        clip_dt: wp.array(dtype=float),
        root_qvel_out: wp.array2d(dtype=float),
        total_frames: int,
    ):
        frame_id = wp.tid()
        if frame_id >= total_frames:
            return

        clip_id = frame_clip_ids[frame_id]
        if frame_id + 1 >= total_frames or frame_clip_ids[frame_id + 1] != clip_id:
            for k in range(6):
                root_qvel_out[frame_id, k] = 0.0
            return

        dt = clip_dt[clip_id]
        inv_dt = 1.0 / dt

        root_qvel_out[frame_id, 0] = (qpos_in[frame_id + 1, 0] - qpos_in[frame_id, 0]) * inv_dt
        root_qvel_out[frame_id, 1] = (qpos_in[frame_id + 1, 1] - qpos_in[frame_id, 1]) * inv_dt
        root_qvel_out[frame_id, 2] = (qpos_in[frame_id + 1, 2] - qpos_in[frame_id, 2]) * inv_dt

        w1 = qpos_in[frame_id, 3]
        x1 = -qpos_in[frame_id, 4]
        y1 = -qpos_in[frame_id, 5]
        z1 = -qpos_in[frame_id, 6]
        w2 = qpos_in[frame_id + 1, 3]
        x2 = qpos_in[frame_id + 1, 4]
        y2 = qpos_in[frame_id + 1, 5]
        z2 = qpos_in[frame_id + 1, 6]

        dw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        dx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        dy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        dz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        if dw < 0.0:
            dw = -dw
            dx = -dx
            dy = -dy
            dz = -dz

        sin_half = wp.sqrt(dx * dx + dy * dy + dz * dz)
        cos_half = wp.max(-1.0, wp.min(1.0, dw))

        if sin_half > 1.0e-12:
            angle = 2.0 * wp.atan2(sin_half, cos_half)
            scale = angle * inv_dt / sin_half
            root_qvel_out[frame_id, 3] = dx * scale
            root_qvel_out[frame_id, 4] = dy * scale
            root_qvel_out[frame_id, 5] = dz * scale
        else:
            root_qvel_out[frame_id, 3] = 0.0
            root_qvel_out[frame_id, 4] = 0.0
            root_qvel_out[frame_id, 5] = 0.0

    @wp.kernel
    def _copy_last_root_qvel_kernel(
        clip_starts: wp.array(dtype=int),
        clip_lengths: wp.array(dtype=int),
        root_qvel_out: wp.array2d(dtype=float),
        total_clips: int,
    ):
        clip_id = wp.tid()
        if clip_id >= total_clips:
            return

        clip_start = clip_starts[clip_id]
        clip_length = clip_lengths[clip_id]
        last_frame = clip_start + clip_length - 1

        if clip_length <= 1:
            for k in range(6):
                root_qvel_out[last_frame, k] = 0.0
            return

        prev_frame = last_frame - 1
        for k in range(6):
            root_qvel_out[last_frame, k] = root_qvel_out[prev_frame, k]


@dataclass(frozen=True)
class FilterConfig:
    max_root_qvel: float
    min_frames: int
    all_off_ground_z: float
    max_all_off_ground_seconds: float
    min_max_body_z: float
    batch_size: int


@dataclass(frozen=True)
class MotionCheckResult:
    is_valid: bool
    reasons: tuple[str, ...]
    num_frames: int
    fps: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a unified any4hdmi dataset with FK-based motion sanity checks."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Input dataset root that contains manifest.json.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output dataset root for kept motions. Defaults to <input-root>_filtered.",
    )
    parser.add_argument(
        "--mjcf-path",
        default=None,
        help="Optional MJCF override. Useful when manifest.json points to a stale path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run checks and emit a report without copying kept motions.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit path for the JSON filter report.",
    )
    parser.add_argument(
        "--max-root-qvel",
        type=float,
        default=DEFAULT_MAX_ROOT_QVEL,
        help="Reject clips when any of the first 6 qvel dimensions exceed this absolute value.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help="Reject clips shorter than this many frames.",
    )
    parser.add_argument(
        "--all-off-ground-z",
        type=float,
        default=DEFAULT_ALL_OFF_GROUND_Z,
        help="A frame counts as all-off-ground when every body z is above this threshold.",
    )
    parser.add_argument(
        "--max-all-off-ground-seconds",
        type=float,
        default=DEFAULT_MAX_ALL_OFF_GROUND_SECONDS,
        help="Reject clips when all bodies remain off-ground longer than this duration.",
    )
    parser.add_argument(
        "--min-max-body-z",
        type=float,
        default=DEFAULT_MIN_MAX_BODY_Z,
        help="Reject clips whose maximum body height never exceeds this threshold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="FK batch size. This limits per-call MuJoCo Warp or CPU work.",
    )
    return parser.parse_args()


def _resolve_output_root(input_root: Path, output_root_arg: str | None) -> Path:
    if output_root_arg is not None:
        return Path(output_root_arg).expanduser().resolve()
    return input_root.with_name(f"{input_root.name}_filtered")


def _resolve_mjcf_path(_manifest_root: Path, manifest: Any, mjcf_override: str | None) -> Path:
    if mjcf_override:
        mjcf_path = Path(mjcf_override).expanduser().resolve()
        if not mjcf_path.is_file():
            raise FileNotFoundError(f"MJCF override not found: {mjcf_path}")
        return mjcf_path

    return manifest.mjcf_path


def _load_motion_fps(motion_path: Path, manifest_timestep: float) -> float:
    if manifest_timestep <= 0.0:
        raise ValueError(f"Invalid manifest timestep: {manifest_timestep}")
    return 1.0 / manifest_timestep


def _compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos to be rank 2, got shape {qpos.shape}")
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Motion qpos width {qpos.shape[1]} does not match model.nq={model.nq}")

    root_qvel = np.zeros((qpos.shape[0], 6), dtype=np.float64)
    if qpos.shape[0] <= 1:
        return np.asarray(root_qvel, dtype=np.float32)

    dt = 1.0 / fps
    root_qvel[:-1, :3] = (qpos[1:, :3] - qpos[:-1, :3]) / dt

    q1 = qpos[:-1, 3:7]
    q2 = qpos[1:, 3:7]

    q1_conj = q1.copy()
    q1_conj[:, 1:] *= -1.0

    w1, x1, y1, z1 = np.moveaxis(q1_conj, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    q_delta = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )

    negative_w = q_delta[:, 0] < 0.0
    q_delta[negative_w] *= -1.0

    vec = q_delta[:, 1:]
    sin_half = np.linalg.norm(vec, axis=-1, keepdims=True)
    cos_half = np.clip(q_delta[:, 0:1], -1.0, 1.0)
    angle = 2.0 * np.arctan2(sin_half, cos_half)
    axis = np.divide(vec, sin_half, out=np.zeros_like(vec), where=sin_half > 1e-12)
    root_qvel[:-1, 3:] = axis * (angle / dt)
    root_qvel[-1] = root_qvel[-2]
    return np.asarray(root_qvel, dtype=np.float32)


class FKRunner:
    def __init__(self, mjcf_path: Path, batch_size: int) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.batch_size = max(1, int(batch_size))
        self.backend = "mujoco"
        self._warp_model = None
        self._wp = None
        self._mjw = None
        self._warp_data = None
        self._warp_qpos_host = None
        self._warp_qpos_host_view = None
        self._warp_root_qvel_capacity = 0
        self._warp_root_qvel_clip_capacity = 0
        self._warp_root_qvel_qpos_host = None
        self._warp_root_qvel_qpos_host_view = None
        self._warp_root_qvel_qpos_device = None
        self._warp_root_qvel_frame_clip_ids_host = None
        self._warp_root_qvel_frame_clip_ids_host_view = None
        self._warp_root_qvel_frame_clip_ids_device = None
        self._warp_root_qvel_clip_dt_host = None
        self._warp_root_qvel_clip_dt_host_view = None
        self._warp_root_qvel_clip_dt_device = None
        self._warp_root_qvel_clip_starts_host = None
        self._warp_root_qvel_clip_starts_host_view = None
        self._warp_root_qvel_clip_starts_device = None
        self._warp_root_qvel_clip_lengths_host = None
        self._warp_root_qvel_clip_lengths_host_view = None
        self._warp_root_qvel_clip_lengths_device = None
        self._warp_root_qvel_out = None
        self._warned_about_warp_fallback = False

        try:
            import mujoco_warp as mjw

            self._mjw = mjw
            self._wp = wp
            self._warp_model = mjw.put_model(self.model)
            self._warp_data = mjw.make_data(self.model, nworld=self.batch_size)
            self._warp_qpos_host = self._alloc_host_buffer((self.batch_size, self.model.nq), dtype=float)
            self._warp_qpos_host_view = self._warp_qpos_host.numpy()
            self._ensure_root_qvel_capacity(self.batch_size, self.batch_size)
            self.backend = "mujoco_warp"
        except Exception:
            self.backend = "mujoco"

    def _alloc_host_buffer(self, shape, *, dtype):
        assert self._wp is not None
        try:
            return self._wp.zeros(shape, dtype=dtype, device="cpu", pinned=True)
        except Exception:
            return self._wp.zeros(shape, dtype=dtype, device="cpu", pinned=False)

    def _ensure_root_qvel_capacity(self, frame_capacity: int, clip_capacity: int) -> None:
        assert self._wp is not None
        assert self._warp_data is not None

        if frame_capacity > self._warp_root_qvel_capacity:
            self._warp_root_qvel_capacity = frame_capacity
            self._warp_root_qvel_qpos_host = self._alloc_host_buffer(
                (frame_capacity, self.model.nq),
                dtype=float,
            )
            self._warp_root_qvel_qpos_host_view = self._warp_root_qvel_qpos_host.numpy()
            self._warp_root_qvel_qpos_device = self._wp.zeros(
                (frame_capacity, self.model.nq),
                dtype=float,
                device=self._warp_data.qpos.device,
            )
            self._warp_root_qvel_frame_clip_ids_host = self._alloc_host_buffer(
                frame_capacity,
                dtype=self._wp.int32,
            )
            self._warp_root_qvel_frame_clip_ids_host_view = (
                self._warp_root_qvel_frame_clip_ids_host.numpy()
            )
            self._warp_root_qvel_frame_clip_ids_device = self._wp.zeros(
                frame_capacity,
                dtype=self._wp.int32,
                device=self._warp_data.qpos.device,
            )
            self._warp_root_qvel_out = self._wp.zeros(
                (frame_capacity, 6),
                dtype=float,
                device=self._warp_data.qpos.device,
            )

        if clip_capacity > self._warp_root_qvel_clip_capacity:
            self._warp_root_qvel_clip_capacity = clip_capacity
            self._warp_root_qvel_clip_dt_host = self._alloc_host_buffer(
                clip_capacity,
                dtype=float,
            )
            self._warp_root_qvel_clip_dt_host_view = self._warp_root_qvel_clip_dt_host.numpy()
            self._warp_root_qvel_clip_dt_device = self._wp.zeros(
                clip_capacity,
                dtype=float,
                device=self._warp_data.qpos.device,
            )
            self._warp_root_qvel_clip_starts_host = self._alloc_host_buffer(
                clip_capacity,
                dtype=self._wp.int32,
            )
            self._warp_root_qvel_clip_starts_host_view = (
                self._warp_root_qvel_clip_starts_host.numpy()
            )
            self._warp_root_qvel_clip_starts_device = self._wp.zeros(
                clip_capacity,
                dtype=self._wp.int32,
                device=self._warp_data.qpos.device,
            )
            self._warp_root_qvel_clip_lengths_host = self._alloc_host_buffer(
                clip_capacity,
                dtype=self._wp.int32,
            )
            self._warp_root_qvel_clip_lengths_host_view = (
                self._warp_root_qvel_clip_lengths_host.numpy()
            )
            self._warp_root_qvel_clip_lengths_device = self._wp.zeros(
                clip_capacity,
                dtype=self._wp.int32,
                device=self._warp_data.qpos.device,
            )

    def _forward_positions_cpu(self, qpos: np.ndarray) -> np.ndarray:
        xpos = np.zeros((qpos.shape[0], self.model.nbody, 3), dtype=np.float32)
        for frame_idx, qpos_frame in enumerate(qpos):
            self.data.qpos[:] = qpos_frame
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            xpos[frame_idx] = np.asarray(self.data.xpos, dtype=np.float32)
        return xpos

    def _forward_positions_warp(self, qpos: np.ndarray) -> np.ndarray:
        assert self._mjw is not None
        assert self._wp is not None
        assert self._warp_model is not None
        nworld = int(qpos.shape[0])
        assert self._warp_data is not None
        assert self._warp_qpos_host is not None
        assert self._warp_qpos_host_view is not None
        if nworld > self.batch_size:
            raise ValueError(f"Chunk size {nworld} exceeds batch_size {self.batch_size}")

        self._warp_qpos_host_view[:nworld] = np.asarray(qpos, dtype=np.float32)
        if nworld < self.batch_size:
            self._warp_qpos_host_view[nworld:] = 0.0

        self._wp.copy(self._warp_data.qpos, self._warp_qpos_host)
        self._mjw.kinematics(self._warp_model, self._warp_data)
        if hasattr(self._wp, "synchronize"):
            self._wp.synchronize()
        return np.asarray(self._warp_data.xpos.numpy(), dtype=np.float32)[:nworld]

    def forward_positions(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, qpos.shape[0], self.batch_size):
            stop = min(qpos.shape[0], start + self.batch_size)
            qpos_chunk = qpos[start:stop]
            if self.backend == "mujoco_warp":
                try:
                    chunks.append(self._forward_positions_warp(qpos_chunk))
                    continue
                except Exception as exc:
                    if not self._warned_about_warp_fallback:
                        print(f"mujoco_warp FK failed, falling back to CPU MuJoCo: {exc}")
                        self._warned_about_warp_fallback = True
                    self.backend = "mujoco"
            chunks.append(self._forward_positions_cpu(qpos_chunk))
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.model.nbody, 3))

    def _compute_root_qvel_many_warp(
        self,
        qpos_list: list[np.ndarray],
        fps_list: list[float],
    ) -> list[np.ndarray]:
        assert self._wp is not None
        assert self._warp_root_qvel_qpos_host is not None
        assert self._warp_root_qvel_qpos_host_view is not None
        assert self._warp_root_qvel_qpos_device is not None
        assert self._warp_root_qvel_frame_clip_ids_host is not None
        assert self._warp_root_qvel_frame_clip_ids_host_view is not None
        assert self._warp_root_qvel_frame_clip_ids_device is not None
        assert self._warp_root_qvel_clip_dt_host is not None
        assert self._warp_root_qvel_clip_dt_host_view is not None
        assert self._warp_root_qvel_clip_dt_device is not None
        assert self._warp_root_qvel_clip_starts_host is not None
        assert self._warp_root_qvel_clip_starts_host_view is not None
        assert self._warp_root_qvel_clip_starts_device is not None
        assert self._warp_root_qvel_clip_lengths_host is not None
        assert self._warp_root_qvel_clip_lengths_host_view is not None
        assert self._warp_root_qvel_clip_lengths_device is not None
        assert self._warp_root_qvel_out is not None

        lengths = np.asarray([qpos.shape[0] for qpos in qpos_list], dtype=np.int32)
        total_frames = int(lengths.sum())
        total_clips = len(qpos_list)
        self._ensure_root_qvel_capacity(total_frames, total_clips)

        merged_qpos = np.concatenate(qpos_list, axis=0).astype(np.float32, copy=False)
        frame_clip_ids = np.repeat(np.arange(total_clips, dtype=np.int32), lengths)
        clip_dt = np.asarray([1.0 / fps for fps in fps_list], dtype=np.float32)
        clip_starts = np.zeros(total_clips, dtype=np.int32)
        if total_clips > 1:
            clip_starts[1:] = np.cumsum(lengths[:-1], dtype=np.int32)

        self._warp_root_qvel_qpos_host_view[:total_frames] = merged_qpos
        if total_frames < self._warp_root_qvel_capacity:
            self._warp_root_qvel_qpos_host_view[total_frames:] = 0.0

        self._warp_root_qvel_frame_clip_ids_host_view[:total_frames] = frame_clip_ids
        if total_frames < self._warp_root_qvel_capacity:
            self._warp_root_qvel_frame_clip_ids_host_view[total_frames:] = 0

        self._warp_root_qvel_clip_dt_host_view[:total_clips] = clip_dt
        self._warp_root_qvel_clip_starts_host_view[:total_clips] = clip_starts
        self._warp_root_qvel_clip_lengths_host_view[:total_clips] = lengths
        if total_clips < self._warp_root_qvel_clip_capacity:
            self._warp_root_qvel_clip_dt_host_view[total_clips:] = 0.0
            self._warp_root_qvel_clip_starts_host_view[total_clips:] = 0
            self._warp_root_qvel_clip_lengths_host_view[total_clips:] = 0

        self._wp.copy(self._warp_root_qvel_qpos_device, self._warp_root_qvel_qpos_host)
        self._wp.copy(
            self._warp_root_qvel_frame_clip_ids_device,
            self._warp_root_qvel_frame_clip_ids_host,
        )
        self._wp.copy(self._warp_root_qvel_clip_dt_device, self._warp_root_qvel_clip_dt_host)
        self._wp.copy(
            self._warp_root_qvel_clip_starts_device,
            self._warp_root_qvel_clip_starts_host,
        )
        self._wp.copy(
            self._warp_root_qvel_clip_lengths_device,
            self._warp_root_qvel_clip_lengths_host,
        )

        self._wp.launch(
            kernel=_root_qvel_kernel,
            dim=total_frames,
            inputs=[
                self._warp_root_qvel_qpos_device,
                self._warp_root_qvel_frame_clip_ids_device,
                self._warp_root_qvel_clip_dt_device,
                self._warp_root_qvel_out,
                total_frames,
            ],
            device=self._warp_data.qpos.device,
        )
        self._wp.launch(
            kernel=_copy_last_root_qvel_kernel,
            dim=total_clips,
            inputs=[
                self._warp_root_qvel_clip_starts_device,
                self._warp_root_qvel_clip_lengths_device,
                self._warp_root_qvel_out,
                total_clips,
            ],
            device=self._warp_data.qpos.device,
        )
        if hasattr(self._wp, "synchronize"):
            self._wp.synchronize()

        merged_root_qvel = np.asarray(self._warp_root_qvel_out.numpy(), dtype=np.float32)[:total_frames]
        outputs: list[np.ndarray] = []
        start = 0
        for length in lengths.tolist():
            stop = start + length
            outputs.append(merged_root_qvel[start:stop].copy())
            start = stop
        return outputs

    def forward_positions_many(
        self,
        qpos_list: list[np.ndarray],
        fps_list: list[float],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not qpos_list:
            return [], []
        lengths = [int(qpos.shape[0]) for qpos in qpos_list]
        merged_qpos = np.concatenate(qpos_list, axis=0)
        merged_xpos = self.forward_positions(merged_qpos)
        if self.backend == "mujoco_warp" and wp is not None:
            root_qvel_list = self._compute_root_qvel_many_warp(qpos_list, fps_list)
        else:
            root_qvel_list = [
                _compute_qvel(self.model, qpos, fps)
                for qpos, fps in zip(qpos_list, fps_list, strict=True)
            ]
        outputs: list[np.ndarray] = []
        start = 0
        for length in lengths:
            stop = start + length
            outputs.append(merged_xpos[start:stop])
            start = stop
        return outputs, root_qvel_list


def _contiguous_true_run_length(mask: np.ndarray) -> int:
    if mask.size == 0 or not np.any(mask):
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    run_starts = np.where(edges == 1)[0]
    run_ends = np.where(edges == -1)[0]
    if run_starts.size == 0:
        return 0
    return int(np.max(run_ends - run_starts))


def _check_motion(
    *,
    qvel: np.ndarray,
    xpos: np.ndarray,
    fps: float,
    config: FilterConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []

    root_dims = min(6, qvel.shape[1])
    if root_dims > 0 and np.any(np.abs(qvel[:, :root_dims]) > config.max_root_qvel):
        reasons.append("root_qvel_spike")

    if xpos.shape[0] < config.min_frames:
        reasons.append("too_short")

    body_xpos = xpos[:, 1:, :]
    if body_xpos.shape[1] == 0:
        reasons.append("no_dynamic_bodies")
        return tuple(reasons)

    min_body_z = np.min(body_xpos[:, :, 2], axis=1)
    all_off_ground = min_body_z > config.all_off_ground_z
    max_run_frames = _contiguous_true_run_length(all_off_ground)
    max_run_seconds = max_run_frames / fps if fps > 0.0 else 0.0
    if max_run_seconds > config.max_all_off_ground_seconds:
        reasons.append("all_bodies_off_ground_too_long")

    max_body_z = float(np.max(body_xpos[:, :, 2]))
    if max_body_z <= config.min_max_body_z:
        reasons.append("max_body_height_too_low")

    return tuple(reasons)


def _copy_motion(src_motion: Path, dst_motion: Path) -> None:
    dst_motion.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_motion, dst_motion)


def _build_filter_source(manifest_payload: dict[str, Any], config: FilterConfig, backend: str) -> dict[str, Any]:
    source = dict(manifest_payload.get("source", {}))
    source["filter"] = {
        "type": "fk_sanity_filter",
        "backend": backend,
        "max_root_qvel": config.max_root_qvel,
        "min_frames": config.min_frames,
        "all_off_ground_z": config.all_off_ground_z,
        "max_all_off_ground_seconds": config.max_all_off_ground_seconds,
        "min_max_body_z": config.min_max_body_z,
        "batch_size": config.batch_size,
    }
    return source


def _process_motion_batch(
    *,
    batch_items: list[dict[str, Any]],
    fk_runner: FKRunner,
    config: FilterConfig,
    output_root: Path,
    dry_run: bool,
    motion_entries: list[dict[str, Any]],
    reason_counts: Counter[str],
) -> int:
    if not batch_items:
        return 0

    xpos_list, qvel_list = fk_runner.forward_positions_many(
        [item["qpos"] for item in batch_items],
        [item["fps"] for item in batch_items],
    )
    kept_count = 0

    for item, xpos, qvel in zip(batch_items, xpos_list, qvel_list, strict=True):
        reasons = _check_motion(
            qvel=qvel,
            xpos=xpos,
            fps=item["fps"],
            config=config,
        )
        is_valid = len(reasons) == 0
        result = MotionCheckResult(
            is_valid=is_valid,
            reasons=reasons,
            num_frames=int(item["qpos"].shape[0]),
            fps=float(item["fps"]),
        )

        entry = {
            "motion": item["rel_motion"].as_posix(),
            "status": "kept" if result.is_valid else "rejected",
            "reasons": list(result.reasons),
            "num_frames": result.num_frames,
            "fps": result.fps,
        }
        motion_entries.append(entry)

        if result.is_valid:
            kept_count += 1
            if not dry_run:
                dst_motion = output_root / item["rel_motion"]
                _copy_motion(item["motion_path"], dst_motion)
        else:
            reason_counts.update(result.reasons)

    batch_items.clear()
    return kept_count


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    manifest = load_manifest(input_root)
    output_root = _resolve_output_root(input_root, args.output_root)

    if not args.dry_run and output_root == input_root:
        raise ValueError("--output-root must differ from --input-root unless --dry-run is used")

    mjcf_path = _resolve_mjcf_path(input_root, manifest, args.mjcf_path)
    fk_runner = FKRunner(mjcf_path=mjcf_path, batch_size=args.batch_size)
    config = FilterConfig(
        max_root_qvel=float(args.max_root_qvel),
        min_frames=int(args.min_frames),
        all_off_ground_z=float(args.all_off_ground_z),
        max_all_off_ground_seconds=float(args.max_all_off_ground_seconds),
        min_max_body_z=float(args.min_max_body_z),
        batch_size=int(args.batch_size),
    )

    motions_dir = input_root / manifest.payload["motions_subdir"]
    motion_paths = sorted(motions_dir.rglob("*.npz"))
    if not motion_paths:
        raise FileNotFoundError(f"No motion files found under {motions_dir}")

    if not args.dry_run:
        ensure_dir(output_root)

    kept_count = 0
    motion_entries: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    batch_items: list[dict[str, Any]] = []
    batch_frames = 0

    with tqdm(total=len(motion_paths), desc="Filtering", unit="motion") as progress:
        for motion_path in motion_paths:
            rel_motion = motion_path.relative_to(input_root)
            qpos = load_motion(motion_path)
            fps = _load_motion_fps(motion_path, manifest.timestep)

            batch_items.append(
                {
                    "motion_path": motion_path,
                    "rel_motion": rel_motion,
                    "qpos": qpos,
                    "fps": fps,
                }
            )
            batch_frames += int(qpos.shape[0])

            if batch_frames >= config.batch_size:
                processed_count = len(batch_items)
                kept_count += _process_motion_batch(
                    batch_items=batch_items,
                    fk_runner=fk_runner,
                    config=config,
                    output_root=output_root,
                    dry_run=args.dry_run,
                    motion_entries=motion_entries,
                    reason_counts=reason_counts,
                )
                progress.update(processed_count)
                batch_frames = 0

        if batch_items:
            processed_count = len(batch_items)
            kept_count += _process_motion_batch(
                batch_items=batch_items,
                fk_runner=fk_runner,
                config=config,
                output_root=output_root,
                dry_run=args.dry_run,
                motion_entries=motion_entries,
                reason_counts=reason_counts,
            )
            progress.update(processed_count)

    report = {
        "input_root": str(input_root),
        "output_root": None if args.dry_run else str(output_root),
        "mjcf_path": str(mjcf_path),
        "backend": fk_runner.backend,
        "summary": {
            "total_motions": len(motion_paths),
            "kept_motions": kept_count,
            "rejected_motions": len(motion_paths) - kept_count,
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "thresholds": {
            "max_root_qvel": config.max_root_qvel,
            "min_frames": config.min_frames,
            "all_off_ground_z": config.all_off_ground_z,
            "max_all_off_ground_seconds": config.max_all_off_ground_seconds,
            "min_max_body_z": config.min_max_body_z,
            "batch_size": config.batch_size,
        },
        "motions": motion_entries,
    }

    report_path: Path | None
    if args.report_path is not None:
        report_path = Path(args.report_path).expanduser().resolve()
    elif args.dry_run:
        report_path = None
    else:
        report_path = output_root / REPORT_NAME

    if not args.dry_run:
        source = _build_filter_source(manifest.payload, config, fk_runner.backend)
        write_manifest(
            output_root,
            dataset_name=manifest.dataset_name,
            mjcf=manifest.payload["mjcf"],
            timestep=manifest.timestep,
            qpos_names=list(manifest.payload["qpos_names"]),
            num_motions=kept_count,
            source=source,
        )
        source_manifest_path = output_root / "manifest.source.json"
        source_manifest_path.write_text(
            json.dumps(
                {
                    "copied_from": str(input_root / "manifest.json"),
                    "resolved_original_mjcf_cache_path": str(mjcf_path),
                    "original_manifest": manifest.payload,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "backend": fk_runner.backend,
                "total_motions": len(motion_paths),
                "kept_motions": kept_count,
                "rejected_motions": len(motion_paths) - kept_count,
                "report_path": None if report_path is None else str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
