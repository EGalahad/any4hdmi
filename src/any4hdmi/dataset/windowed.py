from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from any4hdmi.dataset.base import BaseDataset, DatasetIndex, MotionData, MotionSample
from any4hdmi.dataset.fk_cache import FKCacheEntry


RUNTIME_MOTION_MAX_LEN = 512
# NEXT_WINDOW_DEVICE = torch.device("cpu")
NEXT_WINDOW_DEVICE = None
NEXT_WINDOW_FLOAT_DTYPE = torch.float16
CURRENT_WINDOW_FLOAT_DTYPE = torch.float32

_MOTION_DATA_FIELD_NAMES = (
    "motion_id",
    "step",
    "body_pos_w",
    "body_lin_vel_w",
    "body_quat_w",
    "body_ang_vel_w",
    "joint_pos",
    "joint_vel",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass
class _PrefetchProfileStats:
    calls: int = 0
    current_hit_calls: int = 0
    prev_hit_calls: int = 0
    next_hit_calls: int = 0
    cross_window_calls: int = 0
    sample_return_calls: int = 0
    next_wait_calls: int = 0
    sync_refill_calls: int = 0
    background_jobs_submitted: int = 0
    background_frames_total: int = 0
    background_load_wall_time_s: float = 0.0
    blocking_wait_wall_time_s: float = 0.0
    pool_hit_motions: int = 0
    pool_miss_motions: int = 0
    wait_next_ready_calls: int = 0
    wait_next_ready_wall_time_s: float = 0.0


@dataclass
class _MotionLoadResult:
    env_ids: torch.Tensor
    motion: MotionData
    load_wall_time_s: float


@dataclass
class _PendingMotionLoad:
    env_ids: torch.Tensor
    profile_name: str
    future: Future[_MotionLoadResult]




class WindowedMotionDataset(BaseDataset):
    def __init__(
        self,
        *,
        body_names: list[str],
        joint_names: list[str],
        motion_paths: list[Path],
        starts: list[int],
        ends: list[int],
        storage_fields: dict[str, torch.Tensor],
        num_envs: int,
        device: torch.device | str | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.body_names = list(body_names)
        self.joint_names = list(joint_names)
        self.motion_paths = list(motion_paths)
        self._num_envs = int(num_envs)
        self._storage_cpu = storage_fields
        self._motion_id_index_cpu = storage_fields["motion_id"]
        self._step_index_cpu = storage_fields["step"]
        self._storage_total_length = int(self._motion_id_index_cpu.shape[0])
        self._window_steps_cpu = torch.arange(RUNTIME_MOTION_MAX_LEN, dtype=torch.long, device="cpu")
        self._storage_starts_cpu = torch.as_tensor(starts, dtype=torch.long, device="cpu")
        self._storage_ends_cpu = torch.as_tensor(ends, dtype=torch.long, device="cpu")
        self.starts = self._storage_starts_cpu.clone()
        self.ends = self._storage_ends_cpu.clone()
        self.lengths = self.ends - self.starts
        self.data = DatasetIndex(motion_id=self._motion_id_index_cpu, step=self._step_index_cpu)

        self._fake_sample_motion = os.environ.get("ANY4HDMI_FAKE_SAMPLE_MOTION", "0") == "1"
        self._fake_get_slice = os.environ.get("ANY4HDMI_FAKE_GET_SLICE", "0") == "1"
        self._profile_enabled = os.environ.get("ANY4HDMI_PROFILE_CACHE_IDS", "0") == "1"
        self._profile_print_every = _env_int("ANY4HDMI_PROFILE_CACHE_IDS_PRINT_EVERY", 50)
        self._profile_focus = os.environ.get("ANY4HDMI_PROFILE_CACHE_IDS_FOCUS")
        self._profile_stats: dict[str, _PrefetchProfileStats] = {}

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="any4hdmi-motion-pool")
        self._pending_loads: list[_PendingMotionLoad] = []

        self.to(device if device is not None else torch.device("cpu"))
        if self._fake_sample_motion:
            print("[any4hdmi][online_dataset] ANY4HDMI_FAKE_SAMPLE_MOTION=1, returning canonical MotionSample")
        if self._fake_get_slice:
            print("[any4hdmi][online_dataset] ANY4HDMI_FAKE_GET_SLICE=1, returning default pose MotionData")

    @classmethod
    def from_cache_entry(
        cls,
        entry: FKCacheEntry,
        *,
        num_envs: int,
        device: torch.device | str | None = None,
    ) -> WindowedMotionDataset:
        return cls(
            body_names=entry.body_names,
            joint_names=entry.joint_names,
            motion_paths=entry.motion_paths,
            starts=entry.starts,
            ends=entry.ends,
            storage_fields=entry.storage_fields,
            num_envs=num_envs,
            device=device,
        )

    def __del__(self) -> None:
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _allocate_window_pool(
        self,
        *,
        device: torch.device,
        float_dtype: torch.dtype,
    ) -> MotionData:
        pool_shape = (self._num_envs, RUNTIME_MOTION_MAX_LEN)
        body_count = len(self.body_names)
        joint_count = len(self.joint_names)
        pool = MotionData(
            motion_id=torch.full(pool_shape, -1, dtype=torch.long, device=device),
            step=torch.zeros(pool_shape, dtype=torch.long, device=device),
            body_pos_w=torch.empty(
                (self._num_envs, RUNTIME_MOTION_MAX_LEN, body_count, 3),
                dtype=float_dtype,
                device=device,
            ),
            body_lin_vel_w=torch.empty(
                (self._num_envs, RUNTIME_MOTION_MAX_LEN, body_count, 3),
                dtype=float_dtype,
                device=device,
            ),
            body_quat_w=torch.empty(
                (self._num_envs, RUNTIME_MOTION_MAX_LEN, body_count, 4),
                dtype=float_dtype,
                device=device,
            ),
            body_ang_vel_w=torch.empty(
                (self._num_envs, RUNTIME_MOTION_MAX_LEN, body_count, 3),
                dtype=float_dtype,
                device=device,
            ),
            joint_pos=torch.empty(
                (self._num_envs, RUNTIME_MOTION_MAX_LEN, joint_count),
                dtype=float_dtype,
                device=device,
            ),
            joint_vel=torch.empty(
                (self._num_envs, RUNTIME_MOTION_MAX_LEN, joint_count),
                dtype=float_dtype,
                device=device,
            ),
            batch_size=pool_shape,
            device=device,
        )
        pool.zero_()
        pool.body_pos_w[..., 2] = 1.0
        pool.body_quat_w[..., 0] = 1.0
        return pool

    def _reset_runtime_state(self) -> None:
        self._current_window = self._allocate_window_pool(
            device=self.device,
            float_dtype=CURRENT_WINDOW_FLOAT_DTYPE,
        )
        self._next_window = self._allocate_window_pool(
            device=self.next_window_device,
            float_dtype=NEXT_WINDOW_FLOAT_DTYPE,
        )
        self._pending_loads = []
        self._env_current_motion_id = torch.full((self._num_envs,), -1, dtype=torch.long, device=self.device)
        self._env_current_source_start_t = torch.zeros((self._num_envs,), dtype=torch.long, device=self.device)
        self._env_current_window_len = torch.zeros((self._num_envs,), dtype=torch.long, device=self.device)
        self._env_next_motion_id = torch.full((self._num_envs,), -1, dtype=torch.long, device=self.device)
        self._env_next_source_start_t = torch.zeros((self._num_envs,), dtype=torch.long, device=self.device)
        self._env_next_window_len = torch.zeros((self._num_envs,), dtype=torch.long, device=self.device)

        all_env_ids = torch.arange(self._num_envs, device=self.device, dtype=torch.long)
        current_motion_ids, current_source_start_t, current_window_len = self._draw_uniform_window_specs(
            self._num_envs
        )
        self._assign_current_window_metadata(
            all_env_ids,
            motion_ids=current_motion_ids,
            source_start_t=current_source_start_t,
            window_len=current_window_len,
        )
        self._load_current_windows_sync(
            all_env_ids,
            motion_ids=current_motion_ids,
            source_start_t=current_source_start_t,
        )
        next_motion_ids, next_source_start_t, next_window_len = self._draw_uniform_window_specs(self._num_envs)
        self._assign_next_window_metadata(
            all_env_ids,
            motion_ids=next_motion_ids,
            source_start_t=next_source_start_t,
            window_len=next_window_len,
        )
        self._schedule_next_window_prefetch(
            all_env_ids,
            motion_ids=next_motion_ids,
            source_start_t=next_source_start_t,
            window_len=next_window_len,
            profile_name="bootstrap",
        )

    def to(self, device: torch.device | str) -> WindowedMotionDataset:
        if hasattr(self, "_pending_loads") and self._pending_loads:
            self._drain_pending_loads(wait=True)
        self.device = torch.device(device)
        self.next_window_device = NEXT_WINDOW_DEVICE or self.device
        self.data = DatasetIndex(
            motion_id=self._motion_id_index_cpu.to(self.device),
            step=self._step_index_cpu.to(self.device),
        )
        self.starts = self._storage_starts_cpu.to(self.device)
        self.ends = self._storage_ends_cpu.to(self.device)
        self.lengths = (self.ends - self.starts).to(self.device)
        self._pending_loads = []
        self._reset_runtime_state()
        return self

    def _empty_motion_sample(self) -> MotionSample:
        empty = torch.empty((0,), dtype=torch.long, device=self.device)
        return MotionSample(
            motion_id=empty,
            motion_len=empty,
            start_t=empty,
        )

    def _canonical_motion_sample(self, count: int) -> MotionSample:
        if count == 0:
            return self._empty_motion_sample()
        motion_id = torch.zeros((count,), dtype=torch.long, device=self.device)
        source_len = self.lengths[motion_id]
        window_len = torch.minimum(source_len, torch.full_like(source_len, RUNTIME_MOTION_MAX_LEN))
        start_t = torch.zeros((count,), dtype=torch.long, device=self.device)
        return MotionSample(
            motion_id=motion_id,
            motion_len=window_len,
            start_t=start_t,
        )

    def _build_default_pose_motion_data(self, env_ids: torch.Tensor, local_idx: torch.Tensor):
        batch_size, steps_count = local_idx.shape
        motion_id = self._env_current_motion_id[env_ids].unsqueeze(1).expand(-1, steps_count)
        step = local_idx
        body_pos_w = torch.zeros(
            (batch_size, steps_count, len(self.body_names), 3),
            device=self.device,
            dtype=torch.float32,
        )
        body_pos_w[..., 2] = 1.0
        body_lin_vel_w = torch.zeros_like(body_pos_w)
        body_quat_w = torch.zeros(
            (batch_size, steps_count, len(self.body_names), 4),
            device=self.device,
            dtype=torch.float32,
        )
        body_quat_w[..., 0] = 1.0
        body_ang_vel_w = torch.zeros_like(body_pos_w)
        joint_pos = torch.zeros(
            (batch_size, steps_count, len(self.joint_names)),
            device=self.device,
            dtype=torch.float32,
        )
        joint_vel = torch.zeros_like(joint_pos)
        return MotionData(
            motion_id=motion_id,
            step=step,
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_quat_w=body_quat_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            device=motion_id.device,
            batch_size=list(motion_id.shape),
        )

    def _stats_for(self, profile_name: str) -> _PrefetchProfileStats:
        stats = self._profile_stats.get(profile_name)
        if stats is None:
            stats = _PrefetchProfileStats()
            self._profile_stats[profile_name] = stats
        return stats

    def _record_stats(
        self,
        profile_name: str,
        *,
        current_hit: bool = False,
        sample_return: bool = False,
        blocking_wait_wall_time_s: float = 0.0,
        background_jobs: int = 0,
        background_frames: int = 0,
        background_load_wall_time_s: float = 0.0,
        pool_hit_motions: int = 0,
        pool_miss_motions: int = 0,
        wait_next_ready_calls: int = 0,
        wait_next_ready_wall_time_s: float = 0.0,
    ) -> None:
        stats = self._stats_for(profile_name)
        stats.calls += 1
        stats.current_hit_calls += int(current_hit)
        stats.sample_return_calls += int(sample_return)
        stats.background_jobs_submitted += int(background_jobs)
        stats.background_frames_total += int(background_frames)
        stats.background_load_wall_time_s += float(background_load_wall_time_s)
        stats.blocking_wait_wall_time_s += float(blocking_wait_wall_time_s)
        stats.pool_hit_motions += int(pool_hit_motions)
        stats.pool_miss_motions += int(pool_miss_motions)
        stats.wait_next_ready_calls += int(wait_next_ready_calls)
        stats.wait_next_ready_wall_time_s += float(wait_next_ready_wall_time_s)
        if not self._profile_enabled:
            return
        if self._profile_focus and self._profile_focus not in profile_name:
            return
        if stats.calls % self._profile_print_every != 0:
            return

        avg_wait_ms = 1000.0 * stats.blocking_wait_wall_time_s / max(1, stats.calls)
        avg_load_ms = 1000.0 * stats.background_load_wall_time_s / max(1, stats.background_jobs_submitted)
        avg_wait_next_ready_ms = (
            1000.0 * stats.wait_next_ready_wall_time_s / max(1, stats.wait_next_ready_calls)
        )
        print(
            "[any4hdmi][cache_profile]"
            f" profile={profile_name}"
            f" calls={stats.calls}"
            f" current_hit_calls={stats.current_hit_calls}"
            f" sample_return_calls={stats.sample_return_calls}"
            f" sync_refill_calls={stats.sync_refill_calls}"
            f" next_wait_calls={stats.next_wait_calls}"
            f" background_jobs={stats.background_jobs_submitted}"
            f" background_frames={stats.background_frames_total}"
            f" pool_hit_motions={stats.pool_hit_motions}"
            f" pool_miss_motions={stats.pool_miss_motions}"
            f" wait_next_ready_calls={stats.wait_next_ready_calls}"
            f" total_wait_next_ready_s={stats.wait_next_ready_wall_time_s:.4f}"
            f" avg_wait_next_ready_ms={avg_wait_next_ready_ms:.2f}"
            f" avg_background_load_ms={avg_load_ms:.2f}"
            f" avg_wait_ms={avg_wait_ms:.2f}"
        )

    def _draw_uniform_window_specs(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if count == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty, empty
        sampled_frame_ids = torch.randint(0, self.num_steps, size=(count,), device=self.device)
        sampled_motion_ids = self.data.motion_id[sampled_frame_ids].long()
        sampled_source_start_t = self.data.step[sampled_frame_ids].long()
        sampled_source_len = self.lengths[sampled_motion_ids].long()
        sampled_window_len = (sampled_source_len - sampled_source_start_t).clamp_max_(RUNTIME_MOTION_MAX_LEN)
        return sampled_motion_ids, sampled_source_start_t, sampled_window_len

    def _assign_current_window_metadata(
        self,
        env_ids: torch.Tensor,
        *,
        motion_ids: torch.Tensor,
        source_start_t: torch.Tensor,
        window_len: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return
        self._env_current_motion_id.index_copy_(0, env_ids, motion_ids)
        self._env_current_source_start_t.index_copy_(0, env_ids, source_start_t)
        self._env_current_window_len.index_copy_(0, env_ids, window_len)

    def _assign_next_window_metadata(
        self,
        env_ids: torch.Tensor,
        *,
        motion_ids: torch.Tensor,
        source_start_t: torch.Tensor,
        window_len: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return
        self._env_next_motion_id.index_copy_(0, env_ids, motion_ids)
        self._env_next_source_start_t.index_copy_(0, env_ids, source_start_t)
        self._env_next_window_len.index_copy_(0, env_ids, window_len)

    def _load_window_batch_data(
        self,
        *,
        motion_ids: torch.Tensor,
        global_start: torch.Tensor,
        target_device: torch.device,
        float_dtype: torch.dtype,
    ) -> MotionData:
        motion_ids = motion_ids.to(device="cpu", dtype=torch.long)
        global_start = global_start.to(device="cpu", dtype=torch.long)
        batch_size = int(motion_ids.shape[0])
        if batch_size == 0:
            raise ValueError("window batch load requires a non-empty batch")

        global_index = global_start.unsqueeze(1) + self._window_steps_cpu.unsqueeze(0)
        global_index.clamp_max_(self._storage_total_length - 1)
        flat_index = global_index.reshape(-1)

        def gather_float_field(field_name: str) -> torch.Tensor:
            field = self._storage_cpu[field_name].index_select(0, flat_index)
            field = field.reshape(batch_size, RUNTIME_MOTION_MAX_LEN, *field.shape[1:])
            if target_device.type == "cpu":
                return field.to(dtype=float_dtype).clone()
            return field.to(device=target_device, dtype=float_dtype, non_blocking=True).contiguous()

        step = self._window_steps_cpu.to(device=target_device).unsqueeze(0).expand(batch_size, -1).contiguous()
        motion_id = motion_ids.to(device=target_device).unsqueeze(1).expand(-1, RUNTIME_MOTION_MAX_LEN).contiguous()
        return MotionData(
            motion_id=motion_id,
            step=step,
            body_pos_w=gather_float_field("body_pos_w"),
            body_lin_vel_w=gather_float_field("body_lin_vel_w"),
            body_quat_w=gather_float_field("body_quat_w"),
            body_ang_vel_w=gather_float_field("body_ang_vel_w"),
            joint_pos=gather_float_field("joint_pos"),
            joint_vel=gather_float_field("joint_vel"),
            batch_size=(batch_size, RUNTIME_MOTION_MAX_LEN),
            device=target_device,
        )

    def _load_current_windows_sync(
        self,
        env_ids: torch.Tensor,
        *,
        motion_ids: torch.Tensor,
        source_start_t: torch.Tensor,
    ) -> None:
        global_start = self._storage_starts_cpu[motion_ids.detach().cpu()] + source_start_t.detach().cpu()
        window = self._load_window_batch_data(
            motion_ids=motion_ids.detach().cpu(),
            global_start=global_start,
            target_device=self.device,
            float_dtype=CURRENT_WINDOW_FLOAT_DTYPE,
        )
        self._index_copy_motion_data(
            self._current_window,
            env_ids,
            window,
        )

    @staticmethod
    def _index_copy_motion_data(
        dst: MotionData,
        env_ids: torch.Tensor,
        src: MotionData,
    ) -> None:
        env_ids = env_ids.to(device=dst.motion_id.device, dtype=torch.long)
        for field_name in _MOTION_DATA_FIELD_NAMES:
            dst_field = getattr(dst, field_name)
            src_field = getattr(src, field_name).to(
                device=dst_field.device,
                dtype=dst_field.dtype,
                non_blocking=True,
            )
            dst_field.index_copy_(0, env_ids, src_field)

    def _load_motion_batch_to_runtime(
        self,
        env_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        source_start_t: torch.Tensor,
    ) -> _MotionLoadResult:
        start_time = time.perf_counter()
        motion_ids_cpu = motion_ids.detach().to(device="cpu", dtype=torch.long)
        source_start_t_cpu = source_start_t.detach().to(device="cpu", dtype=torch.long)
        global_start = self._storage_starts_cpu[motion_ids_cpu] + source_start_t_cpu
        motion = self._load_window_batch_data(
            motion_ids=motion_ids_cpu,
            global_start=global_start,
            target_device=self.next_window_device,
            float_dtype=NEXT_WINDOW_FLOAT_DTYPE,
        )
        # self._next_window[env_ids].copy_(motion)
        return _MotionLoadResult(
            env_ids=env_ids,
            motion=motion,
            load_wall_time_s=time.perf_counter() - start_time,
        )

    def _drain_pending_loads(
        self,
        *,
        wait: bool,
        env_ids: torch.Tensor | None = None,
    ) -> float:
        if env_ids is not None:
            env_ids = env_ids.detach().to(dtype=torch.long)
        blocking_wait_wall_time_s = 0.0

        remaining: list[_PendingMotionLoad] = []
        for job in self._pending_loads:
            should_wait = wait and (
                env_ids is None
                or torch.any(torch.isin(env_ids, job.env_ids.to(device=env_ids.device, dtype=torch.long)))
            )
            if not should_wait and not job.future.done():
                remaining.append(job)
                continue
            wait_start = time.perf_counter()
            result = job.future.result()
            if should_wait:
                blocking_wait_wall_time_s += time.perf_counter() - wait_start
            self._index_copy_motion_data(
                self._next_window,
                result.env_ids,
                result.motion,
            )
            self._record_stats(
                job.profile_name,
                background_load_wall_time_s=result.load_wall_time_s,
            )
        self._pending_loads = remaining
        return blocking_wait_wall_time_s

    def _schedule_next_window_prefetch(
        self,
        env_ids: torch.Tensor,
        *,
        motion_ids: torch.Tensor,
        source_start_t: torch.Tensor,
        window_len: torch.Tensor,
        profile_name: str,
    ) -> None:
        if env_ids.numel() == 0:
            return
        future = self._executor.submit(
            self._load_motion_batch_to_runtime,
            env_ids,
            motion_ids,
            source_start_t,
        )
        self._pending_loads.append(
            _PendingMotionLoad(
                env_ids=env_ids,
                profile_name=profile_name,
                future=future,
            )
        )
        self._record_stats(
            profile_name,
            background_jobs=1,
            background_frames=int(window_len.sum().item()),
        )

    def _promote_next_windows_to_current(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        next_env_ids = env_ids.to(device=self.next_window_device)
        self._index_copy_motion_data(
            self._current_window,
            env_ids,
            self._next_window[next_env_ids].to(device=self.device),
        )
        self._assign_current_window_metadata(
            env_ids,
            motion_ids=self._env_next_motion_id[env_ids],
            source_start_t=self._env_next_source_start_t[env_ids],
            window_len=self._env_next_window_len[env_ids],
        )

    def sample_motion(
        self,
        env_ids: torch.Tensor,
        *,
        terminated_t: torch.Tensor,
        rewind_mask: torch.Tensor,
        rewind_steps: torch.Tensor,
    ) -> MotionSample:
        if env_ids.numel() == 0:
            return self._empty_motion_sample()

        if self._fake_sample_motion:
            result = self._canonical_motion_sample(int(env_ids.numel()))
            self._assign_current_window_metadata(
                env_ids,
                motion_ids=result.motion_id,
                source_start_t=torch.zeros_like(result.motion_id),
                window_len=result.motion_len,
            )
            return result

        if not torch.all(rewind_mask):
            non_rewind_env_ids = env_ids[~rewind_mask]
            wait_time = self._drain_pending_loads(
                wait=True,
                env_ids=non_rewind_env_ids,
            )

            self._promote_next_windows_to_current(non_rewind_env_ids)

            next_motion_ids, next_source_start_t, next_window_len = self._draw_uniform_window_specs(
                non_rewind_env_ids.numel()
            )

            self._assign_next_window_metadata(
                non_rewind_env_ids,
                motion_ids=next_motion_ids,
                source_start_t=next_source_start_t,
                window_len=next_window_len,
            )
            self._schedule_next_window_prefetch(
                non_rewind_env_ids,
                motion_ids=next_motion_ids,
                source_start_t=next_source_start_t,
                window_len=next_window_len,
                profile_name="sample_motion",
            )
            self._record_stats(
                "sample_motion",
                sample_return=True,
                blocking_wait_wall_time_s=wait_time,
                wait_next_ready_calls=1,
                wait_next_ready_wall_time_s=wait_time,
            )

        result_motion_len = self._env_current_window_len[env_ids]
        result_start_t = terminated_t - rewind_steps
        result_start_t.clamp_min_(0)
        result_start_t.masked_fill_(~rewind_mask, 0)

        return MotionSample(
            motion_id=env_ids,
            motion_len=result_motion_len,
            start_t=result_start_t,
        )

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
        *,
        profile_name: str | None = None,
    ):
        motion_lengths = self._env_current_window_len[motion_ids]
        local_idx = starts.unsqueeze(1) + steps.unsqueeze(0)

        local_idx.clamp_min_(0)
        local_idx.clamp_max_(motion_lengths.unsqueeze(1) - 1)
        if self._fake_get_slice:
            return self._build_default_pose_motion_data(motion_ids, local_idx)
        motion_data = self._current_window[motion_ids.unsqueeze(1), local_idx]
        self._record_stats(profile_name or "get_slice", current_hit=True)
        return motion_data



OnlineQposDataset = WindowedMotionDataset
