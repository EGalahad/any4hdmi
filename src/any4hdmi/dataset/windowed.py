from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from any4hdmi.dataset.base import BaseDataset, MotionData, MotionSample
from any4hdmi.dataset.fk_cache import FKCacheEntry


RUNTIME_MOTION_MAX_LEN = 512
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
_FLOAT_MOTION_DATA_FIELD_NAMES = _MOTION_DATA_FIELD_NAMES[2:]


def _contiguous_id_runs(sorted_ids: torch.Tensor) -> list[tuple[int, int, int, int]]:
    """Return source and destination slices for consecutive destination IDs."""
    ids = sorted_ids.tolist()
    if not ids:
        return []
    runs = []
    source_start = 0
    destination_start = ids[0]
    for source_end in range(1, len(ids) + 1):
        if source_end == len(ids) or ids[source_end] != ids[source_end - 1] + 1:
            runs.append(
                (source_start, source_end, destination_start, ids[source_end - 1] + 1)
            )
            if source_end < len(ids):
                source_start = source_end
                destination_start = ids[source_end]
    return runs


@dataclass
class _PendingMotionLoad:
    env_ids: torch.Tensor
    future: Future[None]


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
        next_window_device: str | torch.device | None = "current",
        pin_window_load: bool = True,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.body_names = list(body_names)
        self.joint_names = list(joint_names)
        self.motion_paths = list(motion_paths)
        self._num_envs = int(num_envs)
        self._storage_cpu = dict(storage_fields)
        self._storage_total_length = int(self._storage_cpu["motion_id"].shape[0])
        self._storage_starts_cpu = torch.as_tensor(starts, dtype=torch.long)
        self._storage_ends_cpu = torch.as_tensor(ends, dtype=torch.long)
        self._window_steps_cpu = torch.arange(RUNTIME_MOTION_MAX_LEN, dtype=torch.long)
        self._configured_next_window_device = next_window_device
        self._pin_window_load = bool(pin_window_load)
        self._window_load_scratch: dict[str, torch.Tensor] = {}
        self._pin_window_load_disabled = False
        self._prefetch_cuda_stream: torch.cuda.Stream | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="any4hdmi-motion-pool",
        )
        self._pending_loads: list[_PendingMotionLoad] = []
        self.to(device if device is not None else torch.device("cpu"))
        print(
            "[any4hdmi][windowed]"
            f" backing_dtype={self._storage_cpu['body_pos_w'].dtype}"
            f" next_window_device={self.next_window_device}"
            f" pinned_scratch={int(self._pin_window_load)}"
        )

    @classmethod
    def from_cache_entry(
        cls,
        entry: FKCacheEntry,
        *,
        num_envs: int,
        device: torch.device | str | None = None,
        next_window_device: str | torch.device | None = "current",
        pin_window_load: bool = True,
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
            next_window_device=next_window_device,
            pin_window_load=pin_window_load,
        )

    def __del__(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is None:
            return
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    @property
    def num_steps(self) -> int:
        return self._storage_total_length

    @property
    def sample_id_span(self) -> int:
        return max(self.num_motions, self._num_envs)

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
                (*pool_shape, body_count, 3), dtype=float_dtype, device=device
            ),
            body_lin_vel_w=torch.empty(
                (*pool_shape, body_count, 3), dtype=float_dtype, device=device
            ),
            body_quat_w=torch.empty(
                (*pool_shape, body_count, 4), dtype=float_dtype, device=device
            ),
            body_ang_vel_w=torch.empty(
                (*pool_shape, body_count, 3), dtype=float_dtype, device=device
            ),
            joint_pos=torch.empty(
                (*pool_shape, joint_count), dtype=float_dtype, device=device
            ),
            joint_vel=torch.empty(
                (*pool_shape, joint_count), dtype=float_dtype, device=device
            ),
            batch_size=pool_shape,
            device=device,
        )
        pool.zero_()
        pool.body_pos_w[..., 2] = 1.0
        pool.body_quat_w[..., 0] = 1.0
        return pool

    def to(self, device: torch.device | str) -> WindowedMotionDataset:
        if self._pending_loads:
            self._drain_pending_loads(wait=True)
        self.device = torch.device(device)
        configured = self._configured_next_window_device
        if configured is None or configured == "current":
            self.next_window_device = self.device
        else:
            self.next_window_device = torch.device(configured)
        self._prefetch_cuda_stream = (
            torch.cuda.Stream(device=self.next_window_device)
            if self.next_window_device.type == "cuda"
            else None
        )
        self.starts = self._storage_starts_cpu.to(self.device)
        self.ends = self._storage_ends_cpu.to(self.device)
        self.lengths = self.ends - self.starts
        self._current_window = self._allocate_window_pool(
            device=self.device,
            float_dtype=CURRENT_WINDOW_FLOAT_DTYPE,
        )
        self._next_window = self._allocate_window_pool(
            device=self.next_window_device,
            float_dtype=NEXT_WINDOW_FLOAT_DTYPE,
        )
        self._env_current_motion_id = torch.full(
            (self._num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._env_current_source_start_t = torch.zeros(
            (self._num_envs,), dtype=torch.long, device=self.device
        )
        self._env_current_window_len = torch.zeros(
            (self._num_envs,), dtype=torch.long, device=self.device
        )
        self._env_next_motion_id = torch.full(
            (self._num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._env_next_source_start_t = torch.zeros(
            (self._num_envs,), dtype=torch.long, device=self.device
        )
        self._env_next_window_len = torch.zeros(
            (self._num_envs,), dtype=torch.long, device=self.device
        )
        self._pending_loads = []

        all_env_ids = torch.arange(self._num_envs, device=self.device)
        motion_ids, source_starts, window_lengths = self._draw_uniform_window_specs(
            self._num_envs
        )
        self._env_current_motion_id.copy_(motion_ids)
        self._env_current_source_start_t.copy_(source_starts)
        self._env_current_window_len.copy_(window_lengths)
        self._load_current_windows_sync(all_env_ids, motion_ids, source_starts)

        motion_ids, source_starts, window_lengths = self._draw_uniform_window_specs(
            self._num_envs
        )
        self._env_next_motion_id.copy_(motion_ids)
        self._env_next_source_start_t.copy_(source_starts)
        self._env_next_window_len.copy_(window_lengths)
        self._schedule_next_window_prefetch(all_env_ids, motion_ids, source_starts)
        return self

    def _draw_uniform_window_specs(
        self, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if count == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty, empty
        sampled_frame_ids = torch.randint(
            0,
            self._storage_total_length,
            size=(count,),
            device=self.device,
        )
        motion_ids = torch.searchsorted(self.ends, sampled_frame_ids, right=True)
        source_starts = sampled_frame_ids - self.starts[motion_ids]
        max_source_start = (
            self.lengths[motion_ids].long() - RUNTIME_MOTION_MAX_LEN
        ).clamp_min_(0)
        source_starts = torch.minimum(source_starts, max_source_start)
        return (
            motion_ids,
            source_starts,
            torch.full_like(source_starts, RUNTIME_MOTION_MAX_LEN),
        )

    def _get_window_load_scratch(
        self,
        field_name: str,
        source: torch.Tensor,
        flat_count: int,
    ) -> torch.Tensor | None:
        if self._pin_window_load_disabled:
            return None
        scratch = self._window_load_scratch.get(field_name)
        shape = (flat_count, *source.shape[1:])
        if (
            scratch is not None
            and scratch.shape[0] >= flat_count
            and scratch.shape[1:] == source.shape[1:]
            and scratch.dtype == source.dtype
        ):
            return scratch[:flat_count]
        try:
            scratch = torch.empty(shape, dtype=source.dtype, pin_memory=True)
        except RuntimeError as error:
            print(f"[any4hdmi][windowed] disabling pinned scratch: {error}")
            self._pin_window_load_disabled = True
            self._window_load_scratch.clear()
            return None
        self._window_load_scratch[field_name] = scratch
        return scratch

    def _load_window_batch_into_pool(
        self,
        pool: MotionData,
        env_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        global_starts: torch.Tensor,
    ) -> None:
        env_ids_cpu, order = env_ids.detach().to(device="cpu", dtype=torch.long).sort()
        motion_ids = motion_ids.detach().to(device="cpu", dtype=torch.long)[order]
        global_starts = global_starts.detach().to(device="cpu", dtype=torch.long)[order]
        global_index = global_starts.unsqueeze(1) + self._window_steps_cpu.unsqueeze(0)
        global_end = self._storage_ends_cpu[motion_ids].sub(1).unsqueeze(1)
        flat_index = torch.minimum(global_index, global_end).reshape(-1)
        batch_size = int(env_ids_cpu.numel())
        runs = _contiguous_id_runs(env_ids_cpu)

        for field_name in _FLOAT_MOTION_DATA_FIELD_NAMES:
            source = self._storage_cpu[field_name]
            scratch = (
                self._get_window_load_scratch(field_name, source, int(flat_index.numel()))
                if self._pin_window_load and pool.device.type != "cpu"
                else None
            )
            if scratch is not None:
                torch.index_select(source, 0, flat_index, out=scratch)
                field = scratch
            else:
                field = source.index_select(0, flat_index)
            field = field.reshape(batch_size, RUNTIME_MOTION_MAX_LEN, *field.shape[1:])
            destination = getattr(pool, field_name)
            for source_start, source_end, destination_start, destination_end in runs:
                destination[destination_start:destination_end].copy_(
                    field[source_start:source_end],
                    non_blocking=True,
                )

        motion_ids = motion_ids.unsqueeze(1).expand(-1, RUNTIME_MOTION_MAX_LEN)
        steps = self._window_steps_cpu.unsqueeze(0).expand(batch_size, -1)
        for destination, source in (
            (pool.motion_id, motion_ids),
            (pool.step, steps),
        ):
            for source_start, source_end, destination_start, destination_end in runs:
                destination[destination_start:destination_end].copy_(
                    source[source_start:source_end],
                    non_blocking=True,
                )

    def _load_current_windows_sync(
        self,
        env_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        source_starts: torch.Tensor,
    ) -> None:
        motion_ids_cpu = motion_ids.detach().to(device="cpu", dtype=torch.long)
        global_starts = (
            self._storage_starts_cpu[motion_ids_cpu]
            + source_starts.detach().to(device="cpu", dtype=torch.long)
        )
        self._load_window_batch_into_pool(
            self._current_window,
            env_ids,
            motion_ids_cpu,
            global_starts,
        )

    def _load_next_windows(
        self,
        env_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        source_starts: torch.Tensor,
    ) -> None:
        motion_ids_cpu = motion_ids.detach().to(device="cpu", dtype=torch.long)
        global_starts = (
            self._storage_starts_cpu[motion_ids_cpu]
            + source_starts.detach().to(device="cpu", dtype=torch.long)
        )
        stream_context = (
            torch.cuda.stream(self._prefetch_cuda_stream)
            if self._prefetch_cuda_stream is not None
            else nullcontext()
        )
        with stream_context:
            self._load_window_batch_into_pool(
                self._next_window,
                env_ids,
                motion_ids_cpu,
                global_starts,
            )
        if self._prefetch_cuda_stream is not None:
            self._prefetch_cuda_stream.synchronize()

    def _drain_pending_loads(
        self,
        *,
        wait: bool,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids is not None:
            env_ids = env_ids.detach().to(dtype=torch.long)
        remaining = []
        for job in self._pending_loads:
            should_wait = wait and (
                env_ids is None
                or torch.any(
                    torch.isin(
                        env_ids,
                        job.env_ids.to(device=env_ids.device, dtype=torch.long),
                    )
                )
            )
            if not should_wait and not job.future.done():
                remaining.append(job)
                continue
            job.future.result()
        self._pending_loads = remaining

    def _schedule_next_window_prefetch(
        self,
        env_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        source_starts: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return
        self._pending_loads.append(
            _PendingMotionLoad(
                env_ids=env_ids,
                future=self._executor.submit(
                    self._load_next_windows,
                    env_ids,
                    motion_ids,
                    source_starts,
                ),
            )
        )

    @staticmethod
    def _copy_motion_rows(
        destination: MotionData,
        destination_ids: torch.Tensor,
        source: MotionData,
    ) -> None:
        destination_ids = destination_ids.to(
            device=destination.device,
            dtype=torch.long,
        )
        for field_name in _MOTION_DATA_FIELD_NAMES:
            destination_field = getattr(destination, field_name)
            destination_field.index_copy_(
                0,
                destination_ids,
                getattr(source, field_name).to(
                    device=destination_field.device,
                    dtype=destination_field.dtype,
                    non_blocking=True,
                ),
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
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return MotionSample(motion_id=empty, motion_len=empty, start_t=empty)

        if not torch.all(rewind_mask):
            active_env_ids = env_ids[~rewind_mask]
            self._drain_pending_loads(wait=True, env_ids=active_env_ids)
            next_env_ids = active_env_ids.to(device=self.next_window_device)
            self._copy_motion_rows(
                self._current_window,
                active_env_ids,
                self._next_window[next_env_ids].to(device=self.device),
            )
            self._env_current_motion_id.index_copy_(
                0,
                active_env_ids,
                self._env_next_motion_id[active_env_ids],
            )
            self._env_current_source_start_t.index_copy_(
                0,
                active_env_ids,
                self._env_next_source_start_t[active_env_ids],
            )
            self._env_current_window_len.index_copy_(
                0,
                active_env_ids,
                self._env_next_window_len[active_env_ids],
            )

            motion_ids, source_starts, window_lengths = self._draw_uniform_window_specs(
                int(active_env_ids.numel())
            )
            self._env_next_motion_id.index_copy_(0, active_env_ids, motion_ids)
            self._env_next_source_start_t.index_copy_(0, active_env_ids, source_starts)
            self._env_next_window_len.index_copy_(0, active_env_ids, window_lengths)
            self._schedule_next_window_prefetch(
                active_env_ids,
                motion_ids,
                source_starts,
            )

        start_t = (terminated_t - rewind_steps).clamp_min_(0)
        start_t.masked_fill_(~rewind_mask, 0)
        return MotionSample(
            motion_id=env_ids,
            motion_len=self._env_current_window_len[env_ids],
            start_t=start_t,
        )

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
    ) -> MotionData:
        motion_lengths = self._env_current_window_len[motion_ids]
        local_index = starts.unsqueeze(1) + steps.unsqueeze(0)
        local_index.clamp_min_(0)
        local_index.clamp_max_(motion_lengths.unsqueeze(1) - 1)
        return self._current_window[motion_ids.unsqueeze(1), local_index]
