from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tensordict import MemoryMappedTensor
from tqdm import tqdm

from any4hdmi.core.model import body_names_from_model, hinge_joint_info, load_model
from any4hdmi.dataset.base import MotionData
from any4hdmi.dataset.interpolation import interpolate_qpos_qvel_batch_torch, resampled_length
from any4hdmi.dataset.loading import (
    resolve_any4hdmi_dataset_context,
    resolve_any4hdmi_motion_paths,
    resolve_source_fps,
)
from any4hdmi.fk.runner import FKRunner
from any4hdmi.utils.dataset import (
    DEFAULT_MOTION_LOADER_NUM_WORKERS,
    DEFAULT_MOTION_LOADER_PREFETCH_FACTOR,
    build_motion_loader,
)
from any4hdmi.utils.mjcf import build_hf_mjcf_reference, resolve_mjcf_path


QPOS_CACHE_VERSION = 5
QPOS_CACHE_SUBDIR = ".cache/motion/qpos_online_v2"
QPOS_CACHE_INDEX_NAME = "motion_index.json"
QPOS_CACHE_META_NAME = "cache_meta.json"
QPOS_CACHE_READY_NAME = "ready.flag"
QPOS_CACHE_TD_SUBDIR = "td"
QPOS_CACHE_INITIAL_CAPACITY = 1024

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


@dataclass(frozen=True)
class FKCacheEntry:
    cache_entry_dir: Path
    body_names: list[str]
    joint_names: list[str]
    motion_paths: list[Path]
    starts: list[int]
    ends: list[int]
    storage_fields: dict[str, torch.Tensor]

    def as_motion_data(self) -> MotionData:
        motion_id = self.storage_fields["motion_id"]
        return MotionData(
            motion_id=self.storage_fields["motion_id"],
            step=self.storage_fields["step"],
            body_pos_w=self.storage_fields["body_pos_w"],
            body_lin_vel_w=self.storage_fields["body_lin_vel_w"],
            body_quat_w=self.storage_fields["body_quat_w"],
            body_ang_vel_w=self.storage_fields["body_ang_vel_w"],
            joint_pos=self.storage_fields["joint_pos"],
            joint_vel=self.storage_fields["joint_vel"],
            device=motion_id.device,
            batch_size=list(motion_id.shape),
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


def _cache_root(base_dir: Path) -> Path:
    cache_root = base_dir / QPOS_CACHE_SUBDIR
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _stat_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _content_fingerprint(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return {
        "size": int(path.stat().st_size),
        "sha256": hasher.hexdigest(),
    }


def _fingerprint_motion_entry(motion_path: Path) -> dict[str, Any]:
    entry = {"motion": _stat_fingerprint(motion_path)}
    sidecar_path = motion_path.with_suffix(".json")
    if sidecar_path.is_file():
        entry["sidecar"] = _stat_fingerprint(sidecar_path)
    return entry


def _make_qpos_cache_key(
    *,
    dataset_root: Path,
    motion_paths: list[Path],
    mjcf_path: Path,
    target_fps: int,
) -> str:
    payload = {
        "cache_version": QPOS_CACHE_VERSION,
        "dataset_root": str(dataset_root),
        "manifest": _stat_fingerprint(dataset_root / "manifest.json"),
        "mjcf": _content_fingerprint(mjcf_path),
        "target_fps": int(target_fps),
        "motions": [_fingerprint_motion_entry(path) for path in motion_paths],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _acquire_cache_lock(lock_dir: Path, ready_flag: Path, timeout_s: float = 600.0) -> bool:
    start_time = time.monotonic()
    while True:
        if ready_flag.is_file():
            return False
        try:
            lock_dir.mkdir(parents=False, exist_ok=False)
            return True
        except FileExistsError:
            if time.monotonic() - start_time > timeout_s:
                raise TimeoutError(f"Timed out waiting for cache lock {lock_dir}")
            time.sleep(0.5)


def _resolve_any4hdmi_mjcf_path(dataset_root: Path, manifest: dict[str, Any]) -> Path:
    mjcf_ref = manifest.get("mjcf")
    if mjcf_ref is not None:
        if isinstance(mjcf_ref, dict):
            if mjcf_ref.get("kind") == "huggingface":
                return resolve_mjcf_path(
                    build_hf_mjcf_reference(
                        repo_id=str(mjcf_ref["repo_id"]),
                        path=str(mjcf_ref["path"]),
                        revision=str(mjcf_ref.get("revision", "main")),
                    ),
                    dataset_root=dataset_root,
                )
            if "path" in mjcf_ref:
                return resolve_mjcf_path(str(mjcf_ref["path"]), dataset_root=dataset_root)
            raise TypeError(f"Unsupported structured mjcf payload: {mjcf_ref!r}")
        return resolve_mjcf_path(mjcf_ref, dataset_root=dataset_root)

    mjcf_path_raw = manifest.get("mjcf_path")
    if mjcf_path_raw is None:
        raise KeyError("manifest.json is missing mjcf or mjcf_path")
    mjcf_path = Path(mjcf_path_raw).expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
    return mjcf_path


def _storage_field_specs(
    *,
    body_count: int,
    joint_count: int,
) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    return {
        "motion_id": (torch.long, ()),
        "step": (torch.long, ()),
        "body_pos_w": (torch.float32, (body_count, 3)),
        "body_lin_vel_w": (torch.float32, (body_count, 3)),
        "body_quat_w": (torch.float32, (body_count, 4)),
        "body_ang_vel_w": (torch.float32, (body_count, 3)),
        "joint_pos": (torch.float32, (joint_count,)),
        "joint_vel": (torch.float32, (joint_count,)),
    }


class _GrowableMotionStorage:
    def __init__(
        self,
        *,
        cache_entry_dir: Path,
        body_count: int,
        joint_count: int,
        initial_capacity: int = QPOS_CACHE_INITIAL_CAPACITY,
    ) -> None:
        self.root = cache_entry_dir / QPOS_CACHE_TD_SUBDIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.specs = _storage_field_specs(body_count=body_count, joint_count=joint_count)
        self.capacity = max(1, int(initial_capacity))
        self.fields = self._open_fields(self.capacity)

    def _field_path(self, field_name: str) -> Path:
        return self.root / f"{field_name}.memmap"

    def _field_shape(self, field_name: str, capacity: int) -> tuple[int, ...]:
        _, tail_shape = self.specs[field_name]
        return (capacity, *tail_shape)

    def _open_fields(self, capacity: int) -> dict[str, torch.Tensor]:
        fields: dict[str, torch.Tensor] = {}
        for field_name, (dtype, _) in self.specs.items():
            field_path = self._field_path(field_name)
            field_shape = self._field_shape(field_name, capacity)
            if field_path.exists():
                fields[field_name] = MemoryMappedTensor.from_filename(
                    str(field_path),
                    dtype=dtype,
                    shape=field_shape,
                )
            else:
                fields[field_name] = MemoryMappedTensor.empty(
                    field_shape,
                    dtype=dtype,
                    filename=str(field_path),
                )
        return fields

    def ensure_capacity(self, required_length: int) -> None:
        if required_length <= self.capacity:
            return
        new_capacity = self.capacity
        while new_capacity < required_length:
            new_capacity *= 2
        self.fields = self._open_fields(new_capacity)
        self.capacity = new_capacity


def _cache_build_device() -> torch.device | None:
    raw = os.environ.get("ANY4HDMI_CACHE_BUILD_DEVICE")
    if raw is None:
        return None
    return torch.device(raw)


def _cache_build_num_workers() -> int:
    raw = os.environ.get("ANY4HDMI_CACHE_BUILD_NUM_WORKERS")
    if raw is None:
        return DEFAULT_MOTION_LOADER_NUM_WORKERS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MOTION_LOADER_NUM_WORKERS


def _cache_build_prefetch_factor() -> int:
    raw = os.environ.get("ANY4HDMI_CACHE_BUILD_PREFETCH_FACTOR")
    if raw is None:
        return DEFAULT_MOTION_LOADER_PREFETCH_FACTOR
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MOTION_LOADER_PREFETCH_FACTOR


def _cache_build_multiprocessing_context(num_workers: int) -> str | None:
    if int(num_workers) <= 0:
        return None
    raw = os.environ.get("ANY4HDMI_CACHE_BUILD_MULTIPROCESSING_CONTEXT")
    if raw is not None:
        value = raw.strip()
        return value or None
    if os.name == "posix" and sys.platform.startswith("linux"):
        return "fork"
    return None


def _cache_build_write_buffer_bytes() -> int:
    raw = os.environ.get("ANY4HDMI_CACHE_BUILD_WRITE_BUFFER_BYTES")
    default = 10 * 1024 * 1024 * 1024
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _motion_data_num_bytes(data: MotionData) -> int:
    total = 0
    for field_name in _MOTION_DATA_FIELD_NAMES:
        field = getattr(data, field_name)
        total += int(field.numel()) * int(field.element_size())
    return total


def _concat_motion_data_chunks(chunks: list[MotionData]) -> MotionData:
    if not chunks:
        raise ValueError("Expected at least one MotionData chunk to concatenate")
    if len(chunks) == 1:
        return chunks[0]
    motion_id = torch.cat([chunk.motion_id for chunk in chunks], dim=0)
    return MotionData(
        motion_id=motion_id,
        step=torch.cat([chunk.step for chunk in chunks], dim=0),
        body_pos_w=torch.cat([chunk.body_pos_w for chunk in chunks], dim=0),
        body_lin_vel_w=torch.cat([chunk.body_lin_vel_w for chunk in chunks], dim=0),
        body_quat_w=torch.cat([chunk.body_quat_w for chunk in chunks], dim=0),
        body_ang_vel_w=torch.cat([chunk.body_ang_vel_w for chunk in chunks], dim=0),
        joint_pos=torch.cat([chunk.joint_pos for chunk in chunks], dim=0),
        joint_vel=torch.cat([chunk.joint_vel for chunk in chunks], dim=0),
        device=motion_id.device,
        batch_size=list(motion_id.shape),
    )


def _build_fk_cache(
    *,
    dataset_root: Path,
    manifest: dict[str, Any],
    motion_paths: list[Path],
    mjcf_path: Path,
    cache_entry_dir: Path,
    target_fps: int,
) -> None:
    model = load_model(mjcf_path)
    body_names = body_names_from_model(model)
    joint_names, _, _ = hinge_joint_info(model)
    source_fps = resolve_source_fps(manifest)
    storage = _GrowableMotionStorage(
        cache_entry_dir=cache_entry_dir,
        body_count=len(body_names),
        joint_count=len(joint_names),
    )
    fk_runner = FKRunner(
        mjcf_path=mjcf_path,
        batch_size=_env_int("ANY4HDMI_CACHE_BUILD_BATCH_SIZE", 51200),
        device=_cache_build_device(),
    )
    num_workers = _cache_build_num_workers()
    motion_loader = build_motion_loader(
        input_root=dataset_root,
        motion_paths=motion_paths,
        mjcf_path=mjcf_path,
        fps=float(source_fps),
        num_workers=num_workers,
        prefetch_factor=_cache_build_prefetch_factor(),
        pin_memory=fk_runner.device.type == "cuda",
        multiprocessing_context=_cache_build_multiprocessing_context(num_workers),
        tensor_device=fk_runner.device,
    )

    starts: list[int] = []
    ends: list[int] = []
    start_idx = 0
    batched_motion_ids: list[int] = []
    batched_qpos: list[torch.Tensor] = []
    batched_qvel: list[torch.Tensor] = []
    batched_frames = 0
    staged_motion_chunks: list[MotionData] = []
    staged_motion_lengths: list[int] = []
    staged_motion_bytes = 0
    write_buffer_bytes = _cache_build_write_buffer_bytes()

    def flush_batched_motions() -> tuple[MotionData, list[int]] | None:
        nonlocal batched_frames
        if not batched_qpos:
            return None

        qpos_batch, qvel_batch = interpolate_qpos_qvel_batch_torch(
            batched_qpos,
            batched_qvel,
            source_fps=float(source_fps),
            target_fps=float(target_fps),
        )
        lengths = [int(qpos.shape[0]) for qpos in qpos_batch]
        packed_qpos = torch.cat(qpos_batch, dim=0)
        packed_qvel = torch.cat(qvel_batch, dim=0)
        fk_start_time = time.perf_counter()
        packed_outputs = fk_runner.forward_kinematics(packed_qpos, packed_qvel)
        fk_elapsed_s = time.perf_counter() - fk_start_time
        print(f"FK: {fk_elapsed_s:.2f}s for {len(batched_motion_ids)} motions / {int(packed_qpos.shape[0])} frames")
        device = packed_qpos.device
        length_t = torch.as_tensor(lengths, device=device, dtype=torch.long)
        motion_ids_t = torch.as_tensor(batched_motion_ids, device=device, dtype=torch.long)
        local_steps = [torch.arange(length, device=device, dtype=torch.long) for length in lengths]
        motion_id = torch.repeat_interleave(motion_ids_t, length_t).to(device="cpu")
        motion_chunk = MotionData(
            motion_id=motion_id,
            step=torch.cat(local_steps, dim=0).to(device="cpu"),
            body_pos_w=packed_outputs["body_pos_w"].to(device="cpu"),
            body_lin_vel_w=packed_outputs["body_lin_vel_w"].to(device="cpu"),
            body_quat_w=packed_outputs["body_quat_w"].to(device="cpu"),
            body_ang_vel_w=packed_outputs["body_ang_vel_w"].to(device="cpu"),
            joint_pos=packed_outputs["joint_pos"].to(device="cpu"),
            joint_vel=packed_outputs["joint_vel"].to(device="cpu"),
            device=motion_id.device,
            batch_size=list(motion_id.shape),
        )

        batched_motion_ids.clear()
        batched_qpos.clear()
        batched_qvel.clear()
        batched_frames = 0
        return motion_chunk, lengths

    def flush_staged_motion_chunks() -> None:
        nonlocal start_idx, staged_motion_bytes
        if not staged_motion_chunks:
            return
        write_start_time = time.perf_counter()
        packed_motion = _concat_motion_data_chunks(staged_motion_chunks)
        total_length = int(packed_motion.motion_id.shape[0])
        end_idx = start_idx + total_length
        storage.ensure_capacity(end_idx)

        storage.fields["motion_id"][start_idx:end_idx] = packed_motion.motion_id
        storage.fields["step"][start_idx:end_idx] = packed_motion.step
        storage.fields["body_pos_w"][start_idx:end_idx] = packed_motion.body_pos_w
        storage.fields["body_lin_vel_w"][start_idx:end_idx] = packed_motion.body_lin_vel_w
        storage.fields["body_quat_w"][start_idx:end_idx] = packed_motion.body_quat_w
        storage.fields["body_ang_vel_w"][start_idx:end_idx] = packed_motion.body_ang_vel_w
        storage.fields["joint_pos"][start_idx:end_idx] = packed_motion.joint_pos
        storage.fields["joint_vel"][start_idx:end_idx] = packed_motion.joint_vel

        for length in staged_motion_lengths:
            starts.append(start_idx)
            start_idx += length
            ends.append(start_idx)

        write_elapsed_s = time.perf_counter() - write_start_time
        print(
            f"Write to storage: {write_elapsed_s:.2f}s for {len(staged_motion_lengths)} motions / {total_length} frames"
        )
        staged_motion_chunks.clear()
        staged_motion_lengths.clear()
        staged_motion_bytes = 0

    motion_iter = iter(motion_loader)
    for motion_idx, item in enumerate(tqdm(motion_iter, total=len(motion_paths), desc="Building FK cache", unit="file")):
        qpos = item["qpos"]
        qvel = item["qvel"]
        motion_length = resampled_length(
            int(qpos.shape[0]),
            source_fps=float(source_fps),
            target_fps=float(target_fps),
        )

        if batched_frames > 0 and batched_frames + motion_length >= fk_runner.batch_size:
            flushed = flush_batched_motions()
            if flushed is not None:
                motion_chunk, lengths = flushed
                staged_motion_bytes += _motion_data_num_bytes(motion_chunk)
                staged_motion_chunks.append(motion_chunk)
                staged_motion_lengths.extend(lengths)
                if staged_motion_bytes >= write_buffer_bytes:
                    print(
                        f"Staged motion data size {staged_motion_bytes} bytes exceeds buffer limit "
                        f"{write_buffer_bytes} bytes, flushing to storage"
                    )
                    flush_staged_motion_chunks()

        batched_motion_ids.append(motion_idx)
        batched_qpos.append(qpos)
        batched_qvel.append(qvel)
        batched_frames += motion_length

    flushed = flush_batched_motions()
    if flushed is not None:
        motion_chunk, lengths = flushed
        staged_motion_bytes += _motion_data_num_bytes(motion_chunk)
        staged_motion_chunks.append(motion_chunk)
        staged_motion_lengths.extend(lengths)
    flush_staged_motion_chunks()

    index_payload = {
        "body_names": body_names,
        "joint_names": joint_names,
        "starts": starts,
        "ends": ends,
        "motion_paths": [str(path) for path in motion_paths],
        "source_fps": float(source_fps),
        "target_fps": int(target_fps),
        "total_length": int(start_idx),
        "allocated_capacity": int(storage.capacity),
        "num_motions": len(starts),
    }
    (cache_entry_dir / QPOS_CACHE_INDEX_NAME).write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    cache_meta = {
        "cache_version": QPOS_CACHE_VERSION,
        "dataset_root": str(dataset_root),
        "manifest_path": str((dataset_root / "manifest.json").resolve()),
        "mjcf_path": str(mjcf_path),
        "target_fps": int(target_fps),
    }
    (cache_entry_dir / QPOS_CACHE_META_NAME).write_text(json.dumps(cache_meta, indent=2), encoding="utf-8")
    (cache_entry_dir / QPOS_CACHE_READY_NAME).write_text("ready\n", encoding="utf-8")


class FKCache:
    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest: dict[str, Any],
        motion_paths: list[Path],
        mjcf_path: Path,
        target_fps: int,
        base_dir: Path,
    ) -> None:
        self.dataset_root = dataset_root
        self.manifest = manifest
        self.motion_paths = list(motion_paths)
        self.mjcf_path = mjcf_path
        self.target_fps = int(target_fps)
        self.cache_root = _cache_root(base_dir)
        self.cache_key = _make_qpos_cache_key(
            dataset_root=dataset_root,
            motion_paths=self.motion_paths,
            mjcf_path=mjcf_path,
            target_fps=target_fps,
        )
        self.cache_entry_dir = self.cache_root / self.cache_key

    @classmethod
    def from_inputs(
        cls,
        *,
        input_paths: list[Path],
        target_fps: int,
        base_dir: Path,
    ) -> FKCache:
        dataset_root, manifest = resolve_any4hdmi_dataset_context(input_paths)
        _, _, motion_paths = resolve_any4hdmi_motion_paths(input_paths)
        mjcf_path = _resolve_any4hdmi_mjcf_path(dataset_root, manifest)
        return cls(
            dataset_root=dataset_root,
            manifest=manifest,
            motion_paths=motion_paths,
            mjcf_path=mjcf_path,
            target_fps=target_fps,
            base_dir=base_dir,
        )

    @property
    def ready_flag(self) -> Path:
        return self.cache_entry_dir / QPOS_CACHE_READY_NAME

    @property
    def lock_dir(self) -> Path:
        return self.cache_root / f"{self.cache_key}.lock"

    def get_or_build(self) -> FKCacheEntry:
        if not self.ready_flag.is_file():
            owns_lock = _acquire_cache_lock(self.lock_dir, self.ready_flag)
            if owns_lock:
                tmp_entry_dir = self.cache_root / f"{self.cache_key}.tmp-{os.getpid()}-{time.time_ns()}"
                try:
                    if tmp_entry_dir.exists():
                        shutil.rmtree(tmp_entry_dir)
                    tmp_entry_dir.mkdir(parents=True, exist_ok=False)
                    _build_fk_cache(
                        dataset_root=self.dataset_root,
                        manifest=self.manifest,
                        motion_paths=self.motion_paths,
                        mjcf_path=self.mjcf_path,
                        cache_entry_dir=tmp_entry_dir,
                        target_fps=self.target_fps,
                    )
                    if self.cache_entry_dir.exists():
                        shutil.rmtree(tmp_entry_dir)
                    else:
                        tmp_entry_dir.rename(self.cache_entry_dir)
                finally:
                    if tmp_entry_dir.exists():
                        shutil.rmtree(tmp_entry_dir, ignore_errors=True)
                    if self.lock_dir.exists():
                        shutil.rmtree(self.lock_dir, ignore_errors=True)
            elif not self.ready_flag.is_file():
                raise RuntimeError(f"Cache lock released but cache is not ready: {self.cache_entry_dir}")

        print(f"Loading qpos cache from {self.cache_entry_dir}")
        return self.load()

    def load(self, motion_paths: list[Path] | None = None) -> FKCacheEntry:
        index_payload = json.loads((self.cache_entry_dir / QPOS_CACHE_INDEX_NAME).read_text(encoding="utf-8"))
        total_length = int(index_payload["total_length"])
        allocated_capacity = int(index_payload.get("allocated_capacity", total_length))
        body_count = len(index_payload["body_names"])
        joint_count = len(index_payload["joint_names"])
        specs = _storage_field_specs(body_count=body_count, joint_count=joint_count)
        storage_fields: dict[str, torch.Tensor] = {}
        for field_name, (dtype, tail_shape) in specs.items():
            storage_fields[field_name] = MemoryMappedTensor.from_filename(
                str(self.cache_entry_dir / QPOS_CACHE_TD_SUBDIR / f"{field_name}.memmap"),
                dtype=dtype,
                shape=(allocated_capacity, *tail_shape),
            )[:total_length]

        resolved_motion_paths = motion_paths
        if resolved_motion_paths is None:
            resolved_motion_paths = [Path(path) for path in index_payload["motion_paths"]]
        return FKCacheEntry(
            cache_entry_dir=self.cache_entry_dir,
            body_names=list(index_payload["body_names"]),
            joint_names=list(index_payload["joint_names"]),
            motion_paths=resolved_motion_paths,
            starts=list(index_payload["starts"]),
            ends=list(index_payload["ends"]),
            storage_fields=storage_fields,
        )
