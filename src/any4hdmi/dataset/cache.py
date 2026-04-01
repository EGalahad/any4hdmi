from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from tensordict import TensorDict
from tqdm import tqdm

from any4hdmi.core.model import body_names_from_model, hinge_joint_info, load_model
from any4hdmi.dataset.interpolation import interpolate
from any4hdmi.dataset.loading import (
    LoadedDatasetPayload,
    apply_joint_mapping,
    build_motion_data_from_fields,
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


QPOS_CACHE_VERSION = 2
QPOS_CACHE_SUBDIR = ".cache/motion/qpos_fk"
QPOS_CACHE_INDEX_NAME = "motion_index.json"
QPOS_CACHE_META_NAME = "cache_meta.json"
QPOS_CACHE_DATA_NAME = "motion_data.pt"
QPOS_CACHE_READY_NAME = "ready.flag"
QPOS_CACHE_TD_SUBDIR = "td"
CACHE_GPU_PROMOTE_THRESHOLD_BYTES = 16 * 1024**3
QPOS_CACHE_BUILD_NUM_WORKERS = DEFAULT_MOTION_LOADER_NUM_WORKERS
QPOS_CACHE_BUILD_PREFETCH_FACTOR = DEFAULT_MOTION_LOADER_PREFETCH_FACTOR
MOTION_DATA_FIELD_NAMES = (
    "motion_id",
    "step",
    "body_pos_w",
    "body_lin_vel_w",
    "body_quat_w",
    "body_ang_vel_w",
    "joint_pos",
    "joint_vel",
)


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
        "mjcf": _stat_fingerprint(mjcf_path),
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


def _compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float32)
    if qpos.shape[0] <= 1:
        return qvel

    dt = 1.0 / fps
    work = np.zeros(model.nv, dtype=np.float64)
    for frame_idx in range(qpos.shape[0] - 1):
        mujoco.mj_differentiatePos(model, work, dt, qpos[frame_idx], qpos[frame_idx + 1])
        qvel[frame_idx] = np.asarray(work, dtype=np.float32)
    qvel[-1] = qvel[-2]
    return qvel


def _interpolated_length(num_frames: int, source_fps: int, target_fps: int) -> int:
    if num_frames <= 0:
        return 0
    if source_fps == target_fps:
        return int(num_frames)
    return ((int(num_frames) - 1) * int(target_fps)) // int(source_fps) + 1


def _scan_motion_lengths(
    *,
    motion_paths: list[Path],
    source_fps: int,
    target_fps: int,
) -> tuple[list[int], int]:
    motion_lengths: list[int] = []
    total_length = 0
    for motion_path in tqdm(motion_paths, desc="Scanning motion lengths", unit="file"):
        with np.load(motion_path, allow_pickle=False) as payload:
            if "qpos" not in payload:
                raise KeyError(f"Motion file {motion_path} does not contain a qpos array")
            qpos_shape = payload["qpos"].shape
        if len(qpos_shape) == 1:
            frame_count = 1
        elif len(qpos_shape) == 2:
            frame_count = int(qpos_shape[0])
        else:
            raise ValueError(f"Expected qpos to be rank 1 or 2, got shape {qpos_shape}")
        output_length = _interpolated_length(frame_count, source_fps, target_fps)
        motion_lengths.append(output_length)
        total_length += output_length
    return motion_lengths, total_length


def _estimate_materialized_bytes(total_length: int, *, nbody: int, njoint: int) -> int:
    float32_size = 4
    int64_size = 8
    return (
        total_length * int64_size +  # motion_id
        total_length * int64_size +  # step
        total_length * nbody * 3 * float32_size +  # body_pos_w
        total_length * nbody * 3 * float32_size +  # body_lin_vel_w
        total_length * nbody * 4 * float32_size +  # body_quat_w
        total_length * nbody * 3 * float32_size +  # body_ang_vel_w
        total_length * njoint * float32_size +  # joint_pos
        total_length * njoint * float32_size    # joint_vel
    )


def _allocate_storage(
    *,
    cache_entry_dir: Path,
    total_length: int,
    nbody: int,
    njoint: int,
) -> tuple[dict[str, torch.Tensor], str]:
    td = TensorDict(
        {
            "motion_id": torch.empty(total_length, dtype=torch.long),
            "step": torch.empty(total_length, dtype=torch.long),
            "body_pos_w": torch.empty(total_length, nbody, 3, dtype=torch.float32),
            "body_lin_vel_w": torch.empty(total_length, nbody, 3, dtype=torch.float32),
            "body_quat_w": torch.empty(total_length, nbody, 4, dtype=torch.float32),
            "body_ang_vel_w": torch.empty(total_length, nbody, 3, dtype=torch.float32),
            "joint_pos": torch.empty(total_length, njoint, dtype=torch.float32),
            "joint_vel": torch.empty(total_length, njoint, dtype=torch.float32),
        },
        batch_size=[total_length],
    )
    td = td.memmap(prefix=str(cache_entry_dir / QPOS_CACHE_TD_SUBDIR))
    return {
        "motion_id": td["motion_id"],
        "step": td["step"],
        "body_pos_w": td["body_pos_w"],
        "body_lin_vel_w": td["body_lin_vel_w"],
        "body_quat_w": td["body_quat_w"],
        "body_ang_vel_w": td["body_ang_vel_w"],
        "joint_pos": td["joint_pos"],
        "joint_vel": td["joint_vel"],
    }, "memmap"


def _should_promote_loaded_data_to_gpu(estimated_bytes: int) -> bool:
    if not torch.cuda.is_available():
        return False
    if estimated_bytes > CACHE_GPU_PROMOTE_THRESHOLD_BYTES:
        return False
    free_bytes, _ = torch.cuda.mem_get_info()
    return estimated_bytes <= int(free_bytes)


def _maybe_promote_loaded_data_to_gpu(data, estimated_bytes: int):
    if _should_promote_loaded_data_to_gpu(estimated_bytes):
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            promoted_fields = {
                field_name: data_field.pin_memory().to(device=device, non_blocking=True)
                for field_name, data_field in data.__dict__.items()
            }
        torch.cuda.current_stream(device=device).wait_stream(stream)
        return build_motion_data_from_fields(**promoted_fields)
    return data


def _flush_motion_batch(
    *,
    batch_items: list[dict[str, Any]],
    batch_motion_ids: list[int],
    batch_output_lengths: list[int],
    model: mujoco.MjModel,
    fk_runner: FKRunner,
    joint_qpos_addrs_t: torch.Tensor,
    joint_dof_addrs_t: torch.Tensor,
    source_fps_int: int,
    target_fps: int,
    storage_fields: dict[str, torch.Tensor],
    starts: list[int],
    ends: list[int],
    start_idx: int,
) -> int:
    if not batch_items:
        return start_idx

    qpos_list = [item["qpos"] for item in batch_items]
    qvel_list = [
        torch.from_numpy(_compute_qvel(model, item["qpos"].numpy(), source_fps_int)).contiguous()
        for item in batch_items
    ]
    motion_tensors_list = fk_runner.forward_kinematics_many(
        qpos_list,
        qvel_list,
        joint_qpos_addrs=joint_qpos_addrs_t,
        joint_dof_addrs=joint_dof_addrs_t,
    )

    for motion_id, output_length, motion_tensors in zip(
        batch_motion_ids,
        batch_output_lengths,
        motion_tensors_list,
        strict=True,
    ):
        motion = interpolate(
            motion_tensors,
            source_fps=source_fps_int,
            target_fps=target_fps,
        )
        end_idx = start_idx + output_length
        storage_fields["motion_id"][start_idx:end_idx] = motion_id
        storage_fields["step"][start_idx:end_idx] = torch.arange(output_length, dtype=torch.long)
        storage_fields["body_pos_w"][start_idx:end_idx] = motion["body_pos_w"].to(device="cpu")
        storage_fields["body_lin_vel_w"][start_idx:end_idx] = motion["body_lin_vel_w"].to(device="cpu")
        storage_fields["body_quat_w"][start_idx:end_idx] = motion["body_quat_w"].to(device="cpu")
        storage_fields["body_ang_vel_w"][start_idx:end_idx] = motion["body_ang_vel_w"].to(device="cpu")
        storage_fields["joint_pos"][start_idx:end_idx] = motion["joint_pos"].to(device="cpu")
        storage_fields["joint_vel"][start_idx:end_idx] = motion["joint_vel"].to(device="cpu")
        starts.append(start_idx)
        ends.append(end_idx)
        start_idx = end_idx

    batch_items.clear()
    batch_motion_ids.clear()
    batch_output_lengths.clear()
    return start_idx


def _save_cache_entry(
    *,
    cache_entry_dir: Path,
    dataset_root: Path,
    mjcf_path: Path,
    target_fps: int,
    motion_paths: list[Path],
    body_names: list[str],
    joint_names: list[str],
    starts: list[int],
    ends: list[int],
    data,
    source_fps: float,
    storage: str,
    estimated_bytes: int,
) -> None:
    index_payload = {
        "body_names": body_names,
        "joint_names": joint_names,
        "starts": starts,
        "ends": ends,
        "motion_paths": [str(path) for path in motion_paths],
        "source_fps": float(source_fps),
        "target_fps": int(target_fps),
        "total_length": len(data),
        "storage": storage,
        "estimated_bytes": int(estimated_bytes),
    }
    (cache_entry_dir / QPOS_CACHE_INDEX_NAME).write_text(
        json.dumps(index_payload, indent=2),
        encoding="utf-8",
    )
    cache_meta = {
        "cache_version": QPOS_CACHE_VERSION,
        "dataset_root": str(dataset_root),
        "manifest_path": str((dataset_root / "manifest.json").resolve()),
        "mjcf_path": str(mjcf_path),
        "target_fps": int(target_fps),
        "storage": storage,
        "estimated_bytes": int(estimated_bytes),
    }
    (cache_entry_dir / QPOS_CACHE_META_NAME).write_text(
        json.dumps(cache_meta, indent=2),
        encoding="utf-8",
    )
    (cache_entry_dir / QPOS_CACHE_READY_NAME).write_text("ready\n", encoding="utf-8")


def _build_qpos_cache(
    *,
    dataset_root: Path,
    manifest: dict[str, Any],
    motion_paths: list[Path],
    mjcf_path: Path,
    cache_entry_dir: Path,
    target_fps: int,
) -> None:
    model = load_model(mjcf_path)
    fk_runner = FKRunner(mjcf_path=mjcf_path, batch_size=2048)
    body_names = body_names_from_model(model)
    joint_names, joint_qpos_addrs, joint_dof_addrs = hinge_joint_info(model)
    joint_qpos_addrs_t = torch.as_tensor(joint_qpos_addrs, dtype=torch.long)
    joint_dof_addrs_t = torch.as_tensor(joint_dof_addrs, dtype=torch.long)
    source_fps = resolve_source_fps(manifest)
    source_fps_int = int(round(source_fps))
    motion_lengths, total_length = _scan_motion_lengths(
        motion_paths=motion_paths,
        source_fps=source_fps_int,
        target_fps=target_fps,
    )
    estimated_bytes = _estimate_materialized_bytes(
        total_length,
        nbody=len(body_names),
        njoint=len(joint_names),
    )
    storage_fields, storage = _allocate_storage(
        cache_entry_dir=cache_entry_dir,
        total_length=total_length,
        nbody=len(body_names),
        njoint=len(joint_names),
    )

    starts: list[int] = []
    ends: list[int] = []
    start_idx = 0
    motion_loader = build_motion_loader(
        input_root=dataset_root,
        motion_paths=motion_paths,
        fps=source_fps,
        num_workers=QPOS_CACHE_BUILD_NUM_WORKERS,
        prefetch_factor=QPOS_CACHE_BUILD_PREFETCH_FACTOR,
        pin_memory=fk_runner.device.type == "cuda",
    )
    batch_items: list[dict[str, Any]] = []
    batch_motion_ids: list[int] = []
    batch_output_lengths: list[int] = []
    batch_frames = 0
    for motion_idx, item in enumerate(
        tqdm(motion_loader, total=len(motion_paths), desc="Building qpos FK cache", unit="file")
    ):
        batch_items.append(item)
        batch_motion_ids.append(motion_idx)
        batch_output_lengths.append(motion_lengths[motion_idx])
        batch_frames += int(item["qpos"].shape[0])
        if batch_frames >= fk_runner.batch_size:
            start_idx = _flush_motion_batch(
                batch_items=batch_items,
                batch_motion_ids=batch_motion_ids,
                batch_output_lengths=batch_output_lengths,
                model=model,
                fk_runner=fk_runner,
                joint_qpos_addrs_t=joint_qpos_addrs_t,
                joint_dof_addrs_t=joint_dof_addrs_t,
                source_fps_int=source_fps_int,
                target_fps=target_fps,
                storage_fields=storage_fields,
                starts=starts,
                ends=ends,
                start_idx=start_idx,
            )
            batch_frames = 0
    start_idx = _flush_motion_batch(
        batch_items=batch_items,
        batch_motion_ids=batch_motion_ids,
        batch_output_lengths=batch_output_lengths,
        model=model,
        fk_runner=fk_runner,
        joint_qpos_addrs_t=joint_qpos_addrs_t,
        joint_dof_addrs_t=joint_dof_addrs_t,
        source_fps_int=source_fps_int,
        target_fps=target_fps,
        storage_fields=storage_fields,
        starts=starts,
        ends=ends,
        start_idx=start_idx,
    )

    built_data = build_motion_data_from_fields(
        motion_id=storage_fields["motion_id"],
        step=storage_fields["step"],
        body_pos_w=storage_fields["body_pos_w"],
        body_lin_vel_w=storage_fields["body_lin_vel_w"],
        body_quat_w=storage_fields["body_quat_w"],
        body_ang_vel_w=storage_fields["body_ang_vel_w"],
        joint_pos=storage_fields["joint_pos"],
        joint_vel=storage_fields["joint_vel"],
    )
    _save_cache_entry(
        cache_entry_dir=cache_entry_dir,
        dataset_root=dataset_root,
        mjcf_path=mjcf_path,
        target_fps=target_fps,
        motion_paths=motion_paths,
        body_names=body_names,
        joint_names=joint_names,
        starts=starts,
        ends=ends,
        data=built_data,
        source_fps=source_fps,
        storage=storage,
        estimated_bytes=estimated_bytes,
    )


def _load_qpos_cache_entry(
    *,
    cache_entry_dir: Path,
    motion_paths: list[Path] | None,
    asset_joint_names: list[str] | None,
) -> LoadedDatasetPayload:
    index_payload = json.loads((cache_entry_dir / QPOS_CACHE_INDEX_NAME).read_text(encoding="utf-8"))
    storage = str(index_payload.get("storage", "torch"))
    estimated_bytes = int(index_payload.get("estimated_bytes", 0))
    if storage == "memmap":
        td = TensorDict.load_memmap(cache_entry_dir / QPOS_CACHE_TD_SUBDIR)
        tensor_payload = {field_name: td[field_name] for field_name in MOTION_DATA_FIELD_NAMES}
    else:
        tensor_payload = torch.load(cache_entry_dir / QPOS_CACHE_DATA_NAME, map_location="cpu")

    joint_names = list(index_payload["joint_names"])
    joint_pos = tensor_payload["joint_pos"]
    joint_vel = tensor_payload["joint_vel"]
    if asset_joint_names is not None and joint_names != list(asset_joint_names):
        motions = [
            {
                "joint_pos": joint_pos.numpy(),
                "joint_vel": joint_vel.numpy(),
            }
        ]
        joint_names = apply_joint_mapping(motions, joint_names, asset_joint_names)
        joint_pos = torch.from_numpy(motions[0]["joint_pos"])
        joint_vel = torch.from_numpy(motions[0]["joint_vel"])

    loaded_data = build_motion_data_from_fields(
        motion_id=tensor_payload["motion_id"],
        step=tensor_payload["step"],
        body_pos_w=tensor_payload["body_pos_w"],
        body_lin_vel_w=tensor_payload["body_lin_vel_w"],
        body_quat_w=tensor_payload["body_quat_w"],
        body_ang_vel_w=tensor_payload["body_ang_vel_w"],
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )
    loaded_data = _maybe_promote_loaded_data_to_gpu(loaded_data, estimated_bytes)
    resolved_motion_paths = motion_paths
    if resolved_motion_paths is None:
        resolved_motion_paths = [Path(path) for path in index_payload["motion_paths"]]
    return LoadedDatasetPayload(
        body_names=list(index_payload["body_names"]),
        joint_names=joint_names,
        motion_paths=resolved_motion_paths,
        starts=list(index_payload["starts"]),
        ends=list(index_payload["ends"]),
        data=loaded_data,
    )


def load_cached_any4hdmi_dataset(
    *,
    input_paths: list[Path],
    asset_joint_names: list[str] | None,
    target_fps: int,
    base_dir: Path,
) -> LoadedDatasetPayload:
    dataset_root, manifest = resolve_any4hdmi_dataset_context(input_paths)
    _, _, motion_paths = resolve_any4hdmi_motion_paths(input_paths)
    mjcf_path = _resolve_any4hdmi_mjcf_path(dataset_root, manifest)
    cache_root = _cache_root(base_dir)
    cache_key = _make_qpos_cache_key(
        dataset_root=dataset_root,
        motion_paths=motion_paths,
        mjcf_path=mjcf_path,
        target_fps=target_fps,
    )
    cache_entry_dir = cache_root / cache_key
    ready_flag = cache_entry_dir / QPOS_CACHE_READY_NAME
    lock_dir = cache_root / f"{cache_key}.lock"

    if not ready_flag.is_file():
        owns_lock = _acquire_cache_lock(lock_dir, ready_flag)
        if owns_lock:
            tmp_entry_dir = cache_root / f"{cache_key}.tmp-{os.getpid()}-{time.time_ns()}"
            try:
                if tmp_entry_dir.exists():
                    shutil.rmtree(tmp_entry_dir)
                tmp_entry_dir.mkdir(parents=True, exist_ok=False)
                _build_qpos_cache(
                    dataset_root=dataset_root,
                    manifest=manifest,
                    motion_paths=motion_paths,
                    mjcf_path=mjcf_path,
                    cache_entry_dir=tmp_entry_dir,
                    target_fps=target_fps,
                )
                if cache_entry_dir.exists():
                    shutil.rmtree(tmp_entry_dir)
                else:
                    tmp_entry_dir.rename(cache_entry_dir)
            finally:
                if tmp_entry_dir.exists():
                    shutil.rmtree(tmp_entry_dir, ignore_errors=True)
                if lock_dir.exists():
                    shutil.rmtree(lock_dir, ignore_errors=True)
        elif not ready_flag.is_file():
            raise RuntimeError(f"Cache lock released but cache is not ready: {cache_entry_dir}")

    return _load_qpos_cache_entry(
        cache_entry_dir=cache_entry_dir,
        motion_paths=motion_paths,
        asset_joint_names=asset_joint_names,
    )
