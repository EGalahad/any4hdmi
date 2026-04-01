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
from tqdm import tqdm

from any4hdmi.core.model import body_names_from_model, hinge_joint_info, load_model
from any4hdmi.dataset.interpolation import interpolate
from any4hdmi.dataset.loading import (
    LoadedDatasetPayload,
    apply_joint_mapping,
    build_motion_data,
    build_motion_data_from_fields,
    resolve_any4hdmi_dataset_context,
    resolve_any4hdmi_motion_paths,
    resolve_source_fps,
)
from any4hdmi.fk.runner import FKRunner
from any4hdmi.utils.mjcf import build_hf_mjcf_reference, resolve_mjcf_path


QPOS_CACHE_VERSION = 1
QPOS_CACHE_SUBDIR = ".cache/motion/qpos_fk"
QPOS_CACHE_INDEX_NAME = "motion_index.json"
QPOS_CACHE_META_NAME = "cache_meta.json"
QPOS_CACHE_DATA_NAME = "motion_data.pt"
QPOS_CACHE_READY_NAME = "ready.flag"


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
    }
    (cache_entry_dir / QPOS_CACHE_META_NAME).write_text(
        json.dumps(cache_meta, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "motion_id": data.motion_id.cpu(),
            "step": data.step.cpu(),
            "body_pos_w": data.body_pos_w.cpu(),
            "body_lin_vel_w": data.body_lin_vel_w.cpu(),
            "body_quat_w": data.body_quat_w.cpu(),
            "body_ang_vel_w": data.body_ang_vel_w.cpu(),
            "joint_pos": data.joint_pos.cpu(),
            "joint_vel": data.joint_vel.cpu(),
        },
        cache_entry_dir / QPOS_CACHE_DATA_NAME,
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
    fk_runner = FKRunner(mjcf_path=mjcf_path, batch_size=2048, device="cpu")
    body_names = body_names_from_model(model)
    joint_names, joint_qpos_addrs, joint_dof_addrs = hinge_joint_info(model)
    joint_qpos_addrs_t = torch.as_tensor(joint_qpos_addrs, dtype=torch.long)
    joint_dof_addrs_t = torch.as_tensor(joint_dof_addrs, dtype=torch.long)
    source_fps = resolve_source_fps(manifest)

    motions: list[dict[str, np.ndarray]] = []
    for motion_path in tqdm(motion_paths, desc="Building qpos FK cache", unit="file"):
        qpos = load_motion(motion_path)
        qvel = _compute_qvel(model, qpos, source_fps)
        motion_tensors = fk_runner.forward_kinematics_many(
            [torch.from_numpy(qpos)],
            [torch.from_numpy(qvel)],
            joint_qpos_addrs=joint_qpos_addrs_t,
            joint_dof_addrs=joint_dof_addrs_t,
        )[0]
        motion = {
            key: value.detach().cpu().numpy()
            for key, value in motion_tensors.items()
        }
        motions.append(
            interpolate(
                motion,
                source_fps=int(round(source_fps)),
                target_fps=target_fps,
            )
        )

    built_data, starts, ends = build_motion_data(
        motions,
        body_names=body_names,
        joint_names=joint_names,
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
    )


def _load_qpos_cache_entry(
    *,
    cache_entry_dir: Path,
    motion_paths: list[Path] | None,
    asset_joint_names: list[str] | None,
) -> LoadedDatasetPayload:
    tensor_payload = torch.load(cache_entry_dir / QPOS_CACHE_DATA_NAME, map_location="cpu")
    index_payload = json.loads((cache_entry_dir / QPOS_CACHE_INDEX_NAME).read_text(encoding="utf-8"))

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
