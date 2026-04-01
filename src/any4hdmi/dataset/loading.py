from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from any4hdmi.core.format import MANIFEST_NAME, load_motion
from any4hdmi.dataset.types import MotionData


@dataclass(frozen=True)
class LoadedDatasetPayload:
    body_names: list[str]
    joint_names: list[str]
    motion_paths: list[Path]
    starts: list[int]
    ends: list[int]
    data: MotionData


def resolve_input_paths(base_dir: Path, root_path: str | list[str] | Path | list[Path]) -> list[Path]:
    if isinstance(root_path, (str, Path)):
        raw_paths = [Path(root_path)]
    else:
        raw_paths = [Path(path) for path in root_path]

    resolved_paths: list[Path] = []
    for path in raw_paths:
        expanded = path.expanduser()
        if not expanded.is_absolute():
            expanded = base_dir / expanded
        resolved_paths.append(expanded.resolve())
    return resolved_paths


def find_any4hdmi_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
    return None


def load_any4hdmi_manifest(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / MANIFEST_NAME).read_text(encoding="utf-8"))


def resolve_any4hdmi_dataset_context(input_paths: list[Path]) -> tuple[Path, dict[str, Any]]:
    dataset_root: Path | None = None
    dataset_manifest: dict[str, Any] | None = None

    for input_path in input_paths:
        current_root = find_any4hdmi_root(input_path)
        if current_root is None:
            raise RuntimeError(f"Could not find {MANIFEST_NAME} above {input_path}")
        if dataset_root is None:
            dataset_root = current_root
            dataset_manifest = load_any4hdmi_manifest(current_root)
        elif current_root != dataset_root:
            raise ValueError(
                f"All any4hdmi inputs must belong to one dataset root, got {dataset_root} and {current_root}"
            )

    if dataset_root is None or dataset_manifest is None:
        raise RuntimeError("Failed to resolve any4hdmi dataset root")
    return dataset_root, dataset_manifest


def resolve_any4hdmi_motion_paths(input_paths: list[Path]) -> tuple[Path, dict[str, Any], list[Path]]:
    dataset_root, dataset_manifest = resolve_any4hdmi_dataset_context(input_paths)
    motion_paths: set[Path] = set()
    motions_root = dataset_root / dataset_manifest.get("motions_subdir", "motions")

    for input_path in input_paths:
        if input_path.is_file():
            if input_path.suffix != ".npz":
                raise ValueError(f"Expected a .npz motion file under any4hdmi root, got {input_path}")
            motion_paths.add(input_path.resolve())
            continue

        scan_root = motions_root if input_path == dataset_root else input_path
        motion_paths.update(path.resolve() for path in scan_root.rglob("*.npz"))

    if not motion_paths:
        motion_paths.update(path.resolve() for path in motions_root.rglob("*.npz"))
    motion_paths_list = sorted(motion_paths)
    if not motion_paths_list:
        raise RuntimeError(f"No qpos motions found under {dataset_root}")
    return dataset_root, dataset_manifest, motion_paths_list


def build_motion_data(
    motions: list[dict[str, np.ndarray]],
    *,
    body_names: list[str],
    joint_names: list[str],
) -> tuple[MotionData, list[int], list[int]]:
    total_length = sum(int(motion["body_pos_w"].shape[0]) for motion in motions)

    step = torch.empty(total_length, dtype=torch.long)
    motion_id = torch.empty(total_length, dtype=torch.long)
    body_pos_w = torch.empty(total_length, len(body_names), 3, dtype=torch.float32)
    body_lin_vel_w = torch.empty(total_length, len(body_names), 3, dtype=torch.float32)
    body_quat_w = torch.empty(total_length, len(body_names), 4, dtype=torch.float32)
    body_ang_vel_w = torch.empty(total_length, len(body_names), 3, dtype=torch.float32)
    joint_pos = torch.empty(total_length, len(joint_names), dtype=torch.float32)
    joint_vel = torch.empty(total_length, len(joint_names), dtype=torch.float32)

    starts: list[int] = []
    ends: list[int] = []
    start_idx = 0
    for motion_idx, motion in enumerate(motions):
        motion_length = int(motion["body_pos_w"].shape[0])
        end_idx = start_idx + motion_length
        step[start_idx:end_idx] = torch.arange(motion_length, dtype=torch.long)
        motion_id[start_idx:end_idx] = motion_idx
        body_pos_w[start_idx:end_idx] = torch.from_numpy(motion["body_pos_w"])
        body_lin_vel_w[start_idx:end_idx] = torch.from_numpy(motion["body_lin_vel_w"])
        body_quat_w[start_idx:end_idx] = torch.from_numpy(motion["body_quat_w"])
        body_ang_vel_w[start_idx:end_idx] = torch.from_numpy(motion["body_ang_vel_w"])
        joint_pos[start_idx:end_idx] = torch.from_numpy(motion["joint_pos"])
        joint_vel[start_idx:end_idx] = torch.from_numpy(motion["joint_vel"])
        starts.append(start_idx)
        ends.append(end_idx)
        start_idx = end_idx

    return (
        MotionData(
            motion_id=motion_id,
            step=step,
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_quat_w=body_quat_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        ),
        starts,
        ends,
    )


def build_motion_data_from_fields(
    *,
    motion_id: torch.Tensor,
    step: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> MotionData:
    return MotionData(
        motion_id=motion_id,
        step=step,
        body_pos_w=body_pos_w,
        body_lin_vel_w=body_lin_vel_w,
        body_quat_w=body_quat_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )


def apply_joint_mapping(
    motions: list[dict[str, np.ndarray]],
    joint_names: list[str],
    asset_joint_names: list[str] | None,
) -> list[str]:
    if asset_joint_names is None:
        return joint_names

    asset_joint_names_list = list(asset_joint_names)
    shared_joint_names = [name for name in joint_names if name in asset_joint_names_list]
    src_joint_indices = [joint_names.index(name) for name in shared_joint_names]
    dst_joint_indices = [asset_joint_names_list.index(name) for name in shared_joint_names]

    extra_joint_names = [name for name in joint_names if name not in asset_joint_names_list]
    src_joint_indices.extend(joint_names.index(name) for name in extra_joint_names)
    dst_joint_indices.extend(len(asset_joint_names_list) + i for i in range(len(extra_joint_names)))

    remapped_joint_names = asset_joint_names_list + extra_joint_names
    for motion in motions:
        motion_length = motion["joint_pos"].shape[0]
        joint_pos = np.zeros((motion_length, len(remapped_joint_names)), dtype=np.float32)
        joint_vel = np.zeros((motion_length, len(remapped_joint_names)), dtype=np.float32)
        joint_pos[:, dst_joint_indices] = motion["joint_pos"][:, src_joint_indices]
        joint_vel[:, dst_joint_indices] = motion["joint_vel"][:, src_joint_indices]
        motion["joint_pos"] = joint_pos
        motion["joint_vel"] = joint_vel
    return remapped_joint_names


def resolve_source_fps(manifest: dict[str, Any]) -> float:
    source_fps = float(manifest.get("fps", 0.0))
    if source_fps > 0.0:
        return source_fps
    timestep = float(manifest.get("timestep", 0.0))
    if timestep <= 0.0:
        raise ValueError("any4hdmi manifest must contain fps or timestep")
    return 1.0 / timestep
