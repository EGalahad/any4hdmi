from __future__ import annotations

from pathlib import Path

import torch

from any4hdmi.dataset.base import BaseDataset
from any4hdmi.dataset.fk_cache import FKCache, FKCacheEntry
from any4hdmi.dataset.loading import resolve_input_paths
from any4hdmi.dataset.full import FullMotionDataset
from any4hdmi.dataset.windowed import OnlineQposDataset, WindowedMotionDataset


_BODY_FIELD_NAMES = (
    "body_pos_w",
    "body_lin_vel_w",
    "body_quat_w",
    "body_ang_vel_w",
)
_JOINT_FIELD_NAMES = ("joint_pos", "joint_vel")


def _indices_for_names(available: list[str], requested: list[str], *, label: str) -> list[int]:
    name_to_index = {name: index for index, name in enumerate(available)}
    missing = [name for name in requested if name not in name_to_index]
    if missing:
        raise ValueError(f"Requested {label} names are missing from motion cache: {missing}")
    return [name_to_index[name] for name in requested]


def _prune_cache_entry(
    entry: FKCacheEntry,
    *,
    body_names: list[str] | None,
    joint_names: list[str] | None,
) -> FKCacheEntry:
    if body_names is None and joint_names is None:
        return entry

    storage_fields = dict(entry.storage_fields)
    next_body_names = entry.body_names
    next_joint_names = entry.joint_names

    if body_names is not None:
        body_indices = _indices_for_names(entry.body_names, body_names, label="body")
        index = storage_fields["body_pos_w"].new_tensor(body_indices, dtype=torch.long)
        for field_name in _BODY_FIELD_NAMES:
            storage_fields[field_name] = storage_fields[field_name].index_select(1, index)
        next_body_names = list(body_names)

    if joint_names is not None:
        joint_indices = _indices_for_names(entry.joint_names, joint_names, label="joint")
        index = storage_fields["joint_pos"].new_tensor(joint_indices, dtype=torch.long)
        for field_name in _JOINT_FIELD_NAMES:
            storage_fields[field_name] = storage_fields[field_name].index_select(1, index)
        next_joint_names = list(joint_names)

    return FKCacheEntry(
        cache_entry_dir=entry.cache_entry_dir,
        body_names=next_body_names,
        joint_names=next_joint_names,
        motion_paths=entry.motion_paths,
        starts=entry.starts,
        ends=entry.ends,
        storage_fields=storage_fields,
    )


def load_any4hdmi_dataset(
    *,
    root_path: str | Path | list[str] | list[Path],
    target_fps: int,
    base_dir: Path,
    body_names: list[str] | None = None,
    joint_names: list[str] | None = None,
    num_envs: int,
    full_motion: bool = True,
    windowed_next_window_device: str | torch.device | None = "current",
    windowed_pin_window_load: bool = True,
) -> BaseDataset:
    input_paths = resolve_input_paths(base_dir, root_path)
    cache_entry = FKCache.from_inputs(
        input_paths=input_paths,
        target_fps=target_fps,
        base_dir=base_dir,
    ).get_or_build()
    cache_entry = _prune_cache_entry(
        cache_entry,
        body_names=body_names,
        joint_names=joint_names,
    )
    if full_motion:
        return FullMotionDataset.from_cache_entry(cache_entry, num_envs=num_envs)
    return WindowedMotionDataset.from_cache_entry(
        cache_entry,
        num_envs=num_envs,
        next_window_device=windowed_next_window_device,
        pin_window_load=windowed_pin_window_load,
    )


__all__ = [
    "FKCache",
    "FKCacheEntry",
    "FullMotionDataset",
    "OnlineQposDataset",
    "WindowedMotionDataset",
    "load_any4hdmi_dataset",
]
