from __future__ import annotations

from pathlib import Path

from any4hdmi.dataset.base import BaseDataset
from any4hdmi.dataset.fk_cache import FKCache, FKCacheEntry
from any4hdmi.dataset.loading import resolve_input_paths
from any4hdmi.dataset.full import FullMotionDataset
from any4hdmi.dataset.windowed import OnlineQposDataset, WindowedMotionDataset


def load_any4hdmi_dataset(
    *,
    root_path: str | Path | list[str] | list[Path],
    target_fps: int,
    base_dir: Path,
    asset_joint_names: list[str] | None = None,
    num_envs: int,
    full_motion: bool = True,
) -> BaseDataset:
    del asset_joint_names
    input_paths = resolve_input_paths(base_dir, root_path)
    cache_entry = FKCache.from_inputs(
        input_paths=input_paths,
        target_fps=target_fps,
        base_dir=base_dir,
    ).get_or_build()
    if full_motion:
        return FullMotionDataset.from_cache_entry(cache_entry, num_envs=num_envs)
    return WindowedMotionDataset.from_cache_entry(cache_entry, num_envs=num_envs)


__all__ = [
    "FKCache",
    "FKCacheEntry",
    "FullMotionDataset",
    "OnlineQposDataset",
    "WindowedMotionDataset",
    "load_any4hdmi_dataset",
]
