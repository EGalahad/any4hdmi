from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from any4hdmi.core.format import MANIFEST_NAME
from tqdm import tqdm


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
        motion_paths.update(
            path.resolve()
            for path in tqdm(scan_root.rglob("*.npz"), desc=f"Scanning {scan_root.name}", unit="file")
        )

    if not motion_paths:
        motion_paths.update(
            path.resolve()
            for path in tqdm(motions_root.rglob("*.npz"), desc=f"Scanning {motions_root.name}", unit="file")
        )
    motion_paths_list = sorted(motion_paths)
    if not motion_paths_list:
        raise RuntimeError(f"No qpos motions found under {dataset_root}")
    return dataset_root, dataset_manifest, motion_paths_list


def resolve_source_fps(manifest: dict[str, Any]) -> float:
    source_fps = float(manifest.get("fps", 0.0))
    if source_fps > 0.0:
        return source_fps
    timestep = float(manifest.get("timestep", 0.0))
    if timestep <= 0.0:
        raise ValueError("any4hdmi manifest must contain fps or timestep")
    return 1.0 / timestep
