from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from any4hdmi.core.format import MOTIONS_SUBDIR, load_manifest, write_manifest
from any4hdmi.dataset.loading import resolve_input_paths


DEFAULT_INPUT_ROOT = Path("output/amass")
DEFAULT_OUTPUT_ROOT = Path("output/limmt_amass")
DEFAULT_PASS_DATASET_NAME = "amass_limmt_pass"
DEFAULT_RATIOS = (0.04, 0.08, 0.16, 0.32)


def resolve_dataset_root(path: str | Path, *, base_dir: Path | None = None) -> Path:
    base = Path.cwd() if base_dir is None else base_dir
    return resolve_input_paths(base, path)[0]


def motion_paths_for_root(dataset_root: Path) -> list[Path]:
    manifest = load_manifest(dataset_root)
    motions_dir = dataset_root / manifest.payload.get("motions_subdir", MOTIONS_SUBDIR)
    motion_paths = sorted(motions_dir.rglob("*.npz"))
    if not motion_paths:
        raise FileNotFoundError(f"No .npz motions found under {motions_dir}")
    return motion_paths


def relative_motion_path(dataset_root: Path, motion_path: Path) -> Path:
    return motion_path.resolve().relative_to(dataset_root.resolve())


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def copy_any4hdmi_subset(
    *,
    input_root: Path,
    output_root: Path,
    selected_rel_paths: list[str],
    dataset_name: str,
    source_update: dict[str, Any],
) -> Path:
    manifest = load_manifest(input_root)
    if output_root.resolve() == input_root.resolve():
        raise ValueError(f"Refusing to overwrite input dataset root: {input_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    total_frames = 0

    for rel_raw in selected_rel_paths:
        rel_path = Path(rel_raw)
        src = input_root / rel_path
        dst = output_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        try:
            import numpy as np

            with np.load(src, allow_pickle=False) as data:
                total_frames += int(data["qpos"].shape[0])
        except Exception:
            pass

    source = dict(manifest.payload.get("source", {}))
    source.update(source_update)
    timestep = manifest.timestep
    total_hours = float(total_frames * timestep / 3600.0)
    return write_manifest(
        output_root,
        dataset_name=dataset_name,
        mjcf=manifest.payload["mjcf"],
        timestep=timestep,
        qpos_names=list(manifest.payload["qpos_names"]),
        num_motions=len(selected_rel_paths),
        source=source,
        total_hours=total_hours,
    )
