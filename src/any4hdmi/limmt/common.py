from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from any4hdmi.core.format import MOTIONS_SUBDIR, load_manifest, write_manifest
from any4hdmi.dataset.loading import resolve_input_paths


def resolve_dataset_root(path: str | Path, *, base_dir: Path | None = None) -> Path:
    base = Path.cwd() if base_dir is None else base_dir
    return resolve_input_paths(base, path)[0]


def resolve_project_root(path: str | Path, *, base_dir: Path | None = None) -> Path:
    base = Path.cwd() if base_dir is None else base_dir
    project_path = Path(path).expanduser()
    if not project_path.is_absolute():
        project_path = base / project_path
    return project_path.resolve()


def default_project_root_for_input(input_root: Path) -> Path:
    return input_root.parent / f"{input_root.name}_limmt"


def project_pass_root(project_root: Path, pass_dataset_name: str = "passed") -> Path:
    return project_root / pass_dataset_name


def project_hme_root(project_root: Path, hme_folder: str = "hme") -> Path:
    return project_root / hme_folder


def project_embeddings_root(project_root: Path, embeddings_folder: str = "embeddings") -> Path:
    return project_root / embeddings_folder


def project_subsets_root(project_root: Path, subsets_folder: str = "subsets") -> Path:
    return project_root / subsets_folder


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


def write_json(path: str | Path, json_payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
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

            with np.load(src, allow_pickle=False) as motion_file:
                total_frames += int(motion_file["qpos"].shape[0])
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
