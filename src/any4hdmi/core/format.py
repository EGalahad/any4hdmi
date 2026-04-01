from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from any4hdmi.utils.mjcf import MjcfInput, normalize_mjcf_reference, resolve_mjcf_path


FORMAT_VERSION = 2
MANIFEST_NAME = "manifest.json"
MOTIONS_SUBDIR = "motions"
MOTION_DTYPE = np.float32


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    root: Path
    payload: dict[str, Any]

    @property
    def dataset_name(self) -> str:
        return str(self.payload["dataset_name"])

    @property
    def timestep(self) -> float:
        return float(self.payload["timestep"])

    @property
    def mjcf(self) -> MjcfInput:
        return normalize_mjcf_reference(self.payload["mjcf"], dataset_root=self.root)

    @property
    def mjcf_path(self) -> Path:
        return resolve_mjcf_path(self.mjcf)

    @property
    def qpos_dim(self) -> int:
        return int(self.payload["qpos_dim"])


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_root(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def _manifest_mjcf_payload(mjcf: MjcfInput, dataset_root: Path) -> str:
    normalized = normalize_mjcf_reference(mjcf)
    if isinstance(normalized, Path):
        return relative_to_root(normalized, dataset_root)
    return normalized


def write_manifest(
    dataset_root: str | Path,
    *,
    dataset_name: str,
    mjcf: MjcfInput,
    timestep: float,
    qpos_names: list[str],
    num_motions: int,
    source: dict[str, Any],
) -> Path:
    dataset_root = ensure_dir(dataset_root).resolve()
    manifest_path = dataset_root / MANIFEST_NAME
    payload = {
        "format_version": FORMAT_VERSION,
        "dataset_name": dataset_name,
        "mjcf": _manifest_mjcf_payload(mjcf, dataset_root),
        "motions_subdir": MOTIONS_SUBDIR,
        "timestep": float(timestep),
        "qpos_dim": len(qpos_names),
        "qpos_names": qpos_names,
        "num_motions": int(num_motions),
        "source": source,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_manifest(dataset_root: str | Path) -> DatasetManifest:
    dataset_root = Path(dataset_root).expanduser().resolve()
    manifest_path = dataset_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DatasetManifest(path=manifest_path, root=dataset_root, payload=payload)


def find_dataset_root(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    current = path if path.is_dir() else path.parent
    for candidate in [current, *current.parents]:
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {MANIFEST_NAME} above {path}")


def load_motion(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Motion file not found: {path}")
    payload = np.load(path, allow_pickle=False)
    if "qpos" not in payload:
        raise KeyError(f"Motion file {path} does not contain a qpos array")
    qpos = np.asarray(payload["qpos"], dtype=MOTION_DTYPE)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos to be rank 2, got shape {qpos.shape}")
    return qpos


def save_motion(path: str | Path, qpos: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, qpos=np.asarray(qpos, dtype=MOTION_DTYPE))
    return path
