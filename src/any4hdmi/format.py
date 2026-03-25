from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from huggingface_hub import snapshot_download


FORMAT_VERSION = 2
MANIFEST_NAME = "manifest.json"
MOTIONS_SUBDIR = "motions"
DEFAULT_BASE_JOINT_NAME = "floating_base_joint"
MOTION_DTYPE = np.float32
DEFAULT_MJCF_REPO_ID = "elijahgalahad/g1_xmls"
DEFAULT_MJCF_PATH = "g1.xml"
DEFAULT_MJCF_REVISION = "main"


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
    def mjcf(self) -> Any:
        return self.payload["mjcf"]

    @property
    def mjcf_path(self) -> Path:
        return resolve_mjcf_reference(self.payload["mjcf"], dataset_root=self.root)

    @property
    def qpos_dim(self) -> int:
        return int(self.payload["qpos_dim"])


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_root(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def build_hf_mjcf_reference(
    *,
    repo_id: str = DEFAULT_MJCF_REPO_ID,
    path: str = DEFAULT_MJCF_PATH,
    revision: str = DEFAULT_MJCF_REVISION,
) -> dict[str, str]:
    return {
        "kind": "huggingface",
        "repo_id": repo_id,
        "path": path,
        "revision": revision,
    }


def _resolve_huggingface_mjcf(reference: dict[str, Any]) -> Path:
    repo_id = str(reference["repo_id"])
    path = str(reference["path"])
    revision = str(reference.get("revision", DEFAULT_MJCF_REVISION))
    repo_root = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
        )
    )
    # Keep the snapshot-relative symlink path so MuJoCo can still resolve
    # meshes like assets/*.STL relative to the XML location.
    mjcf_path = repo_root / path
    if not mjcf_path.is_file():
        raise FileNotFoundError(
            f"MJCF path {path!r} was not found in Hugging Face repo {repo_id!r} at revision {revision!r}"
        )
    return mjcf_path


def resolve_mjcf_reference(mjcf: Any, *, dataset_root: str | Path | None = None) -> Path:
    if isinstance(mjcf, dict):
        if mjcf.get("kind") != "huggingface":
            raise ValueError(f"Unsupported mjcf reference kind: {mjcf.get('kind')!r}")
        return _resolve_huggingface_mjcf(mjcf)

    if dataset_root is None:
        raise ValueError("dataset_root is required when resolving a local mjcf path")

    mjcf_path = (Path(dataset_root).expanduser().resolve() / Path(mjcf)).resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
    return mjcf_path


def qpos_names_from_model(
    model: mujoco.MjModel, base_joint_name: str = DEFAULT_BASE_JOINT_NAME
) -> list[str]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            if joint_name == base_joint_name:
                names.extend(
                    [
                        "root_tx",
                        "root_ty",
                        "root_tz",
                        "root_qw",
                        "root_qx",
                        "root_qy",
                        "root_qz",
                    ]
                )
            else:
                names.extend(
                    [
                        f"{joint_name}_tx",
                        f"{joint_name}_ty",
                        f"{joint_name}_tz",
                        f"{joint_name}_qw",
                        f"{joint_name}_qx",
                        f"{joint_name}_qy",
                        f"{joint_name}_qz",
                    ]
                )
        elif joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            names.append(str(joint_name))
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            names.extend(
                [f"{joint_name}_qw", f"{joint_name}_qx", f"{joint_name}_qy", f"{joint_name}_qz"]
            )
        else:
            raise ValueError(f"Unsupported joint type {joint_type} for {joint_name}")
    if len(names) != model.nq:
        raise ValueError(f"Expected {model.nq} qpos names, got {len(names)}")
    return names


def write_manifest(
    dataset_root: str | Path,
    *,
    dataset_name: str,
    mjcf: Any,
    timestep: float,
    qpos_names: list[str],
    num_motions: int,
    source: dict[str, Any],
) -> Path:
    dataset_root = ensure_dir(dataset_root).resolve()
    manifest_path = dataset_root / MANIFEST_NAME
    if isinstance(mjcf, (str, Path)):
        mjcf_payload: Any = relative_to_root(Path(mjcf).resolve(), dataset_root)
    elif isinstance(mjcf, dict):
        mjcf_payload = dict(mjcf)
    else:
        raise TypeError(f"Unsupported mjcf payload type: {type(mjcf)!r}")
    payload = {
        "format_version": FORMAT_VERSION,
        "dataset_name": dataset_name,
        "mjcf": mjcf_payload,
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
