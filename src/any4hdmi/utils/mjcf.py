from __future__ import annotations

from pathlib import Path

import mujoco
from mjhub import resolve_mjcf_reference as _resolve_mjcf_reference


DEFAULT_BASE_JOINT_NAME = "floating_base_joint"
DEFAULT_MJCF_REPO_ID = "elijahgalahad/g1_xmls"
DEFAULT_MJCF_PATH = "g1.xml"
DEFAULT_MJCF_REVISION = "main"
MjcfInput = str | Path


def build_hf_mjcf_reference(
    *,
    repo_id: str = DEFAULT_MJCF_REPO_ID,
    path: str = DEFAULT_MJCF_PATH,
    revision: str = DEFAULT_MJCF_REVISION,
) -> str:
    return f"hf://{repo_id}@{revision}/{path}"


def resolve_mjcf_path(mjcf: MjcfInput, *, dataset_root: str | Path | None = None) -> Path:
    normalized = normalize_mjcf_reference(mjcf, dataset_root=dataset_root)
    if isinstance(normalized, Path):
        return normalized
    return _resolve_mjcf_reference(normalized)


def normalize_mjcf_reference(
    mjcf: MjcfInput,
    *,
    dataset_root: str | Path | None = None,
) -> MjcfInput:
    if isinstance(mjcf, Path):
        return mjcf.expanduser().resolve()
    if isinstance(mjcf, str):
        if mjcf.startswith("hf://"):
            return mjcf
        base_dir = Path(dataset_root).expanduser().resolve() if dataset_root else Path.cwd()
        return (base_dir / mjcf).resolve()
    raise TypeError(f"Unsupported mjcf payload type: {type(mjcf)!r}")


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
