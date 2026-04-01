from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


G1_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def load_model(mjcf_path: str | Path) -> mujoco.MjModel:
    # Keep Hugging Face snapshot symlinks intact so MuJoCo resolves relative
    # mesh/asset paths against the snapshot directory instead of the blob store.
    mjcf_path = Path(mjcf_path).expanduser().absolute()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
    return mujoco.MjModel.from_xml_path(str(mjcf_path))


def body_names_from_model(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(model.nbody)
    ]


def base_qpos_adr(model: mujoco.MjModel, joint_name: str = "floating_base_joint") -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Base joint not found: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def joint_qpos_adrs(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    addrs = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in model: {joint_name}")
        addrs.append(model.jnt_qposadr[joint_id])
    return np.asarray(addrs, dtype=np.int32)


def hinge_joint_info(model: mujoco.MjModel) -> tuple[list[str], np.ndarray, np.ndarray]:
    joint_names: list[str] = []
    joint_qpos_addrs: list[int] = []
    joint_dof_addrs: list[int] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        joint_qpos_addrs.append(int(model.jnt_qposadr[joint_id]))
        joint_dof_addrs.append(int(model.jnt_dofadr[joint_id]))
    return (
        joint_names,
        np.asarray(joint_qpos_addrs, dtype=np.int32),
        np.asarray(joint_dof_addrs, dtype=np.int32),
    )
