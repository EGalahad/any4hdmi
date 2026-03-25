from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from any4hdmi.format import MOTION_DTYPE


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
    mjcf_path = Path(mjcf_path).expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
    return mujoco.MjModel.from_xml_path(str(mjcf_path))


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


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return quat[..., [1, 2, 3, 0]]


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.asarray(
        np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
        ),
        dtype=MOTION_DTYPE,
    )


def axis_angle_quat(axis: str, angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=MOTION_DTYPE)
    half = angle / MOTION_DTYPE(2.0)
    quat = np.zeros(angle.shape + (4,), dtype=MOTION_DTYPE)
    quat[..., 0] = np.cos(half, dtype=MOTION_DTYPE)
    sin_half = np.sin(half, dtype=MOTION_DTYPE)
    if axis == "x":
        quat[..., 1] = sin_half
    elif axis == "y":
        quat[..., 2] = sin_half
    elif axis == "z":
        quat[..., 3] = sin_half
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return quat


def euler_to_quat_wxyz(
    angles: np.ndarray, order: str, frame: str = "intrinsic"
) -> np.ndarray:
    angles = np.asarray(angles, dtype=MOTION_DTYPE)
    order = order.lower()
    if len(order) != 3 or sorted(order) != ["x", "y", "z"]:
        raise ValueError(f"Euler order must be a permutation of xyz, got {order!r}")
    frame = frame.lower()
    if frame not in {"intrinsic", "extrinsic"}:
        raise ValueError(f"Euler frame must be intrinsic or extrinsic, got {frame!r}")
    quat = np.zeros(angles.shape[:-1] + (4,), dtype=MOTION_DTYPE)
    quat[..., 0] = MOTION_DTYPE(1.0)
    for axis_index, axis in enumerate(order):
        axis_quat = axis_angle_quat(axis, angles[..., axis_index])
        if frame == "intrinsic":
            quat = quat_mul(quat, axis_quat)
        else:
            quat = quat_mul(axis_quat, quat)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = np.asarray(quat / norm, dtype=MOTION_DTYPE)
    return quat


def maybe_degrees_to_radians(values: np.ndarray, unit: str) -> np.ndarray:
    values = np.asarray(values, dtype=MOTION_DTYPE)
    unit = unit.lower()
    if unit == "deg":
        return np.asarray(np.deg2rad(values), dtype=MOTION_DTYPE)
    if unit == "rad":
        return values
    raise ValueError(f"Unsupported angle unit: {unit}")
