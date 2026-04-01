from __future__ import annotations

import numpy as np

from any4hdmi.core.format import MOTION_DTYPE


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat[..., [1, 2, 3, 0]], dtype=MOTION_DTYPE)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat[..., [3, 0, 1, 2]], dtype=MOTION_DTYPE)


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
    angles: np.ndarray,
    order: str,
    frame: str = "intrinsic",
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
    return np.asarray(quat / norm, dtype=MOTION_DTYPE)


def maybe_degrees_to_radians(values: np.ndarray, unit: str) -> np.ndarray:
    values = np.asarray(values, dtype=MOTION_DTYPE)
    unit = unit.lower()
    if unit == "deg":
        return np.asarray(np.deg2rad(values), dtype=MOTION_DTYPE)
    if unit == "rad":
        return values
    raise ValueError(f"Unsupported angle unit: {unit}")
