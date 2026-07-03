from __future__ import annotations

import numpy as np
import torch


HME_FEATURE_TYPE = "joint_pos_joint_vel_root_pose6d_window_initial_heading_root_vel_current_root_local"
HME_CACHE_FEATURE_TYPE = f"{HME_FEATURE_TYPE}_final_window_cache"


def compute_win_len(win_sec: float, downsample_rate: int, frequency: float = 50.0) -> int:
    win_len = int(float(win_sec) * float(frequency) / int(downsample_rate)) + 1
    return win_len + (0 if win_len % 2 == 1 else 1)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-8)


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _quat_conj_wxyz(quat: np.ndarray) -> np.ndarray:
    conjugated = np.asarray(quat, dtype=np.float32).copy()
    conjugated[..., 1:] *= -1.0
    return conjugated


def _rotate_by_quat_wxyz(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    vec_in = np.asarray(vec, dtype=np.float32)
    quat_in = _normalize_quat_wxyz(np.asarray(quat, dtype=np.float32))
    zeros = np.zeros((*vec_in.shape[:-1], 1), dtype=np.float32)
    vec_quat = np.concatenate([zeros, vec_in], axis=-1)
    rotated = _quat_mul_wxyz(_quat_mul_wxyz(quat_in, vec_quat), _quat_conj_wxyz(quat_in))
    return rotated[..., 1:].astype(np.float32, copy=False)


def _rotate_by_inverse_quat_wxyz(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return _rotate_by_quat_wxyz(vec, _quat_conj_wxyz(_normalize_quat_wxyz(np.asarray(quat, dtype=np.float32))))


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float | np.ndarray:
    quat = _normalize_quat_wxyz(np.asarray(quat, dtype=np.float64))
    w, x, y, z = np.moveaxis(quat, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(yaw) if np.ndim(yaw) == 0 else yaw.astype(np.float32, copy=False)


def _yaw_quat_wxyz(yaw: float | np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(yaw, dtype=np.float32)
    quat = np.zeros((*half.shape, 4), dtype=np.float32)
    quat[..., 0] = np.cos(half)
    quat[..., 3] = np.sin(half)
    return quat


def _rotate_by_inverse_yaw(vec: np.ndarray, yaw: float | np.ndarray) -> np.ndarray:
    rotated = np.asarray(vec, dtype=np.float32).copy()
    if rotated.shape[-1] < 2:
        return rotated
    yaw_arr = np.asarray(yaw, dtype=np.float32)
    cos_y = np.cos(yaw_arr)
    sin_y = np.sin(yaw_arr)
    while np.ndim(cos_y) < rotated.ndim - 1:
        cos_y = np.expand_dims(cos_y, axis=-1)
        sin_y = np.expand_dims(sin_y, axis=-1)
    x = rotated[..., 0].copy()
    y = rotated[..., 1].copy()
    rotated[..., 0] = cos_y * x + sin_y * y
    rotated[..., 1] = -sin_y * x + cos_y * y
    return rotated


def _quat_to_rot6d_wxyz(quat: np.ndarray) -> np.ndarray:
    quat_in = _normalize_quat_wxyz(np.asarray(quat, dtype=np.float32))
    w, x, y, z = np.moveaxis(quat_in, -1, 0)
    col0 = np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        axis=-1,
    )
    col1 = np.stack(
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        axis=-1,
    )
    return np.concatenate([col0, col1], axis=-1).astype(np.float32, copy=False)


def root_pose_in_initial_heading_frame(qpos: np.ndarray) -> np.ndarray:
    """Express root pose in each HME window's first-frame heading frame.

    `qpos` is expected to be `[num_windows, win_len, nq]`; each window uses its
    own first frame as the translation/yaw reference.
    """
    qpos_in = np.asarray(qpos, dtype=np.float32)
    if qpos_in.ndim != 3 or qpos_in.shape[-1] < 7:
        raise ValueError(f"Expected windowed free-root qpos [num_windows, win_len, nq>=7], got shape {qpos_in.shape}")

    ref_pos = qpos_in[:, :1, :3]
    ref_quat = qpos_in[:, 0, 3:7]
    first_yaw = _yaw_from_quat_wxyz(ref_quat)
    inv_yaw = _yaw_quat_wxyz(-first_yaw)[:, None, :]

    root_pose = np.empty((*qpos_in.shape[:-1], 9), dtype=np.float32)

    root_delta = qpos_in[..., :3] - ref_pos
    root_pose[..., :3] = _rotate_by_inverse_yaw(root_delta, first_yaw)

    inv_yaws = np.broadcast_to(inv_yaw, qpos_in[..., 3:7].shape)
    root_quat_local = _normalize_quat_wxyz(_quat_mul_wxyz(inv_yaws, qpos_in[..., 3:7]))
    root_pose[..., 3:9] = _quat_to_rot6d_wxyz(root_quat_local)
    return root_pose


def root_vel_in_current_root_frame(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    qpos_in = np.asarray(qpos, dtype=np.float32)
    qvel_in = np.asarray(qvel, dtype=np.float32)
    if qpos_in.shape[1] < 7 or qvel_in.shape[1] < 6:
        raise ValueError(f"Expected free-root qpos/qvel with at least 7/6 columns, got {qpos_in.shape}/{qvel_in.shape}")
    root_vel = qvel_in[:, :6].copy()
    root_vel[:, :3] = _rotate_by_inverse_quat_wxyz(root_vel[:, :3], qpos_in[:, 3:7])
    return root_vel


def motion_features(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    qpos_in = np.asarray(qpos, dtype=np.float32)
    qvel_in = np.asarray(qvel, dtype=np.float32)
    if qpos_in.shape[1] < 7 or qvel_in.shape[1] < 6:
        raise ValueError(f"Expected free-root qpos/qvel with at least 7/6 columns, got {qpos_in.shape}/{qvel_in.shape}")
    joint_pos = qpos_in[:, 7:]
    joint_vel = qvel_in[:, 6:]
    root_pose = root_pose_in_initial_heading_frame(qpos_in[None, ...])[0]
    root_vel = root_vel_in_current_root_frame(qpos_in, qvel_in)
    return np.concatenate([joint_pos, joint_vel, root_pose, root_vel], axis=-1)


def motion_frame_features(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    qpos_in = np.asarray(qpos, dtype=np.float32)
    qvel_in = np.asarray(qvel, dtype=np.float32)
    if qpos_in.shape[1] < 7 or qvel_in.shape[1] < 6:
        raise ValueError(f"Expected free-root qpos/qvel with at least 7/6 columns, got {qpos_in.shape}/{qvel_in.shape}")
    joint_pos = qpos_in[:, 7:]
    joint_vel = qvel_in[:, 6:]
    root_pos_quat = qpos_in[:, :7]
    root_vel = root_vel_in_current_root_frame(qpos_in, qvel_in)
    return np.concatenate([joint_pos, joint_vel, root_pos_quat, root_vel], axis=-1)


def motion_frame_features_from_fk(
    qpos: np.ndarray | torch.Tensor,
    qvel: np.ndarray | torch.Tensor,
    fk_results: dict[str, np.ndarray | torch.Tensor],
) -> np.ndarray:
    qpos_in = qpos.detach().to(device="cpu", dtype=torch.float32).numpy() if isinstance(qpos, torch.Tensor) else np.asarray(qpos, dtype=np.float32)
    qvel_in = qvel.detach().to(device="cpu", dtype=torch.float32).numpy() if isinstance(qvel, torch.Tensor) else np.asarray(qvel, dtype=np.float32)
    joint_pos_raw = fk_results["joint_pos"]
    joint_vel_raw = fk_results["joint_vel"]
    joint_pos = (
        joint_pos_raw.detach().to(device="cpu", dtype=torch.float32).numpy()
        if isinstance(joint_pos_raw, torch.Tensor)
        else np.asarray(joint_pos_raw, dtype=np.float32)
    )
    joint_vel = (
        joint_vel_raw.detach().to(device="cpu", dtype=torch.float32).numpy()
        if isinstance(joint_vel_raw, torch.Tensor)
        else np.asarray(joint_vel_raw, dtype=np.float32)
    )
    if qpos_in.shape[1] < 7 or qvel_in.shape[1] < 6:
        raise ValueError(f"Expected free-root qpos/qvel with at least 7/6 columns, got {qpos_in.shape}/{qvel_in.shape}")
    if joint_pos.shape[0] != qpos_in.shape[0] or joint_vel.shape[0] != qpos_in.shape[0]:
        raise ValueError(
            "Expected FK joint outputs to match qpos frame count, got "
            f"qpos={qpos_in.shape[0]} joint_pos={joint_pos.shape[0]} joint_vel={joint_vel.shape[0]}"
        )
    root_pos_quat = qpos_in[:, :7]
    root_vel = root_vel_in_current_root_frame(qpos_in, qvel_in)
    return np.concatenate([joint_pos, joint_vel, root_pos_quat, root_vel], axis=-1)


def hme_features_from_raw_windows(raw_windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(raw_windows, dtype=np.float32)
    if windows.ndim not in (2, 3):
        raise ValueError(f"Expected raw windows with shape [win_len, dim] or [batch, win_len, dim], got {windows.shape}")
    squeeze_batch = windows.ndim == 2
    if squeeze_batch:
        windows = windows[None, ...]
    raw_dim = int(windows.shape[-1])
    if raw_dim < 15 or (raw_dim - 13) % 2 != 0:
        raise ValueError(f"Expected raw feature dim 2 * joints + 13, got {raw_dim}")
    joint_dim = (raw_dim - 13) // 2
    joint_pos = windows[..., :joint_dim]
    joint_vel = windows[..., joint_dim : 2 * joint_dim]
    root_pos_quat = windows[..., 2 * joint_dim : 2 * joint_dim + 7]
    root_vel = windows[..., 2 * joint_dim + 7 :]
    root_pose = root_pose_in_initial_heading_frame(root_pos_quat)
    hme_windows = np.concatenate([joint_pos, joint_vel, root_pose, root_vel], axis=-1)
    return hme_windows[0] if squeeze_batch else hme_windows


def hme_features_from_raw(features: np.ndarray, window_ids: np.ndarray) -> np.ndarray:
    return hme_features_from_raw_windows(np.asarray(features, dtype=np.float32)[window_ids])


def window_indices(length: int, *, win_len: int, downsample_rate: int, stride: int = 1) -> np.ndarray:
    win_half = (win_len - 1) // 2
    padding = win_half * int(downsample_rate)
    seq_len = int(length) - 2 * padding
    if seq_len <= 0:
        return np.empty((0, win_len), dtype=np.int64)
    centers = padding + np.arange(0, seq_len, max(1, int(stride)), dtype=np.int64)
    offsets = (np.arange(-win_half, win_half + 1, dtype=np.int64) * int(downsample_rate))[None, :]
    return centers[:, None] + offsets


def normalize_motion_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
