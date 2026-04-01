from __future__ import annotations

from typing import Any

import numpy as np
import torch


def lerp(ts_target, ts_source, x):
    return np.stack(
        [np.interp(ts_target, ts_source, x[:, i]) for i in range(x.shape[1])],
        axis=-1,
    )


def _lerp_torch(ts_target: torch.Tensor, ts_source: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    right_idx = torch.searchsorted(ts_source, ts_target, right=False)
    right_idx = right_idx.clamp(1, ts_source.numel() - 1)
    left_idx = right_idx - 1

    t_left = ts_source[left_idx]
    t_right = ts_source[right_idx]
    denom = torch.where(t_right > t_left, t_right - t_left, torch.ones_like(t_right))
    alpha = ((ts_target - t_left) / denom).unsqueeze(1)

    x0 = x[left_idx]
    x1 = x[right_idx]
    return (1.0 - alpha) * x0 + alpha * x1


def slerp(ts_target, ts_source, quat):
    batch_shape = quat.shape[1:-1]
    quat_dim = quat.shape[-1]
    if quat_dim != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {quat.shape}")

    steps_target = ts_target.shape[0]
    steps_source = ts_source.shape[0]

    quat = np.asarray(quat, dtype=np.float64).reshape(steps_source, -1, quat_dim)
    ts_source = np.asarray(ts_source)
    ts_target = np.asarray(ts_target)

    if steps_source == 0:
        raise ValueError("Cannot interpolate empty quaternion sequence")
    if steps_source == 1:
        out = np.broadcast_to(quat[:1], (steps_target, *quat[:1].shape[1:])).copy()
        return out.reshape(steps_target, *batch_shape, quat_dim)

    right_idx = np.searchsorted(ts_source, ts_target, side="left")
    right_idx = np.clip(right_idx, 1, steps_source - 1)
    left_idx = right_idx - 1

    t_left = ts_source[left_idx]
    t_right = ts_source[right_idx]
    denom = np.where(t_right > t_left, t_right - t_left, 1.0)
    alpha = ((ts_target - t_left) / denom).astype(np.float64)[:, None, None]

    q0 = quat[left_idx]
    q1 = quat[right_idx]
    q0 /= np.linalg.norm(q0, axis=-1, keepdims=True).clip(min=1e-12)
    q1 /= np.linalg.norm(q1, axis=-1, keepdims=True).clip(min=1e-12)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    flip_mask = dot < 0.0
    q1 = np.where(flip_mask, -q1, q1)
    dot = np.where(flip_mask, -dot, dot)
    dot = np.clip(dot, -1.0, 1.0)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha

    safe_denom = np.where(sin_theta_0 > 1e-8, sin_theta_0, 1.0)
    s0 = np.sin(theta_0 - theta) / safe_denom
    s1 = np.sin(theta) / safe_denom
    slerp_out = s0 * q0 + s1 * q1

    nlerp_out = (1.0 - alpha) * q0 + alpha * q1
    out = np.where(dot > 0.9995, nlerp_out, slerp_out)
    out /= np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-12)
    return out.reshape(steps_target, *batch_shape, quat_dim)


def _slerp_torch(ts_target: torch.Tensor, ts_source: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    batch_shape = quat.shape[1:-1]
    quat_dim = quat.shape[-1]
    if quat_dim != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {tuple(quat.shape)}")

    steps_target = ts_target.shape[0]
    steps_source = ts_source.shape[0]
    quat = quat.reshape(steps_source, -1, quat_dim)

    if steps_source == 0:
        raise ValueError("Cannot interpolate empty quaternion sequence")
    if steps_source == 1:
        out = quat[:1].expand(steps_target, -1, -1).clone()
        return out.reshape(steps_target, *batch_shape, quat_dim)

    right_idx = torch.searchsorted(ts_source, ts_target, right=False)
    right_idx = right_idx.clamp(1, steps_source - 1)
    left_idx = right_idx - 1

    t_left = ts_source[left_idx]
    t_right = ts_source[right_idx]
    denom = torch.where(t_right > t_left, t_right - t_left, torch.ones_like(t_right))
    alpha = ((ts_target - t_left) / denom).view(-1, 1, 1)

    q0 = quat[left_idx]
    q1 = quat[right_idx]
    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    flip_mask = dot < 0.0
    q1 = torch.where(flip_mask, -q1, q1)
    dot = torch.where(flip_mask, -dot, dot).clamp(-1.0, 1.0)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * alpha

    safe_denom = torch.where(sin_theta_0 > 1e-8, sin_theta_0, torch.ones_like(sin_theta_0))
    s0 = torch.sin(theta_0 - theta) / safe_denom
    s1 = torch.sin(theta) / safe_denom
    slerp_out = s0 * q0 + s1 * q1

    nlerp_out = (1.0 - alpha) * q0 + alpha * q1
    out = torch.where(dot > 0.9995, nlerp_out, slerp_out)
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out.reshape(steps_target, *batch_shape, quat_dim)


def interpolate(motion: dict[str, Any], source_fps: int, target_fps: int):
    if source_fps == target_fps:
        return motion

    in_keys = [
        "body_pos_w",
        "body_lin_vel_w",
        "body_quat_w",
        "body_ang_vel_w",
        "joint_pos",
        "joint_vel",
    ]
    extra_keys = set(motion.keys()) - set(in_keys)
    if extra_keys:
        raise NotImplementedError(f"interpolation is not fully implemented for keys: {extra_keys}")

    length = motion["joint_pos"].shape[0]
    if isinstance(motion["joint_pos"], torch.Tensor):
        device = motion["joint_pos"].device
        dtype = motion["joint_pos"].dtype
        ts_source = torch.arange(0, (length - 1) * target_fps + 1, target_fps, device=device, dtype=dtype)
        ts_target = torch.arange(0, (length - 1) * target_fps + 1, source_fps, device=device, dtype=dtype)
        motion["body_pos_w"] = _lerp_torch(
            ts_target,
            ts_source,
            motion["body_pos_w"].reshape(length, -1),
        ).reshape(len(ts_target), -1, 3)
        motion["body_lin_vel_w"] = _lerp_torch(
            ts_target,
            ts_source,
            motion["body_lin_vel_w"].reshape(length, -1),
        ).reshape(len(ts_target), -1, 3)
        motion["body_quat_w"] = _slerp_torch(ts_target, ts_source, motion["body_quat_w"])
        motion["body_ang_vel_w"] = _lerp_torch(
            ts_target,
            ts_source,
            motion["body_ang_vel_w"].reshape(length, -1),
        ).reshape(len(ts_target), -1, 3)
        motion["joint_pos"] = _lerp_torch(ts_target, ts_source, motion["joint_pos"])
        motion["joint_vel"] = _lerp_torch(ts_target, ts_source, motion["joint_vel"])
        return motion

    ts_source = np.arange(0, (length - 1) * target_fps + 1, target_fps)
    ts_target = np.arange(0, (length - 1) * target_fps + 1, source_fps)
    motion["body_pos_w"] = lerp(ts_target, ts_source, motion["body_pos_w"].reshape(length, -1)).reshape(
        len(ts_target), -1, 3
    )
    motion["body_lin_vel_w"] = lerp(
        ts_target,
        ts_source,
        motion["body_lin_vel_w"].reshape(length, -1),
    ).reshape(len(ts_target), -1, 3)
    motion["body_quat_w"] = slerp(ts_target, ts_source, motion["body_quat_w"])
    motion["body_ang_vel_w"] = lerp(
        ts_target,
        ts_source,
        motion["body_ang_vel_w"].reshape(length, -1),
    ).reshape(len(ts_target), -1, 3)
    motion["joint_pos"] = lerp(ts_target, ts_source, motion["joint_pos"])
    motion["joint_vel"] = lerp(ts_target, ts_source, motion["joint_vel"])
    return motion
