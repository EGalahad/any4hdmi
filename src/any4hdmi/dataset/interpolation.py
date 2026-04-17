from __future__ import annotations

import math
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


def _resample_times_torch(
    length: int,
    *,
    source_fps: float,
    target_fps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_length = resampled_length(
        length,
        source_fps=source_fps,
        target_fps=target_fps,
    )
    if length <= 0:
        raise ValueError(f"Expected positive sequence length, got {length}")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"fps must be positive, got source_fps={source_fps}, target_fps={target_fps}")

    source_times = torch.arange(length, device=device, dtype=dtype) / float(source_fps)
    if length == 1:
        return source_times, source_times.clone()

    target_times = torch.arange(target_length, device=device, dtype=dtype) / float(target_fps)
    return source_times, target_times


def resampled_length(
    length: int,
    *,
    source_fps: float,
    target_fps: float,
) -> int:
    if length <= 0:
        raise ValueError(f"Expected positive sequence length, got {length}")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"fps must be positive, got source_fps={source_fps}, target_fps={target_fps}")
    if length == 1 or source_fps == target_fps:
        return int(length)
    duration = (length - 1) / float(source_fps)
    return max(1, int(math.floor(duration * float(target_fps) + 1e-9)) + 1)


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


def _packed_interp_plan_torch(
    clip_lengths: list[int],
    *,
    source_fps: float,
    target_fps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    clip_lengths_t = torch.as_tensor(clip_lengths, device=device, dtype=torch.long)
    output_lengths = [
        resampled_length(
            int(length),
            source_fps=source_fps,
            target_fps=target_fps,
        )
        for length in clip_lengths
    ]
    output_lengths_t = torch.as_tensor(output_lengths, device=device, dtype=torch.long)
    total_output = int(output_lengths_t.sum().item())

    input_starts = torch.cumsum(
        torch.cat([clip_lengths_t.new_zeros(1), clip_lengths_t[:-1]]),
        dim=0,
    )
    output_starts = torch.cumsum(
        torch.cat([output_lengths_t.new_zeros(1), output_lengths_t[:-1]]),
        dim=0,
    )

    clip_ids = torch.repeat_interleave(
        torch.arange(len(clip_lengths), device=device, dtype=torch.long),
        output_lengths_t,
    )
    output_indices = torch.arange(total_output, device=device, dtype=torch.long)
    local_output_idx = output_indices - output_starts.index_select(0, clip_ids)

    source_pos = local_output_idx.to(dtype=dtype) * (float(source_fps) / float(target_fps))
    single_frame_mask = clip_lengths_t.index_select(0, clip_ids) <= 1
    max_right = (clip_lengths_t.index_select(0, clip_ids) - 1).clamp_min(0)

    right_local = torch.ceil(source_pos).to(dtype=torch.long)
    right_local = torch.where(single_frame_mask, torch.zeros_like(right_local), right_local.clamp_min(1))
    right_local = torch.minimum(right_local, max_right)
    left_local = torch.where(single_frame_mask, torch.zeros_like(right_local), right_local - 1)

    alpha = torch.where(
        single_frame_mask,
        torch.zeros_like(source_pos),
        source_pos - left_local.to(dtype=dtype),
    )

    global_left = input_starts.index_select(0, clip_ids) + left_local
    global_right = input_starts.index_select(0, clip_ids) + right_local
    return output_lengths, global_left, global_right, alpha


def _packed_slerp_by_indices_torch(
    quat: torch.Tensor,
    *,
    global_left: torch.Tensor,
    global_right: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    batch_shape = quat.shape[1:-1]
    quat_dim = quat.shape[-1]
    flat = quat.reshape(quat.shape[0], -1, quat_dim)
    q0 = flat[global_left]
    q1 = flat[global_right]

    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    flip_mask = dot < 0.0
    q1 = torch.where(flip_mask, -q1, q1)
    dot = torch.where(flip_mask, -dot, dot).clamp(-1.0, 1.0)

    alpha_quat = alpha.view(-1, 1, 1)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * alpha_quat

    safe_denom = torch.where(sin_theta_0 > 1e-8, sin_theta_0, torch.ones_like(sin_theta_0))
    s0 = torch.sin(theta_0 - theta) / safe_denom
    s1 = torch.sin(theta) / safe_denom
    slerp_out = s0 * q0 + s1 * q1

    nlerp_out = (1.0 - alpha_quat) * q0 + alpha_quat * q1
    out = torch.where(dot > 0.9995, nlerp_out, slerp_out)
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out.reshape(alpha.shape[0], *batch_shape, quat_dim)


def interpolate_qpos_torch(
    qpos: torch.Tensor,
    *,
    source_fps: float,
    target_fps: float,
) -> torch.Tensor:
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos to be rank 2, got shape {tuple(qpos.shape)}")
    if qpos.shape[0] <= 1 or source_fps == target_fps:
        return qpos.to(dtype=torch.float32).contiguous()

    qpos = qpos.to(dtype=torch.float32).contiguous()
    ts_source, ts_target = _resample_times_torch(
        int(qpos.shape[0]),
        source_fps=source_fps,
        target_fps=target_fps,
        device=qpos.device,
        dtype=qpos.dtype,
    )

    if qpos.shape[1] >= 7:
        pieces = [
            _lerp_torch(ts_target, ts_source, qpos[:, 0:3]),
            _slerp_torch(ts_target, ts_source, qpos[:, 3:7]),
        ]
        if qpos.shape[1] > 7:
            pieces.append(_lerp_torch(ts_target, ts_source, qpos[:, 7:]))
        return torch.cat(pieces, dim=1).contiguous()

    return _lerp_torch(ts_target, ts_source, qpos).contiguous()


def interpolate_qvel_torch(
    qvel: torch.Tensor,
    *,
    source_fps: float,
    target_fps: float,
) -> torch.Tensor:
    if qvel.ndim != 2:
        raise ValueError(f"Expected qvel to be rank 2, got shape {tuple(qvel.shape)}")
    if qvel.shape[0] <= 1 or source_fps == target_fps:
        return qvel.to(dtype=torch.float32).contiguous()

    qvel = qvel.to(dtype=torch.float32).contiguous()
    ts_source, ts_target = _resample_times_torch(
        int(qvel.shape[0]),
        source_fps=source_fps,
        target_fps=target_fps,
        device=qvel.device,
        dtype=qvel.dtype,
    )
    return _lerp_torch(ts_target, ts_source, qvel).contiguous()


def interpolate_qpos_qvel_batch_torch(
    qpos_list: list[torch.Tensor],
    qvel_list: list[torch.Tensor],
    *,
    source_fps: float,
    target_fps: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if len(qpos_list) != len(qvel_list):
        raise ValueError(f"Expected qpos/qvel list lengths to match, got {len(qpos_list)} and {len(qvel_list)}")
    if not qpos_list:
        return [], []

    prepared_qpos: list[torch.Tensor] = []
    prepared_qvel: list[torch.Tensor] = []
    clip_lengths: list[int] = []
    device = qpos_list[0].device

    for qpos, qvel in zip(qpos_list, qvel_list, strict=True):
        if qpos.ndim != 2:
            raise ValueError(f"Expected qpos to be rank 2, got shape {tuple(qpos.shape)}")
        if qvel.ndim != 2:
            raise ValueError(f"Expected qvel to be rank 2, got shape {tuple(qvel.shape)}")
        if qpos.device != device or qvel.device != device:
            raise ValueError("Expected all qpos/qvel tensors in a batch to share the same device")
        if qpos.shape[0] != qvel.shape[0]:
            raise ValueError(
                f"Expected qpos/qvel frame counts to match, got {tuple(qpos.shape)} and {tuple(qvel.shape)}"
            )
        prepared_qpos.append(qpos.to(dtype=torch.float32).contiguous())
        prepared_qvel.append(qvel.to(dtype=torch.float32).contiguous())
        clip_lengths.append(int(qpos.shape[0]))

    if source_fps == target_fps or all(length <= 1 for length in clip_lengths):
        return prepared_qpos, prepared_qvel

    packed_qpos = torch.cat(prepared_qpos, dim=0)
    packed_qvel = torch.cat(prepared_qvel, dim=0)
    output_lengths, global_left, global_right, alpha = _packed_interp_plan_torch(
        clip_lengths,
        source_fps=source_fps,
        target_fps=target_fps,
        device=device,
        dtype=packed_qpos.dtype,
    )

    def _packed_lerp(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        x0 = flat[global_left]
        x1 = flat[global_right]
        out = (1.0 - alpha.unsqueeze(1)) * x0 + alpha.unsqueeze(1) * x1
        return out.reshape(alpha.shape[0], *x.shape[1:]).contiguous()

    if packed_qpos.shape[1] >= 7:
        qpos_pieces = [
            _packed_lerp(packed_qpos[:, 0:3]),
            _packed_slerp_by_indices_torch(
                packed_qpos[:, 3:7],
                global_left=global_left,
                global_right=global_right,
                alpha=alpha,
            ),
        ]
        if packed_qpos.shape[1] > 7:
            qpos_pieces.append(_packed_lerp(packed_qpos[:, 7:]))
        packed_qpos_interp = torch.cat(qpos_pieces, dim=1).contiguous()
    else:
        packed_qpos_interp = _packed_lerp(packed_qpos)

    packed_qvel_interp = _packed_lerp(packed_qvel)
    return (
        list(packed_qpos_interp.split(output_lengths, dim=0)),
        list(packed_qvel_interp.split(output_lengths, dim=0)),
    )


def interpolate_packed_torch(
    motion: dict[str, torch.Tensor],
    clip_lengths: list[int],
    *,
    source_fps: int,
    target_fps: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    if not clip_lengths:
        return (
            {
                key: value[:0]
                for key, value in motion.items()
            },
            [],
        )

    if source_fps == target_fps:
        return motion, [int(length) for length in clip_lengths]

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

    clip_lengths_t = torch.as_tensor(clip_lengths, device=motion["joint_pos"].device, dtype=torch.long)
    output_lengths_t = ((clip_lengths_t - 1).clamp_min(0) * int(target_fps)) // int(source_fps) + 1
    total_output = int(output_lengths_t.sum().item())

    input_starts = torch.cumsum(
        torch.cat([clip_lengths_t.new_zeros(1), clip_lengths_t[:-1]]),
        dim=0,
    )
    output_starts = torch.cumsum(
        torch.cat([output_lengths_t.new_zeros(1), output_lengths_t[:-1]]),
        dim=0,
    )

    clip_ids = torch.repeat_interleave(
        torch.arange(len(clip_lengths), device=clip_lengths_t.device, dtype=torch.long),
        output_lengths_t,
    )
    output_indices = torch.arange(total_output, device=clip_lengths_t.device, dtype=torch.long)
    local_output_idx = output_indices - output_starts.index_select(0, clip_ids)

    single_frame_mask = clip_lengths_t.index_select(0, clip_ids) <= 1
    numer = local_output_idx * int(source_fps)

    right_local = torch.div(
        numer + int(target_fps) - 1,
        int(target_fps),
        rounding_mode="floor",
    )
    max_right = (clip_lengths_t.index_select(0, clip_ids) - 1).clamp_min(0)
    right_local = right_local.clamp_min(1)
    right_local = torch.minimum(right_local, max_right)
    right_local = torch.where(single_frame_mask, torch.zeros_like(right_local), right_local)
    left_local = torch.where(single_frame_mask, torch.zeros_like(right_local), right_local - 1)

    t_left = left_local.to(dtype=motion["joint_pos"].dtype) * float(target_fps)
    t_right = right_local.to(dtype=motion["joint_pos"].dtype) * float(target_fps)
    denom = torch.where(t_right > t_left, t_right - t_left, torch.ones_like(t_right))
    alpha = torch.where(
        single_frame_mask,
        torch.zeros_like(t_left),
        (numer.to(dtype=motion["joint_pos"].dtype) - t_left) / denom,
    )

    global_left = input_starts.index_select(0, clip_ids) + left_local
    global_right = input_starts.index_select(0, clip_ids) + right_local

    def _packed_lerp(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        x0 = flat[global_left]
        x1 = flat[global_right]
        out = (1.0 - alpha.unsqueeze(1)) * x0 + alpha.unsqueeze(1) * x1
        return out.reshape(total_output, *x.shape[1:])

    def _packed_slerp(quat: torch.Tensor) -> torch.Tensor:
        batch_shape = quat.shape[1:-1]
        quat_dim = quat.shape[-1]
        flat = quat.reshape(quat.shape[0], -1, quat_dim)
        q0 = flat[global_left]
        q1 = flat[global_right]

        q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        flip_mask = dot < 0.0
        q1 = torch.where(flip_mask, -q1, q1)
        dot = torch.where(flip_mask, -dot, dot).clamp(-1.0, 1.0)

        alpha_quat = alpha.view(-1, 1, 1)
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * alpha_quat

        safe_denom = torch.where(sin_theta_0 > 1e-8, sin_theta_0, torch.ones_like(sin_theta_0))
        s0 = torch.sin(theta_0 - theta) / safe_denom
        s1 = torch.sin(theta) / safe_denom
        slerp_out = s0 * q0 + s1 * q1

        nlerp_out = (1.0 - alpha_quat) * q0 + alpha_quat * q1
        out = torch.where(dot > 0.9995, nlerp_out, slerp_out)
        out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return out.reshape(total_output, *batch_shape, quat_dim)

    interpolated = {
        "body_pos_w": _packed_lerp(motion["body_pos_w"]),
        "body_lin_vel_w": _packed_lerp(motion["body_lin_vel_w"]),
        "body_quat_w": _packed_slerp(motion["body_quat_w"]),
        "body_ang_vel_w": _packed_lerp(motion["body_ang_vel_w"]),
        "joint_pos": _packed_lerp(motion["joint_pos"]),
        "joint_vel": _packed_lerp(motion["joint_vel"]),
    }
    return interpolated, output_lengths_t.tolist()


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
