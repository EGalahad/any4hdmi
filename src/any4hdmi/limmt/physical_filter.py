from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import tyro
from mjhub import temp_mjcf_with_floor
from tqdm import tqdm

from any4hdmi.core.format import load_manifest
from any4hdmi.fk.runner import FKRunner
from any4hdmi.limmt.common import (
    default_project_root_for_input,
    resolve_dataset_root,
    resolve_project_root,
    write_json,
)
from any4hdmi.utils.dataset import build_motion_loader


@dataclass(frozen=True)
class PhysicalScoreWeights:
    foot_sliding: float = 1.70
    velocity_violation: float = 44.22
    self_collision: float = 0.17
    jerk: float = 0.28
    penetration: float = 216.62
    floating_frames_ratio: float = 24.19


@dataclass(frozen=True)
class PhysicalFilterConfig:
    pass_threshold: float = 90.0
    batch_frames: int = 131072
    fk_batch_size: int = 8192
    max_joint_vel: float = 30.0
    foot_height: float = 0.05
    foot_slide_speed: float = 0.10
    foot_slide_multiplier: float = 5.0
    floating_distance: float = 0.05
    floating_window_sec: float = 1.0
    penetration_margin: float = 0.01
    self_collision_count_clip: float = 10.0
    contact_nconmax: int = 128


@dataclass(frozen=True)
class PhysicalFilterArgs:
    """Run LIMMT physical quality filtering on an any4hdmi dataset."""

    input_path: str
    project_path: str | None = None
    scoring_mjcf_path: str | None = None
    pass_threshold: float = 90.0
    batch_frames: int = 131072
    fk_batch_size: int = 8192
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = max(1, min(16, (os.cpu_count() or 2) - 1))
    prefetch_factor: int = 4
    contact_nconmax: int = 128
    start: int = 0
    limit: int | None = None


G1_JOINT_VELOCITY_LIMITS = {
    "left_hip_pitch_joint": 32.0,
    "right_hip_pitch_joint": 32.0,
    "waist_yaw_joint": 32.0,
    "left_hip_roll_joint": 32.0,
    "right_hip_roll_joint": 32.0,
    "left_hip_yaw_joint": 32.0,
    "right_hip_yaw_joint": 32.0,
    "left_knee_joint": 20.0,
    "right_knee_joint": 20.0,
    "left_shoulder_pitch_joint": 37.0,
    "right_shoulder_pitch_joint": 37.0,
    "left_ankle_pitch_joint": 37.0,
    "right_ankle_pitch_joint": 37.0,
    "left_shoulder_roll_joint": 37.0,
    "right_shoulder_roll_joint": 37.0,
    "left_ankle_roll_joint": 37.0,
    "right_ankle_roll_joint": 37.0,
    "left_shoulder_yaw_joint": 37.0,
    "right_shoulder_yaw_joint": 37.0,
    "left_elbow_joint": 37.0,
    "right_elbow_joint": 37.0,
    "left_wrist_roll_joint": 37.0,
    "right_wrist_roll_joint": 37.0,
    "left_wrist_pitch_joint": 22.0,
    "right_wrist_pitch_joint": 22.0,
    "left_wrist_yaw_joint": 22.0,
    "right_wrist_yaw_joint": 22.0,
}


def _resolve_floor_geom_id(model: mujoco.MjModel) -> tuple[int, str]:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        return int(floor_id), "floor"
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            return int(geom_id), name
    raise ValueError("No floor geom named 'floor' or plane geom found in scoring MJCF")


def _body_id_by_name(model: mujoco.MjModel, name: str) -> int | None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return int(body_id) if body_id >= 0 else None


def _joint_velocity_limits(joint_names: list[str], *, default: float, device: torch.device) -> torch.Tensor:
    limits = [float(G1_JOINT_VELOCITY_LIMITS.get(joint_name, default)) for joint_name in joint_names]
    return torch.as_tensor(limits, dtype=torch.float32, device=device)


def _official_foot_sliding(
    *,
    body_pos: torch.Tensor,
    body_lin: torch.Tensor,
    foot_body_ids: tuple[int, int],
    config: PhysicalFilterConfig,
) -> float:
    left_id, right_id = foot_body_ids
    left_speed = torch.linalg.vector_norm(body_lin[:, left_id, :2], dim=-1)
    right_speed = torch.linalg.vector_norm(body_lin[:, right_id, :2], dim=-1)
    left_contact = body_pos[:, left_id, 2] < config.foot_height
    right_contact = body_pos[:, right_id, 2] < config.foot_height
    p_slide = torch.where(left_contact, torch.relu(left_speed - config.foot_slide_speed), left_speed.new_zeros(()))
    p_slide = p_slide + torch.where(
        right_contact,
        torch.relu(right_speed - config.foot_slide_speed),
        right_speed.new_zeros(()),
    )
    return float((p_slide * float(config.foot_slide_multiplier)).mean().item())


def _sustained_air_ratio(floor_min_dist: torch.Tensor, *, fps: float, config: PhysicalFilterConfig) -> float:
    valid_air = (floor_min_dist.detach().to(device="cpu", dtype=torch.float32).numpy() > float(config.floating_distance)).astype(np.float32)
    if valid_air.size == 0:
        return 0.0
    window_size = max(1, int(float(config.floating_window_sec) * float(fps)))
    if window_size <= 1:
        return float(valid_air.mean())
    conv = np.convolve(valid_air, np.ones(window_size, dtype=np.float32), mode="same")
    return float(np.mean(conv >= (float(window_size) - 0.1)))


def _score_motion(
    *,
    rel_motion: str,
    fk_results: dict[str, torch.Tensor],
    qvel: torch.Tensor,
    fps: float,
    config: PhysicalFilterConfig,
    weights: PhysicalScoreWeights,
    contact_summary: dict[str, torch.Tensor],
    foot_body_ids: tuple[int, int],
    joint_vel_limits: torch.Tensor | None = None,
) -> dict[str, Any]:
    frames = int(qvel.shape[0])
    body_pos = fk_results["body_pos_w"]
    body_lin = fk_results["body_lin_vel_w"]
    joint_vel = fk_results["joint_vel"]

    foot_sliding = _official_foot_sliding(body_pos=body_pos, body_lin=body_lin, foot_body_ids=foot_body_ids, config=config)

    if joint_vel.numel():
        if joint_vel_limits is None:
            joint_vel_limits = torch.full((joint_vel.shape[1],), float(config.max_joint_vel), dtype=torch.float32, device=joint_vel.device)
        else:
            joint_vel_limits = joint_vel_limits.to(device=joint_vel.device, dtype=torch.float32)
        velocity_violation = float(torch.relu(torch.abs(joint_vel) - joint_vel_limits).mean().item())
    else:
        velocity_violation = 0.0

    if qvel.shape[0] >= 2:
        prev_qvel = torch.cat([qvel[:1], qvel[:-1]], dim=0)
        accel = (qvel - prev_qvel) * float(fps)
        jerk = float((torch.linalg.vector_norm(accel, dim=1) * 0.01).mean().item())
    else:
        jerk = 0.0

    floor_min_dist = contact_summary["floor_min_dist"].to(device=body_pos.device, dtype=torch.float32)
    non_floor_contact_count = contact_summary["non_floor_contact_count"].to(device=body_pos.device, dtype=torch.float32)
    penetration = float(torch.relu(-floor_min_dist - float(config.penetration_margin)).mean().item())
    floating_frames_ratio = _sustained_air_ratio(floor_min_dist, fps=fps, config=config)
    self_collision = float(torch.clamp(non_floor_contact_count, max=float(config.self_collision_count_clip)).mean().item())

    penalties = {
        "foot_sliding": foot_sliding,
        "velocity_violation": velocity_violation,
        "self_collision": self_collision,
        "jerk": jerk,
        "penetration": penetration,
        "floating_frames_ratio": floating_frames_ratio,
    }
    deduction = sum(float(getattr(weights, name)) * value for name, value in penalties.items())
    physical_score = max(0.0, 100.0 - deduction)
    return {
        "motion": rel_motion,
        "num_frames": frames,
        "fps": float(fps),
        **penalties,
        "physical_score": physical_score,
        "status": "kept" if physical_score >= config.pass_threshold else "rejected",
    }


def run_filter(args: PhysicalFilterArgs) -> dict[str, Any]:
    start_time = time.perf_counter()
    input_root = resolve_dataset_root(args.input_path)
    project_root = (
        resolve_project_root(args.project_path)
        if args.project_path is not None
        else default_project_root_for_input(input_root)
    )
    project_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(input_root)
    scoring_mjcf_path = (
        manifest.mjcf_path
        if args.scoring_mjcf_path is None
        else Path(args.scoring_mjcf_path).expanduser().resolve()
    )
    if not scoring_mjcf_path.is_file():
        raise FileNotFoundError(f"Scoring MJCF not found: {scoring_mjcf_path}")
    fps = 1.0 / manifest.timestep
    motion_paths = sorted((input_root / manifest.payload.get("motions_subdir", "motions")).rglob("*.npz"))
    if args.start < 0:
        raise ValueError(f"start must be non-negative, got {args.start}")
    motion_paths = motion_paths[int(args.start) :]
    if args.limit is not None:
        motion_paths = motion_paths[: int(args.limit)]
    if not motion_paths:
        raise FileNotFoundError(f"No motions found under {input_root}")

    config = PhysicalFilterConfig(
        pass_threshold=float(args.pass_threshold),
        batch_frames=int(args.batch_frames),
        fk_batch_size=int(args.fk_batch_size),
        contact_nconmax=int(args.contact_nconmax),
    )
    weights = PhysicalScoreWeights()
    with temp_mjcf_with_floor(scoring_mjcf_path) as scoring_mjcf_with_floor_path:
        runner = FKRunner(
            mjcf_path=scoring_mjcf_with_floor_path,
            batch_size=config.fk_batch_size,
            device=args.device,
            contact_nconmax=config.contact_nconmax,
        )
    if runner.model.nq != len(manifest.payload["qpos_names"]):
        raise ValueError(
            f"Floor-augmented scoring model nq={runner.model.nq} does not match manifest qpos width "
            f"{len(manifest.payload['qpos_names'])}"
        )
    if str(args.device).startswith("cuda") and runner.backend != "mujoco_warp":
        print(
            "[any4hdmi-limmt-score] MuJoCo Warp is unavailable; falling back to CPU MuJoCo. "
            "Install with `uv sync --extra warp` for the AMASS <15min target."
        )
    floor_geom_id, floor_geom_name = _resolve_floor_geom_id(runner.model)
    left_foot_body_id = _body_id_by_name(runner.model, "left_ankle_roll_link")
    right_foot_body_id = _body_id_by_name(runner.model, "right_ankle_roll_link")
    missing_foot_bodies = [
        name
        for name, body_id in (
            ("left_ankle_roll_link", left_foot_body_id),
            ("right_ankle_roll_link", right_foot_body_id),
        )
        if body_id is None
    ]
    if missing_foot_bodies:
        raise ValueError(
            "G1 foot body ids are required for LIMMT scoring; missing body names: "
            + ", ".join(missing_foot_bodies)
        )
    assert left_foot_body_id is not None and right_foot_body_id is not None
    foot_body_ids = (left_foot_body_id, right_foot_body_id)
    joint_vel_limits = _joint_velocity_limits(runner.joint_names, default=config.max_joint_vel, device=runner.device)
    contact_buffer_saturation_count = 0

    score_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    motion_batch: list[dict[str, Any]] = []
    batch_frames = 0

    def flush() -> None:
        nonlocal motion_batch, batch_frames, contact_buffer_saturation_count
        if not motion_batch:
            return
        qpos_list = [motion_sample["qpos"] for motion_sample in motion_batch]
        qvel_list = [motion_sample["qvel"] for motion_sample in motion_batch]
        lengths = [int(qpos.shape[0]) for qpos in qpos_list]
        packed_qpos = torch.cat(qpos_list, dim=0)
        packed_qvel = torch.cat(qvel_list, dim=0)
        fk_results = runner.forward_kinematics(packed_qpos, packed_qvel)
        contact_summary = runner.forward_contact_summary(
            packed_qpos,
            packed_qvel,
            floor_geom_id=floor_geom_id,
        )
        contact_saturation = int(contact_summary["contact_buffer_saturated"].to(device="cpu").sum().item())
        fk_result_splits = {key: value.split(lengths, dim=0) for key, value in fk_results.items()}
        contact_summary_splits = {key: value.split(lengths, dim=0) for key, value in contact_summary.items()}
        qvel_splits = packed_qvel.split(lengths, dim=0)
        contact_buffer_saturation_count += contact_saturation
        for motion_idx, motion_sample in enumerate(motion_batch):
            score_row = _score_motion(
                rel_motion=motion_sample["rel_motion"].as_posix(),
                fk_results={key: fk_result_splits[key][motion_idx] for key in fk_result_splits},
                qvel=qvel_splits[motion_idx],
                fps=fps,
                config=config,
                weights=weights,
                contact_summary={key: contact_summary_splits[key][motion_idx] for key in contact_summary_splits},
                joint_vel_limits=joint_vel_limits,
                foot_body_ids=foot_body_ids,
            )
            score_rows.append(score_row)
            if score_row["status"] != "kept":
                reason_counts.update(["physical_score_below_threshold"])
        motion_batch = []
        batch_frames = 0

    loader = build_motion_loader(
        input_root=input_root,
        motion_paths=motion_paths,
        mjcf_path=scoring_mjcf_path,
        fps=fps,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=runner.device.type == "cuda",
        tensor_device=runner.device,
    )
    for motion_sample in tqdm(loader, total=len(motion_paths), desc="LIMMT physical score", unit="motion"):
        frames = int(motion_sample["qpos"].shape[0])
        if motion_batch and batch_frames + frames > config.batch_frames:
            flush()
        motion_batch.append(motion_sample)
        batch_frames += frames
    flush()

    score_rows.sort(key=lambda score_row: score_row["motion"])
    motions_subdir = Path(manifest.payload.get("motions_subdir", "motions"))
    kept_motions = [
        Path(score_row["motion"]).relative_to(motions_subdir).as_posix()
        for score_row in score_rows
        if score_row["status"] == "kept"
    ]
    scores_json = project_root / "scores.json"
    filenames_path = project_root / "filenames.txt"
    elapsed = time.perf_counter() - start_time
    filter_summary = {
        "input_root": str(input_root),
        "project_root": str(project_root),
        "filenames_path": str(filenames_path),
        "scoring_mjcf_path": str(scoring_mjcf_path),
        "input_start": int(args.start),
        "input_limit": None if args.limit is None else int(args.limit),
        "backend": runner.backend,
        "device": str(runner.device),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "total_motions": len(score_rows),
        "kept_motions": len(kept_motions),
        "rejected_motions": len(score_rows) - len(kept_motions),
        "reason_counts": dict(reason_counts),
        "config": config.__dict__,
        "weights": weights.__dict__,
        "contact_based": True,
        "floor_geom_id": floor_geom_id,
        "floor_geom_name": floor_geom_name,
        "contact_backend": runner.backend,
        "nconmax": config.contact_nconmax,
        "contact_buffer_saturation_count": contact_buffer_saturation_count,
        "foot_body_ids": list(foot_body_ids),
    }
    write_json(scores_json, {"summary": filter_summary, "details": score_rows})
    filenames_path.write_text("\n".join(kept_motions) + ("\n" if kept_motions else ""), encoding="utf-8")
    write_json(project_root / "summary.json", filter_summary)
    print(json.dumps(filter_summary, indent=2))
    return filter_summary


def main() -> None:
    run_filter(tyro.cli(PhysicalFilterArgs))


if __name__ == "__main__":
    main()
