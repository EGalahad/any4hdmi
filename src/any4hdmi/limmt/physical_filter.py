from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from any4hdmi.core.format import ensure_dir, load_manifest, load_motion, write_manifest
from any4hdmi.fk.runner import FKRunner
from any4hdmi.limmt.common import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_ROOT, relative_motion_path, resolve_dataset_root, write_json
from any4hdmi.utils.dataset import build_motion_loader


@dataclass(frozen=True)
class PhysicalScoreWeights:
    foot_sliding: float = 8.0
    velocity_violation: float = 10.0
    self_collision: float = 5.0
    jerk: float = 0.02
    penetration: float = 20.0
    floating_frames_ratio: float = 25.0


@dataclass(frozen=True)
class PhysicalFilterConfig:
    pass_threshold: float = 90.0
    batch_frames: int = 131072
    fk_batch_size: int = 8192
    max_root_qvel: float = 10.0
    max_joint_vel: float = 30.0
    max_body_lin_vel: float = 20.0
    max_body_ang_vel: float = 40.0
    foot_height: float = 0.08
    foot_slide_speed: float = 0.15
    floating_height: float = 0.20
    penetration_height: float = -0.02
    self_collision_distance: float = 0.045
    contact_sample_stride: int = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIMMT physical quality filtering on an any4hdmi dataset.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pass-dataset-name", default="amass_limmt_pass")
    parser.add_argument("--pass-threshold", type=float, default=90.0)
    parser.add_argument("--batch-frames", type=int, default=131072)
    parser.add_argument("--fk-batch-size", type=int, default=8192)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Preferred FK device. cuda uses MuJoCo Warp when the optional warp extra is installed; otherwise FKRunner falls back to CPU MuJoCo.",
    )
    parser.add_argument("--num-workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--contact-sample-stride", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit on number of motions.")
    return parser.parse_args()


def _body_indices(names: list[str], tokens: tuple[str, ...]) -> list[int]:
    out = []
    for idx, name in enumerate(names):
        lowered = (name or "").lower()
        if any(token in lowered for token in tokens):
            out.append(idx)
    return out


def _self_collision_pairs(body_names: list[str], body_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    skip_tokens = ("world", "pelvis", "torso", "logo", "head")
    candidate = [
        idx
        for idx, name in enumerate(body_names)
        if idx > 0 and not any(token in (name or "").lower() for token in skip_tokens)
    ]
    left: list[int] = []
    right: list[int] = []
    for offset, a in enumerate(candidate):
        side_a = "left" if "left" in (body_names[a] or "").lower() else "right" if "right" in (body_names[a] or "").lower() else ""
        for b in candidate[offset + 1 :]:
            name_b = (body_names[b] or "").lower()
            side_b = "left" if "left" in name_b else "right" if "right" in name_b else ""
            if side_a and side_a == side_b:
                continue
            left.append(a)
            right.append(b)
    if not left:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.as_tensor(left, dtype=torch.long), torch.as_tensor(right, dtype=torch.long)


def _safe_mean_positive(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.to(dtype=torch.float32).mean().item())


def _score_motion(
    *,
    rel_motion: str,
    outputs: dict[str, torch.Tensor],
    qvel: torch.Tensor,
    fps: float,
    config: PhysicalFilterConfig,
    weights: PhysicalScoreWeights,
    foot_indices: torch.Tensor,
    pair_left: torch.Tensor,
    pair_right: torch.Tensor,
) -> dict[str, Any]:
    frames = int(qvel.shape[0])
    body_pos = outputs["body_pos_w"]
    body_lin = outputs["body_lin_vel_w"]
    body_ang = outputs["body_ang_vel_w"]
    joint_vel = outputs["joint_vel"]

    if foot_indices.numel() > 0:
        foot_pos = body_pos.index_select(1, foot_indices.to(body_pos.device))
        foot_vel = body_lin.index_select(1, foot_indices.to(body_lin.device))
        near_ground = foot_pos[..., 2] < config.foot_height
        slide_speed = torch.linalg.vector_norm(foot_vel[..., :2], dim=-1)
        foot_sliding = _safe_mean_positive(torch.relu(slide_speed - config.foot_slide_speed) * near_ground)
    else:
        foot_sliding = 0.0

    root_width = min(6, qvel.shape[1])
    root_violation = torch.relu(torch.abs(qvel[:, :root_width]) - config.max_root_qvel).mean() if root_width else qvel.new_tensor(0.0)
    joint_violation = torch.relu(torch.abs(joint_vel) - config.max_joint_vel).mean() if joint_vel.numel() else qvel.new_tensor(0.0)
    body_lin_violation = torch.relu(torch.linalg.vector_norm(body_lin, dim=-1) - config.max_body_lin_vel).mean()
    body_ang_violation = torch.relu(torch.linalg.vector_norm(body_ang, dim=-1) - config.max_body_ang_vel).mean()
    velocity_violation = float((root_violation + joint_violation + body_lin_violation + body_ang_violation).item())

    if qvel.shape[0] >= 3:
        qacc = torch.diff(qvel, dim=0) * float(fps)
        jerk = float(torch.diff(qacc, dim=0).abs().mean().item())
    else:
        jerk = 0.0

    dynamic_body_z = body_pos[:, 1:, 2] if body_pos.shape[1] > 1 else body_pos[:, :, 2]
    min_body_z = dynamic_body_z.min(dim=1).values
    floating_frames_ratio = float((min_body_z > config.floating_height).to(torch.float32).mean().item())
    penetration = float(torch.relu(config.penetration_height - min_body_z).mean().item())

    if pair_left.numel() > 0 and config.contact_sample_stride > 0:
        sampled = body_pos[:: config.contact_sample_stride]
        left_pos = sampled.index_select(1, pair_left.to(sampled.device))
        right_pos = sampled.index_select(1, pair_right.to(sampled.device))
        dists = torch.linalg.vector_norm(left_pos - right_pos, dim=-1)
        self_collision = float((dists < config.self_collision_distance).to(torch.float32).mean().item())
    else:
        self_collision = 0.0

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


def _write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _copy_pass_dataset(input_root: Path, pass_root: Path, rows: list[dict[str, Any]], *, pass_threshold: float) -> Path:
    manifest = load_manifest(input_root)
    selected = [row["motion"] for row in rows if row["status"] == "kept"]
    if pass_root.exists():
        shutil.rmtree(pass_root)
    ensure_dir(pass_root)
    total_frames = 0
    for rel in selected:
        src = input_root / rel
        dst = pass_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total_frames += int(load_motion(src).shape[0])
    source = dict(manifest.payload.get("source", {}))
    source["limmt_physical_filter"] = {"pass_threshold": float(pass_threshold), "input_root": str(input_root)}
    return write_manifest(
        pass_root,
        dataset_name=pass_root.name,
        mjcf=manifest.payload["mjcf"],
        timestep=manifest.timestep,
        qpos_names=list(manifest.payload["qpos_names"]),
        num_motions=len(selected),
        source=source,
        total_hours=float(total_frames * manifest.timestep / 3600.0),
    )


def run_filter(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.perf_counter()
    input_root = resolve_dataset_root(args.input_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(input_root)
    fps = 1.0 / manifest.timestep
    motion_paths = sorted((input_root / manifest.payload.get("motions_subdir", "motions")).rglob("*.npz"))
    if args.limit is not None:
        motion_paths = motion_paths[: int(args.limit)]
    if not motion_paths:
        raise FileNotFoundError(f"No motions found under {input_root}")

    config = PhysicalFilterConfig(
        pass_threshold=float(args.pass_threshold),
        batch_frames=int(args.batch_frames),
        fk_batch_size=int(args.fk_batch_size),
        contact_sample_stride=int(args.contact_sample_stride),
    )
    weights = PhysicalScoreWeights()
    runner = FKRunner(mjcf_path=manifest.mjcf_path, batch_size=config.fk_batch_size, device=args.device)
    if str(args.device).startswith("cuda") and runner.backend != "mujoco_warp":
        print(
            "[any4hdmi-limmt-score] MuJoCo Warp is unavailable; falling back to CPU MuJoCo. "
            "Install with `uv sync --extra warp` for the AMASS <15min target."
        )
    foot_indices = torch.as_tensor(_body_indices(runner.body_names, ("foot", "ankle")), dtype=torch.long, device=runner.device)
    pair_left, pair_right = _self_collision_pairs(runner.body_names, len(runner.body_names))

    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    batch_items: list[dict[str, Any]] = []
    batch_frames = 0

    def flush() -> None:
        nonlocal batch_items, batch_frames
        if not batch_items:
            return
        qpos_list = [item["qpos"] for item in batch_items]
        qvel_list = [item["qvel"] for item in batch_items]
        lengths = [int(qpos.shape[0]) for qpos in qpos_list]
        outputs = runner.forward_kinematics(torch.cat(qpos_list, dim=0), torch.cat(qvel_list, dim=0))
        splits = {key: value.split(lengths, dim=0) for key, value in outputs.items()}
        qvel_splits = torch.cat(qvel_list, dim=0).split(lengths, dim=0)
        for idx, item in enumerate(batch_items):
            row = _score_motion(
                rel_motion=item["rel_motion"].as_posix(),
                outputs={key: splits[key][idx] for key in splits},
                qvel=qvel_splits[idx],
                fps=fps,
                config=config,
                weights=weights,
                foot_indices=foot_indices,
                pair_left=pair_left,
                pair_right=pair_right,
            )
            rows.append(row)
            if row["status"] != "kept":
                reason_counts.update(["physical_score_below_threshold"])
        batch_items = []
        batch_frames = 0

    loader = build_motion_loader(
        input_root=input_root,
        motion_paths=motion_paths,
        mjcf_path=manifest.mjcf_path,
        fps=fps,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=runner.device.type == "cuda",
        tensor_device=runner.device,
    )
    for item in tqdm(loader, total=len(motion_paths), desc="LIMMT physical score", unit="motion"):
        frames = int(item["qpos"].shape[0])
        if batch_items and batch_frames + frames > config.batch_frames:
            flush()
        batch_items.append(item)
        batch_frames += frames
    flush()

    rows.sort(key=lambda row: row["motion"])
    kept = [row["motion"] for row in rows if row["status"] == "kept"]
    scores_json = output_dir / "scores.json"
    scores_csv = output_dir / "scores.csv"
    pass_filenames = output_dir / "pass_filenames.txt"
    elapsed = time.perf_counter() - start_time
    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "backend": runner.backend,
        "device": str(runner.device),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "total_motions": len(rows),
        "kept_motions": len(kept),
        "rejected_motions": len(rows) - len(kept),
        "reason_counts": dict(reason_counts),
        "config": config.__dict__,
        "weights": weights.__dict__,
    }
    write_json(scores_json, {"summary": summary, "details": rows})
    _write_scores_csv(scores_csv, rows)
    pass_filenames.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    pass_root = output_dir / args.pass_dataset_name
    manifest_path = _copy_pass_dataset(input_root, pass_root, rows, pass_threshold=config.pass_threshold)
    summary["pass_dataset_root"] = str(pass_root)
    summary["pass_manifest"] = str(manifest_path)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run_filter(_parse_args())


if __name__ == "__main__":
    main()
