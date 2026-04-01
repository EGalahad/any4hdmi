from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from mjhub import resolve_mjcf_reference
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from any4hdmi.core.format import ensure_dir, load_manifest, write_manifest
from any4hdmi.fk.runner import FKRunner, compute_root_qvel_many_torch
from any4hdmi.utils.dataset import (
    DEFAULT_MOTION_LOADER_NUM_WORKERS,
    DEFAULT_MOTION_LOADER_PREFETCH_FACTOR,
    build_motion_loader,
)


REPORT_NAME = "filter_report.json"
DEFAULT_MAX_ROOT_QVEL = 10.0
DEFAULT_MIN_FRAMES = 250
DEFAULT_ALL_OFF_GROUND_Z = 0.2
DEFAULT_MAX_ALL_OFF_GROUND_SECONDS = 1.0
DEFAULT_MIN_MAX_BODY_Z = 0.2
DEFAULT_BATCH_SIZE = 2048
DEFAULT_NUM_WORKERS = DEFAULT_MOTION_LOADER_NUM_WORKERS
DEFAULT_PREFETCH_FACTOR = DEFAULT_MOTION_LOADER_PREFETCH_FACTOR


@dataclass(frozen=True)
class FilterConfig:
    max_root_qvel: float
    min_frames: int
    all_off_ground_z: float
    max_all_off_ground_seconds: float
    min_max_body_z: float
    batch_size: int


@dataclass(frozen=True)
class MotionCheckResult:
    is_valid: bool
    reasons: tuple[str, ...]
    num_frames: int
    fps: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a unified any4hdmi dataset with FK-based motion sanity checks."
    )
    parser.add_argument("--input-root", required=True, help="Input dataset root that contains manifest.json.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output dataset root for kept motions. Defaults to <input-root>_filtered.",
    )
    parser.add_argument(
        "--mjcf-path",
        default=None,
        help=(
            "Optional MJCF override. Defaults to the manifest.json mjcf reference resolved via mjhub. "
            "Accepts either a local XML path or an hf://... MJCF reference."
        ),
    )
    parser.add_argument(
        "--keep-filenames-path",
        default=None,
        help=(
            "Optional text file that whitelists motion filenames to keep. "
            "Each line should contain a filename stem such as foo__A001 or foo__A001_M."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Run checks and emit a report without copying kept motions.")
    parser.add_argument("--report-path", default=None, help="Optional explicit path for the JSON filter report.")
    parser.add_argument(
        "--max-root-qvel",
        type=float,
        default=DEFAULT_MAX_ROOT_QVEL,
        help="Reject clips when any of the first 6 qvel dimensions exceed this absolute value.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help="Reject clips shorter than this many frames.",
    )
    parser.add_argument(
        "--all-off-ground-z",
        type=float,
        default=DEFAULT_ALL_OFF_GROUND_Z,
        help="A frame counts as all-off-ground when every body z is above this threshold.",
    )
    parser.add_argument(
        "--max-all-off-ground-seconds",
        type=float,
        default=DEFAULT_MAX_ALL_OFF_GROUND_SECONDS,
        help="Reject clips when all bodies remain off-ground longer than this duration.",
    )
    parser.add_argument(
        "--min-max-body-z",
        type=float,
        default=DEFAULT_MIN_MAX_BODY_Z,
        help="Reject clips whose maximum body height never exceeds this threshold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="FK batch size. This limits per-call MuJoCo Warp or CPU work.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for FK and filtering. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of DataLoader workers used to preload motion tensors.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=DEFAULT_PREFETCH_FACTOR,
        help="DataLoader prefetch factor used when --num-workers is greater than zero.",
    )
    return parser.parse_args()


def _resolve_output_root(input_root: Path, output_root_arg: str | None) -> Path:
    if output_root_arg is not None:
        return Path(output_root_arg).expanduser().resolve()
    return input_root.with_name(f"{input_root.name}_filtered")


def _resolve_mjcf_path(_manifest_root: Path, manifest: Any, mjcf_override: str | None) -> Path:
    if mjcf_override:
        if mjcf_override.startswith("hf://"):
            return resolve_mjcf_reference(mjcf_override, local_root=manifest.root)
        mjcf_path = Path(mjcf_override).expanduser().resolve()
        if mjcf_path.is_file():
            return mjcf_path
        raise FileNotFoundError(f"MJCF override not found: {mjcf_path}")
    return manifest.mjcf_path


def _load_motion_fps(_motions_dir: Path, manifest_timestep: float) -> float:
    if manifest_timestep <= 0.0:
        raise ValueError(f"Invalid manifest timestep: {manifest_timestep}")
    return 1.0 / manifest_timestep


def _normalize_name_token(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    token = token.split("\t", 1)[0]
    token = token.split(" ", 1)[0]
    return Path(token).stem


def _load_keep_filenames(path_arg: str | None) -> tuple[Path | None, frozenset[str] | None]:
    if path_arg is None:
        return None, None
    keep_path = Path(path_arg).expanduser().resolve()
    if not keep_path.is_file():
        raise FileNotFoundError(f"Keep-filenames file not found: {keep_path}")
    keep_names = {
        normalized
        for line in keep_path.read_text(encoding="utf-8").splitlines()
        if (normalized := _normalize_name_token(line))
    }
    if not keep_names:
        raise ValueError(f"Keep-filenames file is empty: {keep_path}")
    return keep_path, frozenset(keep_names)


def _motion_name_is_kept(rel_motion: Path, keep_names: frozenset[str] | None) -> bool:
    if keep_names is None:
        return True
    return rel_motion.stem in keep_names


def _batched_contiguous_true_run_length_torch(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"Expected mask to be rank 2, got shape {tuple(mask.shape)}")
    if mask.shape[1] == 0:
        return torch.zeros(mask.shape[0], dtype=torch.int64, device=mask.device)

    frame_ids = torch.arange(1, mask.shape[1] + 1, device=mask.device, dtype=torch.int64)
    frame_ids = frame_ids.unsqueeze(0).expand(mask.shape[0], -1)
    last_false = torch.where(~mask, frame_ids, 0)
    last_false = torch.cummax(last_false, dim=1).values
    run_lengths = torch.where(mask, frame_ids - last_false, 0)
    return torch.max(run_lengths, dim=1).values


def _check_motion_batch_torch(
    *,
    qvel_list: list[torch.Tensor],
    xpos_list: list[torch.Tensor],
    fps_list: list[float],
    config: FilterConfig,
) -> list[tuple[str, ...]]:
    if not qvel_list:
        return []

    device = qvel_list[0].device
    lengths = torch.as_tensor([int(qvel.shape[0]) for qvel in qvel_list], device=device, dtype=torch.int64)
    fps = torch.as_tensor(fps_list, device=device, dtype=torch.float32)
    qvel_padded = pad_sequence(qvel_list, batch_first=True)
    xpos_padded = pad_sequence(xpos_list, batch_first=True)
    frame_ids = torch.arange(qvel_padded.shape[1], device=device, dtype=torch.int64)
    frame_mask = frame_ids.unsqueeze(0) < lengths.unsqueeze(1)

    root_dims = min(6, qvel_padded.shape[2])
    root_qvel_spike = torch.zeros(len(qvel_list), dtype=torch.bool, device=device)
    if root_dims > 0:
        root_spike_mask = (torch.abs(qvel_padded[:, :, :root_dims]) > config.max_root_qvel).any(dim=2)
        root_qvel_spike = torch.any(root_spike_mask & frame_mask, dim=1)

    too_short = lengths < config.min_frames

    body_xpos = xpos_padded[:, :, 1:, :]
    if body_xpos.shape[2] == 0:
        no_dynamic_bodies = torch.ones(len(qvel_list), dtype=torch.bool, device=device)
        all_bodies_off_ground_too_long = torch.zeros_like(no_dynamic_bodies)
        max_body_height_too_low = torch.zeros_like(no_dynamic_bodies)
    else:
        no_dynamic_bodies = torch.zeros(len(qvel_list), dtype=torch.bool, device=device)
        min_body_z = torch.min(body_xpos[..., 2], dim=2).values
        all_off_ground = (min_body_z > config.all_off_ground_z) & frame_mask
        max_run_frames = _batched_contiguous_true_run_length_torch(all_off_ground)
        max_run_seconds = max_run_frames.to(dtype=torch.float32) / torch.clamp(fps, min=torch.finfo(fps.dtype).eps)
        valid_fps = fps > 0.0
        all_bodies_off_ground_too_long = valid_fps & (
            max_run_seconds > config.max_all_off_ground_seconds
        )

        neg_inf = torch.full((), float("-inf"), dtype=body_xpos.dtype, device=device)
        max_body_z = torch.where(frame_mask.unsqueeze(-1), body_xpos[..., 2], neg_inf)
        max_body_z = torch.amax(max_body_z, dim=(1, 2))
        max_body_height_too_low = max_body_z <= config.min_max_body_z

    reject_matrix = torch.stack(
        [
            root_qvel_spike,
            too_short,
            no_dynamic_bodies,
            all_bodies_off_ground_too_long,
            max_body_height_too_low,
        ],
        dim=1,
    ).cpu()
    reason_names = (
        "root_qvel_spike",
        "too_short",
        "no_dynamic_bodies",
        "all_bodies_off_ground_too_long",
        "max_body_height_too_low",
    )
    return [
        tuple(reason for reason, is_active in zip(reason_names, row.tolist(), strict=True) if is_active)
        for row in reject_matrix
    ]


def _copy_motion(src_motion: Path, dst_motion: Path) -> None:
    dst_motion.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_motion, dst_motion)


def _build_filter_source(manifest_payload: dict[str, Any], config: FilterConfig, backend: str) -> dict[str, Any]:
    source = dict(manifest_payload.get("source", {}))
    source["filter"] = {
        "type": "fk_sanity_filter",
        "backend": backend,
        "max_root_qvel": config.max_root_qvel,
        "min_frames": config.min_frames,
        "all_off_ground_z": config.all_off_ground_z,
        "max_all_off_ground_seconds": config.max_all_off_ground_seconds,
        "min_max_body_z": config.min_max_body_z,
        "batch_size": config.batch_size,
    }
    return source


def _process_motion_batch(
    *,
    batch_items: list[dict[str, Any]],
    fk_runner: FKRunner,
    config: FilterConfig,
    output_root: Path,
    dry_run: bool,
    motion_entries: list[dict[str, Any]],
    reason_counts: Counter[str],
) -> int:
    if not batch_items:
        return 0

    qpos_list = [fk_runner.to_device(item["qpos"]) for item in batch_items]
    fps_list = [float(item["fps"]) for item in batch_items]
    xpos_list = fk_runner.forward_positions_many(qpos_list)
    qvel_list = compute_root_qvel_many_torch(qpos_list, fps_list)
    reasons_list = _check_motion_batch_torch(
        qvel_list=qvel_list,
        xpos_list=xpos_list,
        fps_list=fps_list,
        config=config,
    )
    kept_count = 0

    for item, reasons in zip(batch_items, reasons_list, strict=True):
        is_valid = len(reasons) == 0
        result = MotionCheckResult(
            is_valid=is_valid,
            reasons=reasons,
            num_frames=int(item["qpos"].shape[0]),
            fps=float(item["fps"]),
        )
        motion_entries.append(
            {
                "motion": item["rel_motion"].as_posix(),
                "status": "kept" if result.is_valid else "rejected",
                "reasons": list(result.reasons),
                "num_frames": result.num_frames,
                "fps": result.fps,
            }
        )
        if result.is_valid:
            kept_count += 1
            if not dry_run:
                _copy_motion(item["motion_path"], output_root / item["rel_motion"])
        else:
            reason_counts.update(result.reasons)

    batch_items.clear()
    return kept_count


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    manifest = load_manifest(input_root)
    output_root = _resolve_output_root(input_root, args.output_root)
    keep_filenames_path, keep_names = _load_keep_filenames(args.keep_filenames_path)

    if not args.dry_run and output_root == input_root:
        raise ValueError("--output-root must differ from --input-root unless --dry-run is used")

    mjcf_path = _resolve_mjcf_path(input_root, manifest, args.mjcf_path)
    fk_runner = FKRunner(mjcf_path=mjcf_path, batch_size=args.batch_size, device=args.device)
    config = FilterConfig(
        max_root_qvel=float(args.max_root_qvel),
        min_frames=int(args.min_frames),
        all_off_ground_z=float(args.all_off_ground_z),
        max_all_off_ground_seconds=float(args.max_all_off_ground_seconds),
        min_max_body_z=float(args.min_max_body_z),
        batch_size=int(args.batch_size),
    )

    motions_dir = input_root / manifest.payload["motions_subdir"]
    motion_paths = sorted(motions_dir.rglob("*.npz"))
    if not motion_paths:
        raise FileNotFoundError(f"No motion files found under {motions_dir}")

    fps = _load_motion_fps(motions_dir, manifest.timestep)
    matched_keep_names: set[str] = set()
    if keep_names is not None:
        motion_stems = {motion_path.stem for motion_path in motion_paths}
        matched_keep_names = set(keep_names & motion_stems)

    if not args.dry_run:
        ensure_dir(output_root)

    kept_count = 0
    motion_entries: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    batch_items: list[dict[str, Any]] = []
    batch_frames = 0
    selected_motion_paths: list[Path] = []
    skipped_count = 0

    for motion_path in motion_paths:
        rel_motion = motion_path.relative_to(input_root)
        if _motion_name_is_kept(rel_motion, keep_names):
            selected_motion_paths.append(motion_path)
            continue
        motion_entries.append(
            {
                "motion": rel_motion.as_posix(),
                "status": "rejected",
                "reasons": ["not_in_keep_filenames"],
                "num_frames": None,
                "fps": None,
            }
        )
        reason_counts.update(["not_in_keep_filenames"])
        skipped_count += 1

    motion_loader = build_motion_loader(
        input_root=input_root,
        motion_paths=selected_motion_paths,
        fps=fps,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=fk_runner.device.type == "cuda",
    )

    with tqdm(total=len(motion_paths), desc="Filtering", unit="motion") as progress:
        if skipped_count > 0:
            progress.update(skipped_count)

        for item in motion_loader:
            batch_items.append(item)
            batch_frames += int(item["qpos"].shape[0])
            if batch_frames >= config.batch_size:
                processed_count = len(batch_items)
                kept_count += _process_motion_batch(
                    batch_items=batch_items,
                    fk_runner=fk_runner,
                    config=config,
                    output_root=output_root,
                    dry_run=args.dry_run,
                    motion_entries=motion_entries,
                    reason_counts=reason_counts,
                )
                progress.update(processed_count)
                batch_frames = 0

        if batch_items:
            processed_count = len(batch_items)
            kept_count += _process_motion_batch(
                batch_items=batch_items,
                fk_runner=fk_runner,
                config=config,
                output_root=output_root,
                dry_run=args.dry_run,
                motion_entries=motion_entries,
                reason_counts=reason_counts,
            )
            progress.update(processed_count)

    report = {
        "input_root": str(input_root),
        "output_root": None if args.dry_run else str(output_root),
        "mjcf_path": str(mjcf_path),
        "keep_filenames_path": None if keep_filenames_path is None else str(keep_filenames_path),
        "backend": fk_runner.backend,
        "device": str(fk_runner.device),
        "summary": {
            "total_motions": len(motion_paths),
            "kept_motions": kept_count,
            "rejected_motions": len(motion_paths) - kept_count,
            "reason_counts": dict(sorted(reason_counts.items())),
            "keep_filenames_total": None if keep_names is None else len(keep_names),
            "keep_filenames_matched": None if keep_names is None else len(matched_keep_names),
            "keep_filenames_missing": None
            if keep_names is None
            else len(keep_names) - len(matched_keep_names),
        },
        "thresholds": {
            "max_root_qvel": config.max_root_qvel,
            "min_frames": config.min_frames,
            "all_off_ground_z": config.all_off_ground_z,
            "max_all_off_ground_seconds": config.max_all_off_ground_seconds,
            "min_max_body_z": config.min_max_body_z,
            "batch_size": config.batch_size,
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
        },
        "motions": motion_entries,
    }

    report_path: Path | None
    if args.report_path is not None:
        report_path = Path(args.report_path).expanduser().resolve()
    elif args.dry_run:
        report_path = None
    else:
        report_path = output_root / REPORT_NAME

    if not args.dry_run:
        source = _build_filter_source(manifest.payload, config, fk_runner.backend)
        source["filter"]["device"] = str(fk_runner.device)
        source["filter"]["num_workers"] = int(args.num_workers)
        source["filter"]["prefetch_factor"] = int(args.prefetch_factor)
        if keep_filenames_path is not None:
            source["filter"]["keep_filenames_path"] = str(keep_filenames_path)
            source["filter"]["keep_filenames_total"] = len(keep_names)
            source["filter"]["keep_filenames_matched"] = len(matched_keep_names)
        write_manifest(
            output_root,
            dataset_name=manifest.dataset_name,
            mjcf=manifest.mjcf,
            timestep=manifest.timestep,
            qpos_names=list(manifest.payload["qpos_names"]),
            num_motions=kept_count,
            source=source,
        )
        (output_root / "manifest.source.json").write_text(
            json.dumps(
                {
                    "copied_from": str(input_root / "manifest.json"),
                    "resolved_original_mjcf_cache_path": str(mjcf_path),
                    "original_manifest": manifest.payload,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "backend": fk_runner.backend,
                "device": str(fk_runner.device),
                "total_motions": len(motion_paths),
                "kept_motions": kept_count,
                "rejected_motions": len(motion_paths) - kept_count,
                "report_path": None if report_path is None else str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
