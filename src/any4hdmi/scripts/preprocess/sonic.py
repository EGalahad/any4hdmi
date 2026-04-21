from __future__ import annotations

import argparse
import csv
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from mjhub import resolve_asset_reference
except ImportError:
    from mjhub import resolve_mjcf_reference as resolve_asset_reference

from any4hdmi.core.format import MOTION_DTYPE, MOTIONS_SUBDIR, repo_root, save_motion, write_manifest
from any4hdmi.core.model import G1_JOINT_ORDER, base_qpos_adr, joint_qpos_adrs, load_model
from any4hdmi.utils.math import euler_to_quat_wxyz, maybe_degrees_to_radians
from any4hdmi.utils.mjcf import (
    DEFAULT_MJCF_PATH,
    DEFAULT_MJCF_REPO_ID,
    DEFAULT_MJCF_REVISION,
    build_hf_mjcf_reference,
    qpos_names_from_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert g1_sonic CSV files into the any4hdmi qpos format."
    )
    parser.add_argument(
        "--csv-dir",
        default=str(repo_root().parent / "g1_sonic" / "complete" / "g1" / "csv"),
        help="Directory containing source CSV files.",
    )
    parser.add_argument("--out-dir", default="output/sonic", help="Output dataset root for converted motions.")
    parser.add_argument(
        "--mjcf-repo",
        default=DEFAULT_MJCF_REPO_ID,
        help="Hugging Face repo id that stores the MJCF and mesh assets.",
    )
    parser.add_argument(
        "--mjcf-path",
        default=DEFAULT_MJCF_PATH,
        help="Path to the MJCF file within --mjcf-repo.",
    )
    parser.add_argument(
        "--mjcf-revision",
        default=DEFAULT_MJCF_REVISION,
        help="Revision passed to Hugging Face snapshot_download.",
    )
    parser.add_argument("--fps", type=float, default=120.0, help="Source frame rate.")
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=0.01,
        help="Scale applied to root translations. Default converts centimeters to meters.",
    )
    parser.add_argument(
        "--angle-unit",
        choices=["deg", "rad"],
        default="deg",
        help="Unit used by root Euler angles and joint dof columns.",
    )
    parser.add_argument("--euler-order", default="xyz", help="Euler axis order for root rotations.")
    parser.add_argument(
        "--euler-frame",
        choices=["intrinsic", "extrinsic"],
        default="extrinsic",
        help="Whether root Euler angles are interpreted as intrinsic or extrinsic rotations.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern relative to --csv-dir.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes used for CSV conversion when greater than 1.",
    )
    return parser.parse_args()


def _load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    return header, np.asarray(data, dtype=MOTION_DTYPE)


def _build_column_index(header: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(header)}


def _required_columns() -> list[str]:
    return [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *[f"{joint_name}_dof" for joint_name in G1_JOINT_ORDER],
    ]


def _build_qpos_sequence(
    header: list[str],
    motion: np.ndarray,
    *,
    qpos_dim: int,
    base_adr: int,
    hinge_qpos_adrs: np.ndarray,
    translation_scale: float,
    angle_unit: str,
    euler_order: str,
    euler_frame: str,
) -> np.ndarray:
    column_index = _build_column_index(header)
    missing = [name for name in _required_columns() if name not in column_index]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    qpos = np.zeros((motion.shape[0], qpos_dim), dtype=MOTION_DTYPE)
    translation_cols = [
        column_index["root_translateX"],
        column_index["root_translateY"],
        column_index["root_translateZ"],
    ]
    rotation_cols = [
        column_index["root_rotateX"],
        column_index["root_rotateY"],
        column_index["root_rotateZ"],
    ]
    joint_cols = [column_index[f"{joint_name}_dof"] for joint_name in G1_JOINT_ORDER]

    root_translation = np.asarray(
        motion[:, translation_cols] * MOTION_DTYPE(translation_scale),
        dtype=MOTION_DTYPE,
    )
    root_euler = maybe_degrees_to_radians(motion[:, rotation_cols], angle_unit)
    root_quat = euler_to_quat_wxyz(root_euler, euler_order, euler_frame)
    joint_values = maybe_degrees_to_radians(motion[:, joint_cols], angle_unit)

    qpos[:, base_adr : base_adr + 3] = np.asarray(root_translation, dtype=MOTION_DTYPE)
    qpos[:, base_adr + 3 : base_adr + 7] = np.asarray(root_quat, dtype=MOTION_DTYPE)
    qpos[:, hinge_qpos_adrs] = np.asarray(joint_values, dtype=MOTION_DTYPE)
    return qpos


def _convert_one(
    csv_path: Path,
    *,
    csv_dir: Path,
    out_dir: Path,
    qpos_dim: int,
    base_adr: int,
    hinge_qpos_adrs: np.ndarray,
    translation_scale: float,
    angle_unit: str,
    euler_order: str,
    euler_frame: str,
    start: int,
    end: int,
    stride: int,
) -> int:
    header, motion = _load_csv(csv_path)
    resolved_end = end if end >= 0 else motion.shape[0]
    motion = motion[start : min(resolved_end, motion.shape[0]) : max(1, stride)]
    qpos = _build_qpos_sequence(
        header,
        motion,
        qpos_dim=qpos_dim,
        base_adr=base_adr,
        hinge_qpos_adrs=hinge_qpos_adrs,
        translation_scale=translation_scale,
        angle_unit=angle_unit,
        euler_order=euler_order,
        euler_frame=euler_frame,
    )
    rel_path = csv_path.relative_to(csv_dir).with_suffix(".npz")
    save_motion(out_dir / MOTIONS_SUBDIR / rel_path, qpos)
    return int(qpos.shape[0])


def _available_cpu_ids() -> tuple[int, ...] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    return tuple(sorted(os.sched_getaffinity(0)))


def _init_process_affinity(cpu_ids: tuple[int, ...] | None) -> None:
    if cpu_ids is None:
        return
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU pinning requires os.sched_setaffinity, which is not available here")
    identity = multiprocessing.current_process()._identity
    worker_index = identity[0] - 1 if identity else 0
    cpu_id = cpu_ids[worker_index % len(cpu_ids)]
    os.sched_setaffinity(0, {cpu_id})


def _run_parallel(
    csv_files: list[Path],
    *,
    workers: int,
    task_kwargs: dict[str, object],
    cpu_ids: tuple[int, ...] | None,
) -> int:
    max_in_flight = max(workers * 4, 1)
    submitted = 0
    pending = set()
    total_frames = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_process_affinity,
        initargs=(cpu_ids,),
    ) as executor:
        with tqdm(total=len(csv_files), desc="Converting SONIC", unit="file") as progress:
            while submitted < len(csv_files) or pending:
                while submitted < len(csv_files) and len(pending) < max_in_flight:
                    future = executor.submit(_convert_one, csv_files[submitted], **task_kwargs)
                    pending.add(future)
                    submitted += 1

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    total_frames += int(future.result())
                    progress.update(1)
    return total_frames


def main() -> None:
    args = _parse_args()

    csv_dir = Path(args.csv_dir).expanduser().resolve()
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir = out_dir.resolve()

    mjcf_reference = build_hf_mjcf_reference(
        repo_id=args.mjcf_repo,
        path=args.mjcf_path,
        revision=args.mjcf_revision,
    )
    mjcf_path = resolve_asset_reference(mjcf_reference)
    model = load_model(mjcf_path)
    qpos_names = qpos_names_from_model(model)
    model_qpos_dim = model.nq
    model_base_adr = base_qpos_adr(model)
    model_hinge_qpos_adrs = joint_qpos_adrs(model, G1_JOINT_ORDER)

    csv_files = sorted(csv_dir.rglob(args.pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched {args.pattern!r} under {csv_dir}")

    worker_count = max(1, args.workers)
    cpu_ids = _available_cpu_ids() if worker_count > 1 else None
    if worker_count > 1 and cpu_ids is not None:
        if worker_count > len(cpu_ids):
            raise ValueError(
                f"--workers={worker_count} exceeds available CPUs ({len(cpu_ids)}): {cpu_ids}"
            )
        cpu_ids = cpu_ids[:worker_count]

    task_kwargs = {
        "csv_dir": csv_dir,
        "out_dir": out_dir,
        "qpos_dim": model_qpos_dim,
        "base_adr": model_base_adr,
        "hinge_qpos_adrs": model_hinge_qpos_adrs,
        "translation_scale": args.translation_scale,
        "angle_unit": args.angle_unit,
        "euler_order": args.euler_order,
        "euler_frame": args.euler_frame,
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
    }

    if worker_count == 1:
        total_frames = 0
        for csv_path in tqdm(csv_files, desc="Converting SONIC", unit="file"):
            total_frames += _convert_one(csv_path, **task_kwargs)
    else:
        total_frames = _run_parallel(csv_files, workers=worker_count, task_kwargs=task_kwargs, cpu_ids=cpu_ids)

    write_manifest(
        out_dir,
        dataset_name="sonic",
        mjcf=mjcf_reference,
        timestep=1.0 / args.fps,
        qpos_names=qpos_names,
        num_motions=len(csv_files),
        total_hours=total_frames / args.fps / 3600.0,
        source={
            "csv_dir": str(csv_dir),
            "pattern": args.pattern,
            "fps": args.fps,
            "translation_scale": args.translation_scale,
            "angle_unit": args.angle_unit,
            "euler_order": args.euler_order,
            "euler_frame": args.euler_frame,
            "workers": worker_count,
            "cpu_list": list(cpu_ids) if cpu_ids is not None else None,
            "root_representation": "xyz + euler_xyz_columns",
        },
    )


if __name__ == "__main__":
    main()
