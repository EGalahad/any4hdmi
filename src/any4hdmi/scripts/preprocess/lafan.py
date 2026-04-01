from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mjhub import resolve_mjcf_reference
from tqdm import tqdm

from any4hdmi.core.format import MOTION_DTYPE, MOTIONS_SUBDIR, repo_root, save_motion, write_manifest
from any4hdmi.core.model import G1_JOINT_ORDER, base_qpos_adr, joint_qpos_adrs, load_model
from any4hdmi.utils.mjcf import (
    DEFAULT_MJCF_PATH,
    DEFAULT_MJCF_REPO_ID,
    DEFAULT_MJCF_REVISION,
    build_hf_mjcf_reference,
    qpos_names_from_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LAFAN retargeted G1 CSV files into the any4hdmi qpos format."
    )
    parser.add_argument(
        "--csv-dir",
        default=str(repo_root().parent / "lafan-process" / "LAFAN1_Retargeting_Dataset" / "g1"),
        help="Directory containing source CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="output/lafan",
        help="Output dataset root for converted motions.",
    )
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
    parser.add_argument("--fps", type=float, default=30.0, help="Source frame rate.")
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern relative to --csv-dir.")
    return parser.parse_args()


def _load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    return np.asarray(data, dtype=MOTION_DTYPE)


def _build_qpos_sequence(model, motion: np.ndarray) -> np.ndarray:
    expected_width = 7 + len(G1_JOINT_ORDER)
    if motion.shape[1] != expected_width:
        raise ValueError(
            f"Expected {expected_width} columns (7 root + {len(G1_JOINT_ORDER)} joints), "
            f"got {motion.shape[1]}."
        )

    qpos = np.zeros((motion.shape[0], model.nq), dtype=MOTION_DTYPE)
    base_adr = base_qpos_adr(model)
    hinge_qpos_adrs = joint_qpos_adrs(model, G1_JOINT_ORDER)

    qpos[:, base_adr : base_adr + 3] = motion[:, 0:3]
    qpos[:, base_adr + 3 : base_adr + 7] = motion[:, [6, 3, 4, 5]]
    qpos[:, hinge_qpos_adrs] = motion[:, 7:]
    return qpos


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
    mjcf_path = resolve_mjcf_reference(mjcf_reference)
    model = load_model(mjcf_path)
    qpos_names = qpos_names_from_model(model)

    csv_files = sorted(csv_dir.rglob(args.pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched {args.pattern!r} under {csv_dir}")

    num_motions = 0
    for csv_path in tqdm(csv_files, desc="Converting LAFAN", unit="file"):
        motion = _load_csv(csv_path)
        end = args.end if args.end >= 0 else motion.shape[0]
        motion = motion[args.start : min(end, motion.shape[0]) : max(1, args.stride)]
        qpos = _build_qpos_sequence(model, motion)

        rel_path = csv_path.relative_to(csv_dir).with_suffix(".npz")
        save_motion(out_dir / MOTIONS_SUBDIR / rel_path, qpos)
        num_motions += 1

    write_manifest(
        out_dir,
        dataset_name="lafan",
        mjcf=mjcf_reference,
        timestep=1.0 / args.fps,
        qpos_names=qpos_names,
        num_motions=num_motions,
        source={
            "csv_dir": str(csv_dir),
            "pattern": args.pattern,
            "fps": args.fps,
            "root_representation": "xyz + qx qy qz qw",
            "joint_unit": "rad",
        },
    )


if __name__ == "__main__":
    main()
