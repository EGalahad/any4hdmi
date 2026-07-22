from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from any4hdmi.core.format import MOTION_DTYPE, load_manifest, save_motion, write_manifest


SEGMENT_PATTERN = re.compile(r"^(?P<stem>.+)_(?P<index>[1-9][0-9]*)\.npz$")
GMR_FIELDS = ("root_pos", "root_rot", "dof_pos")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GMR root_pos/root_rot/dof_pos motions into canonical qpos "
            "motions, concatenating numbered segments first."
        )
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def partition_segment_paths(
    motions_root: Path,
) -> tuple[dict[Path, list[Path]], list[Path]]:
    grouped: dict[Path, list[tuple[int, Path]]] = defaultdict(list)
    unsegmented: list[Path] = []
    paths = sorted(motions_root.rglob("*.npz"))
    if not paths:
        raise ValueError(f"No GMR NPZ motions found under {motions_root}")

    for path in paths:
        relative = path.relative_to(motions_root)
        match = SEGMENT_PATTERN.fullmatch(relative.name)
        if match is None:
            unsegmented.append(path)
            continue
        output_relative = relative.with_name(f"{match.group('stem')}.npz")
        grouped[output_relative].append((int(match.group("index")), path))

    segments: dict[Path, list[Path]] = {}
    for output_relative, indexed_paths in grouped.items():
        indexed_paths.sort(key=lambda item: item[0])
        if len(indexed_paths) == 1:
            unsegmented.append(indexed_paths[0][1])
            continue
        indices = [index for index, _ in indexed_paths]
        expected = list(range(1, len(indexed_paths) + 1))
        if indices != expected:
            raise ValueError(
                f"Non-contiguous segment indices for {output_relative}: {indices}"
            )
        segments[output_relative] = [path for _, path in indexed_paths]
    return segments, sorted(unsegmented)


def load_gmr_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(GMR_FIELDS) - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing required GMR fields: {sorted(missing)}")
        motion = {key: np.asarray(archive[key]) for key in GMR_FIELDS}

    root_pos = motion["root_pos"]
    root_rot = motion["root_rot"]
    dof_pos = motion["dof_pos"]
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (T, 3), got {root_pos.shape} in {path}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"Expected root_rot shape (T, 4), got {root_rot.shape} in {path}")
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos to be rank 2, got {dof_pos.shape} in {path}")
    if not (len(root_pos) == len(root_rot) == len(dof_pos)):
        raise ValueError(
            "root_pos, root_rot, and dof_pos must have the same frame count in "
            f"{path}: {len(root_pos)}, {len(root_rot)}, {len(dof_pos)}"
        )
    return motion


def gmr_to_qpos(motion: dict[str, np.ndarray]) -> np.ndarray:
    root_pos = np.asarray(motion["root_pos"], dtype=MOTION_DTYPE)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=MOTION_DTYPE)
    dof_pos = np.asarray(motion["dof_pos"], dtype=MOTION_DTYPE)
    quat_norm = np.linalg.norm(root_rot_xyzw, axis=1, keepdims=True)
    if np.any(quat_norm <= 1e-8):
        raise ValueError("root_rot contains a zero quaternion")
    root_rot_wxyz = (root_rot_xyzw / quat_norm)[:, [3, 0, 1, 2]]
    return np.concatenate((root_pos, root_rot_wxyz, dof_pos), axis=1)


def concatenate_gmr_segments(
    paths: list[Path],
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    segments = [load_gmr_motion(path) for path in paths]
    dof_shape = segments[0]["dof_pos"].shape[1:]
    dof_dtype = segments[0]["dof_pos"].dtype
    for path, segment in zip(paths[1:], segments[1:], strict=True):
        if segment["dof_pos"].shape[1:] != dof_shape:
            raise ValueError(f"dof_pos width differs in {path}")
        if segment["dof_pos"].dtype != dof_dtype:
            raise ValueError(f"dof_pos dtype differs in {path}")

    combined = {
        key: np.concatenate([segment[key] for segment in segments], axis=0)
        for key in GMR_FIELDS
    }
    boundaries: list[dict[str, float]] = []
    for left, right in zip(segments[:-1], segments[1:], strict=True):
        left_quat = np.asarray(left["root_rot"][-1], dtype=np.float64)
        right_quat = np.asarray(right["root_rot"][0], dtype=np.float64)
        left_norm = float(np.linalg.norm(left_quat))
        right_norm = float(np.linalg.norm(right_quat))
        if left_norm <= 1e-8 or right_norm <= 1e-8:
            raise ValueError("Cannot measure a segment boundary with a zero quaternion")
        cosine = abs(float(np.dot(left_quat, right_quat))) / (left_norm * right_norm)
        boundaries.append(
            {
                "root_pos_delta": float(
                    np.linalg.norm(right["root_pos"][0] - left["root_pos"][-1])
                ),
                "root_rot_angle": float(
                    2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))
                ),
                "dof_pos_max_delta": float(
                    np.max(np.abs(right["dof_pos"][0] - left["dof_pos"][-1]))
                ),
            }
        )
    return combined, boundaries


def convert_dataset(input_path: Path, output_path: Path) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Output path already exists: {output_path}")

    manifest = load_manifest(input_path)
    motions_subdir = Path(manifest.payload.get("motions_subdir", "motions"))
    input_motions = input_path / motions_subdir
    segment_groups, unsegmented_paths = partition_segment_paths(input_motions)
    unsegmented_outputs = {
        path.relative_to(input_motions) for path in unsegmented_paths
    }
    collisions = sorted(set(segment_groups) & unsegmented_outputs)
    if collisions:
        raise ValueError(f"Segment outputs collide with unsegmented motions: {collisions}")

    output_motions = output_path / motions_subdir
    output_motions.mkdir(parents=True)
    total_frames = 0
    boundary_rows: list[dict[str, Any]] = []

    for relative_path, segment_paths in tqdm(
        segment_groups.items(), desc="Converting segmented GMR motions", unit="motion"
    ):
        motion, boundaries = concatenate_gmr_segments(segment_paths)
        qpos = gmr_to_qpos(motion)
        save_motion(output_motions / relative_path, qpos)
        total_frames += len(qpos)
        boundary_rows.extend(
            {"motion": relative_path.as_posix(), **row} for row in boundaries
        )

    for source_path in tqdm(
        unsegmented_paths, desc="Converting GMR motions", unit="motion"
    ):
        relative_path = source_path.relative_to(input_motions)
        qpos = gmr_to_qpos(load_gmr_motion(source_path))
        save_motion(output_motions / relative_path, qpos)
        total_frames += len(qpos)

    output_motions_count = len(segment_groups) + len(unsegmented_paths)
    qpos_names = list(manifest.payload["qpos_names"])
    expected_qpos_dim = len(qpos_names)
    for motion_path in output_motions.rglob("*.npz"):
        with np.load(motion_path, allow_pickle=False) as archive:
            if archive["qpos"].shape[1] != expected_qpos_dim:
                raise ValueError(
                    f"Converted qpos width {archive['qpos'].shape[1]} does not match "
                    f"manifest qpos_dim={expected_qpos_dim} in {motion_path}"
                )

    source = dict(manifest.payload.get("source", {}))
    source["gmr_conversion"] = {
        "input_path": str(input_path),
        "input_motion_files": sum(len(paths) for paths in segment_groups.values())
        + len(unsegmented_paths),
        "concatenated_segment_files": sum(
            len(paths) for paths in segment_groups.values()
        ),
        "concatenated_motions": len(segment_groups),
        "unsegmented_motions": len(unsegmented_paths),
        "output_motions": output_motions_count,
        "root_quaternion_conversion": "xyzw -> normalized wxyz",
    }
    write_manifest(
        output_path,
        dataset_name=manifest.dataset_name,
        mjcf=manifest.mjcf,
        timestep=manifest.timestep,
        qpos_names=qpos_names,
        num_motions=output_motions_count,
        source=source,
        total_hours=total_frames * manifest.timestep / 3600.0,
    )

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        **source["gmr_conversion"],
        "total_frames": total_frames,
        "total_hours": total_frames * manifest.timestep / 3600.0,
        "boundary_count": len(boundary_rows),
        "max_root_pos_delta": max(
            (row["root_pos_delta"] for row in boundary_rows), default=0.0
        ),
        "max_root_rot_angle": max(
            (row["root_rot_angle"] for row in boundary_rows), default=0.0
        ),
        "max_dof_pos_delta": max(
            (row["dof_pos_max_delta"] for row in boundary_rows), default=0.0
        ),
    }
    (output_path / "gmr_conversion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_path / "gmr_segment_boundaries.json").write_text(
        json.dumps(boundary_rows, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = _parse_args()
    print(json.dumps(convert_dataset(args.input_path, args.output_path), indent=2))


if __name__ == "__main__":
    main()
