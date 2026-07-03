from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

from any4hdmi.core.format import MOTION_DTYPE, MOTIONS_SUBDIR, repo_root, save_motion, write_manifest
from any4hdmi.core.model import base_qpos_adr, joint_qpos_adrs, load_model
from any4hdmi.utils.mjcf import (
    qpos_names_from_model,
    resolve_mjcf_path,
)


DEFAULT_INPUT_PATH = Path.home() / "Downloads" / "100style.tar"
DEFAULT_DATASET_NAME = "100style"
DEFAULT_MJCF = "hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml"
DEFAULT_SOURCE_ROOT_NAME = "100STYLE"
_TORCH_DTYPE_TO_NUMPY: dict[str, np.dtype[Any]] = {
    "torch.float16": np.float16,
    "torch.float32": np.float32,
    "torch.int32": np.int32,
    "torch.int64": np.int64,
}


@dataclass(frozen=True)
class SegmentRecord:
    source_path: str
    segment_start: int
    segment_end: int
    frame_start: int
    frame_end: int


@dataclass(frozen=True)
class MotionRecord:
    source_path: str
    segment_start: int
    segment_end: int
    segments: tuple[SegmentRecord, ...]
    use_span_suffix: bool = False

    @property
    def frame_count(self) -> int:
        return sum(segment.frame_end - segment.frame_start for segment in self.segments)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an Axellwppr MotionDataset tarball or extracted directory into the any4hdmi qpos format."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to 100style.tar or an extracted 100style directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="output/100style",
        help="Output dataset root for converted motions.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name written to manifest.json.",
    )
    parser.add_argument(
        "--source-root-name",
        default=DEFAULT_SOURCE_ROOT_NAME,
        help="Path component used to strip absolute source prefixes from id_label.json paths.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Frame rate of the MotionDataset export. generate_dataset.py uses 50 fps by default.",
    )
    parser.add_argument(
        "--mjcf",
        default=DEFAULT_MJCF,
        help="Local MJCF/XML path or hf:// reference.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_mjcf_argument(mjcf: str) -> str | Path:
    if mjcf.startswith("hf://"):
        return mjcf
    return Path(mjcf).expanduser().resolve()


def _torch_dtype_to_numpy(name: str) -> np.dtype[Any]:
    try:
        return _TORCH_DTYPE_TO_NUMPY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported TensorDict dtype {name!r}") from exc


def _resolve_dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = [path]
    if path.is_dir():
        candidates.extend(child for child in path.iterdir() if child.is_dir())
    for candidate in candidates:
        if (candidate / "meta_motion.json").is_file() and (candidate / "_tensordict" / "meta.json").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find extracted 100style dataset root under {path}")


def _ensure_within(root: Path, candidate: Path) -> None:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Archive member escapes extraction root: {candidate}") from exc


def _extract_tar_to_temp(tar_path: Path, temp_root: Path) -> Path:
    with tarfile.open(tar_path, mode="r:*") as archive:
        for member in archive.getmembers():
            if member.islnk() or member.issym():
                raise ValueError(f"Refusing to extract links from archive: {member.name}")
            destination = temp_root / member.name
            _ensure_within(temp_root, destination)
        archive.extractall(temp_root)
    return _resolve_dataset_root(temp_root)


@contextmanager
def _prepared_dataset_root(input_path: Path) -> Iterator[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_dir():
        yield _resolve_dataset_root(input_path)
        return
    if not input_path.is_file():
        raise FileNotFoundError(f"100STYLE input not found: {input_path}")
    if not tarfile.is_tarfile(input_path):
        raise ValueError(f"Expected --input to be a tarball or extracted directory, got {input_path}")
    with tempfile.TemporaryDirectory(prefix="any4hdmi-100style-") as temp_dir:
        yield _extract_tar_to_temp(input_path, Path(temp_dir))


def _load_tensordict_field(dataset_root: Path, field_name: str, td_meta: dict[str, Any]) -> np.memmap:
    field_meta = td_meta[field_name]
    return np.memmap(
        dataset_root / "_tensordict" / f"{field_name}.memmap",
        mode="r",
        dtype=_torch_dtype_to_numpy(str(field_meta["dtype"])),
        shape=tuple(int(dim) for dim in field_meta["shape"]),
    )


def _build_segment_records(meta_motion: dict[str, Any], id_labels: list[dict[str, Any]]) -> list[SegmentRecord]:
    starts = [int(value) for value in meta_motion["starts"]]
    ends = [int(value) for value in meta_motion["ends"]]
    if len(starts) != len(ends):
        raise ValueError(f"Expected starts/ends to match, got {len(starts)} and {len(ends)}")
    if len(starts) != len(id_labels):
        raise ValueError(f"Expected id_label length {len(id_labels)} to match {len(starts)} motion segments")

    records: list[SegmentRecord] = []
    for frame_start, frame_end, label in zip(starts, ends, id_labels, strict=True):
        segment_start = int(label["segment_start"])
        segment_end = int(label["segment_end"])
        if frame_end <= frame_start:
            raise ValueError(f"Invalid frame span [{frame_start}, {frame_end})")
        if segment_end <= segment_start:
            raise ValueError(f"Invalid source span [{segment_start}, {segment_end})")
        if frame_end - frame_start != segment_end - segment_start:
            raise ValueError(
                "MotionDataset frame span does not match source segment span: "
                f"{frame_start}:{frame_end} vs {segment_start}:{segment_end}"
            )
        records.append(
            SegmentRecord(
                source_path=str(label["source_path"]),
                segment_start=segment_start,
                segment_end=segment_end,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        )
    return records


def _build_motion_records(records: list[SegmentRecord]) -> list[MotionRecord]:
    groups: dict[str, list[SegmentRecord]] = {}
    source_order: list[str] = []
    for record in records:
        if record.source_path not in groups:
            groups[record.source_path] = []
            source_order.append(record.source_path)
        groups[record.source_path].append(record)

    motion_records: list[MotionRecord] = []
    for source_path in source_order:
        segments = sorted(groups[source_path], key=lambda item: (item.segment_start, item.segment_end))
        runs: list[list[SegmentRecord]] = [[segments[0]]]
        for current in segments[1:]:
            previous = runs[-1][-1]
            if previous.segment_end == current.segment_start:
                runs[-1].append(current)
            else:
                runs.append([current])

        use_span_suffix_for_source = len(runs) > 1
        for run in runs:
            use_span_suffix = use_span_suffix_for_source or run[0].segment_start != 0
            motion_records.append(
                MotionRecord(
                    source_path=source_path,
                    segment_start=run[0].segment_start,
                    segment_end=run[-1].segment_end,
                    segments=tuple(run),
                    use_span_suffix=use_span_suffix,
                )
            )
    return motion_records


def _normalize_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=MOTION_DTYPE)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Encountered a zero-norm root quaternion in 100STYLE input")
    return np.asarray(quaternions / norms, dtype=MOTION_DTYPE)


def _build_qpos_sequence(
    model,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    *,
    base_adr: int,
    hinge_qpos_adrs: np.ndarray,
) -> np.ndarray:
    frames = int(root_pos.shape[0])
    if root_quat_wxyz.shape != (frames, 4):
        raise ValueError(f"Expected root_quat_wxyz shape {(frames, 4)}, got {root_quat_wxyz.shape}")
    if joint_pos.shape != (frames, hinge_qpos_adrs.shape[0]):
        raise ValueError(
            f"Expected joint_pos shape {(frames, hinge_qpos_adrs.shape[0])}, got {joint_pos.shape}"
        )

    qpos = np.zeros((frames, model.nq), dtype=MOTION_DTYPE)
    qpos[:, base_adr : base_adr + 3] = np.asarray(root_pos, dtype=MOTION_DTYPE)
    qpos[:, base_adr + 3 : base_adr + 7] = _normalize_quaternions_wxyz(root_quat_wxyz)
    qpos[:, hinge_qpos_adrs] = np.asarray(joint_pos, dtype=MOTION_DTYPE)
    return qpos


def _source_relative_path(source_path: str, source_root_name: str) -> Path:
    raw_path = Path(source_path)
    parts = raw_path.parts
    for idx, part in enumerate(parts):
        if part.lower() == source_root_name.lower():
            relative = Path(*parts[idx + 1 :])
            if relative.parts:
                return relative
            break
    return Path(raw_path.name)


def _output_rel_path(record: MotionRecord, source_root_name: str) -> Path:
    source_rel = _source_relative_path(record.source_path, source_root_name)
    if not record.use_span_suffix:
        return source_rel
    filename = f"{source_rel.stem}__{record.segment_start:06d}_{record.segment_end:06d}.npz"
    return source_rel.parent / filename


def _validate_meta(
    *,
    meta_motion: dict[str, Any],
    td_meta: dict[str, Any],
    records: list[SegmentRecord],
    root_pos_w: np.memmap,
    root_quat_w: np.memmap,
    joint_pos: np.memmap,
) -> None:
    expected_frames = int(td_meta["shape"][0])
    if root_pos_w.shape != (expected_frames, 3):
        raise ValueError(f"Expected root_pos_w shape {(expected_frames, 3)}, got {root_pos_w.shape}")
    if root_quat_w.shape != (expected_frames, 4):
        raise ValueError(f"Expected root_quat_w shape {(expected_frames, 4)}, got {root_quat_w.shape}")
    expected_joints = len(meta_motion["joint_names"])
    if joint_pos.shape != (expected_frames, expected_joints):
        raise ValueError(f"Expected joint_pos shape {(expected_frames, expected_joints)}, got {joint_pos.shape}")
    max_frame_end = max((record.frame_end for record in records), default=0)
    if max_frame_end > expected_frames:
        raise ValueError(
            f"Segment frame range exceeds TensorDict storage: {max_frame_end} > {expected_frames}"
        )


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir = out_dir.resolve()

    mjcf_reference = _resolve_mjcf_argument(str(args.mjcf))
    mjcf_path = resolve_mjcf_path(mjcf_reference)
    model = load_model(mjcf_path)
    qpos_names = qpos_names_from_model(model)
    base_adr = base_qpos_adr(model)

    with _prepared_dataset_root(Path(args.input)) as dataset_root:
        meta_motion = _load_json(dataset_root / "meta_motion.json")
        td_meta = _load_json(dataset_root / "_tensordict" / "meta.json")
        id_labels = _load_json(dataset_root / "id_label.json")
        records = _build_segment_records(meta_motion, id_labels)
        motion_records = _build_motion_records(records)
        hinge_qpos_adrs = joint_qpos_adrs(model, list(meta_motion["joint_names"]))

        root_pos_w = _load_tensordict_field(dataset_root, "root_pos_w", td_meta)
        root_quat_w = _load_tensordict_field(dataset_root, "root_quat_w", td_meta)
        joint_pos = _load_tensordict_field(dataset_root, "joint_pos", td_meta)
        _validate_meta(
            meta_motion=meta_motion,
            td_meta=td_meta,
            records=records,
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            joint_pos=joint_pos,
        )

        total_frames = 0
        for record in tqdm(motion_records, desc="Converting source motions", unit="motion"):
            qpos_chunks = [
                _build_qpos_sequence(
                    model,
                    np.asarray(root_pos_w[segment.frame_start : segment.frame_end], dtype=MOTION_DTYPE),
                    np.asarray(root_quat_w[segment.frame_start : segment.frame_end], dtype=MOTION_DTYPE),
                    np.asarray(joint_pos[segment.frame_start : segment.frame_end], dtype=MOTION_DTYPE),
                    base_adr=base_adr,
                    hinge_qpos_adrs=hinge_qpos_adrs,
                )
                for segment in record.segments
            ]
            qpos = qpos_chunks[0] if len(qpos_chunks) == 1 else np.concatenate(qpos_chunks, axis=0)
            if qpos.shape[0] != record.frame_count:
                raise ValueError(f"Concatenated qpos length mismatch for {record.source_path}")
            rel_path = _output_rel_path(record, args.source_root_name)
            save_motion(out_dir / MOTIONS_SUBDIR / rel_path, qpos)
            total_frames += int(qpos.shape[0])

    source_payload = {
        "input": str(Path(args.input).expanduser()),
        "fps": args.fps,
        "format": "axell_motiondataset_memmap",
        "segment_naming": "source_path; contiguous chunks concatenated",
        "source_root_name": args.source_root_name,
        "num_input_segments": len(records),
    }
    write_manifest(
        out_dir,
        dataset_name=args.dataset_name,
        mjcf=mjcf_reference,
        timestep=1.0 / args.fps,
        qpos_names=qpos_names,
        num_motions=len(motion_records),
        total_hours=total_frames / args.fps / 3600.0,
        source=source_payload,
    )


if __name__ == "__main__":
    main()
