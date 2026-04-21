from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from any4hdmi.core.format import MANIFEST_NAME
from tqdm import tqdm

LEGACY_MOTION_NAME = "motion.npz"
LEGACY_META_NAME = "meta.json"
HF_DATASET_SCHEME = "hf://"
DEFAULT_HF_DATASET_REVISION = "main"
DatasetKind = Literal["any4hdmi", "legacy"]


@dataclass(frozen=True)
class DatasetContext:
    dataset_kind: DatasetKind
    dataset_root: Path
    motion_paths: list[Path]
    manifest: dict[str, Any] | None = None
    legacy_meta: dict[str, Any] | None = None


def _parse_hf_dataset_reference(root_path: str) -> tuple[str, str, str]:
    raw_reference = root_path[len(HF_DATASET_SCHEME) :]
    parts = raw_reference.split("/", 2)
    if len(parts) < 2:
        raise ValueError(
            "Invalid Hugging Face dataset URI. Expected "
            "'hf://<namespace>/<repo>' or "
            "'hf://<namespace>/<repo>@<revision>/<path>'."
        )

    namespace = parts[0]
    repo_and_revision = parts[1]
    repo_path = parts[2] if len(parts) == 3 else ""
    repo_name, sep, revision = repo_and_revision.partition("@")
    repo_id = f"{namespace}/{repo_name}"
    if not sep:
        revision = DEFAULT_HF_DATASET_REVISION

    if not namespace or not repo_name:
        raise ValueError("Invalid Hugging Face dataset URI. namespace and repo must both be non-empty.")
    return repo_id, revision, repo_path


def _snapshot_download(*, repo_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
    ).resolve()


def _resolve_hf_dataset_path(root_path: str) -> Path:
    repo_id, revision, repo_path = _parse_hf_dataset_reference(root_path)
    snapshot_root = _snapshot_download(repo_id=repo_id, revision=revision)
    resolved_path = (snapshot_root / repo_path) if repo_path else snapshot_root
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Dataset path {repo_path or '.'!r} was not found in Hugging Face dataset repo "
            f"{repo_id!r} at revision {revision!r}"
        )
    return resolved_path.absolute()


def resolve_input_paths(base_dir: Path, root_path: str | list[str] | Path | list[Path]) -> list[Path]:
    raw_paths = [root_path] if isinstance(root_path, (str, Path)) else list(root_path)

    resolved_paths: list[Path] = []
    for raw_path in raw_paths:
        if isinstance(raw_path, str) and raw_path.startswith(HF_DATASET_SCHEME):
            resolved_paths.append(_resolve_hf_dataset_path(raw_path))
            continue

        path = Path(raw_path)
        expanded = path.expanduser()
        if not expanded.is_absolute():
            expanded = base_dir / expanded
        resolved_paths.append(expanded.resolve())
    return resolved_paths


def find_any4hdmi_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
    return None


def load_any4hdmi_manifest(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / MANIFEST_NAME).read_text(encoding="utf-8"))


def _resolve_any4hdmi_dataset_root(input_paths: list[Path]) -> tuple[Path, dict[str, Any]]:
    dataset_root: Path | None = None
    dataset_manifest: dict[str, Any] | None = None

    for input_path in input_paths:
        current_root = find_any4hdmi_root(input_path)
        if current_root is None:
            raise RuntimeError(f"Could not find {MANIFEST_NAME} above {input_path}")
        if dataset_root is None:
            dataset_root = current_root
            dataset_manifest = load_any4hdmi_manifest(current_root)
        elif current_root != dataset_root:
            raise ValueError(
                f"All any4hdmi inputs must belong to one dataset root, got {dataset_root} and {current_root}"
            )

    if dataset_root is None or dataset_manifest is None:
        raise RuntimeError("Failed to resolve any4hdmi dataset root")
    return dataset_root, dataset_manifest


def _collect_any4hdmi_motion_paths(
    *,
    dataset_root: Path,
    dataset_manifest: dict[str, Any],
    input_paths: list[Path],
) -> list[Path]:
    motion_paths: set[Path] = set()
    motions_root = dataset_root / dataset_manifest.get("motions_subdir", "motions")

    for input_path in input_paths:
        if input_path.is_file():
            if input_path.suffix != ".npz":
                raise ValueError(f"Expected a .npz motion file under any4hdmi root, got {input_path}")
            motion_paths.add(input_path.absolute())
            continue

        scan_root = motions_root if input_path == dataset_root else input_path
        motion_paths.update(
            path.absolute()
            for path in tqdm(scan_root.rglob("*.npz"), desc=f"Scanning {scan_root.name}", unit="file")
        )

    if not motion_paths:
        motion_paths.update(
            path.absolute()
            for path in tqdm(motions_root.rglob("*.npz"), desc=f"Scanning {motions_root.name}", unit="file")
        )
    motion_paths_list = sorted(motion_paths)
    if not motion_paths_list:
        raise RuntimeError(f"No qpos motions found under {dataset_root}")
    return motion_paths_list


def resolve_legacy_motion_paths(input_paths: list[Path]) -> list[Path]:
    motion_paths: set[Path] = set()
    for input_path in input_paths:
        if input_path.is_file():
            if input_path.name != LEGACY_MOTION_NAME:
                raise ValueError(
                    f"Expected a legacy {LEGACY_MOTION_NAME} file, got {input_path}"
                )
            motion_paths.add(input_path.absolute())
            continue

        motion_paths.update(
            path.absolute()
            for path in tqdm(
                input_path.rglob(LEGACY_MOTION_NAME),
                desc=f"Scanning {input_path.name or input_path}",
                unit="file",
            )
        )

    motion_paths_list = sorted(motion_paths)
    if not motion_paths_list:
        raise RuntimeError(f"No legacy {LEGACY_MOTION_NAME} files found under {input_paths}")
    return motion_paths_list


def load_legacy_meta(motion_paths: list[Path]) -> dict[str, Any]:
    metas = []
    for motion_path in motion_paths:
        meta_path = motion_path.parent / LEGACY_META_NAME
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        meta.pop("length", None)
        metas.append(meta)

    for idx, meta in enumerate(metas[1:], start=1):
        if meta != metas[0]:
            raise ValueError(
                f"{LEGACY_META_NAME} in {motion_paths[idx].parent} differs from {motion_paths[0].parent}"
            )
    return metas[0]


def _common_parent(paths: list[Path]) -> Path:
    return Path(os.path.commonpath([str(path.absolute()) for path in paths])).absolute()


def resolve_dataset_context(input_paths: list[Path]) -> DatasetContext:
    dataset_roots = [find_any4hdmi_root(path) for path in input_paths]
    if all(root is not None for root in dataset_roots):
        dataset_root, dataset_manifest = _resolve_any4hdmi_dataset_root(input_paths)
        motion_paths = _collect_any4hdmi_motion_paths(
            dataset_root=dataset_root,
            dataset_manifest=dataset_manifest,
            input_paths=input_paths,
        )
        return DatasetContext(
            dataset_kind="any4hdmi",
            dataset_root=dataset_root,
            motion_paths=motion_paths,
            manifest=dataset_manifest,
        )

    if any(root is not None for root in dataset_roots):
        raise ValueError("Cannot mix any4hdmi dataset inputs with legacy motion.npz inputs")

    motion_paths = resolve_legacy_motion_paths(input_paths)
    return DatasetContext(
        dataset_kind="legacy",
        dataset_root=_common_parent([path.parent for path in motion_paths]),
        motion_paths=motion_paths,
        legacy_meta=load_legacy_meta(motion_paths),
    )


def resolve_any4hdmi_dataset_context(input_paths: list[Path]) -> tuple[Path, dict[str, Any]]:
    return _resolve_any4hdmi_dataset_root(input_paths)


def resolve_any4hdmi_motion_paths(input_paths: list[Path]) -> tuple[Path, dict[str, Any], list[Path]]:
    dataset_root, dataset_manifest = _resolve_any4hdmi_dataset_root(input_paths)
    return (
        dataset_root,
        dataset_manifest,
        _collect_any4hdmi_motion_paths(
            dataset_root=dataset_root,
            dataset_manifest=dataset_manifest,
            input_paths=input_paths,
        ),
    )


def resolve_source_fps(manifest: dict[str, Any]) -> float:
    source_fps = float(manifest.get("fps", 0.0))
    if source_fps > 0.0:
        return source_fps
    timestep = float(manifest.get("timestep", 0.0))
    if timestep <= 0.0:
        raise ValueError("any4hdmi manifest must contain fps or timestep")
    return 1.0 / timestep
