from __future__ import annotations

import json
import hashlib
import os
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from any4hdmi.core.format import load_manifest
from any4hdmi.fk.runner import FKRunner
from any4hdmi.limmt.common import motion_paths_for_root
from any4hdmi.limmt.hme.features import (
    HME_CACHE_FEATURE_TYPE,
    HME_FEATURE_TYPE,
    compute_win_len,
    hme_features_from_raw,
    motion_frame_features_from_fk,
    normalize_motion_features,
    window_indices,
)
from any4hdmi.utils.dataset import (
    DEFAULT_MOTION_LOADER_NUM_WORKERS,
    DEFAULT_MOTION_LOADER_PREFETCH_FACTOR,
    build_motion_loader,
)
from any4hdmi.utils.mjcf import resolve_mjcf_path


HME_CACHE_VERSION = 2
HME_CACHE_INDEX_NAME = "index.json"
HME_CACHE_META_NAME = "cache_meta.json"
HME_CACHE_READY_NAME = "ready.flag"
HME_CACHE_SOURCE_KEY_NAME = "source_cache_key"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _cache_build_device() -> torch.device | None:
    raw = os.environ.get("ANY4HDMI_HME_CACHE_BUILD_DEVICE")
    if raw is None:
        raw = os.environ.get("ANY4HDMI_CACHE_BUILD_DEVICE")
    if raw is None:
        return None
    return torch.device(raw)


def _cache_build_num_workers() -> int:
    raw = os.environ.get("ANY4HDMI_HME_CACHE_BUILD_NUM_WORKERS")
    if raw is None:
        raw = os.environ.get("ANY4HDMI_CACHE_BUILD_NUM_WORKERS")
    if raw is None:
        return DEFAULT_MOTION_LOADER_NUM_WORKERS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MOTION_LOADER_NUM_WORKERS


def _cache_build_prefetch_factor() -> int:
    raw = os.environ.get("ANY4HDMI_HME_CACHE_BUILD_PREFETCH_FACTOR")
    if raw is None:
        raw = os.environ.get("ANY4HDMI_CACHE_BUILD_PREFETCH_FACTOR")
    if raw is None:
        return DEFAULT_MOTION_LOADER_PREFETCH_FACTOR
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MOTION_LOADER_PREFETCH_FACTOR


def _cache_build_multiprocessing_context(num_workers: int) -> str | None:
    if int(num_workers) <= 0:
        return None
    raw = os.environ.get("ANY4HDMI_HME_CACHE_BUILD_MULTIPROCESSING_CONTEXT")
    if raw is None:
        raw = os.environ.get("ANY4HDMI_CACHE_BUILD_MULTIPROCESSING_CONTEXT")
    if raw is not None:
        value = raw.strip()
        return value or None
    if os.name == "posix" and sys.platform.startswith("linux"):
        return "fork"
    return None


def _cache_mjcf_path(manifest: Any) -> Path:
    raw = os.environ.get("ANY4HDMI_HME_MJCF_PATH")
    if raw is None:
        raw = os.environ.get("ANY4HDMI_MJCF_PATH")
    if raw is not None:
        return resolve_mjcf_path(raw, dataset_root=manifest.root)
    return manifest.mjcf_path


def _cache_feature_path(cache_dir: Path, rel_motion: str) -> Path:
    return cache_dir / "features" / f"{rel_motion}.npy"


def _lock_dir(cache_dir: Path) -> Path:
    return cache_dir.parent / f"{cache_dir.name}.lock"


def _ready_flag(cache_dir: Path) -> Path:
    return cache_dir / HME_CACHE_READY_NAME


def _index_path(cache_dir: Path) -> Path:
    return cache_dir / HME_CACHE_INDEX_NAME


def _stat_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _content_fingerprint(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return {
        "size": int(path.stat().st_size),
        "sha256": hasher.hexdigest(),
    }


def _fingerprint_motion_entry(motion_path: Path) -> dict[str, Any]:
    entry = {"motion": _stat_fingerprint(motion_path)}
    sidecar_path = motion_path.with_suffix(".json")
    if sidecar_path.is_file():
        entry["sidecar"] = _stat_fingerprint(sidecar_path)
    return entry


def _start_hme_cache_source_key(
    *,
    dataset_root: Path,
    manifest_path: Path,
    mjcf_path: Path,
) -> "hashlib._Hash":
    payload = {
        "cache_version": HME_CACHE_VERSION,
        "dataset_root": str(dataset_root),
        "manifest": _stat_fingerprint(manifest_path),
        "mjcf": _content_fingerprint(mjcf_path),
    }
    hasher = hashlib.sha256()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    hasher.update(encoded)
    return hasher


def _update_hme_cache_source_key(hasher: "hashlib._Hash", motion_path: Path) -> None:
    encoded = json.dumps(_fingerprint_motion_entry(motion_path), sort_keys=True, separators=(",", ":")).encode("utf-8")
    hasher.update(b"\0")
    hasher.update(encoded)


def _finish_hme_cache_source_key(hasher: "hashlib._Hash") -> str:
    return hasher.hexdigest()[:16]


def _make_hme_cache_source_key(
    *,
    dataset_root: Path,
    manifest_path: Path,
    mjcf_path: Path,
    motion_paths: list[Path],
) -> str:
    hasher = _start_hme_cache_source_key(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        mjcf_path=mjcf_path,
    )
    for motion_path in motion_paths:
        _update_hme_cache_source_key(hasher, motion_path)
    return _finish_hme_cache_source_key(hasher)


def _make_hme_cache_source_key_for_root(dataset_root: Path) -> str:
    manifest = load_manifest(dataset_root)
    mjcf_path = _cache_mjcf_path(manifest)
    return _make_hme_cache_source_key(
        dataset_root=dataset_root,
        manifest_path=manifest.path,
        mjcf_path=mjcf_path,
        motion_paths=motion_paths_for_root(dataset_root),
    )


def _acquire_cache_lock(lock_dir: Path, ready_flag: Path, timeout_s: float = 600.0) -> bool:
    start_time = time.monotonic()
    while True:
        if ready_flag.is_file():
            return False
        try:
            lock_dir.mkdir(parents=False, exist_ok=False)
            return True
        except FileExistsError:
            if time.monotonic() - start_time > timeout_s:
                raise TimeoutError(f"Timed out waiting for HME cache lock {lock_dir}")
            time.sleep(0.5)


def _write_feature_file(
    *,
    feature_path: Path,
    raw_features: np.ndarray,
    window_ids: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_memmap = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=expected_shape,
    )
    for start_idx in range(0, window_ids.shape[0], 2048):
        end_idx = min(start_idx + 2048, window_ids.shape[0])
        feature_memmap[start_idx:end_idx] = hme_features_from_raw(raw_features, window_ids[start_idx:end_idx])
    feature_memmap.flush()
    del feature_memmap
    return np.load(feature_path, mmap_mode="r")


def _accumulate_feature_stats(
    *,
    features: np.ndarray,
    feature_sum: np.ndarray | None,
    feature_sumsq: np.ndarray | None,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if feature_sum is None:
        feature_sum = np.zeros(features.shape[-1], dtype=np.float64)
        feature_sumsq = np.zeros(features.shape[-1], dtype=np.float64)
    assert feature_sumsq is not None
    for start_idx in range(0, features.shape[0], 2048):
        feature_windows = features[start_idx : start_idx + 2048]
        flat_features = feature_windows.reshape(-1, feature_windows.shape[-1]).astype(np.float64, copy=False)
        feature_sum += flat_features.sum(axis=0)
        feature_sumsq += np.square(flat_features).sum(axis=0)
        feature_count += int(flat_features.shape[0])
    return feature_sum, feature_sumsq, feature_count


def _build_hme_feature_cache(
    *,
    dataset_root: Path,
    cache_entry_dir: Path,
    win_sec: float,
    downsample_rate: int,
    stride: int,
) -> Path:
    cache_entry_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(dataset_root)
    source_fps = 1.0 / manifest.timestep
    mjcf_path = _cache_mjcf_path(manifest)
    win_len = compute_win_len(win_sec, downsample_rate, source_fps)
    motion_paths = motion_paths_for_root(dataset_root)
    source_key_hasher = _start_hme_cache_source_key(
        dataset_root=dataset_root,
        manifest_path=manifest.path,
        mjcf_path=mjcf_path,
    )
    fk_runner = FKRunner(
        mjcf_path=mjcf_path,
        batch_size=_env_int("ANY4HDMI_HME_CACHE_BUILD_BATCH_SIZE", _env_int("ANY4HDMI_CACHE_BUILD_BATCH_SIZE", 51200)),
        device=_cache_build_device(),
    )
    num_workers = _cache_build_num_workers()
    motion_loader = build_motion_loader(
        input_root=dataset_root,
        motion_paths=motion_paths,
        mjcf_path=mjcf_path,
        fps=float(source_fps),
        num_workers=num_workers,
        prefetch_factor=_cache_build_prefetch_factor(),
        pin_memory=fk_runner.device.type == "cuda",
        multiprocessing_context=_cache_build_multiprocessing_context(num_workers),
        tensor_device=fk_runner.device,
    )

    feature_records: list[dict[str, Any]] = []
    feature_sum: np.ndarray | None = None
    feature_sumsq: np.ndarray | None = None
    feature_count = 0
    cache_feature_dim: int | None = None
    batched_meta: list[tuple[Path, torch.Tensor, torch.Tensor]] = []
    batched_qpos: list[torch.Tensor] = []
    batched_qvel: list[torch.Tensor] = []
    batched_frames = 0
    start_time = time.monotonic()
    print(
        f"building HME feature cache motions={len(motion_paths)} win_len={win_len} "
        f"feature={HME_FEATURE_TYPE} cache_dir={cache_entry_dir}",
        flush=True,
    )

    def flush_batched_motions() -> None:
        nonlocal batched_frames, cache_feature_dim, feature_count, feature_sum, feature_sumsq
        if not batched_qpos:
            return

        total_frames = sum(int(qpos.shape[0]) for qpos in batched_qpos)
        fk_start_time = time.perf_counter()
        motion_fk_outputs = fk_runner.forward_kinematics_many(batched_qpos, batched_qvel)
        fk_elapsed_s = time.perf_counter() - fk_start_time
        print(f"HME FK: {fk_elapsed_s:.2f}s for {len(batched_qpos)} motions / {total_frames} frames")

        for local_idx, (motion_path, qpos, qvel) in enumerate(batched_meta):
            rel_motion = motion_path.relative_to(dataset_root).as_posix()
            length = int(qpos.shape[0])
            window_ids = window_indices(length, win_len=win_len, downsample_rate=downsample_rate, stride=stride)
            if window_ids.shape[0] == 0:
                continue
            fk_motion = motion_fk_outputs[local_idx]
            raw_features = motion_frame_features_from_fk(qpos, qvel, fk_motion)
            expected_shape = (int(window_ids.shape[0]), int(win_len), int(raw_features.shape[-1] + 2))
            feature_path = _cache_feature_path(cache_entry_dir, rel_motion)
            features = _write_feature_file(
                feature_path=feature_path,
                raw_features=raw_features,
                window_ids=window_ids,
                expected_shape=expected_shape,
            )
            if cache_feature_dim is None:
                cache_feature_dim = int(features.shape[-1])
            elif cache_feature_dim != int(features.shape[-1]):
                raise RuntimeError(f"Inconsistent HME cache feature dim: {cache_feature_dim} vs {features.shape[-1]}")
            feature_sum, feature_sumsq, feature_count = _accumulate_feature_stats(
                features=features,
                feature_sum=feature_sum,
                feature_sumsq=feature_sumsq,
                feature_count=feature_count,
            )
            feature_records.append(
                {
                    "motion": rel_motion,
                    "feature_path": str(feature_path.relative_to(cache_entry_dir)),
                    "length": length,
                    "num_windows": int(window_ids.shape[0]),
                }
            )

        batched_meta.clear()
        batched_qpos.clear()
        batched_qvel.clear()
        batched_frames = 0

    motion_iter = iter(motion_loader)
    for motion_idx, item in enumerate(tqdm(motion_iter, total=len(motion_paths), desc="Building HME cache", unit="file"), start=1):
        motion_path = Path(item["motion_path"])
        _update_hme_cache_source_key(source_key_hasher, motion_path)
        qpos = item["qpos"]
        qvel = item["qvel"]
        motion_length = int(qpos.shape[0])
        if batched_frames > 0 and batched_frames + motion_length >= fk_runner.batch_size:
            flush_batched_motions()
        batched_meta.append((motion_path, qpos, qvel))
        batched_qpos.append(qpos)
        batched_qvel.append(qvel)
        batched_frames += motion_length
        if motion_idx % 200 == 0:
            elapsed = time.monotonic() - start_time
            print(
                f"cache_progress motions_seen={motion_idx}/{len(motion_paths)} "
                f"records={len(feature_records)} elapsed_s={elapsed:.1f}",
                flush=True,
            )
    flush_batched_motions()

    if feature_sum is None or feature_sumsq is None or feature_count == 0:
        raise RuntimeError(f"No HME features found under {dataset_root}")
    source_cache_key = _finish_hme_cache_source_key(source_key_hasher)
    feature_mean = feature_sum / float(feature_count)
    feature_var = np.maximum(feature_sumsq / float(feature_count) - np.square(feature_mean), 1e-12)
    feature_std = np.maximum(np.sqrt(feature_var), 1e-6)
    normalization_path = cache_entry_dir / "normalization.npz"
    np.savez_compressed(
        normalization_path,
        mean=feature_mean.astype(np.float32),
        std=feature_std.astype(np.float32),
        count=np.asarray(feature_count, dtype=np.int64),
    )
    cache_index_payload = {
        "cache_version": HME_CACHE_VERSION,
        HME_CACHE_SOURCE_KEY_NAME: source_cache_key,
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest.path),
        "mjcf_path": str(mjcf_path),
        "fps": source_fps,
        "win_sec": float(win_sec),
        "downsample_rate": int(downsample_rate),
        "stride": int(stride),
        "win_len": int(win_len),
        "feature_type": HME_FEATURE_TYPE,
        "cache_feature_type": HME_CACHE_FEATURE_TYPE,
        "cache_shape": "num_windows_win_len_feature_dim",
        "cache_feature_dim": int(cache_feature_dim or 0),
        "feature_dim": int(feature_mean.shape[0]),
        "normalization": {
            "type": "empirical_mean_std",
            "path": str(normalization_path.relative_to(cache_entry_dir)),
            "count": int(feature_count),
            "mean": feature_mean.astype(float).tolist(),
            "std": feature_std.astype(float).tolist(),
        },
        "records": feature_records,
    }
    index_path = _index_path(cache_entry_dir)
    index_path.write_text(json.dumps(cache_index_payload, indent=2) + "\n", encoding="utf-8")
    cache_meta = {
        "cache_version": HME_CACHE_VERSION,
        HME_CACHE_SOURCE_KEY_NAME: source_cache_key,
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest.path),
        "mjcf_path": str(mjcf_path),
        "win_sec": float(win_sec),
        "downsample_rate": int(downsample_rate),
        "stride": int(stride),
        "fk_backend": fk_runner.backend,
    }
    (cache_entry_dir / HME_CACHE_META_NAME).write_text(json.dumps(cache_meta, indent=2) + "\n", encoding="utf-8")
    _ready_flag(cache_entry_dir).write_text("ready\n", encoding="utf-8")
    return index_path


def build_hme_feature_cache(
    *,
    dataset_root: str | Path,
    cache_dir: str | Path,
    win_sec: float,
    downsample_rate: int,
    stride: int,
) -> Path:
    root_path = Path(dataset_root).expanduser().resolve()
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    current_cache = hme_feature_cache_is_current(
        cache_path,
        dataset_root=root_path,
        win_sec=win_sec,
        downsample_rate=downsample_rate,
        stride=stride,
    )
    if current_cache:
        return _index_path(cache_path)
    if _ready_flag(cache_path).is_file():
        shutil.rmtree(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    lock_dir = _lock_dir(cache_path)
    ready_flag = _ready_flag(cache_path)
    owns_lock = _acquire_cache_lock(lock_dir, ready_flag)
    if owns_lock:
        tmp_entry_dir = cache_path.parent / f"{cache_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
        try:
            if tmp_entry_dir.exists():
                shutil.rmtree(tmp_entry_dir)
            index_path = _build_hme_feature_cache(
                dataset_root=root_path,
                cache_entry_dir=tmp_entry_dir,
                win_sec=win_sec,
                downsample_rate=downsample_rate,
                stride=stride,
            )
            if cache_path.exists():
                shutil.rmtree(cache_path)
            tmp_entry_dir.rename(cache_path)
            return cache_path / index_path.name
        finally:
            if tmp_entry_dir.exists():
                shutil.rmtree(tmp_entry_dir, ignore_errors=True)
            if lock_dir.exists():
                shutil.rmtree(lock_dir, ignore_errors=True)
    wait_for_hme_feature_cache(cache_path)
    return _index_path(cache_path)


def wait_for_hme_feature_cache(cache_dir: str | Path, *, timeout_s: float = 3600.0) -> Path:
    cache_path = Path(cache_dir).expanduser().resolve()
    ready_flag = _ready_flag(cache_path)
    index_path = _index_path(cache_path)
    start = time.monotonic()
    while not (ready_flag.is_file() and index_path.is_file()):
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for HME feature cache {cache_path}")
        time.sleep(1.0)
    return index_path


def hme_feature_cache_is_current(
    cache_dir: str | Path,
    *,
    dataset_root: str | Path | None = None,
    win_sec: float | None = None,
    downsample_rate: int | None = None,
    stride: int | None = None,
) -> bool:
    cache_path = Path(cache_dir).expanduser().resolve()
    index_path = _index_path(cache_path)
    if not index_path.is_file() or not _ready_flag(cache_path).is_file():
        return False
    try:
        cache_index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if cache_index_payload.get("cache_version") != HME_CACHE_VERSION:
        return False
    if cache_index_payload.get("feature_type") != HME_FEATURE_TYPE:
        return False
    if cache_index_payload.get("cache_feature_type") != HME_CACHE_FEATURE_TYPE:
        return False
    if dataset_root is not None:
        root_path = Path(dataset_root).expanduser().resolve()
        if Path(cache_index_payload.get("dataset_root", "")).resolve() != root_path:
            return False
        try:
            source_cache_key = _make_hme_cache_source_key_for_root(root_path)
        except Exception:
            return False
        if cache_index_payload.get(HME_CACHE_SOURCE_KEY_NAME) != source_cache_key:
            return False
    if win_sec is not None and float(cache_index_payload.get("win_sec", -1.0)) != float(win_sec):
        return False
    if downsample_rate is not None and int(cache_index_payload.get("downsample_rate", -1)) != int(downsample_rate):
        return False
    if stride is not None and int(cache_index_payload.get("stride", -1)) != int(stride):
        return False
    return True


class CachedHmeWindowDataset(Dataset[torch.Tensor]):
    def __init__(self, cache_dir: str | Path) -> None:
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_index_payload = json.loads(_index_path(cache_path).read_text(encoding="utf-8"))
        feature_type = cache_index_payload.get("feature_type")
        if feature_type != HME_FEATURE_TYPE:
            raise RuntimeError(
                f"HME feature cache {cache_path} has feature_type={feature_type!r}; expected {HME_FEATURE_TYPE!r}. "
                "Rebuild the cache."
            )
        cache_feature_type = cache_index_payload.get("cache_feature_type")
        if cache_feature_type != HME_CACHE_FEATURE_TYPE:
            raise RuntimeError(
                f"HME feature cache {cache_path} has cache_feature_type={cache_feature_type!r}; "
                f"expected {HME_CACHE_FEATURE_TYPE!r}. Rebuild the cache."
            )
        self.win_len = int(cache_index_payload["win_len"])
        self.downsample_rate = int(cache_index_payload["downsample_rate"])
        self.stride = int(cache_index_payload["stride"])
        self.file_info: list[tuple[str, int, int]] = []
        for record in cache_index_payload["records"]:
            feature_path = Path(str(record["feature_path"])).expanduser()
            if not feature_path.is_absolute():
                feature_path = cache_path / feature_path
            self.file_info.append((str(feature_path), int(record["length"]), int(record["num_windows"])))
        if "feature_dim" not in cache_index_payload:
            raise RuntimeError(f"HME feature cache {cache_path} does not contain feature_dim; rebuild the cache.")
        normalization_config = cache_index_payload.get("normalization")
        if not normalization_config or "path" not in normalization_config:
            raise RuntimeError(f"HME feature cache {cache_path} does not contain empirical normalization; rebuild the cache.")
        norm_path = Path(normalization_config["path"]).expanduser()
        if not norm_path.is_absolute():
            norm_path = cache_path / norm_path
        if norm_path.is_file():
            normalization_stats = np.load(norm_path)
            self.feature_mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
            self.feature_std = np.asarray(normalization_stats["std"], dtype=np.float32)
        else:
            raise RuntimeError(f"HME feature cache {cache_path} missing normalization file {norm_path}; rebuild the cache.")
        feature_dim = int(cache_index_payload["feature_dim"])
        if self.feature_mean.shape[0] != feature_dim or self.feature_std.shape[0] != feature_dim:
            raise RuntimeError(f"HME feature cache {cache_path} has inconsistent normalization dimension")
        self.feature_dim = feature_dim
        cumsum = [0]
        for _, _, count in self.file_info:
            cumsum.append(cumsum[-1] + count)
        self.cumsum = np.asarray(cumsum, dtype=np.int64)
        self._feature_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._feature_cache_size = 16
        if not self.file_info:
            raise RuntimeError(f"No cached HME features found under {cache_path}")

    def __len__(self) -> int:
        return int(self.cumsum[-1])

    def _load_features(self, path: str) -> np.ndarray:
        cached = self._feature_cache.get(path)
        if cached is not None:
            self._feature_cache.move_to_end(path)
            return cached
        features = np.load(path, mmap_mode="r")
        self._feature_cache[path] = features
        if len(self._feature_cache) > self._feature_cache_size:
            self._feature_cache.popitem(last=False)
        return features

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = int(np.searchsorted(self.cumsum[1:], idx, side="right"))
        local_idx = int(idx - int(self.cumsum[file_idx]))
        feature_path, _length, _ = self.file_info[file_idx]
        features = self._load_features(feature_path)
        window_features = np.asarray(features[local_idx], dtype=np.float32)
        if window_features.shape != (self.win_len, self.feature_dim):
            raise RuntimeError(
                f"HME feature cache {feature_path} returned window shape {window_features.shape}; "
                f"expected {(self.win_len, self.feature_dim)}. Rebuild the cache."
            )
        normalized = normalize_motion_features(window_features, self.feature_mean, self.feature_std)
        return torch.from_numpy(normalized.astype(np.float32, copy=False))
