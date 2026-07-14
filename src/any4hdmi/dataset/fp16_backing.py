from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time

import torch
from tensordict import MemoryMappedTensor

from any4hdmi.dataset.fk_cache import FKCacheEntry


FP16_BACKING_VERSION = 1
FP16_BACKING_DTYPE = torch.float16
FP16_BACKING_CHUNK_FRAMES = 262144
FP16_BACKING_TIMEOUT_S = 3600
_BODY_FIELDS = (
    "body_pos_w",
    "body_lin_vel_w",
    "body_quat_w",
    "body_ang_vel_w",
)
_JOINT_FIELDS = ("joint_pos", "joint_vel")


def _indices_for_names(
    available: list[str], requested: list[str] | None, *, label: str
) -> tuple[list[str], list[int]]:
    names = available if requested is None else requested
    name_to_index = {name: index for index, name in enumerate(available)}
    missing = [name for name in names if name not in name_to_index]
    if missing:
        raise ValueError(
            f"Requested {label} names are missing from motion cache: {missing}"
        )
    return list(names), [name_to_index[name] for name in names]


def _backing_key(
    *, body_names: list[str], joint_names: list[str], dtype: torch.dtype
) -> str:
    payload = json.dumps(
        {
            "version": FP16_BACKING_VERSION,
            "body_names": body_names,
            "joint_names": joint_names,
            "dtype": str(dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _open_backing_fields(
    root: Path,
    *,
    source_fields: dict[str, torch.Tensor],
    body_count: int,
    joint_count: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    total_length = int(source_fields["motion_id"].shape[0])
    fields = {
        "motion_id": source_fields["motion_id"],
        "step": source_fields["step"],
    }
    for field_name in _BODY_FIELDS:
        tail = (body_count, 4 if field_name == "body_quat_w" else 3)
        fields[field_name] = MemoryMappedTensor.from_filename(
            str(root / f"{field_name}.memmap"),
            dtype=dtype,
            shape=(total_length, *tail),
        )
    for field_name in _JOINT_FIELDS:
        fields[field_name] = MemoryMappedTensor.from_filename(
            str(root / f"{field_name}.memmap"),
            dtype=dtype,
            shape=(total_length, joint_count),
        )
    return fields


def _write_backing(
    root: Path,
    *,
    source_fields: dict[str, torch.Tensor],
    body_indices: list[int],
    joint_indices: list[int],
    body_names: list[str],
    joint_names: list[str],
    dtype: torch.dtype,
) -> None:
    total_length = int(source_fields["motion_id"].shape[0])
    chunk_frames = FP16_BACKING_CHUNK_FRAMES
    body_index = torch.tensor(body_indices, dtype=torch.long)
    joint_index = torch.tensor(joint_indices, dtype=torch.long)
    destinations = {}
    for field_name in _BODY_FIELDS:
        source = source_fields[field_name]
        destinations[field_name] = MemoryMappedTensor.empty(
            (total_length, len(body_indices), source.shape[-1]),
            dtype=dtype,
            filename=str(root / f"{field_name}.memmap"),
        )
    for field_name in _JOINT_FIELDS:
        destinations[field_name] = MemoryMappedTensor.empty(
            (total_length, len(joint_indices)),
            dtype=dtype,
            filename=str(root / f"{field_name}.memmap"),
        )

    for start in range(0, total_length, chunk_frames):
        end = min(start + chunk_frames, total_length)
        for field_name in _BODY_FIELDS:
            chunk = source_fields[field_name][start:end].index_select(1, body_index)
            destinations[field_name][start:end].copy_(chunk.to(dtype=dtype))
        for field_name in _JOINT_FIELDS:
            chunk = source_fields[field_name][start:end].index_select(1, joint_index)
            destinations[field_name][start:end].copy_(chunk.to(dtype=dtype))

    (root / "meta.json").write_text(
        json.dumps(
            {
                "version": FP16_BACKING_VERSION,
                "dtype": str(dtype),
                "total_length": total_length,
                "body_names": body_names,
                "joint_names": joint_names,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (root / "ready.flag").touch()


def prepare_fp16_cache_entry(
    entry: FKCacheEntry,
    *,
    body_names: list[str] | None,
    joint_names: list[str] | None,
) -> FKCacheEntry:
    selected_body_names, body_indices = _indices_for_names(
        entry.body_names, body_names, label="body"
    )
    selected_joint_names, joint_indices = _indices_for_names(
        entry.joint_names, joint_names, label="joint"
    )
    dtype = FP16_BACKING_DTYPE
    key = _backing_key(
        body_names=selected_body_names,
        joint_names=selected_joint_names,
        dtype=dtype,
    )
    root = entry.cache_entry_dir / f"windowed_backing_{key}"
    ready = root / "ready.flag"
    lock = root.with_name(root.name + ".lock")

    if not ready.is_file():
        deadline = time.monotonic() + FP16_BACKING_TIMEOUT_S
        acquired = False
        while not ready.is_file() and not acquired:
            try:
                lock.mkdir()
                acquired = True
                break
            except FileExistsError:
                owner_path = lock / "owner.json"
                try:
                    owner = json.loads(owner_path.read_text())
                    owner_pid = int(owner["pid"])
                    same_host = owner.get("host") == socket.gethostname()
                except (
                    FileNotFoundError,
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    same_host = False
                    owner_pid = -1
                owner_alive = True
                if same_host:
                    try:
                        os.kill(owner_pid, 0)
                    except ProcessLookupError:
                        owner_alive = False
                    except PermissionError:
                        pass
                if same_host and not owner_alive:
                    print(
                        "[any4hdmi][backing] recovering stale lock"
                        f" pid={owner_pid} path={lock}"
                    )
                    shutil.rmtree(lock, ignore_errors=True)
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for {ready}")
                time.sleep(0.5)

        if acquired:
            tmp = root.with_name(f"{root.name}.tmp-{os.getpid()}")
            try:
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir()
                (lock / "owner.json").write_text(
                    json.dumps({"host": socket.gethostname(), "pid": os.getpid()})
                )
                print(
                    "[any4hdmi][backing] building shared FP16 backing "
                    f"at {root}"
                )
                _write_backing(
                    tmp,
                    source_fields=entry.storage_fields,
                    body_indices=body_indices,
                    joint_indices=joint_indices,
                    body_names=selected_body_names,
                    joint_names=selected_joint_names,
                    dtype=dtype,
                )
                if root.exists():
                    shutil.rmtree(root)
                tmp.rename(root)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.rmtree(lock, ignore_errors=True)

    fields = _open_backing_fields(
        root,
        source_fields=entry.storage_fields,
        body_count=len(selected_body_names),
        joint_count=len(selected_joint_names),
        dtype=dtype,
    )
    return FKCacheEntry(
        cache_entry_dir=entry.cache_entry_dir,
        body_names=selected_body_names,
        joint_names=selected_joint_names,
        motion_paths=entry.motion_paths,
        starts=entry.starts,
        ends=entry.ends,
        storage_fields=fields,
        motion_id_offset=entry.motion_id_offset,
    )
