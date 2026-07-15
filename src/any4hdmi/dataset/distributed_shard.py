from __future__ import annotations

from bisect import bisect_left
from dataclasses import replace

from any4hdmi.dataset.fk_cache import FKCacheEntry


def balanced_frame_boundaries(
    ends: list[int],
    *,
    world_size: int,
) -> list[int]:
    """Partition contiguous motions into approximately equal frame counts.

    Dataset runtimes sample frames uniformly. Keeping shard frame counts close
    therefore preserves the unsharded sampling distribution when every DDP rank
    contributes the same batch size. Boundaries remain motion-aligned so all
    storage fields stay contiguous views.
    """
    num_motions = len(ends)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if num_motions < world_size:
        raise ValueError(
            f"Cannot split {num_motions} motions across {world_size} ranks"
        )
    if not ends or ends[-1] <= 0:
        raise ValueError("ends must describe at least one non-empty motion")
    if any(left >= right for left, right in zip(ends, ends[1:])):
        raise ValueError("ends must be strictly increasing")

    boundaries = [0]
    total_frames = ends[-1]
    for shard_index in range(1, world_size):
        lower = boundaries[-1] + 1
        upper = num_motions - (world_size - shard_index)
        target = total_frames * shard_index / world_size
        insertion = bisect_left(ends, target, lo=lower - 1, hi=upper) + 1
        candidates = {
            lower,
            upper,
            min(upper, max(lower, insertion)),
            min(upper, max(lower, insertion - 1)),
        }
        boundary = min(
            candidates,
            key=lambda count: (
                abs(ends[count - 1] * world_size - total_frames * shard_index),
                count,
            ),
        )
        boundaries.append(boundary)
    boundaries.append(num_motions)
    return boundaries


def shard_cache_entry(
    entry: FKCacheEntry,
    *,
    rank: int,
    world_size: int,
) -> FKCacheEntry:
    """Return a contiguous rank shard as views into the shared frame storage."""
    if world_size == 1:
        return entry
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    boundaries = balanced_frame_boundaries(entry.ends, world_size=world_size)
    motion_start = boundaries[rank]
    motion_end = boundaries[rank + 1]
    frame_start = 0 if motion_start == 0 else entry.ends[motion_start - 1]
    frame_end = entry.ends[motion_end - 1]

    storage_fields = {
        name: tensor[frame_start:frame_end]
        for name, tensor in entry.storage_fields.items()
    }
    starts = [value - frame_start for value in entry.starts[motion_start:motion_end]]
    ends = [value - frame_start for value in entry.ends[motion_start:motion_end]]
    return replace(
        entry,
        motion_paths=entry.motion_paths[motion_start:motion_end],
        starts=starts,
        ends=ends,
        storage_fields=storage_fields,
        motion_id_offset=entry.motion_id_offset + motion_start,
    )
