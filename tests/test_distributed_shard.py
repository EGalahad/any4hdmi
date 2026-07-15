from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

import torch

from any4hdmi.dataset.distributed_shard import balanced_frame_boundaries
from any4hdmi.dataset.fk_cache import FKCacheEntry
from any4hdmi.dataset.full import FullMotionDataset
from any4hdmi.dataset.loaders import partition_cache_entry


def _entry() -> FKCacheEntry:
    lengths = [3, 5, 2, 6]
    ends = torch.tensor(lengths).cumsum(0).tolist()
    starts = [0, *ends[:-1]]
    motion_id = torch.repeat_interleave(
        torch.arange(len(lengths)), torch.tensor(lengths)
    )
    step = torch.cat([torch.arange(length) for length in lengths])
    frames = sum(lengths)
    body_shape = (frames, 1, 3)
    return FKCacheEntry(
        cache_entry_dir=Path("/tmp/test-cache"),
        body_names=["pelvis"],
        joint_names=["j0", "j1"],
        motion_paths=[Path(f"motion_{index}.npz") for index in range(4)],
        starts=starts,
        ends=ends,
        storage_fields={
            "motion_id": motion_id,
            "step": step,
            "body_pos_w": torch.arange(
                torch.tensor(body_shape).prod()
            ).reshape(body_shape).half(),
            "body_lin_vel_w": torch.zeros(body_shape, dtype=torch.float16),
            "body_quat_w": torch.zeros((frames, 1, 4), dtype=torch.float16),
            "body_ang_vel_w": torch.zeros(body_shape, dtype=torch.float16),
            "joint_pos": torch.zeros((frames, 2), dtype=torch.float16),
            "joint_vel": torch.zeros((frames, 2), dtype=torch.float16),
        },
    )


class PartitionCacheEntryTest(unittest.TestCase):
    def test_boundaries_balance_frames_without_splitting_motions(self) -> None:
        lengths = [8, 1, 1, 1, 1, 1, 1, 1]
        ends = torch.tensor(lengths).cumsum(0).tolist()

        boundaries = balanced_frame_boundaries(ends, world_size=2)

        self.assertEqual(boundaries, [0, 1, 8])
        frame_boundaries = [0, *(ends[index - 1] for index in boundaries[1:])]
        self.assertEqual(
            [
                right - left
                for left, right in zip(frame_boundaries, frame_boundaries[1:])
            ],
            [8, 7],
        )

    def test_identity_does_not_construct_or_copy(self) -> None:
        entry = _entry()
        with mock.patch(
            "any4hdmi.dataset.loaders.shard_cache_entry"
        ) as shard_cache_entry:
            result = partition_cache_entry(
                entry, shard=False, rank=0, world_size=8
            )
        self.assertIs(result, entry)
        shard_cache_entry.assert_not_called()

        result = partition_cache_entry(
            entry, shard=False, rank=99, world_size=0
        )
        self.assertIs(result, entry)

    def test_single_rank_shard_returns_full_entry_view(self) -> None:
        entry = _entry()
        result = partition_cache_entry(entry, shard=True, rank=0, world_size=1)
        self.assertIs(result, entry)
        self.assertEqual(result.motion_paths, entry.motion_paths)
        self.assertEqual(result.starts, entry.starts)
        self.assertEqual(result.ends, entry.ends)

    def test_multi_rank_shards_are_disjoint_and_cover_all_motions(self) -> None:
        entry = _entry()
        with (
            mock.patch(
                "any4hdmi.dataset.loaders.FullMotionDataset.from_cache_entry"
            ) as full_runtime,
            mock.patch(
                "any4hdmi.dataset.loaders.WindowedMotionDataset.from_cache_entry"
            ) as windowed_runtime,
        ):
            shards = [
                partition_cache_entry(entry, shard=True, rank=rank, world_size=2)
                for rank in range(2)
            ]
        full_runtime.assert_not_called()
        windowed_runtime.assert_not_called()
        paths = [set(shard.motion_paths) for shard in shards]
        self.assertTrue(paths[0].isdisjoint(paths[1]))
        self.assertEqual(paths[0] | paths[1], set(entry.motion_paths))
        self.assertEqual(shards[1].starts, [0, 2])
        self.assertEqual(shards[1].ends, [2, 8])
        self.assertEqual(shards[1].motion_id_offset, 2)
        self.assertEqual(
            shards[1].storage_fields["motion_id"].tolist(),
            [2, 2, 3, 3, 3, 3, 3, 3],
        )
        self.assertEqual(
            shards[1].storage_fields["motion_id"].untyped_storage().data_ptr(),
            entry.storage_fields["motion_id"].untyped_storage().data_ptr(),
        )

    def test_invalid_rank_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank must be"):
            partition_cache_entry(_entry(), shard=True, rank=2, world_size=2)

    def test_fewer_motions_than_ranks_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cannot split 4 motions"):
            partition_cache_entry(_entry(), shard=True, rank=0, world_size=5)

    def test_sharded_full_dataset_keeps_fp16_storage_and_returns_fp32(self) -> None:
        shard = partition_cache_entry(_entry(), shard=True, rank=1, world_size=2)
        dataset = FullMotionDataset.from_cache_entry(
            shard,
            num_envs=2,
            output_float_dtype=torch.float32,
        )
        self.assertEqual(dataset.data.body_pos_w.dtype, torch.float16)
        result = dataset.get_slice(
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
        )
        self.assertEqual(result.body_pos_w.dtype, torch.float32)
        self.assertEqual(result.motion_id.dtype, torch.long)
        self.assertEqual(result.motion_id.tolist(), [[0, 0], [1, 1]])
        with mock.patch(
            "any4hdmi.dataset.full.torch.randint",
            return_value=torch.tensor([0, 2]),
        ):
            sampled = dataset.sample_motion(
                torch.tensor([0, 1]),
                terminated_t=torch.zeros(2, dtype=torch.long),
                rewind_mask=torch.zeros(2, dtype=torch.bool),
                rewind_steps=torch.zeros(2, dtype=torch.long),
            )
        self.assertEqual(sampled.motion_id.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
