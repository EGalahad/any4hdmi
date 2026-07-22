from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from any4hdmi.dataset.base import MotionData
from any4hdmi.dataset.fk_cache import (
    _GrowableMotionStorage,
    _write_motion_chunks_to_storage,
)


FIELD_NAMES = (
    "motion_id",
    "step",
    "body_pos_w",
    "body_lin_vel_w",
    "body_quat_w",
    "body_ang_vel_w",
    "joint_pos",
    "joint_vel",
)


def _motion_data(motion_id_value: int, length: int) -> MotionData:
    motion_id = torch.full((length,), motion_id_value, dtype=torch.long)
    base = torch.arange(length, dtype=torch.float32).view(length, 1, 1)
    body_xyz = base.expand(length, 2, 3) + float(motion_id_value * 10)
    body_quat = base.expand(length, 2, 4) + float(motion_id_value * 20)
    joint = base.view(length, 1).expand(length, 3) + float(motion_id_value * 30)
    return MotionData(
        motion_id=motion_id,
        step=torch.arange(length, dtype=torch.long),
        body_pos_w=body_xyz.clone(),
        body_lin_vel_w=(body_xyz + 1).clone(),
        body_quat_w=body_quat.clone(),
        body_ang_vel_w=(body_xyz + 2).clone(),
        joint_pos=joint.clone(),
        joint_vel=(joint + 3).clone(),
        device=motion_id.device,
        batch_size=[length],
    )


class FKCacheBuildTest(unittest.TestCase):
    def test_write_motion_chunks_preserves_cache_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = _GrowableMotionStorage(
                cache_entry_dir=Path(temp_dir),
                body_count=2,
                joint_count=3,
                initial_capacity=1,
            )
            chunks = [_motion_data(4, 2), _motion_data(7, 3)]
            expected = {
                field_name: torch.cat(
                    [getattr(chunk, field_name) for chunk in chunks], dim=0
                )
                for field_name in FIELD_NAMES
            }
            lengths = [2, 3]
            starts: list[int] = []
            ends: list[int] = []

            cursor = _write_motion_chunks_to_storage(
                storage=storage,
                staged_motion_chunks=chunks,
                staged_motion_lengths=lengths,
                starts=starts,
                ends=ends,
                start_idx=11,
            )

            self.assertEqual(cursor, 16)
            self.assertEqual(starts, [11, 13])
            self.assertEqual(ends, [13, 16])
            self.assertEqual(chunks, [])
            self.assertEqual(lengths, [])
            self.assertEqual(set(storage.fields), set(FIELD_NAMES))
            for field_name, expected_value in expected.items():
                actual = storage.fields[field_name][11:16]
                self.assertEqual(actual.dtype, expected_value.dtype)
                self.assertEqual(tuple(actual.shape), tuple(expected_value.shape))
                torch.testing.assert_close(actual, expected_value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
