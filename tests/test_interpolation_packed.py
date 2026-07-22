from __future__ import annotations

import unittest

import torch

from any4hdmi.dataset.interpolation import (
    interpolate_qpos_qvel_batch_torch,
    interpolate_qpos_qvel_packed_torch,
)


class PackedQposQvelInterpolationTest(unittest.TestCase):
    def test_packed_output_matches_clip_api_exactly(self) -> None:
        generator = torch.Generator().manual_seed(7)
        qpos = [torch.randn((length, 36), generator=generator) for length in (1, 7, 13)]
        for clip in qpos:
            clip[:, 3:7] /= clip[:, 3:7].norm(dim=1, keepdim=True)
        qvel = [torch.randn((length, 35), generator=generator) for length in (1, 7, 13)]

        expected_qpos, expected_qvel = interpolate_qpos_qvel_batch_torch(
            qpos,
            qvel,
            source_fps=120.0,
            target_fps=50.0,
        )
        packed_qpos, packed_qvel, lengths = interpolate_qpos_qvel_packed_torch(
            qpos,
            qvel,
            source_fps=120.0,
            target_fps=50.0,
        )

        self.assertEqual(lengths, [len(clip) for clip in expected_qpos])
        torch.testing.assert_close(
            packed_qpos,
            torch.cat(expected_qpos),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            packed_qvel,
            torch.cat(expected_qvel),
            rtol=0,
            atol=0,
        )

    def test_no_resampling_preserves_values_and_lengths(self) -> None:
        qpos = [torch.randn((2, 8)), torch.randn((3, 8))]
        qvel = [torch.randn((2, 7)), torch.randn((3, 7))]

        packed_qpos, packed_qvel, lengths = interpolate_qpos_qvel_packed_torch(
            qpos,
            qvel,
            source_fps=50.0,
            target_fps=50.0,
        )

        self.assertEqual(lengths, [2, 3])
        torch.testing.assert_close(packed_qpos, torch.cat(qpos), rtol=0, atol=0)
        torch.testing.assert_close(packed_qvel, torch.cat(qvel), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
