from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest import mock

import mujoco
import numpy as np
import torch

from any4hdmi.utils.dataset import (
    MotionTensorDataset,
    compute_motion_qvel,
    pack_motion_items,
)


def _reference_motion_qvel(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: float,
) -> np.ndarray:
    qpos64 = np.asarray(qpos, dtype=np.float64)
    qvel = np.zeros((qpos64.shape[0], model.nv), dtype=np.float32)
    if qpos64.shape[0] <= 1:
        return qvel
    work = np.zeros(model.nv, dtype=np.float64)
    for frame_idx in range(qpos64.shape[0] - 1):
        mujoco.mj_differentiatePos(
            model,
            work,
            1.0 / float(fps),
            qpos64[frame_idx],
            qpos64[frame_idx + 1],
        )
        qvel[frame_idx] = work
    qvel[-1] = qvel[-2]
    return qvel


class MotionQvelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body>
                  <freejoint/>
                  <geom type="sphere" size="0.1"/>
                  <body>
                    <joint type="ball"/>
                    <geom type="sphere" size="0.1"/>
                    <body>
                      <joint type="slide"/>
                      <geom type="sphere" size="0.1"/>
                      <body>
                        <joint type="hinge"/>
                        <geom type="sphere" size="0.1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </worldbody>
            </mujoco>
            """
        )

    def test_matches_mujoco_reference_for_every_joint_type(self) -> None:
        rng = np.random.default_rng(42)
        qpos = rng.normal(size=(4096, self.model.nq)).astype(np.float32)
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
        qpos[:, 7:11] /= np.linalg.norm(qpos[:, 7:11], axis=1, keepdims=True)

        expected = _reference_motion_qvel(self.model, qpos, 120.0)
        actual = compute_motion_qvel(self.model, qpos, 120.0)

        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (4096, self.model.nv))
        np.testing.assert_array_equal(actual, expected)

    def test_empty_and_single_frame_inputs(self) -> None:
        for frames in (0, 1):
            with self.subTest(frames=frames):
                qpos = np.zeros((frames, self.model.nq), dtype=np.float32)
                actual = compute_motion_qvel(self.model, qpos, 50.0)
                np.testing.assert_array_equal(
                    actual,
                    np.zeros((frames, self.model.nv), dtype=np.float32),
                )

    def test_rejects_invalid_qpos_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank 2"):
            compute_motion_qvel(self.model, np.zeros(self.model.nq), 50.0)
        with self.assertRaisesRegex(ValueError, "does not match"):
            compute_motion_qvel(
                self.model,
                np.zeros((2, self.model.nq + 1)),
                50.0,
            )

    def test_pack_motion_items_preserves_order_and_boundaries(self) -> None:
        first_qpos = np.arange(21, dtype=np.float32).reshape(3, 7)
        second_qpos = np.arange(14, dtype=np.float32).reshape(2, 7) + 100
        first_qvel = np.arange(18, dtype=np.float32).reshape(3, 6)
        second_qvel = np.arange(12, dtype=np.float32).reshape(2, 6) + 100
        packed = pack_motion_items(
            [
                {
                    "motion_path": "first.npz",
                    "qpos": torch.from_numpy(first_qpos),
                    "qvel": torch.from_numpy(first_qvel),
                    "fps": 120.0,
                },
                {
                    "motion_path": "second.npz",
                    "qpos": torch.from_numpy(second_qpos),
                    "qvel": torch.from_numpy(second_qvel),
                    "fps": 120.0,
                },
            ]
        )

        np.testing.assert_array_equal(packed["lengths"].numpy(), [3, 2])
        np.testing.assert_array_equal(
            packed["qpos"].numpy(),
            np.concatenate((first_qpos, second_qpos)),
        )
        np.testing.assert_array_equal(
            packed["qvel"].numpy(),
            np.concatenate((first_qvel, second_qvel)),
        )
        self.assertEqual(packed["motion_path"], ["first.npz", "second.npz"])
        self.assertEqual(packed["fps"], [120.0, 120.0])

    def test_prevalidated_qpos_path_skips_revalidation_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            motion_path = Path(temp_dir) / "motion.npz"
            expected = np.arange(21, dtype=np.float32).reshape(3, 7)
            np.savez_compressed(motion_path, qpos=expected)
            dataset = MotionTensorDataset(
                input_root=None,
                motion_paths=[motion_path],
                prevalidated_paths=True,
            )

            with mock.patch(
                "any4hdmi.utils.dataset.load_motion",
                side_effect=AssertionError("prevalidated qpos should load directly"),
            ):
                actual = dataset[0]["qpos"].numpy()

            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
