from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from any4hdmi.core.format import load_motion
from any4hdmi.scripts.preprocess.gmr import convert_dataset, gmr_to_qpos


def _write_gmr_motion(path: Path, start: int, *, frames: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(start, start + frames, dtype=np.float64)
    np.savez_compressed(
        path,
        root_pos=np.stack((values, values + 1, values + 2), axis=-1),
        root_rot=np.tile(np.array([0.0, 0.0, 0.0, 2.0]), (frames, 1)),
        dof_pos=np.stack((values, values + 1), axis=-1),
    )


def _write_manifest(path: Path, *, num_motions: int, total_frames: int) -> None:
    (path / "model.xml").write_text("<mujoco/>")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "dataset_name": "gmr-test",
                "mjcf": "model.xml",
                "motions_subdir": "motions",
                "timestep": 0.02,
                "qpos_dim": 9,
                "qpos_names": [
                    "root_tx",
                    "root_ty",
                    "root_tz",
                    "root_qw",
                    "root_qx",
                    "root_qy",
                    "root_qz",
                    "joint_a",
                    "joint_b",
                ],
                "num_motions": num_motions,
                "source": {"schema": "motion_retarget_gmr_v1"},
                "total_hours": total_frames * 0.02 / 3600.0,
            }
        )
    )


class GmrConversionTest(unittest.TestCase):
    def test_converts_xyzw_gmr_arrays_to_canonical_qpos(self) -> None:
        qpos = gmr_to_qpos(
            {
                "root_pos": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "root_rot": np.array(
                    [[0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 1.0]]
                ),
                "dof_pos": np.array([[0.1, 0.2], [0.3, 0.4]]),
            }
        )

        np.testing.assert_allclose(qpos[:, :3], [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(
            qpos[:, 3:7],
            [[1.0, 0.0, 0.0, 0.0], [2**-0.5, 0.0, 0.0, 2**-0.5]],
        )
        np.testing.assert_allclose(qpos[:, 7:], [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(qpos.dtype, np.float32)

    def test_converts_and_concatenates_dataset_to_qpos_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input"
            output_path = root / "output"
            _write_gmr_motion(input_path / "motions" / "walk_1.npz", 0)
            _write_gmr_motion(input_path / "motions" / "walk_2.npz", 3)
            _write_gmr_motion(input_path / "motions" / "jump.npz", 6)
            _write_manifest(input_path, num_motions=3, total_frames=9)

            summary = convert_dataset(input_path, output_path)

            self.assertEqual(summary["input_motion_files"], 3)
            self.assertEqual(summary["concatenated_motions"], 1)
            self.assertEqual(summary["output_motions"], 2)
            self.assertEqual(summary["total_frames"], 9)
            np.testing.assert_array_equal(
                load_motion(output_path / "motions" / "walk.npz")[:, 0],
                np.arange(6),
            )
            with np.load(
                output_path / "motions" / "jump.npz", allow_pickle=False
            ) as archive:
                self.assertEqual(archive.files, ["qpos"])
            manifest = json.loads((output_path / "manifest.json").read_text())
            self.assertEqual(manifest["num_motions"], 2)
            self.assertEqual(manifest["qpos_dim"], 9)
            self.assertEqual(
                manifest["source"]["gmr_conversion"]["output_motions"], 2
            )

    def test_rejects_missing_segment_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input"
            _write_gmr_motion(input_path / "motions" / "walk_1.npz", 0)
            _write_gmr_motion(input_path / "motions" / "walk_3.npz", 3)
            _write_manifest(input_path, num_motions=2, total_frames=6)

            with self.assertRaisesRegex(ValueError, "Non-contiguous segment indices"):
                convert_dataset(input_path, root / "output")

    def test_raw_gmr_motion_is_not_accepted_by_common_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            motion_path = Path(temp_dir) / "motion.npz"
            _write_gmr_motion(motion_path, 0)

            with self.assertRaisesRegex(KeyError, "does not contain a qpos array"):
                load_motion(motion_path)


if __name__ == "__main__":
    unittest.main()
