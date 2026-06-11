from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import mujoco

from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.limmt import gqs, physical_filter
from any4hdmi.limmt.common import copy_any4hdmi_subset
from any4hdmi.limmt.hme import (
    PeriodicAutoencoder,
    CachedHmeWindowDataset,
    HME_FEATURE_TYPE,
    compute_win_len,
    motion_features,
    root_pose_in_initial_heading_frame,
    root_vel_in_current_root_frame,
    window_indices,
)


class LimmtHmeTest(unittest.TestCase):
    def test_window_indices_use_centered_downsampled_window(self) -> None:
        win_len = compute_win_len(4.0, 5, 50.0)
        self.assertEqual(win_len, 41)
        ids = window_indices(250, win_len=win_len, downsample_rate=5, stride=1)
        self.assertEqual(ids.shape, (50, 41))
        self.assertEqual(ids[0, 0], 0)
        self.assertEqual(ids[0, 20], 100)
        self.assertEqual(ids[0, -1], 200)

    def test_periodic_autoencoder_outputs_expected_hme_shapes(self) -> None:
        model = PeriodicAutoencoder(inp_ch=73, latent_ch=8, win_len=41, win_sec=4.0)
        model.eval()
        batch = torch.randn(2, 73, 41)
        with torch.no_grad():
            encoded = model.encode(batch)
            out = model(batch)
        self.assertEqual(encoded["amp"].shape, (2, 8, 1))
        self.assertEqual(encoded["freq"].shape, (2, 8, 1))
        self.assertEqual(encoded["shift"].shape, (2, 8, 1))
        self.assertEqual(encoded["offset"].shape, (2, 8, 1))
        self.assertEqual(out["pred"].shape, batch.shape)

    def test_motion_features_use_joint_state_and_initial_heading_root_pose(self) -> None:
        qpos = np.zeros((3, 36), dtype=np.float32)
        qpos[:, :3] = np.asarray([[1.0, 2.0, 0.8], [1.4, 2.1, 0.82], [1.8, 2.4, 0.81]], dtype=np.float32)
        qpos[:, 3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        qpos[:, 7:] = np.arange(87, dtype=np.float32).reshape(3, 29)
        qvel = np.zeros((3, 35), dtype=np.float32)
        qvel[:, 6:] = 100.0 + np.arange(87, dtype=np.float32).reshape(3, 29)
        qvel[:, :6] = 200.0 + np.arange(18, dtype=np.float32).reshape(3, 6)

        features = motion_features(qpos, qvel)
        self.assertEqual(features.shape, (3, 73))
        np.testing.assert_allclose(features[:, :29], qpos[:, 7:])
        np.testing.assert_allclose(features[:, 29:58], qvel[:, 6:])
        np.testing.assert_allclose(features[:, 58:67], root_pose_in_initial_heading_frame(qpos))
        np.testing.assert_allclose(features[:, 67:], root_vel_in_current_root_frame(qpos, qvel))

    def test_motion_features_are_invariant_to_global_yaw(self) -> None:
        def yaw_quat(yaw: float) -> np.ndarray:
            return np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)

        def quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
            lw, lx, ly, lz = lhs
            rw, rx, ry, rz = rhs
            return np.asarray(
                [
                    lw * rw - lx * rx - ly * ry - lz * rz,
                    lw * rx + lx * rw + ly * rz - lz * ry,
                    lw * ry - lx * rz + ly * rw + lz * rx,
                    lw * rz + lx * ry - ly * rx + lz * rw,
                ],
                dtype=np.float32,
            )

        def rotate_xy(vec: np.ndarray, yaw: float) -> np.ndarray:
            out = vec.copy()
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            x = out[:, 0].copy()
            y = out[:, 1].copy()
            out[:, 0] = cos_y * x - sin_y * y
            out[:, 1] = sin_y * x + cos_y * y
            return out

        root_yaws = [0.3, 0.6, 1.0]
        qpos = np.zeros((3, 36), dtype=np.float32)
        qpos[:, :3] = np.asarray([[1.0, 2.0, 0.8], [1.4, 2.1, 0.82], [1.8, 2.4, 0.81]], dtype=np.float32)
        qpos[:, 3:7] = np.stack([yaw_quat(yaw) for yaw in root_yaws])
        qpos[:, 7:] = np.asarray([0.1, 0.2, 0.4], dtype=np.float32)[:, None]
        qvel = np.zeros((3, 35), dtype=np.float32)
        qvel[:, 6:] = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)[:, None]
        qvel[:, :6] = np.asarray(
            [
                [0.2, 0.1, 0.0, 0.0, 0.0, 0.3],
                [0.4, 0.2, 0.0, 0.0, 0.0, 0.4],
                [0.3, 0.3, 0.0, 0.0, 0.0, 0.5],
            ],
            dtype=np.float32,
        )

        global_yaw = 1.2
        qpos_rot = qpos.copy()
        qpos_rot[:, :3] = rotate_xy(qpos[:, :3], global_yaw)
        qpos_rot[:, 3:7] = np.stack([quat_mul(yaw_quat(global_yaw), quat) for quat in qpos[:, 3:7]])
        qvel_rot = qvel.copy()
        qvel_rot[:, :3] = rotate_xy(qvel[:, :3], global_yaw)
        qvel_rot[:, 3:6] = qvel[:, 3:6]

        np.testing.assert_allclose(motion_features(qpos, qvel), motion_features(qpos_rot, qvel_rot), atol=1e-5)

    def test_motion_features_are_invariant_to_global_yaw_with_mujoco_freejoint_qvel(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="b"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
        )

        def axis_quat(axis: np.ndarray, angle: float) -> np.ndarray:
            axis = axis / np.linalg.norm(axis)
            return np.asarray([np.cos(angle / 2), *(np.sin(angle / 2) * axis)], dtype=np.float64)

        def quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
            lw, lx, ly, lz = lhs
            rw, rx, ry, rz = rhs
            return np.asarray(
                [
                    lw * rw - lx * rx - ly * ry - lz * rz,
                    lw * rx + lx * rw + ly * rz - lz * ry,
                    lw * ry - lx * rz + ly * rw + lz * rx,
                    lw * rz + lx * ry - ly * rx + lz * rw,
                ],
                dtype=np.float64,
            )

        def rotate_xy(vec: np.ndarray, yaw: float) -> np.ndarray:
            out = vec.copy()
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            x = out[:, 0].copy()
            y = out[:, 1].copy()
            out[:, 0] = cos_y * x - sin_y * y
            out[:, 1] = sin_y * x + cos_y * y
            return out

        def free_qvel(qpos7: np.ndarray) -> np.ndarray:
            out = np.zeros((qpos7.shape[0], 6), dtype=np.float64)
            work = np.zeros(6, dtype=np.float64)
            for frame_idx in range(qpos7.shape[0] - 1):
                mujoco.mj_differentiatePos(model, work, 0.02, qpos7[frame_idx], qpos7[frame_idx + 1])
                out[frame_idx] = work
            out[-1] = out[-2]
            return out.astype(np.float32)

        qpos7 = np.zeros((3, 7), dtype=np.float64)
        qpos7[:, :3] = np.asarray([[1.0, 2.0, 0.8], [1.1, 2.2, 0.9], [1.3, 2.25, 0.85]], dtype=np.float64)
        base = quat_mul(
            axis_quat(np.asarray([0.0, 0.0, 1.0]), 0.4),
            quat_mul(axis_quat(np.asarray([1.0, 0.0, 0.0]), 0.2), axis_quat(np.asarray([0.0, 1.0, 0.0]), -0.1)),
        )
        qpos7[0, 3:7] = base
        qpos7[1, 3:7] = quat_mul(base, axis_quat(np.asarray([0.0, 1.0, 0.0]), 0.2))
        qpos7[2, 3:7] = quat_mul(base, axis_quat(np.asarray([1.0, 0.0, 0.0]), 0.3))
        qvel6 = free_qvel(qpos7)

        global_yaw = 0.9
        yaw_quat = axis_quat(np.asarray([0.0, 0.0, 1.0]), global_yaw)
        qpos7_rot = qpos7.copy()
        qpos7_rot[:, :3] = rotate_xy(qpos7[:, :3], global_yaw)
        qpos7_rot[:, 3:7] = np.stack([quat_mul(yaw_quat, quat) for quat in qpos7[:, 3:7]])
        qvel6_rot = free_qvel(qpos7_rot)

        qpos = np.zeros((3, 36), dtype=np.float32)
        qpos_rot = np.zeros((3, 36), dtype=np.float32)
        qpos[:, :7] = qpos7.astype(np.float32)
        qpos_rot[:, :7] = qpos7_rot.astype(np.float32)
        qvel = np.zeros((3, 35), dtype=np.float32)
        qvel_rot = np.zeros((3, 35), dtype=np.float32)
        qvel[:, :6] = qvel6
        qvel_rot[:, :6] = qvel6_rot

        np.testing.assert_allclose(motion_features(qpos, qvel), motion_features(qpos_rot, qvel_rot), atol=1e-5)

    def test_root_pose_uses_first_frame_heading_not_first_frame_roll_pitch(self) -> None:
        def axis_quat(axis: np.ndarray, angle: float) -> np.ndarray:
            axis = axis / np.linalg.norm(axis)
            return np.asarray(
                [np.cos(angle / 2), *(np.sin(angle / 2) * axis)],
                dtype=np.float32,
            )

        def quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
            lw, lx, ly, lz = lhs
            rw, rx, ry, rz = rhs
            return np.asarray(
                [
                    lw * rw - lx * rx - ly * ry - lz * rz,
                    lw * rx + lx * rw + ly * rz - lz * ry,
                    lw * ry - lx * rz + ly * rw + lz * rx,
                    lw * rz + lx * ry - ly * rx + lz * rw,
                ],
                dtype=np.float32,
            )

        yaw = axis_quat(np.asarray([0.0, 0.0, 1.0]), 0.7)
        roll = axis_quat(np.asarray([1.0, 0.0, 0.0]), 0.2)
        first = quat_mul(yaw, roll)
        second_delta = axis_quat(np.asarray([0.0, 1.0, 0.0]), -0.3)
        qpos = np.zeros((2, 36), dtype=np.float32)
        qpos[:, :3] = np.asarray([[0.0, 0.0, 0.8], [0.2, 0.1, 0.82]], dtype=np.float32)
        qpos[0, 3:7] = first
        qpos[1, 3:7] = quat_mul(first, second_delta)

        root_pose = root_pose_in_initial_heading_frame(qpos)
        self.assertEqual(root_pose.shape, (2, 9))
        self.assertGreater(np.linalg.norm(root_pose[1, 3:9] - root_pose[0, 3:9]), 0.1)

    def test_root_pose_6d_orientation_is_invariant_to_quaternion_sign(self) -> None:
        qpos = np.zeros((2, 36), dtype=np.float32)
        qpos[:, :3] = np.asarray([[0.0, 0.0, 0.8], [0.1, 0.2, 0.9]], dtype=np.float32)
        qpos[:, 3:7] = np.asarray(
            [
                [0.9, 0.1, 0.2, 0.3],
                [0.8, -0.2, 0.1, 0.5],
            ],
            dtype=np.float32,
        )
        qpos_flipped = qpos.copy()
        qpos_flipped[:, 3:7] *= -1.0

        np.testing.assert_allclose(
            root_pose_in_initial_heading_frame(qpos),
            root_pose_in_initial_heading_frame(qpos_flipped),
            atol=1e-5,
        )

    def test_cached_hme_dataset_returns_empirically_normalized_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir)
            feature_path = cache / "features" / "motion.npy"
            feature_path.parent.mkdir(parents=True)
            raw = np.arange(12, dtype=np.float32).reshape(4, 3)
            np.save(feature_path, raw)
            mean = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
            std = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
            np.savez_compressed(cache / "normalization.npz", mean=mean, std=std, count=np.asarray(raw.shape[0]))
            (cache / "index.json").write_text(
                json.dumps(
                    {
                        "win_len": 1,
                        "downsample_rate": 1,
                        "stride": 1,
                        "feature_type": HME_FEATURE_TYPE,
                        "feature_dim": int(raw.shape[1]),
                        "normalization": {
                            "type": "empirical_mean_std",
                            "path": str(cache / "normalization.npz"),
                            "count": int(raw.shape[0]),
                        },
                        "records": [
                            {
                                "motion": "motion.npy",
                                "feature_path": str(feature_path),
                                "length": int(raw.shape[0]),
                                "num_windows": int(raw.shape[0]),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = CachedHmeWindowDataset(cache)
            np.testing.assert_allclose(dataset[2].numpy(), ((raw[2:3] - mean) / std), atol=1e-6)


class LimmtPhysicalFilterTest(unittest.TestCase):
    def test_score_motion_aggregates_penalties_and_threshold(self) -> None:
        outputs = {
            "body_pos_w": torch.tensor(
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.01], [0.02, 0.0, 0.03]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.01], [0.02, 0.0, 0.03]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.01], [0.02, 0.0, 0.03]],
                ],
                dtype=torch.float32,
            ),
            "body_lin_vel_w": torch.tensor(
                [
                    [[0, 0, 0], [0.4, 0.0, 0.0], [0, 0, 0]],
                    [[0, 0, 0], [0.4, 0.0, 0.0], [0, 0, 0]],
                    [[0, 0, 0], [0.4, 0.0, 0.0], [0, 0, 0]],
                ],
                dtype=torch.float32,
            ),
            "body_ang_vel_w": torch.zeros((3, 3, 3), dtype=torch.float32),
            "joint_vel": torch.zeros((3, 2), dtype=torch.float32),
        }
        row = physical_filter._score_motion(
            rel_motion="motions/a.npz",
            outputs=outputs,
            qvel=torch.zeros((3, 3), dtype=torch.float32),
            fps=50.0,
            config=physical_filter.PhysicalFilterConfig(pass_threshold=90.0, contact_sample_stride=1),
            weights=physical_filter.PhysicalScoreWeights(),
            foot_indices=torch.tensor([1], dtype=torch.long),
            pair_left=torch.tensor([1], dtype=torch.long),
            pair_right=torch.tensor([2], dtype=torch.long),
        )
        self.assertEqual(row["motion"], "motions/a.npz")
        self.assertGreater(row["foot_sliding"], 0.0)
        self.assertGreater(row["self_collision"], 0.0)
        self.assertLessEqual(row["physical_score"], 100.0)


class LimmtGqsTest(unittest.TestCase):
    def test_weighted_fps_is_deterministic_and_respects_count(self) -> None:
        embeddings = np.asarray([[0, 0], [1, 0], [0, 1], [10, 10]], dtype=np.float32)
        complexities = np.asarray([0.1, 0.4, 0.2, 1.0], dtype=np.float32)
        selected = gqs.weighted_fps_global(embeddings, complexities, 2, alpha=0.6)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0], 3)
        self.assertEqual(selected, gqs.weighted_fps_global(embeddings, complexities, 2, alpha=0.6))

    def test_copy_any4hdmi_subset_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            root = temp / "input"
            save_motion(root / "motions" / "a.npz", np.zeros((5, 3), dtype=np.float32))
            save_motion(root / "motions" / "b.npz", np.ones((7, 3), dtype=np.float32))
            write_manifest(
                root,
                dataset_name="input",
                mjcf="hf://example/repo@main/model.xml",
                timestep=0.02,
                qpos_names=["a", "b", "c"],
                num_motions=2,
                source={"kind": "test"},
                total_hours=12 * 0.02 / 3600.0,
            )
            stale = temp / "subset" / "motions" / "stale.npz"
            save_motion(stale, np.full((3, 3), 2.0, dtype=np.float32))
            manifest_path = copy_any4hdmi_subset(
                input_root=root,
                output_root=temp / "subset",
                selected_rel_paths=["motions/b.npz"],
                dataset_name="subset",
                source_update={"limmt_gqs": {"ratio": 0.5}},
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_name"], "subset")
            self.assertEqual(payload["num_motions"], 1)
            self.assertTrue((temp / "subset" / "motions" / "b.npz").is_file())
            self.assertFalse(stale.exists())
            self.assertEqual(payload["source"]["limmt_gqs"]["ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
