from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import mujoco

from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.fk.runner import FKRunner
from any4hdmi.limmt import gqs, physical_filter
from any4hdmi.limmt.common import copy_any4hdmi_subset
from any4hdmi.limmt.hme import (
    PeriodicAutoencoder,
    CachedHmeWindowDataset,
    HME_CACHE_FEATURE_TYPE,
    HME_FEATURE_TYPE,
    compute_win_len,
    hme_features_from_raw,
    motion_frame_features,
    motion_features,
    root_pose_in_initial_heading_frame,
    root_vel_in_current_root_frame,
    window_indices,
)
from any4hdmi.limmt.hme.loading import (
    HME_CACHE_SOURCE_KEY_NAME,
    HME_CACHE_VERSION,
    _make_hme_cache_source_key_for_root,
    hme_feature_cache_is_current,
)


class LimmtHmeTest(unittest.TestCase):
    def test_window_indices_use_centered_downsampled_window(self) -> None:
        win_len = compute_win_len(4.0, 5, 50.0)
        self.assertEqual(win_len, 41)
        window_ids = window_indices(250, win_len=win_len, downsample_rate=5, stride=1)
        self.assertEqual(window_ids.shape, (50, 41))
        self.assertEqual(window_ids[0, 0], 0)
        self.assertEqual(window_ids[0, 20], 100)
        self.assertEqual(window_ids[0, -1], 200)

    def test_periodic_autoencoder_outputs_expected_hme_shapes(self) -> None:
        model = PeriodicAutoencoder(inp_ch=73, latent_ch=8, win_len=41, win_sec=4.0)
        model.eval()
        batch = torch.randn(2, 73, 41)
        with torch.no_grad():
            encoded = model.encode(batch)
            model_output = model(batch)
        self.assertEqual(encoded["amp"].shape, (2, 8, 1))
        self.assertEqual(encoded["freq"].shape, (2, 8, 1))
        self.assertEqual(encoded["shift"].shape, (2, 8, 1))
        self.assertEqual(encoded["offset"].shape, (2, 8, 1))
        self.assertEqual(model_output["pred"].shape, batch.shape)

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
        np.testing.assert_allclose(features[:, 58:67], root_pose_in_initial_heading_frame(qpos[None, ...])[0])
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
            rotated_vec = vec.copy()
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            x = rotated_vec[:, 0].copy()
            y = rotated_vec[:, 1].copy()
            rotated_vec[:, 0] = cos_y * x - sin_y * y
            rotated_vec[:, 1] = sin_y * x + cos_y * y
            return rotated_vec

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
            rotated_vec = vec.copy()
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            x = rotated_vec[:, 0].copy()
            y = rotated_vec[:, 1].copy()
            rotated_vec[:, 0] = cos_y * x - sin_y * y
            rotated_vec[:, 1] = sin_y * x + cos_y * y
            return rotated_vec

        def free_qvel(qpos7: np.ndarray) -> np.ndarray:
            qvel_frames = np.zeros((qpos7.shape[0], 6), dtype=np.float64)
            work = np.zeros(6, dtype=np.float64)
            for frame_idx in range(qpos7.shape[0] - 1):
                mujoco.mj_differentiatePos(model, work, 0.02, qpos7[frame_idx], qpos7[frame_idx + 1])
                qvel_frames[frame_idx] = work
            qvel_frames[-1] = qvel_frames[-2]
            return qvel_frames.astype(np.float32)

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

        root_pose = root_pose_in_initial_heading_frame(qpos[None, ...])[0]
        self.assertEqual(root_pose.shape, (2, 9))
        self.assertGreater(np.linalg.norm(root_pose[1, 3:9] - root_pose[0, 3:9]), 0.1)

    def test_root_pose_window_batch_uses_each_window_initial_heading(self) -> None:
        def yaw_quat(yaw: float) -> np.ndarray:
            return np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)

        qpos = np.zeros((2, 3, 36), dtype=np.float32)
        qpos[0, :, :3] = np.asarray([[1.0, 2.0, 0.8], [2.0, 2.0, 0.8], [1.0, 3.0, 0.8]], dtype=np.float32)
        qpos[0, :, 3:7] = yaw_quat(0.0)
        qpos[1, :, :3] = np.asarray([[10.0, 10.0, 0.8], [10.0, 11.0, 0.8], [9.0, 10.0, 0.8]], dtype=np.float32)
        qpos[1, :, 3:7] = yaw_quat(np.pi / 2)

        root_pose = root_pose_in_initial_heading_frame(qpos)

        self.assertEqual(root_pose.shape, (2, 3, 9))
        np.testing.assert_allclose(root_pose[:, 0, :3], 0.0, atol=1e-6)
        expected_identity_rot6d = np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * 2, dtype=np.float32)
        np.testing.assert_allclose(root_pose[:, 0, 3:9], expected_identity_rot6d, atol=1e-6)
        np.testing.assert_allclose(root_pose[0, 1:, :3], np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-6)
        np.testing.assert_allclose(root_pose[1, 1:, :3], np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-6)

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
            root_pose_in_initial_heading_frame(qpos[None, ...]),
            root_pose_in_initial_heading_frame(qpos_flipped[None, ...]),
            atol=1e-5,
        )

    def test_root_pose_rejects_unwindowed_qpos(self) -> None:
        qpos = np.zeros((2, 36), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Expected windowed free-root qpos"):
            root_pose_in_initial_heading_frame(qpos)

    def test_cached_hme_dataset_returns_empirically_normalized_features(self) -> None:
        def yaw_quat(yaw: float) -> np.ndarray:
            return np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir)
            feature_path = cache / "features" / "motion.npy"
            feature_path.parent.mkdir(parents=True)
            qpos = np.zeros((4, 8), dtype=np.float32)
            qpos[:, :3] = np.asarray(
                [[10.0, 0.0, 0.8], [10.0, 1.0, 0.8], [10.0, 2.0, 0.8], [9.0, 2.0, 0.8]],
                dtype=np.float32,
            )
            qpos[:, 3:7] = yaw_quat(np.pi / 2)
            qpos[:, 7] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            qvel = np.zeros((4, 7), dtype=np.float32)
            qvel[:, 6] = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            raw = motion_frame_features(qpos, qvel)
            window_ids = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
            final_features = hme_features_from_raw(raw, window_ids)
            np.save(feature_path, final_features)
            mean = np.linspace(0.0, 1.6, 17, dtype=np.float32)
            std = np.linspace(1.0, 2.6, 17, dtype=np.float32)
            np.savez_compressed(
                cache / "normalization.npz",
                mean=mean,
                std=std,
                count=np.asarray(final_features.reshape(-1, final_features.shape[-1]).shape[0]),
            )
            (cache / "index.json").write_text(
                json.dumps(
                    {
                        "win_len": 3,
                        "downsample_rate": 1,
                        "stride": 1,
                        "feature_type": HME_FEATURE_TYPE,
                        "cache_feature_type": HME_CACHE_FEATURE_TYPE,
                        "cache_shape": "num_windows_win_len_feature_dim",
                        "cache_feature_dim": int(final_features.shape[-1]),
                        "feature_dim": int(mean.shape[0]),
                        "normalization": {
                            "type": "empirical_mean_std",
                            "path": str(cache / "normalization.npz"),
                            "count": int(final_features.reshape(-1, final_features.shape[-1]).shape[0]),
                        },
                        "records": [
                            {
                                "motion": "motion.npy",
                                "feature_path": str(feature_path),
                                "length": int(raw.shape[0]),
                                "num_windows": int(final_features.shape[0]),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = CachedHmeWindowDataset(cache)
            expected_final = final_features[1]
            expected = (expected_final - mean) / std
            np.testing.assert_allclose(dataset[1].numpy(), expected, atol=1e-6)
            np.testing.assert_allclose(expected_final[0, 2:5], 0.0, atol=1e-6)

    def test_hme_feature_cache_current_check_tracks_source_motion_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            mjcf_path = dataset_root / "model.xml"
            mjcf_path.parent.mkdir(parents=True, exist_ok=True)
            mjcf_path.write_text("<mujoco><worldbody/></mujoco>\n", encoding="utf-8")
            motion_path = save_motion(dataset_root / "motions" / "a.npz", np.zeros((5, 3), dtype=np.float32))
            write_manifest(
                dataset_root,
                dataset_name="input",
                mjcf=mjcf_path,
                timestep=0.02,
                qpos_names=["x", "y", "z"],
                num_motions=1,
                source={"kind": "test"},
                total_hours=5 * 0.02 / 3600.0,
            )

            cache_dir = temp / "cache"
            cache_dir.mkdir(parents=True)
            (cache_dir / "index.json").write_text(
                json.dumps(
                    {
                        "cache_version": HME_CACHE_VERSION,
                        HME_CACHE_SOURCE_KEY_NAME: _make_hme_cache_source_key_for_root(dataset_root.resolve()),
                        "dataset_root": str(dataset_root.resolve()),
                        "win_sec": 4.0,
                        "downsample_rate": 5,
                        "stride": 10,
                        "feature_type": HME_FEATURE_TYPE,
                        "cache_feature_type": HME_CACHE_FEATURE_TYPE,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (cache_dir / "ready.flag").write_text("ready\n", encoding="utf-8")

            self.assertTrue(
                hme_feature_cache_is_current(
                    cache_dir,
                    dataset_root=dataset_root,
                    win_sec=4.0,
                    downsample_rate=5,
                    stride=10,
                )
            )

            os.utime(motion_path, ns=(motion_path.stat().st_atime_ns, motion_path.stat().st_mtime_ns + 1))
            self.assertFalse(
                hme_feature_cache_is_current(
                    cache_dir,
                    dataset_root=dataset_root,
                    win_sec=4.0,
                    downsample_rate=5,
                    stride=10,
                )
            )


class LimmtPhysicalFilterTest(unittest.TestCase):
    def test_run_filter_writes_filenames_view_without_copying_dataset(self) -> None:
        class FakeRunner:
            backend = "test"

            def __init__(self, mjcf_path, *, batch_size, device, contact_nconmax) -> None:
                del batch_size, device, contact_nconmax
                self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
                self.device = torch.device("cpu")
                self.joint_names: list[str] = []

            def forward_kinematics(self, qpos: torch.Tensor, qvel: torch.Tensor) -> dict[str, torch.Tensor]:
                del qvel
                frames = qpos.shape[0]
                return {
                    "body_pos_w": torch.zeros((frames, self.model.nbody, 3)),
                    "body_lin_vel_w": torch.zeros((frames, self.model.nbody, 3)),
                    "body_ang_vel_w": torch.zeros((frames, self.model.nbody, 3)),
                    "joint_vel": torch.zeros((frames, 0)),
                }

            def forward_contact_summary(
                self,
                qpos: torch.Tensor,
                qvel: torch.Tensor,
                *,
                floor_geom_id: int,
            ) -> dict[str, torch.Tensor]:
                del qvel, floor_geom_id
                floor_min_dist = torch.zeros(qpos.shape[0])
                floor_min_dist[qpos.shape[0] // 2 :] = -1.0
                return {
                    "floor_min_dist": floor_min_dist,
                    "non_floor_contact_count": torch.zeros(qpos.shape[0]),
                    "contact_buffer_saturated": torch.zeros(qpos.shape[0], dtype=torch.bool),
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            model_path = dataset_root / "model.xml"
            model_path.parent.mkdir(parents=True)
            model_path.write_text(
                """
                <mujoco>
                  <asset>
                    <material name="test" rgba="1 1 1 1"/>
                  </asset>
                  <worldbody>
                    <body name="base">
                      <freejoint name="root"/>
                      <body name="left_ankle_roll_link"><geom type="sphere" size="0.01"/></body>
                      <body name="right_ankle_roll_link"><geom type="sphere" size="0.01"/></body>
                    </body>
                  </worldbody>
                </mujoco>
                """,
                encoding="utf-8",
            )
            qpos = np.zeros((2, 7), dtype=np.float32)
            qpos[:, 3] = 1.0
            save_motion(dataset_root / "motions" / "a.npz", qpos)
            save_motion(dataset_root / "motions" / "b.npz", qpos)
            write_manifest(
                dataset_root,
                dataset_name="input",
                mjcf=model_path,
                timestep=0.02,
                qpos_names=["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz"],
                num_motions=2,
                source={"kind": "test"},
                total_hours=4 * 0.02 / 3600.0,
            )
            motion_samples = [
                {"qpos": torch.from_numpy(qpos), "qvel": torch.zeros((2, 6)), "rel_motion": Path("motions/a.npz")},
                {"qpos": torch.from_numpy(qpos), "qvel": torch.zeros((2, 6)), "rel_motion": Path("motions/b.npz")},
            ]
            project_root = temp / "filter"
            with (
                patch.object(physical_filter, "FKRunner", FakeRunner),
                patch.object(physical_filter, "build_motion_loader", return_value=motion_samples),
            ):
                summary = physical_filter.run_filter(
                    physical_filter.PhysicalFilterArgs(
                        input_path=str(dataset_root),
                        project_path=str(project_root),
                        device="cpu",
                        num_workers=0,
                    )
                )

            self.assertEqual((project_root / "filenames.txt").read_text(), "a.npz\n")
            self.assertEqual(summary["filenames_path"], str(project_root / "filenames.txt"))
            self.assertFalse((project_root / "passed").exists())
            self.assertNotIn("pass_dataset_root", summary)

    def test_fk_runner_cpu_contact_summary_reports_floor_distances(self) -> None:
        xml = """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 .1"/>
            <body name="ball" pos="0 0 1">
              <freejoint/>
              <geom name="ball_geom" type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "sphere.xml"
            model_path.write_text(xml, encoding="utf-8")
            runner = FKRunner(model_path, batch_size=2, device="cpu", contact_nconmax=8)
            floor_id = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            qpos = torch.tensor(
                [
                    [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.00, 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
            qvel = torch.zeros((2, 6), dtype=torch.float32)

            summary = runner.forward_contact_summary(qpos, qvel, floor_geom_id=floor_id)

        self.assertLess(float(summary["floor_min_dist"][0]), -0.04)
        self.assertEqual(float(summary["floor_min_dist"][1]), 100.0)
        self.assertEqual(float(summary["non_floor_contact_count"].sum()), 0.0)

    def test_fk_runner_cpu_contact_summary_counts_non_floor_contacts(self) -> None:
        xml = """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" pos="0 0 -1" size="2 2 .1"/>
            <geom name="fixed_ball" type="sphere" pos="0 0 0.5" size="0.1"/>
            <body name="ball" pos="0 0 0.5">
              <freejoint/>
              <geom name="moving_ball" type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "self_collision.xml"
            model_path.write_text(xml, encoding="utf-8")
            runner = FKRunner(model_path, batch_size=1, device="cpu", contact_nconmax=8)
            floor_id = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            qpos = torch.tensor([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            qvel = torch.zeros((1, 6), dtype=torch.float32)

            summary = runner.forward_contact_summary(qpos, qvel, floor_geom_id=floor_id)

        self.assertEqual(float(summary["floor_min_dist"][0]), 100.0)
        self.assertGreaterEqual(float(summary["non_floor_contact_count"][0]), 1.0)

    def test_fk_runner_warp_contact_summary_clears_stale_contacts(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        try:
            import mujoco_warp  # noqa: F401
            import warp  # noqa: F401
        except ImportError:
            self.skipTest("mujoco_warp/warp is not available")

        xml = """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 .1"/>
            <body name="ball" pos="0 0 1">
              <freejoint/>
              <geom name="ball_geom" type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "warp_stale_contact.xml"
            model_path.write_text(xml, encoding="utf-8")
            runner = FKRunner(model_path, batch_size=1, device="cuda", contact_nconmax=8)
            floor_id = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            penetrating = torch.tensor([[0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            high = torch.tensor([[0.0, 0.0, 1.00, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            qvel = torch.zeros((1, 6), dtype=torch.float32)

            first = runner.forward_contact_summary(penetrating, qvel, floor_geom_id=floor_id)
            runner.forward_kinematics(high, qvel)
            second = runner.forward_contact_summary(high, qvel, floor_geom_id=floor_id)

        self.assertLess(float(first["floor_min_dist"][0]), -0.04)
        self.assertEqual(float(second["floor_min_dist"][0]), 100.0)

    def test_score_motion_aggregates_penalties_and_threshold(self) -> None:
        fk_results = {
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
        contact_summary = {
            "floor_min_dist": torch.tensor([100.0, 100.0, 100.0], dtype=torch.float32),
            "non_floor_contact_count": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
            "contact_buffer_saturated": torch.zeros((3,), dtype=torch.bool),
        }
        score_row = physical_filter._score_motion(
            rel_motion="motions/a.npz",
            fk_results=fk_results,
            qvel=torch.zeros((3, 3), dtype=torch.float32),
            fps=50.0,
            config=physical_filter.PhysicalFilterConfig(pass_threshold=90.0),
            weights=physical_filter.PhysicalScoreWeights(),
            contact_summary=contact_summary,
            foot_body_ids=(1, 2),
        )
        self.assertEqual(score_row["motion"], "motions/a.npz")
        self.assertGreater(score_row["foot_sliding"], 0.0)
        self.assertGreater(score_row["self_collision"], 0.0)
        self.assertLessEqual(score_row["physical_score"], 100.0)

    def test_score_motion_uses_contact_summary_with_official_averaging(self) -> None:
        fk_results = {
            "body_pos_w": torch.zeros((3, 3, 3), dtype=torch.float32),
            "body_lin_vel_w": torch.zeros((3, 3, 3), dtype=torch.float32),
            "body_ang_vel_w": torch.zeros((3, 3, 3), dtype=torch.float32),
            "joint_vel": torch.zeros((3, 2), dtype=torch.float32),
        }
        contact_summary = {
            "floor_min_dist": torch.tensor([100.0, -0.05, -0.005], dtype=torch.float32),
            "non_floor_contact_count": torch.tensor([0.0, 2.0, 12.0], dtype=torch.float32),
            "contact_buffer_saturated": torch.zeros((3,), dtype=torch.bool),
        }
        config = physical_filter.PhysicalFilterConfig(
            pass_threshold=90.0,
            floating_window_sec=1.0,
            floating_distance=0.05,
            penetration_margin=0.01,
            self_collision_count_clip=10.0,
        )

        score_row = physical_filter._score_motion(
            rel_motion="motions/contact.npz",
            fk_results=fk_results,
            qvel=torch.zeros((3, 3), dtype=torch.float32),
            fps=1.0,
            config=config,
            weights=physical_filter.PhysicalScoreWeights(),
            contact_summary=contact_summary,
            foot_body_ids=(1, 2),
        )

        self.assertAlmostEqual(score_row["penetration"], 0.04 / 3.0, places=6)
        self.assertAlmostEqual(score_row["floating_frames_ratio"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(score_row["self_collision"], 4.0, places=6)
        self.assertEqual(physical_filter.PhysicalScoreWeights().penetration, 216.62)


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
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["dataset_name"], "subset")
            self.assertEqual(manifest_payload["num_motions"], 1)
            self.assertTrue((temp / "subset" / "motions" / "b.npz").is_file())
            self.assertFalse(stale.exists())
            self.assertEqual(manifest_payload["source"]["limmt_gqs"]["ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
