from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from any4hdmi.dataset.fk_cache import FKCacheEntry
from any4hdmi.dataset.full import FullMotionDataset
from any4hdmi.dataset.loaders import (
    build_runtime_dataset,
    load_any4hdmi_dataset,
    partition_cache_entry,
    prepare_cache_entry,
)
from any4hdmi.dataset.loading import resolve_dataset_context, resolve_motion_filenames
from any4hdmi.dataset.windowed import WindowedMotionDataset


def _write_motion(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, qpos=np.zeros((2, 3), dtype=np.float32))


def _entry(root: Path) -> FKCacheEntry:
    lengths = [3, 5, 2, 6]
    ends = torch.tensor(lengths).cumsum(0).tolist()
    starts = [0, *ends[:-1]]
    frames = sum(lengths)
    body_shape = (frames, 1, 3)
    return FKCacheEntry(
        cache_entry_dir=root,
        body_names=["pelvis"],
        joint_names=["joint"],
        motion_paths=[root / f"motion_{index}.npz" for index in range(4)],
        starts=starts,
        ends=ends,
        storage_fields={
            "motion_id": torch.repeat_interleave(
                torch.arange(4), torch.tensor(lengths)
            ),
            "step": torch.cat([torch.arange(length) for length in lengths]),
            "body_pos_w": torch.zeros(body_shape, dtype=torch.float16),
            "body_lin_vel_w": torch.zeros(body_shape, dtype=torch.float16),
            "body_quat_w": torch.zeros((frames, 1, 4), dtype=torch.float16),
            "body_ang_vel_w": torch.zeros(body_shape, dtype=torch.float16),
            "joint_pos": torch.zeros((frames, 1), dtype=torch.float16),
            "joint_vel": torch.zeros((frames, 1), dtype=torch.float16),
        },
    )


class DatasetLoadingTest(unittest.TestCase):
    def test_filenames_path_resolves_relative_to_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            _write_motion(dataset_root / "motions" / "a.npz")
            _write_motion(dataset_root / "motions" / "nested" / "b.npz")
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n"
            )
            filenames_path = dataset_root / "views" / "view.txt"
            filenames_path.parent.mkdir()
            filenames_path.write_text("nested/b.npz\nmotions/a.npz\n")

            filenames = resolve_motion_filenames(
                dataset_root, filenames_path="views/view.txt"
            )
            context = resolve_dataset_context(
                [dataset_root], motion_filenames=filenames
            )

            self.assertEqual(
                [
                    path.relative_to(dataset_root / "motions").as_posix()
                    for path in context.motion_paths
                ],
                ["nested/b.npz", "a.npz"],
            )

    def test_prepare_resolves_filenames_then_builds_neutral_fp16_backing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            base_dir = temp / "base"
            base_dir.mkdir()
            _write_motion(dataset_root / "motions" / "nested" / "b.npz")
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n"
            )
            filenames_path = dataset_root / "views" / "view.txt"
            filenames_path.parent.mkdir()
            filenames_path.write_text("nested/b.npz\n")
            raw_entry = object()
            prepared_entry = object()
            cache = mock.Mock()
            cache.get_or_build.return_value = raw_entry

            with (
                mock.patch(
                    "any4hdmi.dataset.loaders.FKCache.from_inputs",
                    return_value=cache,
                ) as from_inputs,
                mock.patch(
                    "any4hdmi.dataset.loaders.prepare_fp16_cache_entry",
                    return_value=prepared_entry,
                ) as prepare,
            ):
                result = prepare_cache_entry(
                    root_path=dataset_root,
                    target_fps=50,
                    base_dir=base_dir,
                    filenames_path="views/view.txt",
                    body_names=["pelvis"],
                    joint_names=["joint"],
                )

            self.assertIs(result, prepared_entry)
            self.assertEqual(
                from_inputs.call_args.kwargs["motion_filenames"], ["nested/b.npz"]
            )
            prepare.assert_called_once_with(
                raw_entry,
                body_names=["pelvis"],
                joint_names=["joint"],
            )

    def test_runtime_factory_is_independent_of_distributed_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            entry = _entry(Path(temp_dir))
            with mock.patch.dict(
                "os.environ", {"RANK": "invalid", "WORLD_SIZE": "invalid"}
            ):
                resident = build_runtime_dataset(
                    entry, full_motion=True, num_envs=2
                )
                windowed = build_runtime_dataset(
                    entry,
                    full_motion=False,
                    num_envs=2,
                    windowed_next_window_device="cpu",
                    windowed_pin_window_load=False,
                )

            self.assertIsInstance(resident, FullMotionDataset)
            self.assertIsInstance(windowed, WindowedMotionDataset)

    def test_four_combinations_preserve_partition_and_runtime_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            entry = _entry(Path(temp_dir))
            for shard in (False, True):
                partitioned = partition_cache_entry(
                    entry,
                    shard=shard,
                    rank=1 if shard else 0,
                    world_size=2 if shard else 1,
                )
                expected_paths = entry.motion_paths[2:] if shard else entry.motion_paths
                self.assertEqual(partitioned.motion_paths, expected_paths)
                self.assertEqual(
                    partitioned.storage_fields["body_pos_w"].dtype, torch.float16
                )
                for full_motion in (False, True):
                    dataset = build_runtime_dataset(
                        partitioned,
                        full_motion=full_motion,
                        num_envs=2,
                        windowed_next_window_device="cpu",
                        windowed_pin_window_load=False,
                    )
                    self.assertEqual(dataset.motion_paths, expected_paths)
                    if full_motion:
                        self.assertIsInstance(dataset, FullMotionDataset)
                        self.assertEqual(dataset.data.body_pos_w.dtype, torch.float16)
                        output = dataset.get_slice(
                            torch.tensor([0]),
                            torch.tensor([0]),
                            torch.tensor([0]),
                        )
                        self.assertEqual(output.body_pos_w.dtype, torch.float32)
                    else:
                        self.assertIsInstance(dataset, WindowedMotionDataset)
                        self.assertEqual(
                            dataset._storage_cpu["body_pos_w"].dtype, torch.float16
                        )
                        self.assertEqual(
                            dataset._current_window.body_pos_w.dtype, torch.float32
                        )

    def test_loader_passes_explicit_context_only_to_partition_stage(self) -> None:
        entry = mock.Mock()
        dataset = object()
        with (
            mock.patch(
                "any4hdmi.dataset.loaders.prepare_cache_entry", return_value=entry
            ),
            mock.patch(
                "any4hdmi.dataset.loaders.partition_cache_entry", return_value=entry
            ) as partition,
            mock.patch(
                "any4hdmi.dataset.loaders.build_runtime_dataset",
                return_value=dataset,
            ),
        ):
            result = load_any4hdmi_dataset(
                root_path="unused",
                target_fps=50,
                base_dir=Path("."),
                num_envs=1,
                full_motion=False,
                shard=False,
                rank=3,
                world_size=8,
            )

        self.assertIs(result, dataset)
        partition.assert_called_once_with(
            entry, shard=False, rank=3, world_size=8
        )

    def test_loader_never_reads_distributed_environment(self) -> None:
        with mock.patch.dict(
            "os.environ", {"RANK": "invalid", "WORLD_SIZE": "invalid"}
        ):
            with (
                mock.patch(
                    "any4hdmi.dataset.loaders.prepare_cache_entry",
                    return_value=_entry(Path(".")),
                ),
                mock.patch(
                    "any4hdmi.dataset.loaders.build_runtime_dataset",
                    return_value=object(),
                ),
            ):
                load_any4hdmi_dataset(
                    root_path="unused",
                    target_fps=50,
                    base_dir=Path("."),
                    num_envs=1,
                    full_motion=True,
                    shard=False,
                )

    def test_filenames_reject_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            _write_motion(dataset_root / "motions" / "a.npz")
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "Invalid motion filename"):
                resolve_dataset_context(
                    [dataset_root], motion_filenames=["../a.npz"]
                )

    def test_filenames_accept_hf_snapshot_blob_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_cache = Path(temp_dir) / "datasets--org--motions"
            snapshot_root = repo_cache / "snapshots" / "revision"
            motion_path = snapshot_root / "motions" / "a.npz"
            blob_path = repo_cache / "blobs" / "motion-blob"
            blob_path.parent.mkdir(parents=True)
            with blob_path.open("wb") as blob_file:
                np.savez_compressed(
                    blob_file, qpos=np.zeros((2, 3), dtype=np.float32)
                )
            motion_path.parent.mkdir(parents=True)
            motion_path.symlink_to("../../../blobs/motion-blob")
            (snapshot_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n"
            )

            context = resolve_dataset_context(
                [snapshot_root], motion_filenames=["a.npz"]
            )

            self.assertEqual(context.motion_paths, [motion_path.absolute()])
            self.assertTrue(context.motion_paths[0].is_symlink())

    def test_filenames_reject_non_hf_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            motion_path = dataset_root / "motions" / "a.npz"
            outside_path = temp / "outside.npz"
            _write_motion(outside_path)
            motion_path.parent.mkdir(parents=True)
            motion_path.symlink_to(outside_path)
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n"
            )

            with self.assertRaisesRegex(ValueError, "escapes motions root"):
                resolve_dataset_context(
                    [dataset_root], motion_filenames=["a.npz"]
                )

    def test_filenames_reject_parent_directory_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            outside_dir = temp / "outside"
            _write_motion(outside_dir / "a.npz")
            motions_root = dataset_root / "motions"
            motions_root.mkdir(parents=True)
            (motions_root / "linked").symlink_to(outside_dir, target_is_directory=True)
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n"
            )

            with self.assertRaisesRegex(ValueError, "escapes motions root"):
                resolve_dataset_context(
                    [dataset_root], motion_filenames=["linked/a.npz"]
                )

    def test_filenames_path_missing_entry_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            _write_motion(dataset_root / "motions" / "a.npz")
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n"
            )
            filenames_path = dataset_root / "views" / "view.txt"
            filenames_path.parent.mkdir()
            filenames_path.write_text("missing.npz\n")
            filenames = resolve_motion_filenames(
                dataset_root, filenames_path="views/view.txt"
            )
            with self.assertRaisesRegex(FileNotFoundError, "missing.npz"):
                resolve_dataset_context(
                    [dataset_root], motion_filenames=filenames
                )

    def test_empty_subdir_does_not_fallback_to_dataset_motions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            _write_motion(dataset_root / "motions" / "a.npz")
            empty_subset = dataset_root / "empty_subset"
            empty_subset.mkdir()
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n"
            )
            with self.assertRaisesRegex(RuntimeError, "empty_subset"):
                resolve_dataset_context([empty_subset])


if __name__ == "__main__":
    unittest.main()
