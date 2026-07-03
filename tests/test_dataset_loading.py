from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from any4hdmi.dataset.loaders import load_any4hdmi_dataset
from any4hdmi.dataset.loading import resolve_dataset_context, resolve_motion_filenames


def _write_motion(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, qpos=np.zeros((2, 3), dtype=np.float32))


class DatasetLoadingTest(unittest.TestCase):
    def test_filenames_path_resolves_relative_to_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            _write_motion(dataset_root / "motions" / "a.npz")
            _write_motion(dataset_root / "motions" / "nested" / "b.npz")
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n",
                encoding="utf-8",
            )
            filenames_path = dataset_root / "views" / "view.txt"
            filenames_path.parent.mkdir()
            filenames_path.write_text("nested/b.npz\nmotions/a.npz\n", encoding="utf-8")

            filenames = resolve_motion_filenames(dataset_root, filenames_path="views/view.txt")
            context = resolve_dataset_context([dataset_root], motion_filenames=filenames)

            self.assertEqual(
                [path.relative_to(dataset_root / "motions").as_posix() for path in context.motion_paths],
                ["nested/b.npz", "a.npz"],
            )

    def test_loader_resolves_filenames_path_from_dataset_root_not_base_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            base_dir = temp / "base"
            base_dir.mkdir()
            _write_motion(dataset_root / "motions" / "nested" / "b.npz")
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n",
                encoding="utf-8",
            )
            filenames_path = dataset_root / "views" / "view.txt"
            filenames_path.parent.mkdir()
            filenames_path.write_text("nested/b.npz\n", encoding="utf-8")

            cache = mock.Mock()
            cache_entry = object()
            cache.get_or_build.return_value = cache_entry
            with (
                mock.patch("any4hdmi.dataset.loaders.FKCache.from_inputs", return_value=cache) as from_inputs,
                mock.patch("any4hdmi.dataset.loaders._prune_cache_entry", return_value=cache_entry),
                mock.patch("any4hdmi.dataset.loaders.FullMotionDataset.from_cache_entry", return_value="dataset"),
            ):
                result = load_any4hdmi_dataset(
                    root_path=dataset_root,
                    target_fps=50,
                    base_dir=base_dir,
                    filenames_path="views/view.txt",
                    num_envs=1,
                )

            self.assertEqual(result, "dataset")
            self.assertEqual(from_inputs.call_args.kwargs["motion_filenames"], ["nested/b.npz"])

    def test_filenames_reject_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            _write_motion(dataset_root / "motions" / "a.npz")
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid motion filename"):
                resolve_dataset_context([dataset_root], motion_filenames=["../a.npz"])

    def test_filenames_path_missing_entry_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            _write_motion(dataset_root / "motions" / "a.npz")
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n",
                encoding="utf-8",
            )
            filenames_path = dataset_root / "views" / "view.txt"
            filenames_path.parent.mkdir()
            filenames_path.write_text("missing.npz\n", encoding="utf-8")

            filenames = resolve_motion_filenames(dataset_root, filenames_path="views/view.txt")
            with self.assertRaisesRegex(FileNotFoundError, "missing.npz"):
                resolve_dataset_context([dataset_root], motion_filenames=filenames)

    def test_empty_subdir_does_not_fallback_to_dataset_motions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            _write_motion(dataset_root / "motions" / "a.npz")
            empty_subset = dataset_root / "empty_subset"
            empty_subset.mkdir()
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "empty_subset"):
                resolve_dataset_context([empty_subset])


if __name__ == "__main__":
    unittest.main()
