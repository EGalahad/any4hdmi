from __future__ import annotations

import json
import socket
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch

from any4hdmi.dataset.fk_cache import FKCacheEntry
from any4hdmi.dataset.fp16_backing import (
    FP16_BACKING_DTYPE,
    _backing_key,
    prepare_fp16_cache_entry,
)


class FP16BackingTest(unittest.TestCase):
    def _entry(self, root: Path) -> FKCacheEntry:
        frames = 5
        values = torch.arange(frames * 2 * 3, dtype=torch.float32).reshape(
            frames, 2, 3
        )
        storage = {
            "motion_id": torch.zeros(frames, dtype=torch.long),
            "step": torch.arange(frames, dtype=torch.long),
            "body_pos_w": values,
            "body_lin_vel_w": values + 1,
            "body_quat_w": torch.arange(
                frames * 2 * 4, dtype=torch.float32
            ).reshape(frames, 2, 4),
            "body_ang_vel_w": values + 2,
            "joint_pos": torch.arange(frames * 3, dtype=torch.float32).reshape(
                frames, 3
            ),
            "joint_vel": torch.arange(frames * 3, dtype=torch.float32).reshape(
                frames, 3
            )
            + 3,
        }
        return FKCacheEntry(
            cache_entry_dir=root,
            body_names=["a", "b"],
            joint_names=["j0", "j1", "j2"],
            motion_paths=[],
            starts=[0],
            ends=[frames],
            storage_fields=storage,
        )

    def test_fp16_backing_is_shared_memmap_and_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            entry = self._entry(Path(tmp_dir))
            first = prepare_fp16_cache_entry(
                entry, body_names=["b"], joint_names=["j2", "j0"]
            )
            second = prepare_fp16_cache_entry(
                entry, body_names=["b"], joint_names=["j2", "j0"]
            )

            self.assertEqual(first.body_names, ["b"])
            self.assertEqual(first.joint_names, ["j2", "j0"])
            self.assertEqual(first.storage_fields["body_pos_w"].dtype, torch.float16)
            self.assertEqual(first.storage_fields["body_pos_w"].shape, (5, 1, 3))
            torch.testing.assert_close(
                first.storage_fields["body_pos_w"].float(),
                entry.storage_fields["body_pos_w"][:, [1]],
            )
            torch.testing.assert_close(
                first.storage_fields["joint_pos"].float(),
                entry.storage_fields["joint_pos"][:, [2, 0]],
            )
            self.assertEqual(entry.storage_fields["body_pos_w"].dtype, torch.float32)
            torch.testing.assert_close(
                second.storage_fields["joint_vel"], first.storage_fields["joint_vel"]
            )
            backing_roots = list(Path(tmp_dir).glob("windowed_backing_*"))
            self.assertEqual(len(backing_roots), 1)
            self.assertTrue((backing_roots[0] / "ready.flag").is_file())

    def test_dead_same_host_owner_lock_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            entry = self._entry(root)
            key = _backing_key(
                body_names=entry.body_names,
                joint_names=entry.joint_names,
                dtype=FP16_BACKING_DTYPE,
            )
            lock = root / f"windowed_backing_{key}.lock"
            lock.mkdir()
            (lock / "owner.json").write_text(
                json.dumps({"host": socket.gethostname(), "pid": 999999999})
            )

            with mock.patch(
                "any4hdmi.dataset.fp16_backing.os.kill",
                side_effect=ProcessLookupError,
            ):
                prepared = prepare_fp16_cache_entry(
                    entry,
                    body_names=None,
                    joint_names=None,
                )

            self.assertFalse(lock.exists())
            self.assertEqual(
                prepared.storage_fields["body_pos_w"].dtype, torch.float16
            )


if __name__ == "__main__":
    unittest.main()
