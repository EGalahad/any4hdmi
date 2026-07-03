from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from any4hdmi.scripts.viewer import _resolve_motion_path


class ViewerTest(unittest.TestCase):
    def test_resolve_motion_path_accepts_hf_uri(self) -> None:
        expected = Path("/tmp/hf-cache/snapshot/motions/example.npz")
        motion_uri = "hf://example/dataset/motions/example.npz"

        with mock.patch(
            "any4hdmi.scripts.viewer.resolve_input_paths",
            return_value=[expected],
        ) as resolve_input_paths:
            resolved = _resolve_motion_path(motion_uri)

        self.assertEqual(resolved, expected)
        resolve_input_paths.assert_called_once_with(Path.cwd(), motion_uri)


if __name__ == "__main__":
    unittest.main()
