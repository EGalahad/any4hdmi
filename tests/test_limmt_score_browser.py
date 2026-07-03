from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from any4hdmi.limmt.score_browser import (
    build_score_bins,
    flatten_bin_motion_choices,
    load_score_rows,
    plot_score_histogram,
    score_values_for_names,
)


class LimmtScoreBrowserTest(unittest.TestCase):
    def test_load_score_rows_and_score_values_for_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            score_path = self._write_scores(
                Path(temp_dir),
                [
                    {"motion": "motions/a.npz", "physical_score": "98.5", "status": "kept"},
                    {"motion": "motions/b.npz", "physical_score": 73, "status": "rejected"},
                ],
            )

            rows = load_score_rows(score_path)
            self.assertEqual(rows[0]["motion"], "motions/a.npz")
            self.assertEqual(rows[0]["physical_score"], 98.5)
            self.assertEqual(rows[1]["physical_score"], 73.0)

            scores = score_values_for_names(score_path, names=["motions/b.npz", "missing.npz", "motions/a.npz"])
            self.assertEqual(list(scores), ["motions/b.npz", "missing.npz", "motions/a.npz"])
            self.assertEqual(scores["motions/b.npz"], 73.0)
            self.assertTrue(math.isnan(scores["missing.npz"]))

    def test_build_score_bins_assigns_shared_boundaries_to_higher_range(self) -> None:
        rows = [
            {"motion": "above_100.npz", "physical_score": 100.001},
            {"motion": "edge_100.npz", "physical_score": 100.0},
            {"motion": "edge_095.npz", "physical_score": 95.0},
            {"motion": "inside_095_090.npz", "physical_score": 94.999},
            {"motion": "edge_090.npz", "physical_score": 90.0},
            {"motion": "edge_085.npz", "physical_score": 85.0},
            {"motion": "edge_080.npz", "physical_score": 80.0},
            {"motion": "edge_075.npz", "physical_score": 75.0},
            {"motion": "edge_070.npz", "physical_score": 70.0},
            {"motion": "edge_065.npz", "physical_score": 65.0},
            {"motion": "edge_060.npz", "physical_score": 60.0},
            {"motion": "below_060.npz", "physical_score": 59.999},
        ]

        score_bins = build_score_bins(rows, samples_per_bin=10, seed=1)

        self.assertEqual(
            [score_bin.label for score_bin in score_bins],
            ["100-95", "95-90", "90-85", "85-80", "80-75", "75-70", "70-65", "65-60"],
        )
        motions_by_label = {score_bin.label: set(score_bin.motions) for score_bin in score_bins}
        self.assertEqual(motions_by_label["100-95"], {"edge_095.npz", "edge_100.npz"})
        self.assertEqual(motions_by_label["95-90"], {"edge_090.npz", "inside_095_090.npz"})
        self.assertEqual(motions_by_label["90-85"], {"edge_085.npz"})
        self.assertEqual(motions_by_label["85-80"], {"edge_080.npz"})
        self.assertEqual(motions_by_label["80-75"], {"edge_075.npz"})
        self.assertEqual(motions_by_label["75-70"], {"edge_070.npz"})
        self.assertEqual(motions_by_label["70-65"], {"edge_065.npz"})
        self.assertEqual(motions_by_label["65-60"], {"edge_060.npz"})
        self.assertFalse(any("above_100.npz" in score_bin.motions for score_bin in score_bins))
        self.assertFalse(any("below_060.npz" in score_bin.motions for score_bin in score_bins))

    def test_build_score_bins_caps_sampling_deterministically(self) -> None:
        rows = [
            {"motion": f"motions/top_{idx:02d}.npz", "physical_score": 99.0 - idx * 0.01}
            for idx in range(20)
        ]

        first = build_score_bins(rows, samples_per_bin=10, seed=123)[0].motions
        second = build_score_bins(rows, samples_per_bin=10, seed=123)[0].motions

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(first, tuple(sorted(first)))
        self.assertLessEqual(set(first), {row["motion"] for row in rows})

    def test_flatten_bin_motion_choices_includes_score_range(self) -> None:
        score_bins = build_score_bins(
            [
                {"motion": "motions/a.npz", "physical_score": 99.0},
                {"motion": "motions/b.npz", "physical_score": 91.0},
            ],
            samples_per_bin=10,
        )

        choices = flatten_bin_motion_choices(score_bins)

        self.assertEqual(choices, ["100-95 | motions/a.npz", "95-90 | motions/b.npz"])

    def test_plot_score_histogram_creates_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            score_path = self._write_scores(
                temp,
                [
                    {"motion": "motions/a.npz", "physical_score": 99.0},
                    {"motion": "motions/b.npz", "physical_score": 90.0},
                    {"motion": "motions/c.npz", "physical_score": 60.0},
                ],
            )
            out_path = temp / "plots" / "score_histogram.png"

            written_path = plot_score_histogram(load_score_rows(score_path), out_path)

            self.assertEqual(written_path, out_path)
            self.assertTrue(out_path.is_file())
            self.assertGreater(out_path.stat().st_size, 0)

    def _write_scores(self, root: Path, details: list[dict[str, object]]) -> Path:
        score_path = root / "scores.json"
        score_path.write_text(
            json.dumps({"summary": {"total_motions": len(details)}, "details": details}) + "\n",
            encoding="utf-8",
        )
        return score_path


if __name__ == "__main__":
    unittest.main()
