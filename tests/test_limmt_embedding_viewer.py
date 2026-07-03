from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from any4hdmi.limmt.embedding_viewer import (
    make_embedding_projection,
    normalize_points_for_viewer,
    project_embeddings_3d,
    sample_embeddings,
)


class LimmtEmbeddingViewerTest(unittest.TestCase):
    def test_sample_embeddings_is_deterministic_and_source_ordered(self) -> None:
        names = [f"motion_{idx:03d}" for idx in range(20)]
        embeddings = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)

        sampled_names_a, sampled_embeddings_a, source_indices_a = sample_embeddings(
            names,
            embeddings,
            fraction=0.3,
            seed=123,
        )
        sampled_names_b, sampled_embeddings_b, source_indices_b = sample_embeddings(
            names,
            embeddings,
            fraction=0.3,
            seed=123,
        )

        self.assertEqual(sampled_names_a, sampled_names_b)
        np.testing.assert_array_equal(sampled_embeddings_a, sampled_embeddings_b)
        np.testing.assert_array_equal(source_indices_a, source_indices_b)
        self.assertEqual(len(sampled_names_a), 6)
        np.testing.assert_array_equal(source_indices_a, np.sort(source_indices_a))
        np.testing.assert_array_equal(sampled_embeddings_a, embeddings[source_indices_a])

    def test_sample_embeddings_fraction_point_one_keeps_ten_percent(self) -> None:
        names = [f"motion_{idx:03d}" for idx in range(20)]
        embeddings = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)

        sampled_names, sampled_embeddings, source_indices = sample_embeddings(
            names,
            embeddings,
            fraction=0.1,
            seed=7,
        )

        self.assertEqual(len(sampled_names), 2)
        self.assertEqual(sampled_embeddings.shape, (2, 3))
        self.assertEqual(source_indices.shape, (2,))
        self.assertEqual(sampled_names, [names[int(source_idx)] for source_idx in source_indices])

    def test_sample_embeddings_fraction_one_preserves_all_rows(self) -> None:
        names = ["a", "b", "c"]
        embeddings = np.arange(9, dtype=np.float32).reshape(3, 3)

        sampled_names, sampled_embeddings, source_indices = sample_embeddings(
            names,
            embeddings,
            fraction=1.0,
            max_points=None,
            seed=99,
        )

        self.assertEqual(sampled_names, names)
        np.testing.assert_array_equal(sampled_embeddings, embeddings)
        np.testing.assert_array_equal(source_indices, np.asarray([0, 1, 2], dtype=np.int64))

    def test_sample_embeddings_validates_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "fraction"):
            sample_embeddings(["a"], np.zeros((1, 2), dtype=np.float32), fraction=0.0)
        with self.assertRaisesRegex(ValueError, "fraction"):
            sample_embeddings(["a"], np.zeros((1, 2), dtype=np.float32), fraction=1.1)

    def test_project_embeddings_pca_handles_small_n_and_pads_to_three_dims(self) -> None:
        embeddings = np.asarray([[0.0], [2.0]], dtype=np.float32)

        points = project_embeddings_3d(embeddings, method="pca", seed=3)

        self.assertEqual(points.shape, (2, 3))
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(points[:, 1:], 0.0, atol=1e-6)

    def test_project_embeddings_tsne_smoke(self) -> None:
        embeddings = np.asarray(
            [
                [0.0, 0.1, 0.2, 0.3],
                [1.0, 1.1, 1.2, 1.3],
                [2.0, 2.1, 2.2, 2.3],
                [3.0, 3.1, 3.2, 3.3],
                [4.0, 4.1, 4.2, 4.3],
            ],
            dtype=np.float32,
        )

        points = project_embeddings_3d(embeddings, method="tsne", perplexity=1.0, seed=5)

        self.assertEqual(points.shape, (5, 3))
        self.assertTrue(np.isfinite(points).all())

    def test_project_embeddings_umap_reports_missing_optional_dependency(self) -> None:
        embeddings = np.arange(15, dtype=np.float32).reshape(5, 3)
        if importlib.util.find_spec("umap") is not None:
            self.skipTest("umap-learn is installed in this environment")

        with self.assertRaisesRegex(RuntimeError, "umap-learn"):
            project_embeddings_3d(embeddings, method="umap")

    def test_normalize_points_for_viewer_centers_and_scales(self) -> None:
        points = np.asarray([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0], [14.0, 0.0, 0.0]], dtype=np.float32)

        normalized = normalize_points_for_viewer(points)

        self.assertEqual(normalized.shape, (3, 3))
        np.testing.assert_allclose(np.mean(normalized, axis=0), 0.0, atol=1e-6)
        self.assertGreater(np.max(np.abs(normalized[:, 0])), 0.0)

    def test_make_embedding_projection_combines_sampling_projection_and_normalization(self) -> None:
        names = [f"motion_{idx:03d}" for idx in range(10)]
        embeddings = np.arange(40, dtype=np.float32).reshape(10, 4)

        projection = make_embedding_projection(
            names,
            embeddings,
            fraction=0.2,
            method="pca",
            seed=11,
        )

        self.assertEqual(len(projection.names), 2)
        self.assertEqual(projection.points.shape, (2, 3))
        self.assertEqual(projection.source_indices.shape, (2,))


if __name__ == "__main__":
    unittest.main()
