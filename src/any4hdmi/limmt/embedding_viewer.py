from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


ProjectionMethod = Literal["tsne", "pca", "umap"]


@dataclass(frozen=True)
class EmbeddingProjection:
    """Sampled motion names and their projected 3D embedding coordinates."""

    names: list[str]
    points: np.ndarray
    source_indices: np.ndarray


def sample_embeddings(
    names: Sequence[str],
    embeddings: np.ndarray,
    *,
    fraction: float = 1.0,
    max_points: int | None = None,
    seed: int = 42,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return a deterministic random subset of LIMMT embeddings.

    The selected source indices are sorted so downstream plots preserve dataset
    order while still choosing the subset randomly.
    """

    names_list = [str(name) for name in names]
    embedding_matrix = _as_embedding_matrix(embeddings)
    if len(names_list) != embedding_matrix.shape[0]:
        raise ValueError(f"names length {len(names_list)} does not match embeddings rows {embedding_matrix.shape[0]}")
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError("fraction must be in the interval (0, 1]")
    if max_points is not None and int(max_points) < 1:
        raise ValueError("max_points must be positive when provided")

    num_embeddings = len(names_list)
    if num_embeddings == 0:
        source_indices = np.zeros((0,), dtype=np.int64)
        return [], embedding_matrix.copy(), source_indices

    sample_count = min(num_embeddings, max(1, math.ceil(num_embeddings * float(fraction))))
    if max_points is not None:
        sample_count = min(sample_count, int(max_points))

    if sample_count == num_embeddings:
        source_indices = np.arange(num_embeddings, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        source_indices = np.sort(rng.choice(num_embeddings, size=sample_count, replace=False)).astype(np.int64)

    sampled_names = [names_list[int(source_idx)] for source_idx in source_indices]
    return sampled_names, embedding_matrix[source_indices].copy(), source_indices


def project_embeddings_3d(
    embeddings: np.ndarray,
    *,
    method: ProjectionMethod = "tsne",
    perplexity: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """Project an embedding matrix into stable 3D coordinates.

    PCA is used directly for small fast projections. t-SNE uses scikit-learn and
    automatically clamps perplexity below the sample count. UMAP is optional and
    raises a clear error when `umap-learn` is not installed.
    """

    embedding_matrix = _as_embedding_matrix(embeddings)
    method = method.lower()
    if method not in {"tsne", "pca", "umap"}:
        raise ValueError("method must be one of: 'tsne', 'pca', 'umap'")

    num_embeddings, num_features = embedding_matrix.shape
    if num_embeddings == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if num_features == 0:
        raise ValueError("embeddings must include at least one feature column")
    if not np.isfinite(embedding_matrix).all():
        raise ValueError("embeddings must contain only finite values")
    if num_embeddings == 1:
        if method == "umap":
            _import_umap()
        return np.zeros((1, 3), dtype=np.float32)

    if method == "pca":
        return _project_pca_3d(embedding_matrix, seed=seed)
    if method == "tsne":
        return _project_tsne_3d(embedding_matrix, perplexity=perplexity, seed=seed)
    return _project_umap_3d(embedding_matrix, seed=seed)


def normalize_points_for_viewer(points: np.ndarray) -> np.ndarray:
    """Center and scale 3D points so Viser cameras start with a useful view."""

    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points_array.shape}")
    if points_array.shape[0] == 0:
        return points_array.copy()
    if not np.isfinite(points_array).all():
        raise ValueError("points must contain only finite values")

    centered = points_array - np.mean(points_array, axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    positive_radii = radii[radii > 0.0]
    if positive_radii.size == 0:
        return np.zeros_like(centered, dtype=np.float32)
    scale = float(np.percentile(positive_radii, 95))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(positive_radii))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (centered / scale).astype(np.float32, copy=False)


def make_embedding_projection(
    names: Sequence[str],
    embeddings: np.ndarray,
    *,
    fraction: float = 1.0,
    max_points: int | None = None,
    method: ProjectionMethod = "tsne",
    perplexity: float = 30.0,
    seed: int = 42,
    normalize: bool = True,
) -> EmbeddingProjection:
    """Sample embeddings and project them into 3D for viewer consumption."""

    sampled_names, sampled_embeddings, source_indices = sample_embeddings(
        names,
        embeddings,
        fraction=fraction,
        max_points=max_points,
        seed=seed,
    )
    points = project_embeddings_3d(sampled_embeddings, method=method, perplexity=perplexity, seed=seed)
    if normalize:
        points = normalize_points_for_viewer(points)
    return EmbeddingProjection(names=sampled_names, points=points, source_indices=source_indices)


def run_embedding_motion_viewer(
    projection: EmbeddingProjection,
    *,
    server: Any | None = None,
    on_select: Callable[[str, int], None] | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    label: str = "any4hdmi-limmt-embedding-viewer",
    point_size: float = 0.035,
    pick_radius: float = 0.08,
    position: Sequence[float] = (0.0, 0.0, 0.0),
    block: bool = True,
) -> Any:
    """Run a Viser embedding picker and call `on_select` for selected motions.

    This helper owns the embedding side only. A CLI can either pass an existing
    shared Viser `server` or let this function create one, add a
    `mjviser.ViserMujocoScene` to the same page for the right-side MuJoCo motion
    view, and pass an `on_select(name, source_index)` callback that loads the
    selected motion into that scene. Viser point clouds do not expose a per-point
    click id, so scene clicks are resolved to the nearest point ray.

    When `block=False`, the Viser server is returned to the caller so a CLI can
    run its own MuJoCo playback loop.
    """

    if server is None:
        try:
            import viser
        except ImportError as exc:
            raise ImportError("run_embedding_motion_viewer requires the project viewer dependency `viser`.") from exc
        server = viser.ViserServer(host=host, port=port, label=label)

    points = normalize_points_for_viewer(projection.points)
    if len(projection.names) != points.shape[0]:
        raise ValueError(f"projection names length {len(projection.names)} does not match point rows {points.shape[0]}")
    source_indices = np.asarray(projection.source_indices, dtype=np.int64)
    if source_indices.shape != (points.shape[0],):
        raise ValueError(f"source_indices must have shape ({points.shape[0]},), got {source_indices.shape}")
    position_array = np.asarray(position, dtype=np.float32)
    if position_array.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {position_array.shape}")
    world_points = points + position_array[None, :]

    colors = _colors_from_points(points)
    server.scene.add_point_cloud(
        "/limmt_embeddings",
        points=points,
        colors=colors,
        point_size=point_size,
        point_shape="circle",
        precision="float32",
        position=position_array,
    )

    state: dict[str, int | None] = {"selected": None}

    def select_index(point_idx: int) -> None:
        point_idx = int(np.clip(point_idx, 0, points.shape[0] - 1))
        state["selected"] = point_idx
        selected_name.value = projection.names[point_idx]
        selected_source.value = str(int(source_indices[point_idx]))
        if on_select is not None:
            on_select(projection.names[point_idx], int(source_indices[point_idx]))

    with server.gui.add_folder("Embedding", order=0):
        selected_name = server.gui.add_text("Selected motion", initial_value="", disabled=True, order=0)
        selected_source = server.gui.add_text("Source index", initial_value="", disabled=True, order=1)
        if projection.names:
            motion_dropdown = server.gui.add_dropdown(
                "Motion",
                projection.names,
                initial_value=projection.names[0],
                order=2,
            )
        else:
            motion_dropdown = None

    if motion_dropdown is not None:

        @motion_dropdown.on_update
        def _(_) -> None:
            selected_name_raw = str(motion_dropdown.value)
            try:
                point_idx = projection.names.index(selected_name_raw)
            except ValueError:
                return
            select_index(point_idx)

    @server.scene.on_click()
    def _(event: Any) -> None:
        if points.shape[0] == 0:
            return
        nearest_idx, distance = _nearest_point_to_ray(world_points, event.ray_origin, event.ray_direction)
        if nearest_idx is not None and distance <= pick_radius:
            select_index(nearest_idx)

    if projection.names:
        select_index(0)

    print(f"[any4hdmi-limmt-embedding-viewer] Viser server: http://{host}:{port}")
    if not block:
        return server

    print("[any4hdmi-limmt-embedding-viewer] Open the URL in a browser. Press Ctrl+C to quit.")
    while True:
        time.sleep(0.1)


def _as_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    embedding_matrix = np.asarray(embeddings, dtype=np.float32)
    if embedding_matrix.ndim != 2:
        raise ValueError(f"embeddings must have shape (N, D), got {embedding_matrix.shape}")
    return embedding_matrix


def _pad_to_three_columns(points: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 2:
        raise ValueError(f"projected points must be 2D, got {points_array.shape}")
    if points_array.shape[1] == 3:
        return points_array.astype(np.float32, copy=False)
    padded = np.zeros((points_array.shape[0], 3), dtype=np.float32)
    copy_cols = min(3, points_array.shape[1])
    if copy_cols > 0:
        padded[:, :copy_cols] = points_array[:, :copy_cols]
    return padded


def _project_pca_3d(embedding_matrix: np.ndarray, *, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    n_components = min(3, embedding_matrix.shape[0], embedding_matrix.shape[1])
    if n_components < 1:
        return np.zeros((embedding_matrix.shape[0], 3), dtype=np.float32)
    points = PCA(n_components=n_components, random_state=seed).fit_transform(embedding_matrix)
    return _pad_to_three_columns(points)


def _project_tsne_3d(embedding_matrix: np.ndarray, *, perplexity: float, seed: int) -> np.ndarray:
    if float(perplexity) <= 0.0:
        raise ValueError("perplexity must be positive")

    from sklearn.manifold import TSNE

    num_embeddings = embedding_matrix.shape[0]
    safe_perplexity = min(
        float(perplexity),
        max(1.0, (num_embeddings - 1) / 3.0),
        np.nextafter(float(num_embeddings), 0.0),
    )
    init = "pca" if min(embedding_matrix.shape[0], embedding_matrix.shape[1]) >= 3 else "random"
    points = TSNE(
        n_components=3,
        perplexity=safe_perplexity,
        init=init,
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(embedding_matrix)
    return points.astype(np.float32, copy=False)


def _project_umap_3d(embedding_matrix: np.ndarray, *, seed: int) -> np.ndarray:
    umap_module = _import_umap()
    if embedding_matrix.shape[0] < 3:
        return _project_pca_3d(embedding_matrix, seed=seed)
    n_neighbors = min(15, max(2, embedding_matrix.shape[0] - 1))
    points = umap_module.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        random_state=seed,
    ).fit_transform(embedding_matrix)
    return points.astype(np.float32, copy=False)


def _import_umap() -> Any:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP projection requires optional package `umap-learn`; use method='tsne' or method='pca'."
        ) from exc
    return umap


def _colors_from_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    mins = np.min(points, axis=0, keepdims=True)
    spans = np.ptp(points, axis=0, keepdims=True)
    normalized = np.divide(points - mins, spans, out=np.full_like(points, 0.5), where=spans > 1e-8)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def _nearest_point_to_ray(
    points: np.ndarray,
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
) -> tuple[int | None, float]:
    if points.shape[0] == 0:
        return None, math.inf
    origin = np.asarray(ray_origin, dtype=np.float32)
    direction = np.asarray(ray_direction, dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0.0 or not np.isfinite(direction_norm):
        return None, math.inf
    direction = direction / direction_norm
    offsets = points - origin[None, :]
    cross_track = offsets - np.sum(offsets * direction[None, :], axis=1, keepdims=True) * direction[None, :]
    distances = np.linalg.norm(cross_track, axis=1)
    nearest_idx = int(np.argmin(distances))
    return nearest_idx, float(distances[nearest_idx])
