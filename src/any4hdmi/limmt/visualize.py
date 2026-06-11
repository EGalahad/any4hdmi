from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from any4hdmi.limmt.common import DEFAULT_OUTPUT_ROOT, read_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LIMMT HME embeddings with PCA and t-SNE.")
    parser.add_argument("--embeddings", default=str(DEFAULT_OUTPUT_ROOT / "embeddings" / "embeddings.npz"))
    parser.add_argument("--scores-json", default=str(DEFAULT_OUTPUT_ROOT / "scores.json"))
    parser.add_argument("--complexity-csv", default=str(DEFAULT_OUTPUT_ROOT / "subsets" / "complexity.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "visualizations"))
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _load_score_map(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    payload = read_json(path)
    return {str(row["motion"]): float(row.get("physical_score", np.nan)) for row in payload.get("details", [])}


def _load_complexity_map(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    import csv

    out = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["motion"]] = float(row["complexity_raw"])
    return out


def _plot(points: np.ndarray, values: np.ndarray, names: list[str], out_path: Path, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=values, s=8, cmap="viridis", alpha=0.75)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = np.load(Path(args.embeddings).expanduser().resolve(), allow_pickle=False)
    names = [str(name) for name in payload["names"].tolist()]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    lengths = np.asarray(payload["lengths"], dtype=np.float32)
    score_map = _load_score_map(Path(args.scores_json).expanduser().resolve())
    complexity_map = _load_complexity_map(Path(args.complexity_csv).expanduser().resolve())

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca = PCA(n_components=2, random_state=args.random_state).fit_transform(embeddings)
    perplexity = min(float(args.perplexity), max(1.0, (len(names) - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=args.random_state).fit_transform(embeddings)

    score_values = np.asarray([score_map.get(name, np.nan) for name in names], dtype=np.float32)
    complexity_values = np.asarray([complexity_map.get(name, np.nan) for name in names], dtype=np.float32)
    for method_name, points in (("pca", pca), ("tsne", tsne)):
        _plot(points, lengths, names, output_dir / f"hme_{method_name}_length.png", title=f"HME {method_name.upper()} by clip length")
        if np.isfinite(score_values).any():
            _plot(points, score_values, names, output_dir / f"hme_{method_name}_physical_score.png", title=f"HME {method_name.upper()} by physical score")
        if np.isfinite(complexity_values).any():
            _plot(points, complexity_values, names, output_dir / f"hme_{method_name}_complexity.png", title=f"HME {method_name.upper()} by complexity")
    np.savez_compressed(output_dir / "projection_points.npz", names=np.asarray(names), pca=pca, tsne=tsne)
    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()

