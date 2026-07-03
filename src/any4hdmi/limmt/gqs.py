from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from tqdm import tqdm

from any4hdmi.core.format import load_manifest, load_motion
from any4hdmi.core.model import load_model
from any4hdmi.limmt.common import (
    copy_any4hdmi_subset,
    project_embeddings_root,
    project_pass_root,
    project_subsets_root,
    read_json,
    resolve_project_root,
    write_json,
)
from any4hdmi.utils.dataset import compute_motion_qvel


_WORKER_INPUT_ROOT: Path | None = None
_WORKER_MJ_MODEL = None
_WORKER_FPS: float | None = None


@dataclass(frozen=True)
class GqsArgs:
    """Generate LIMMT GQS weighted-FPS subsets."""

    project_path: str
    ratios: tuple[float, ...]
    pass_dataset_name: str = "passed"
    embeddings_folder: str = "embeddings"
    subsets_folder: str = "subsets"
    alpha: float = 0.6
    subset_prefix: str | None = None
    selection_device: str = "cpu"
    complexity_workers: int = 1
    complexity_chunksize: int = 16


def compute_complexity(qvel: np.ndarray, fps: float) -> float:
    if qvel.shape[0] < 3:
        return 0.0
    qacc = np.diff(qvel, axis=0) * float(fps)
    vel_power = float(np.mean(np.sum(qvel * qvel, axis=1)))
    acc_power = float(np.mean(np.sum(qacc * qacc, axis=1)))
    return vel_power + 0.05 * acc_power


def rank_normalize(raw_values: np.ndarray) -> np.ndarray:
    raw_values = np.asarray(raw_values, dtype=np.float32)
    if raw_values.size <= 1:
        return np.zeros_like(raw_values, dtype=np.float32)
    ranks = np.argsort(np.argsort(raw_values))
    return ranks.astype(np.float32) / float(raw_values.size - 1)


def _weighted_fps_numpy(embeddings: np.ndarray, complexities: np.ndarray, n_samples: int, *, alpha: float) -> list[int]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    complexities = np.asarray(complexities, dtype=np.float32)
    num_embeddings = int(embeddings.shape[0])
    if n_samples >= num_embeddings:
        return list(range(num_embeddings))
    if n_samples <= 0:
        return []
    selected_indices = [int(np.argmax(complexities))]
    min_distances = np.linalg.norm(embeddings - embeddings[selected_indices[0]], axis=1)
    for _ in range(n_samples - 1):
        max_distance = float(np.max(min_distances))
        normalized_distances = min_distances / max_distance if max_distance > 1e-8 else min_distances
        scores = float(alpha) * normalized_distances + (1.0 - float(alpha)) * complexities
        scores[selected_indices] = -1.0
        next_idx = int(np.argmax(scores))
        selected_indices.append(next_idx)
        min_distances = np.minimum(min_distances, np.linalg.norm(embeddings - embeddings[next_idx], axis=1))
    return selected_indices


def _weighted_fps_torch(
    embeddings: np.ndarray,
    complexities: np.ndarray,
    n_samples: int,
    *,
    alpha: float,
    device: str,
) -> list[int]:
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.device(device)
    if target.type == "cpu":
        return _weighted_fps_numpy(embeddings, complexities, n_samples, alpha=alpha)

    embedding_tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=target).contiguous()
    complexity_tensor = torch.as_tensor(complexities, dtype=torch.float32, device=target)
    num_embeddings = int(embedding_tensor.shape[0])
    if n_samples >= num_embeddings:
        return list(range(num_embeddings))
    if n_samples <= 0:
        return []

    embedding_norms = torch.sum(embedding_tensor * embedding_tensor, dim=1)
    selected_mask = torch.zeros((num_embeddings,), dtype=torch.bool, device=target)
    first_idx = int(torch.argmax(complexity_tensor).item())
    selected_indices = [first_idx]
    selected_mask[first_idx] = True
    min_distances = torch.clamp(
        embedding_norms + embedding_norms[first_idx] - 2.0 * torch.mv(embedding_tensor, embedding_tensor[first_idx]),
        min=0.0,
    )
    progress = tqdm(total=n_samples - 1, desc="Weighted FPS", unit="sample")
    for _ in range(n_samples - 1):
        max_distance = torch.max(min_distances)
        if float(max_distance.item()) > 1e-8:
            normalized_distances = torch.sqrt(torch.clamp(min_distances / max_distance, min=0.0))
        else:
            normalized_distances = min_distances
        scores = float(alpha) * normalized_distances + (1.0 - float(alpha)) * complexity_tensor
        scores.masked_fill_(selected_mask, -1.0)
        next_idx = int(torch.argmax(scores).item())
        selected_indices.append(next_idx)
        selected_mask[next_idx] = True
        next_distances = torch.clamp(
            embedding_norms + embedding_norms[next_idx] - 2.0 * torch.mv(embedding_tensor, embedding_tensor[next_idx]),
            min=0.0,
        )
        min_distances = torch.minimum(min_distances, next_distances)
        progress.update(1)
    progress.close()
    return selected_indices


def weighted_fps_global(
    embeddings: np.ndarray,
    complexities: np.ndarray,
    n_samples: int,
    *,
    alpha: float = 0.6,
    device: str = "cpu",
) -> list[int]:
    if device == "cpu":
        return _weighted_fps_numpy(embeddings, complexities, n_samples, alpha=alpha)
    return _weighted_fps_torch(embeddings, complexities, n_samples, alpha=alpha, device=device)


def _load_embeddings(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    embedding_archive = np.load(path, allow_pickle=False)
    return (
        [str(name) for name in embedding_archive["names"].tolist()],
        np.asarray(embedding_archive["embeddings"], dtype=np.float32),
        np.asarray(embedding_archive["lengths"]),
    )


def _init_complexity_worker(input_root: str, mjcf_path: str, fps: float) -> None:
    global _WORKER_INPUT_ROOT, _WORKER_MJ_MODEL, _WORKER_FPS
    _WORKER_INPUT_ROOT = Path(input_root)
    _WORKER_MJ_MODEL = load_model(mjcf_path)
    _WORKER_FPS = float(fps)


def _compute_complexity_worker(name: str) -> tuple[str, float]:
    if _WORKER_INPUT_ROOT is None or _WORKER_MJ_MODEL is None or _WORKER_FPS is None:
        raise RuntimeError("Complexity worker was not initialized")
    qpos = load_motion(_WORKER_INPUT_ROOT / name)
    qvel = compute_motion_qvel(_WORKER_MJ_MODEL, qpos, _WORKER_FPS)
    return name, compute_complexity(qvel, _WORKER_FPS)


def _compute_complexities(
    *,
    names: list[str],
    input_root: Path,
    mjcf_path: Path,
    fps: float,
    workers: int,
    chunksize: int,
) -> np.ndarray:
    if workers <= 1:
        mj_model = load_model(mjcf_path)
        complexity_values = []
        for name in tqdm(names, desc="GQS complexity", unit="motion"):
            qpos = load_motion(input_root / name)
            qvel = compute_motion_qvel(mj_model, qpos, fps)
            complexity_values.append(compute_complexity(qvel, fps))
        return np.asarray(complexity_values, dtype=np.float32)

    values_by_name: dict[str, float] = {}
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_complexity_worker,
        initargs=(str(input_root), str(mjcf_path), float(fps)),
    ) as executor:
        complexity_results = executor.map(_compute_complexity_worker, names, chunksize=max(1, int(chunksize)))
        for name, complexity_value in tqdm(complexity_results, total=len(names), desc="GQS complexity", unit="motion"):
            values_by_name[name] = float(complexity_value)
    return np.asarray([values_by_name[name] for name in names], dtype=np.float32)


def main() -> None:
    args = tyro.cli(GqsArgs)
    project_root = resolve_project_root(args.project_path)
    input_root = project_pass_root(project_root, args.pass_dataset_name)
    manifest = load_manifest(input_root)
    fps = 1.0 / manifest.timestep
    embeddings_path = project_embeddings_root(project_root, args.embeddings_folder) / "embeddings.npz"
    scores_json = project_root / "scores.json"
    motion_names, embeddings, motion_lengths = _load_embeddings(embeddings_path)
    score_report = read_json(scores_json) if scores_json.is_file() else None
    score_by_name = {}
    if score_report is not None:
        score_by_name = {score_row["motion"]: score_row for score_row in score_report.get("details", [])}

    complexity_raw = _compute_complexities(
        names=motion_names,
        input_root=input_root,
        mjcf_path=manifest.mjcf_path,
        fps=fps,
        workers=int(args.complexity_workers),
        chunksize=int(args.complexity_chunksize),
    )
    complexity_norm = rank_normalize(complexity_raw)
    output_dir = project_subsets_root(project_root, args.subsets_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "complexity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["motion", "length", "complexity_raw", "complexity_norm", "physical_score"])
        for motion_idx, name in enumerate(motion_names):
            writer.writerow(
                [
                    name,
                    int(motion_lengths[motion_idx]),
                    float(complexity_raw[motion_idx]),
                    float(complexity_norm[motion_idx]),
                    score_by_name.get(name, {}).get("physical_score", ""),
                ]
            )

    ratios = sorted(float(ratio) for ratio in args.ratios)
    target_counts = {ratio: max(1, int(round(len(motion_names) * ratio))) for ratio in ratios}
    max_count = max(target_counts.values()) if target_counts else 0
    global_selected_indices = weighted_fps_global(
        embeddings,
        complexity_norm,
        max_count,
        alpha=args.alpha,
        device=str(args.selection_device),
    )

    subset_reports = []
    for ratio in ratios:
        n_select = max(1, int(round(len(motion_names) * float(ratio))))
        selected_indices = global_selected_indices[:n_select]
        selected_motions = [motion_names[idx] for idx in selected_indices]
        percent = int(round(float(ratio) * 100))
        subset_prefix = args.subset_prefix or f"{project_root.name}_gqs"
        subset_name = f"{subset_prefix}_{percent}"
        subset_root = output_dir / subset_name
        copy_any4hdmi_subset(
            input_root=input_root,
            output_root=subset_root,
            selected_rel_paths=selected_motions,
            dataset_name=subset_name,
            source_update={
                "limmt_gqs": {
                    "ratio": float(ratio),
                    "alpha": float(args.alpha),
                    "source_dataset": str(input_root),
                    "embeddings": str(embeddings_path),
                }
            },
        )
        subset_report = {
            "ratio": float(ratio),
            "selected_count": len(selected_motions),
            "subset_root": str(subset_root),
            "selected": selected_motions,
        }
        subset_reports.append(subset_report)
        write_json(subset_root / "selection_report.json", subset_report)
        (subset_root / "selected_filenames.txt").write_text("\n".join(selected_motions) + "\n", encoding="utf-8")
    write_json(output_dir / "gqs_report.json", {"project_root": str(project_root), "input_root": str(input_root), "reports": subset_reports})
    print(json.dumps({"reports": subset_reports}, indent=2))


if __name__ == "__main__":
    main()
