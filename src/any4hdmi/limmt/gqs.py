from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from any4hdmi.core.format import load_manifest, load_motion
from any4hdmi.core.model import load_model
from any4hdmi.limmt.common import DEFAULT_OUTPUT_ROOT, DEFAULT_RATIOS, copy_any4hdmi_subset, read_json, resolve_dataset_root, write_json
from any4hdmi.utils.dataset import compute_motion_qvel


_WORKER_INPUT_ROOT: Path | None = None
_WORKER_MJ_MODEL = None
_WORKER_FPS: float | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LIMMT GQS weighted-FPS subsets.")
    parser.add_argument("--input-root", default=str(DEFAULT_OUTPUT_ROOT / "amass_limmt_pass"))
    parser.add_argument("--embeddings", default=str(DEFAULT_OUTPUT_ROOT / "embeddings" / "embeddings.npz"))
    parser.add_argument("--scores-json", default=str(DEFAULT_OUTPUT_ROOT / "scores.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "subsets"))
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--subset-prefix", default="amass_limmt_gqs")
    parser.add_argument("--selection-device", default="cpu", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--complexity-workers", type=int, default=1)
    parser.add_argument("--complexity-chunksize", type=int, default=16)
    return parser.parse_args()


def compute_complexity(qvel: np.ndarray, fps: float) -> float:
    if qvel.shape[0] < 3:
        return 0.0
    qacc = np.diff(qvel, axis=0) * float(fps)
    vel_power = float(np.mean(np.sum(qvel * qvel, axis=1)))
    acc_power = float(np.mean(np.sum(qacc * qacc, axis=1)))
    return vel_power + 0.05 * acc_power


def rank_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float32)
    ranks = np.argsort(np.argsort(values))
    return ranks.astype(np.float32) / float(values.size - 1)


def _weighted_fps_numpy(embeddings: np.ndarray, complexities: np.ndarray, n_samples: int, *, alpha: float) -> list[int]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    complexities = np.asarray(complexities, dtype=np.float32)
    n = int(embeddings.shape[0])
    if n_samples >= n:
        return list(range(n))
    if n_samples <= 0:
        return []
    selected = [int(np.argmax(complexities))]
    min_dists = np.linalg.norm(embeddings - embeddings[selected[0]], axis=1)
    for _ in range(n_samples - 1):
        max_dist = float(np.max(min_dists))
        norm_dists = min_dists / max_dist if max_dist > 1e-8 else min_dists
        scores = float(alpha) * norm_dists + (1.0 - float(alpha)) * complexities
        scores[selected] = -1.0
        next_idx = int(np.argmax(scores))
        selected.append(next_idx)
        min_dists = np.minimum(min_dists, np.linalg.norm(embeddings - embeddings[next_idx], axis=1))
    return selected


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

    emb = torch.as_tensor(embeddings, dtype=torch.float32, device=target).contiguous()
    comp = torch.as_tensor(complexities, dtype=torch.float32, device=target)
    n = int(emb.shape[0])
    if n_samples >= n:
        return list(range(n))
    if n_samples <= 0:
        return []

    emb_norm = torch.sum(emb * emb, dim=1)
    selected_mask = torch.zeros((n,), dtype=torch.bool, device=target)
    first = int(torch.argmax(comp).item())
    selected = [first]
    selected_mask[first] = True
    min_dists = torch.clamp(emb_norm + emb_norm[first] - 2.0 * torch.mv(emb, emb[first]), min=0.0)
    progress = tqdm(total=n_samples - 1, desc="Weighted FPS", unit="sample")
    for _ in range(n_samples - 1):
        max_dist = torch.max(min_dists)
        if float(max_dist.item()) > 1e-8:
            norm_dists = torch.sqrt(torch.clamp(min_dists / max_dist, min=0.0))
        else:
            norm_dists = min_dists
        scores = float(alpha) * norm_dists + (1.0 - float(alpha)) * comp
        scores.masked_fill_(selected_mask, -1.0)
        next_idx = int(torch.argmax(scores).item())
        selected.append(next_idx)
        selected_mask[next_idx] = True
        next_dists = torch.clamp(emb_norm + emb_norm[next_idx] - 2.0 * torch.mv(emb, emb[next_idx]), min=0.0)
        min_dists = torch.minimum(min_dists, next_dists)
        progress.update(1)
    progress.close()
    return selected


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
    payload = np.load(path, allow_pickle=False)
    return [str(name) for name in payload["names"].tolist()], np.asarray(payload["embeddings"], dtype=np.float32), np.asarray(payload["lengths"])


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
        values = []
        for name in tqdm(names, desc="GQS complexity", unit="motion"):
            qpos = load_motion(input_root / name)
            qvel = compute_motion_qvel(mj_model, qpos, fps)
            values.append(compute_complexity(qvel, fps))
        return np.asarray(values, dtype=np.float32)

    values_by_name: dict[str, float] = {}
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_complexity_worker,
        initargs=(str(input_root), str(mjcf_path), float(fps)),
    ) as executor:
        iterator = executor.map(_compute_complexity_worker, names, chunksize=max(1, int(chunksize)))
        for name, value in tqdm(iterator, total=len(names), desc="GQS complexity", unit="motion"):
            values_by_name[name] = float(value)
    return np.asarray([values_by_name[name] for name in names], dtype=np.float32)


def main() -> None:
    args = _parse_args()
    input_root = resolve_dataset_root(args.input_root)
    manifest = load_manifest(input_root)
    fps = 1.0 / manifest.timestep
    names, embeddings, lengths = _load_embeddings(Path(args.embeddings).expanduser().resolve())
    score_payload = read_json(args.scores_json) if Path(args.scores_json).expanduser().is_file() else None
    score_by_name = {}
    if score_payload is not None:
        score_by_name = {row["motion"]: row for row in score_payload.get("details", [])}

    complexity_raw = _compute_complexities(
        names=names,
        input_root=input_root,
        mjcf_path=manifest.mjcf_path,
        fps=fps,
        workers=int(args.complexity_workers),
        chunksize=int(args.complexity_chunksize),
    )
    complexity_norm = rank_normalize(complexity_raw)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "complexity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["motion", "length", "complexity_raw", "complexity_norm", "physical_score"])
        for idx, name in enumerate(names):
            writer.writerow([name, int(lengths[idx]), float(complexity_raw[idx]), float(complexity_norm[idx]), score_by_name.get(name, {}).get("physical_score", "")])

    ratios = sorted(float(ratio) for ratio in args.ratios)
    target_counts = {ratio: max(1, int(round(len(names) * ratio))) for ratio in ratios}
    max_count = max(target_counts.values()) if target_counts else 0
    selected_max = weighted_fps_global(
        embeddings,
        complexity_norm,
        max_count,
        alpha=args.alpha,
        device=str(args.selection_device),
    )

    reports = []
    for ratio in ratios:
        n_select = max(1, int(round(len(names) * float(ratio))))
        selected_idx = selected_max[:n_select]
        selected = [names[idx] for idx in selected_idx]
        percent = int(round(float(ratio) * 100))
        subset_name = f"{args.subset_prefix}_{percent}"
        subset_root = output_dir / subset_name
        copy_any4hdmi_subset(
            input_root=input_root,
            output_root=subset_root,
            selected_rel_paths=selected,
            dataset_name=subset_name,
            source_update={
                "limmt_gqs": {
                    "ratio": float(ratio),
                    "alpha": float(args.alpha),
                    "source_dataset": str(input_root),
                    "embeddings": str(Path(args.embeddings).expanduser().resolve()),
                }
            },
        )
        report = {
            "ratio": float(ratio),
            "selected_count": len(selected),
            "subset_root": str(subset_root),
            "selected": selected,
        }
        reports.append(report)
        write_json(subset_root / "selection_report.json", report)
        (subset_root / "selected_filenames.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    write_json(output_dir / "gqs_report.json", {"input_root": str(input_root), "reports": reports})
    print(json.dumps({"reports": reports}, indent=2))


if __name__ == "__main__":
    main()
