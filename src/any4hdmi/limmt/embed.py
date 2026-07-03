from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

from any4hdmi.core.format import load_manifest, load_motion
from any4hdmi.core.model import load_model
from any4hdmi.dataset.interpolation import interpolate_qpos_torch
from any4hdmi.limmt.common import (
    motion_paths_for_root,
    project_embeddings_root,
    project_hme_root,
    project_pass_root,
    relative_motion_path,
    resolve_project_root,
)
from any4hdmi.limmt.hme import (
    HME_FEATURE_TYPE,
    PeriodicAutoencoder,
    hme_features_from_raw,
    motion_frame_features,
    normalize_motion_features,
    window_indices,
)
from any4hdmi.utils.dataset import compute_motion_qvel


@dataclass(frozen=True)
class EmbedArgs:
    """Compute LIMMT HME embeddings for an any4hdmi dataset."""

    project_path: str
    pass_dataset_name: str = "passed"
    hme_folder: str = "hme"
    embeddings_folder: str = "embeddings"
    stride: int = 1
    batch_size: int = 4096
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target_fps: float | None = None


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[PeriodicAutoencoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PeriodicAutoencoder(
        inp_ch=int(checkpoint["state_dim"]),
        latent_ch=int(checkpoint["phase_dim"]),
        win_len=int(checkpoint["win_len"]),
        hidden_dims=tuple(int(dim) for dim in checkpoint.get("hidden_dims", [64, 64])),
        win_sec=float(checkpoint["win_sec"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _load_checkpoint_normalization(checkpoint: dict) -> tuple[np.ndarray, np.ndarray]:
    feature_type = checkpoint.get("feature_type")
    if feature_type != HME_FEATURE_TYPE:
        raise RuntimeError(f"HME checkpoint feature_type={feature_type!r}; expected {HME_FEATURE_TYPE!r}. Retrain HME.")
    normalization = checkpoint.get("feature_normalization")
    if not normalization:
        raise RuntimeError("HME checkpoint does not contain feature_normalization; retrain with empirical normalization.")
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    if mean.shape != std.shape or mean.shape[0] != int(checkpoint["state_dim"]):
        raise RuntimeError("HME checkpoint feature_normalization shape does not match state_dim.")
    return mean, std


def _embedding_window_indices(length: int, *, win_len: int, downsample_rate: int, stride: int) -> np.ndarray:
    window_ids = window_indices(
        int(length),
        win_len=int(win_len),
        downsample_rate=int(downsample_rate),
        stride=int(stride),
    )
    if window_ids.shape[0] > 0:
        return window_ids
    if int(length) <= 0:
        return np.empty((0, int(win_len)), dtype=np.int64)
    win_half = (int(win_len) - 1) // 2
    center = int(length) // 2
    window_offsets = np.arange(-win_half, win_half + 1, dtype=np.int64) * int(downsample_rate)
    return np.clip(center + window_offsets, 0, int(length) - 1)[None, :]


def main() -> None:
    args = tyro.cli(EmbedArgs)
    project_root = resolve_project_root(args.project_path)
    input_root = project_pass_root(project_root, args.pass_dataset_name)
    hme_root = project_hme_root(project_root, args.hme_folder)
    output_dir = project_embeddings_root(project_root, args.embeddings_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, checkpoint = _load_model(hme_root / "hme.pt", device)
    feature_mean, feature_std = _load_checkpoint_normalization(checkpoint)
    manifest = load_manifest(input_root)
    source_fps = 1.0 / manifest.timestep
    feature_fps = float(args.target_fps) if args.target_fps is not None else source_fps
    mj_model = load_model(manifest.mjcf_path)

    motion_names: list[str] = []
    motion_embeddings: list[np.ndarray] = []
    motion_lengths: list[int] = []

    for motion_path in tqdm(motion_paths_for_root(input_root), desc="HME embeddings", unit="motion"):
        qpos = load_motion(motion_path)
        original_length = int(qpos.shape[0])
        if feature_fps != source_fps:
            qpos = (
                interpolate_qpos_torch(
                    torch.from_numpy(qpos),
                    source_fps=source_fps,
                    target_fps=feature_fps,
                )
                .cpu()
                .numpy()
            )
        qvel = compute_motion_qvel(mj_model, qpos, feature_fps)
        features = motion_frame_features(qpos, qvel)
        window_ids = _embedding_window_indices(
            features.shape[0],
            win_len=int(checkpoint["win_len"]),
            downsample_rate=int(checkpoint["downsample_rate"]),
            stride=int(args.stride),
        )
        if window_ids.shape[0] == 0:
            continue
        window_embeddings = []
        with torch.no_grad():
            for start in range(0, window_ids.shape[0], int(args.batch_size)):
                batch_ids = window_ids[start : start + int(args.batch_size)]
                batch_features = hme_features_from_raw(features, batch_ids)
                batch_features = normalize_motion_features(batch_features, feature_mean, feature_std)
                batch = torch.from_numpy(batch_features.astype(np.float32)).permute(0, 2, 1).to(device)
                encoded = model.encode(batch)
                window_embedding = torch.cat([encoded["amp"].squeeze(-1), encoded["freq"].squeeze(-1)], dim=-1)
                window_embeddings.append(window_embedding.cpu())
        motion_embedding = torch.cat(window_embeddings, dim=0).mean(dim=0).numpy()
        motion_names.append(relative_motion_path(input_root, motion_path).as_posix())
        motion_embeddings.append(motion_embedding.astype(np.float32))
        motion_lengths.append(original_length)

    embedding_array = (
        np.stack(motion_embeddings, axis=0)
        if motion_embeddings
        else np.zeros((0, int(checkpoint["phase_dim"]) * 2), dtype=np.float32)
    )
    np.savez_compressed(
        output_dir / "embeddings.npz",
        names=np.asarray(motion_names),
        embeddings=embedding_array,
        lengths=np.asarray(motion_lengths),
    )
    print(f"Saved {len(motion_names)} embeddings to {output_dir}")


if __name__ == "__main__":
    main()
