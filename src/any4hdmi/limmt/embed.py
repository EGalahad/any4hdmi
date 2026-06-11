from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from any4hdmi.core.format import load_manifest, load_motion
from any4hdmi.core.model import load_model
from any4hdmi.dataset.interpolation import interpolate_qpos_torch
from any4hdmi.limmt.common import DEFAULT_OUTPUT_ROOT, motion_paths_for_root, relative_motion_path, resolve_dataset_root
from any4hdmi.limmt.hme import (
    HME_FEATURE_TYPE,
    PeriodicAutoencoder,
    compute_win_len,
    motion_features,
    normalize_motion_features,
    window_indices,
)
from any4hdmi.utils.dataset import compute_motion_qvel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LIMMT HME embeddings for an any4hdmi dataset.")
    parser.add_argument("--input-root", default=str(DEFAULT_OUTPUT_ROOT / "amass_limmt_pass"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_OUTPUT_ROOT / "hme" / "hme.pt"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "embeddings"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Optional qpos resampling FPS before HME features. Use 50 for checkpoints trained on AMASS 50fps.",
    )
    return parser.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[PeriodicAutoencoder, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = PeriodicAutoencoder(
        inp_ch=int(ckpt["state_dim"]),
        latent_ch=int(ckpt["phase_dim"]),
        win_len=int(ckpt["win_len"]),
        hidden_dims=tuple(int(dim) for dim in ckpt.get("hidden_dims", [64, 64])),
        win_sec=float(ckpt["win_sec"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _load_checkpoint_normalization(ckpt: dict) -> tuple[np.ndarray, np.ndarray]:
    feature_type = ckpt.get("feature_type")
    if feature_type != HME_FEATURE_TYPE:
        raise RuntimeError(f"HME checkpoint feature_type={feature_type!r}; expected {HME_FEATURE_TYPE!r}. Retrain HME.")
    normalization = ckpt.get("feature_normalization")
    if not normalization:
        raise RuntimeError("HME checkpoint does not contain feature_normalization; retrain with empirical normalization.")
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    if mean.shape != std.shape or mean.shape[0] != int(ckpt["state_dim"]):
        raise RuntimeError("HME checkpoint feature_normalization shape does not match state_dim.")
    return mean, std


def _embedding_window_indices(length: int, *, win_len: int, downsample_rate: int, stride: int) -> np.ndarray:
    ids = window_indices(
        int(length),
        win_len=int(win_len),
        downsample_rate=int(downsample_rate),
        stride=int(stride),
    )
    if ids.shape[0] > 0:
        return ids
    if int(length) <= 0:
        return np.empty((0, int(win_len)), dtype=np.int64)
    win_half = (int(win_len) - 1) // 2
    center = int(length) // 2
    offsets = np.arange(-win_half, win_half + 1, dtype=np.int64) * int(downsample_rate)
    return np.clip(center + offsets, 0, int(length) - 1)[None, :]


def main() -> None:
    args = _parse_args()
    input_root = resolve_dataset_root(args.input_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt = _load_model(Path(args.checkpoint).expanduser().resolve(), device)
    feature_mean, feature_std = _load_checkpoint_normalization(ckpt)
    manifest = load_manifest(input_root)
    source_fps = 1.0 / manifest.timestep
    feature_fps = float(args.target_fps) if args.target_fps is not None else source_fps
    mj_model = load_model(manifest.mjcf_path)

    names: list[str] = []
    embeddings: list[np.ndarray] = []
    lengths: list[int] = []

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
        features = motion_features(qpos, qvel)
        features = normalize_motion_features(features, feature_mean, feature_std)
        ids = _embedding_window_indices(
            features.shape[0],
            win_len=int(ckpt["win_len"]),
            downsample_rate=int(ckpt["downsample_rate"]),
            stride=int(args.stride),
        )
        if ids.shape[0] == 0:
            continue
        per_motion = []
        with torch.no_grad():
            for start in range(0, ids.shape[0], int(args.batch_size)):
                batch_ids = ids[start : start + int(args.batch_size)]
                batch = torch.from_numpy(features[batch_ids].astype(np.float32)).permute(0, 2, 1).to(device)
                encoded = model.encode(batch)
                emb = torch.cat([encoded["amp"].squeeze(-1), encoded["freq"].squeeze(-1)], dim=-1)
                per_motion.append(emb.cpu())
        global_emb = torch.cat(per_motion, dim=0).mean(dim=0).numpy()
        names.append(relative_motion_path(input_root, motion_path).as_posix())
        embeddings.append(global_emb.astype(np.float32))
        lengths.append(original_length)

    emb_arr = np.stack(embeddings, axis=0) if embeddings else np.zeros((0, int(ckpt["phase_dim"]) * 2), dtype=np.float32)
    np.savez_compressed(output_dir / "embeddings.npz", names=np.asarray(names), embeddings=emb_arr, lengths=np.asarray(lengths))
    with (output_dir / "embeddings.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["motion", "length", *[f"emb_{idx}" for idx in range(emb_arr.shape[1])]])
        for name, length, emb in zip(names, lengths, emb_arr, strict=True):
            writer.writerow([name, length, *[float(x) for x in emb]])
    print(f"Saved {len(names)} embeddings to {output_dir}")


if __name__ == "__main__":
    main()
