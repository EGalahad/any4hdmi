from __future__ import annotations

import math
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from any4hdmi.core.format import load_manifest, load_motion
from any4hdmi.core.model import load_model
from any4hdmi.limmt.common import motion_paths_for_root
from any4hdmi.utils.dataset import compute_motion_qvel


HME_FEATURE_TYPE = "joint_pos_joint_vel_root_pose6d_initial_heading_root_vel_current_root_local"


def compute_win_len(win_sec: float, downsample_rate: int, frequency: float = 50.0) -> int:
    win_len = int(float(win_sec) * float(frequency) / int(downsample_rate)) + 1
    return win_len + (0 if win_len % 2 == 1 else 1)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-8)


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _quat_conj_wxyz(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float32).copy()
    out[..., 1:] *= -1.0
    return out


def _rotate_by_quat_wxyz(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    vec_in = np.asarray(vec, dtype=np.float32)
    quat_in = _normalize_quat_wxyz(np.asarray(quat, dtype=np.float32))
    zeros = np.zeros((*vec_in.shape[:-1], 1), dtype=np.float32)
    vec_quat = np.concatenate([zeros, vec_in], axis=-1)
    rotated = _quat_mul_wxyz(_quat_mul_wxyz(quat_in, vec_quat), _quat_conj_wxyz(quat_in))
    return rotated[..., 1:].astype(np.float32, copy=False)


def _rotate_by_inverse_quat_wxyz(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return _rotate_by_quat_wxyz(vec, _quat_conj_wxyz(_normalize_quat_wxyz(np.asarray(quat, dtype=np.float32))))


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    quat = _normalize_quat_wxyz(np.asarray(quat, dtype=np.float64))
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _rotate_by_inverse_yaw(vec: np.ndarray, yaw: float) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float32).copy()
    if out.shape[-1] < 2:
        return out
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    x = out[..., 0].copy()
    y = out[..., 1].copy()
    out[..., 0] = cos_y * x + sin_y * y
    out[..., 1] = -sin_y * x + cos_y * y
    return out


def _quat_to_rot6d_wxyz(quat: np.ndarray) -> np.ndarray:
    quat_in = _normalize_quat_wxyz(np.asarray(quat, dtype=np.float32))
    w, x, y, z = np.moveaxis(quat_in, -1, 0)
    col0 = np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        axis=-1,
    )
    col1 = np.stack(
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        axis=-1,
    )
    return np.concatenate([col0, col1], axis=-1).astype(np.float32, copy=False)


def root_pose_in_initial_heading_frame(qpos: np.ndarray) -> np.ndarray:
    """Express root pose in the first-frame heading frame, dropping initial roll/pitch from the frame."""
    qpos_in = np.asarray(qpos, dtype=np.float32)
    if qpos_in.shape[1] < 7:
        raise ValueError(f"Expected free-root qpos with at least 7 columns, got shape {qpos_in.shape}")

    first_yaw = _yaw_from_quat_wxyz(qpos_in[0, 3:7])
    root_pose = np.empty((qpos_in.shape[0], 9), dtype=np.float32)

    root_delta = qpos_in[:, :3] - qpos_in[0, :3]
    root_pose[:, :3] = _rotate_by_inverse_yaw(root_delta, first_yaw)

    inv_yaw = _yaw_quat_wxyz(-first_yaw)
    inv_yaws = np.broadcast_to(inv_yaw, qpos_in[:, 3:7].shape)
    root_quat_local = _normalize_quat_wxyz(_quat_mul_wxyz(inv_yaws, qpos_in[:, 3:7]))
    root_pose[:, 3:9] = _quat_to_rot6d_wxyz(root_quat_local)
    return root_pose


def root_vel_in_current_root_frame(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    qpos_in = np.asarray(qpos, dtype=np.float32)
    qvel_in = np.asarray(qvel, dtype=np.float32)
    if qpos_in.shape[1] < 7 or qvel_in.shape[1] < 6:
        raise ValueError(f"Expected free-root qpos/qvel with at least 7/6 columns, got {qpos_in.shape}/{qvel_in.shape}")
    root_vel = qvel_in[:, :6].copy()
    root_vel[:, :3] = _rotate_by_inverse_quat_wxyz(root_vel[:, :3], qpos_in[:, 3:7])
    # MuJoCo free-joint angular qvel is already expressed in the current root frame.
    return root_vel


def motion_features(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    qpos_in = np.asarray(qpos, dtype=np.float32)
    qvel_in = np.asarray(qvel, dtype=np.float32)
    if qpos_in.shape[1] < 7 or qvel_in.shape[1] < 6:
        raise ValueError(f"Expected free-root qpos/qvel with at least 7/6 columns, got {qpos_in.shape}/{qvel_in.shape}")
    joint_pos = qpos_in[:, 7:]
    joint_vel = qvel_in[:, 6:]
    root_pose = root_pose_in_initial_heading_frame(qpos_in)
    root_vel = root_vel_in_current_root_frame(qpos_in, qvel_in)
    return np.concatenate([joint_pos, joint_vel, root_pose, root_vel], axis=-1)


def window_indices(length: int, *, win_len: int, downsample_rate: int, stride: int = 1) -> np.ndarray:
    win_half = (win_len - 1) // 2
    padding = win_half * int(downsample_rate)
    seq_len = int(length) - 2 * padding
    if seq_len <= 0:
        return np.empty((0, win_len), dtype=np.int64)
    centers = padding + np.arange(0, seq_len, max(1, int(stride)), dtype=np.int64)
    offsets = (np.arange(-win_half, win_half + 1, dtype=np.int64) * int(downsample_rate))[None, :]
    return centers[:, None] + offsets


class PeriodicAutoencoder(nn.Module):
    def __init__(
        self,
        inp_ch: int,
        latent_ch: int,
        win_len: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 64),
        win_sec: float = 4.0,
    ) -> None:
        super().__init__()
        self.inp_ch = int(inp_ch)
        self.latent_ch = int(latent_ch)
        self.win_len = int(win_len)
        if self.win_len % 2 == 0:
            raise ValueError("win_len must be odd")
        padding = self.win_len // 2

        self.register_buffer("time_vec", torch.linspace(-win_sec / 2, win_sec / 2, self.win_len), persistent=False)
        self.register_buffer("freqs", torch.fft.rfftfreq(self.win_len)[1:] * self.win_len / win_sec, persistent=False)
        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi, dtype=torch.float32), persistent=False)

        enc_layers: list[nn.Module] = []
        enc_channels = (self.inp_ch, *hidden_dims, self.latent_ch)
        for in_ch, out_ch in zip(enc_channels[:-1], enc_channels[1:], strict=True):
            enc_layers.extend([nn.Conv1d(in_ch, out_ch, self.win_len, padding=padding), nn.BatchNorm1d(out_ch), nn.ELU()])
        self.encoder = nn.Sequential(*enc_layers)

        self.phase_encoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.win_len, 2), nn.BatchNorm1d(2)) for _ in range(self.latent_ch)]
        )

        dec_layers: list[nn.Module] = []
        dec_channels = (self.latent_ch, *hidden_dims)
        for in_ch, out_ch in zip(dec_channels[:-1], dec_channels[1:], strict=True):
            dec_layers.extend([nn.Conv1d(in_ch, out_ch, self.win_len, padding=padding), nn.BatchNorm1d(out_ch), nn.ELU()])
        last_ch = dec_channels[-1]
        dec_layers.append(nn.Conv1d(last_ch, self.inp_ch, self.win_len, padding=padding))
        self.decoder = nn.Sequential(*dec_layers)

    def _compute_fft_params(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rfft = torch.fft.rfft(latent, dim=-1)
        spectrum = torch.abs(rfft[:, :, 1:])
        power = spectrum.square()
        pow_sum = power.sum(dim=-1).add(1e-8)
        freq = (self.freqs * power).sum(dim=-1) / pow_sum
        amp = 2 * torch.sqrt(pow_sum) / self.win_len
        offset = rfft.real[:, :, 0] / self.win_len
        return amp.unsqueeze(-1), freq.unsqueeze(-1), offset.unsqueeze(-1)

    def encode(self, inp: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(inp)
        amp, freq, offset = self._compute_fft_params(latent)
        shifts = []
        for channel, phase_encoder in enumerate(self.phase_encoders):
            z_shift = phase_encoder(latent[:, channel])
            shifts.append(torch.atan2(z_shift[..., 1], z_shift[..., 0]) / self.two_pi)
        shift = torch.stack(shifts, dim=1).unsqueeze(-1)
        return {"latent": latent, "amp": amp, "freq": freq, "offset": offset, "shift": shift}

    def forward(self, inp: torch.Tensor) -> dict[str, torch.Tensor]:
        params = self.encode(inp)
        recon_latent = params["amp"] * torch.sin(self.two_pi * (params["freq"] * self.time_vec + params["shift"])) + params["offset"]
        pred = self.decoder(recon_latent)
        return {"pred": pred, "loss": F.mse_loss(pred, inp)}

def _cache_feature_path(cache_dir: Path, rel_motion: str) -> Path:
    return cache_dir / "features" / f"{rel_motion}.npy"


def normalize_motion_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float32) - mean.astype(np.float32)) / std.astype(np.float32)


def build_hme_feature_cache(
    *,
    dataset_root: str | Path,
    cache_dir: str | Path,
    win_sec: float,
    downsample_rate: int,
    stride: int,
) -> Path:
    root_path = Path(dataset_root).expanduser().resolve()
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(root_path)
    fps = 1.0 / manifest.timestep
    model = load_model(manifest.mjcf_path)
    win_len = compute_win_len(win_sec, downsample_rate, fps)
    records: list[dict[str, Any]] = []
    feature_sum: np.ndarray | None = None
    feature_sumsq: np.ndarray | None = None
    feature_count = 0
    motion_paths = motion_paths_for_root(root_path)
    start = time.monotonic()
    print(
        f"building HME feature cache motions={len(motion_paths)} win_len={win_len} "
        f"feature={HME_FEATURE_TYPE} cache_dir={cache_path}",
        flush=True,
    )
    for motion_index, motion_path in enumerate(motion_paths, start=1):
        rel_motion = motion_path.relative_to(root_path).as_posix()
        qpos = load_motion(motion_path)
        ids = window_indices(qpos.shape[0], win_len=win_len, downsample_rate=downsample_rate, stride=stride)
        if ids.shape[0] == 0:
            continue
        out_path = _cache_feature_path(cache_path, rel_motion)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.is_file():
            qvel = compute_motion_qvel(model, qpos, fps)
            features = motion_features(qpos, qvel)
            np.save(out_path, features.astype(np.float32))
        else:
            features = np.load(out_path, mmap_mode="r")
        features_arr = np.asarray(features, dtype=np.float64)
        if feature_sum is None:
            feature_sum = np.zeros(features_arr.shape[1], dtype=np.float64)
            feature_sumsq = np.zeros(features_arr.shape[1], dtype=np.float64)
        feature_sum += features_arr.sum(axis=0)
        feature_sumsq += np.square(features_arr).sum(axis=0)
        feature_count += int(features_arr.shape[0])
        records.append(
            {
                "motion": rel_motion,
                "feature_path": str(out_path),
                "length": int(qpos.shape[0]),
                "num_windows": int(ids.shape[0]),
            }
        )
        if motion_index % 200 == 0 or motion_index == len(motion_paths):
            elapsed = time.monotonic() - start
            print(
                f"cache_progress motions={motion_index}/{len(motion_paths)} "
                f"records={len(records)} elapsed_s={elapsed:.1f}",
                flush=True,
            )
    if feature_sum is None or feature_sumsq is None or feature_count == 0:
        raise RuntimeError(f"No HME features found under {root_path}")
    feature_mean = feature_sum / float(feature_count)
    feature_var = np.maximum(feature_sumsq / float(feature_count) - np.square(feature_mean), 1e-12)
    feature_std = np.maximum(np.sqrt(feature_var), 1e-6)
    np.savez_compressed(
        cache_path / "normalization.npz",
        mean=feature_mean.astype(np.float32),
        std=feature_std.astype(np.float32),
        count=np.asarray(feature_count, dtype=np.int64),
    )
    payload = {
        "dataset_root": str(root_path),
        "fps": fps,
        "win_sec": float(win_sec),
        "downsample_rate": int(downsample_rate),
        "stride": int(stride),
        "win_len": int(win_len),
        "feature_type": HME_FEATURE_TYPE,
        "feature_dim": int(feature_mean.shape[0]),
        "normalization": {
            "type": "empirical_mean_std",
            "path": str(cache_path / "normalization.npz"),
            "count": int(feature_count),
            "mean": feature_mean.astype(float).tolist(),
            "std": feature_std.astype(float).tolist(),
        },
        "records": records,
    }
    index_path = cache_path / "index.json"
    index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return index_path


def wait_for_hme_feature_cache(cache_dir: str | Path, *, timeout_s: float = 3600.0) -> Path:
    index_path = Path(cache_dir).expanduser().resolve() / "index.json"
    start = time.monotonic()
    while not index_path.is_file():
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for HME feature cache {index_path}")
        time.sleep(1.0)
    return index_path


class CachedHmeWindowDataset(Dataset[torch.Tensor]):
    def __init__(self, cache_dir: str | Path) -> None:
        cache_path = Path(cache_dir).expanduser().resolve()
        payload = json.loads((cache_path / "index.json").read_text(encoding="utf-8"))
        feature_type = payload.get("feature_type")
        if feature_type != HME_FEATURE_TYPE:
            raise RuntimeError(
                f"HME feature cache {cache_path} has feature_type={feature_type!r}; expected {HME_FEATURE_TYPE!r}. "
                "Rebuild the cache."
            )
        self.win_len = int(payload["win_len"])
        self.downsample_rate = int(payload["downsample_rate"])
        self.stride = int(payload["stride"])
        self.file_info: list[tuple[str, int, int]] = [
            (str(record["feature_path"]), int(record["length"]), int(record["num_windows"]))
            for record in payload["records"]
        ]
        if "feature_dim" not in payload:
            raise RuntimeError(f"HME feature cache {cache_path} does not contain feature_dim; rebuild the cache.")
        normalization = payload.get("normalization")
        if not normalization or "path" not in normalization:
            raise RuntimeError(f"HME feature cache {cache_path} does not contain empirical normalization; rebuild the cache.")
        norm_path = Path(normalization["path"]).expanduser()
        if not norm_path.is_absolute():
            norm_path = cache_path / norm_path
        if norm_path.is_file():
            norm = np.load(norm_path)
            self.feature_mean = np.asarray(norm["mean"], dtype=np.float32)
            self.feature_std = np.asarray(norm["std"], dtype=np.float32)
        else:
            raise RuntimeError(f"HME feature cache {cache_path} missing normalization file {norm_path}; rebuild the cache.")
        feature_dim = int(payload["feature_dim"])
        if self.feature_mean.shape[0] != feature_dim or self.feature_std.shape[0] != feature_dim:
            raise RuntimeError(f"HME feature cache {cache_path} has inconsistent normalization dimension")
        self.cumsum = [0]
        for _, _, count in self.file_info:
            self.cumsum.append(self.cumsum[-1] + count)
        self._feature_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._feature_cache_size = 16
        if not self.file_info:
            raise RuntimeError(f"No cached HME features found under {cache_path}")

    def __len__(self) -> int:
        return self.cumsum[-1]

    def _load_features(self, path: str) -> np.ndarray:
        cached = self._feature_cache.get(path)
        if cached is not None:
            self._feature_cache.move_to_end(path)
            return cached
        features = np.load(path, mmap_mode="r")
        self._feature_cache[path] = features
        if len(self._feature_cache) > self._feature_cache_size:
            self._feature_cache.popitem(last=False)
        return features

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = int(np.searchsorted(self.cumsum[1:], idx, side="right"))
        local_idx = idx - self.cumsum[file_idx]
        feature_path, length, _ = self.file_info[file_idx]
        features = self._load_features(feature_path)
        ids = window_indices(
            length,
            win_len=self.win_len,
            downsample_rate=self.downsample_rate,
            stride=self.stride,
        )[local_idx]
        normalized = normalize_motion_features(np.asarray(features[ids], dtype=np.float32), self.feature_mean, self.feature_std)
        return torch.from_numpy(normalized.astype(np.float32, copy=False))
