# Humanoid-GPT HME Reference

Source: `GalaxyGeneralRobotics/Humanoid-GPT`

Local checkout used for this note: `/tmp/Humanoid-GPT`

Commit: `21ec1ac6cbd57d3a7b7fa466deb922efc77e5b32`

Date checked: 2026-06-08

## Relevant Files

| Repo path | Role | Implementation notes |
| --- | --- | --- |
| `projects/hme/model.py` | HME / PeriodicAutoencoder model | Defines `PeriodicAutoencoder(inp_ch, latent_ch, win_len, hidden_dims=(64, 64), win_sec=2.0)`. Encoder is Conv1d + BatchNorm1d + ELU over `[inp_ch, *hidden_dims, latent_ch]` with kernel size equal to `win_len`. Per latent channel, a `Linear(win_len, 2) + BatchNorm1d(2)` predicts phase shift. FFT over the latent sequence gives `amp`, `freq`, and `offset`. Decoder reconstructs a sinusoidal latent trajectory and maps it back to input features. Loss is plain MSE between prediction and input. |
| `projects/hme/dataset.py` | Sliding-window dataset | Defines `compute_win_len(win_sec, downsample_rate, frequency=50) = int(win_sec * frequency / downsample_rate) + 1`. `PAEDataset` scans `*.npz`, reads `qpos`, `qvel`, `gv_vel`, concatenates raw `[qpos, qvel, gv_vel]`, and returns centered sliding windows. With the default G1 data this is `qpos(36) + qvel(35) + gv_vel(3) = 74D`. No heading canonicalization and no feature normalization are applied. |
| `projects/hme/train.py` | HME training entrypoint | Defines `PAETrainConfig` and trains on a single device. Defaults are `batch_size=128`, `num_workers=64`, `lr=1e-3`, `weight_decay=1e-4`, `num_epochs=50`, `max_grad_norm=10.0`, `state_dim=74`, `phase_dim=8`, `win_sec=4.0`, `downsample_rate=5`. Uses `AdamW`, no LR scheduler, no DDP. Saves one checkpoint containing only `model_state_dict`, plus config TOML and reconstruction/loss PNGs under `storage/hme_log/<ckpt_stem>/`. |
| `projects/gqs/global_weighted_fps.py` | GQS HME embedding + weighted FPS | Loads the trained HME, extracts raw 74D windows, calls `model.encode`, concatenates per-window `[amp, freq]`, then averages over all windows in a motion to get the global HME embedding. With `phase_dim=8`, the global embedding is `16D`. Complexity is `mean(sum(qvel^2)) + 0.05 * mean(sum(qacc^2))`, normalized by rank. Weighted FPS starts from the highest-complexity motion and greedily maximizes `alpha * normalized_min_distance + (1 - alpha) * normalized_complexity`, default `alpha=0.6`. |
| `projects/LIMMT.md` | Paper/pipeline notes | Documents the intended stage order: physical filtering, HME training, then GQS global weighted FPS. The documented HME training command is `python -m projects.hme.train --mocap_dir storage/mocap/amass_train_convert --hme_ckpt storage/hme_ckpt/amass.pt`. The documented GQS command uses `projects.gqs.global_weighted_fps` with `--alpha 0.6`. |

## Humanoid-GPT HME Method

Input windows are motion-centered temporal windows. For each center frame, the dataset samples a symmetric window using offsets:

```text
win_len = int(win_sec * frequency / downsample_rate) + 1
win_half = (win_len - 1) / 2
padding = win_half * downsample_rate
window_ids = center + [-win_half, ..., win_half] * downsample_rate
```

For the default `win_sec=4.0`, `frequency=50`, and `downsample_rate=5`, the model sees `41` frames per window, covering 4 seconds.

The PAE encodes each input window into `phase_dim` latent channels. For each latent channel:

- FFT estimates amplitude `amp`, weighted frequency `freq`, and DC offset `offset`.
- A small linear phase encoder estimates `shift`.
- Reconstruction is a periodic latent signal:

```text
latent_recon = amp * sin(2*pi * (freq * time + shift)) + offset
```

The decoder maps this periodic latent signal back to the original input window. Training minimizes reconstruction MSE.

For GQS, Humanoid-GPT does not use the phase-manifold helper. It uses only `amp` and `freq`:

```text
window_embedding = concat(amp, freq)
motion_embedding = mean(window_embedding over all valid windows)
```

With `phase_dim=8`, this makes the global HME embedding `16D`.

## Differences From Current any4hdmi LIMMT

| Area | Humanoid-GPT | Current `any4hdmi.limmt` |
| --- | --- | --- |
| Feature input | Raw `[qpos, qvel, gv_vel]`, `74D`. | Canonicalized `[joint_pos(29), joint_vel(29), root_pose(9), root_vel(6)]`, `73D`. |
| Coordinate frame | No rotation/heading canonicalization. | Root position and 6D root pose are expressed in the first-frame yaw-heading frame. Root linear velocity is expressed in the current root full-orientation local frame; MuJoCo freejoint angular qvel is kept root-local. |
| Root orientation | Raw quaternion in `qpos`. | 6D rotation representation after first-heading transform. |
| Normalization | None. | Empirical mean/std normalization saved in `normalization.npz` and checkpoint metadata. |
| Dataset IO | Loads each `.npz` window lazily and concatenates raw tensors. | Builds a feature cache with normalized windows for training and reuses the cache in DDP. |
| Training parallelism | Single GPU only. | Single GPU and `torchrun` DDP. |
| LR schedule | Fixed `AdamW(lr=1e-3, weight_decay=1e-4)`. | Supports fixed LR, cosine, and OneCycle. Current tuning uses OneCycle variants. |
| Checkpoint metadata | Saves only `model_state_dict`. | Saves `state_dim`, `feature_type`, `phase_dim`, `hidden_dims`, `win_len`, `win_sec`, `downsample_rate`, and feature normalization. |
| Embedding | Global average of per-window `[amp, freq]`. | Same HME embedding definition, but computed on canonicalized and normalized features. |

## Current Tuning Context

The closest architectural baseline to Humanoid-GPT is:

```text
phase_dim=8
hidden_dims=(64, 64)
win_sec=4.0
downsample_rate=5
optimizer=AdamW(lr=1e-3, weight_decay=1e-4)
scheduler=none
```

On the current any4hdmi canonicalized/normalized feature setup, this baseline reached about `0.656` at epoch 20. Increasing capacity has mattered more than only changing LR schedule so far:

| Variant | Epoch 20 loss |
| --- | ---: |
| `phase_dim=8`, `hidden_dims=(64,64)`, fixed `lr=1e-3` | `0.656` |
| `phase_dim=32`, `hidden_dims=(256,256)`, OneCycle `max_lr=3e-3` | `0.462` |

The original Humanoid-GPT implementation should therefore be treated as the architecture/procedure reference, not as a drop-in training recipe for our current 73D normalized feature space.
