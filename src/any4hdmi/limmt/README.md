# LIMMT Pipeline

LIMMT tools select compact any4hdmi motion subsets from AMASS:

1. physical quality filtering
2. HME training
3. HME embedding and visualization
4. GQS weighted FPS subset generation

Example input:

```bash
output/amass
```

Project root:

```bash
output/amass_limmt/
```

`any4hdmi-limmt-score` requires `--input-path`. If `--project-path` is not passed,
it defaults to a sibling directory named `<input-path-name>_limmt`. Inside a
project root, the default folders are `passed`, `hme`, `embeddings`, `subsets`,
and `visualizations`.

## Commands

Score and filter AMASS with MuJoCo Warp:

```bash
PROJECT=output/amass_limmt
any4hdmi-limmt-score \
  --input-path output/amass \
  --project-path "$PROJECT" \
  --device cuda \
  --pass-threshold 90 \
  --batch-frames 131072 \
  --fk-batch-size 8192
```

Train HME:

```bash
cd /home/elijah/Documents/projects/simple-tracking/any4hdmi
any4hdmi-limmt-train-hme \
  --project-path output/amass_limmt \
  --batch-size 256 \
  --epochs 30 \
  --win-sec 4.0 \
  --phase-dim 8 \
  --downsample-rate 5
```

HME windows are built from `joint_pos + joint_vel + root_pose6d_window_initial_heading + root_vel_current_root_local`. `root_pose6d_window_initial_heading` is computed per HME window as `T_window_heading0^-1 * T_root(t)`, where `T_window_heading0` uses that window's first frame root position and yaw but removes first-frame roll/pitch from the reference frame. Root orientation is represented as the first two rotation-matrix columns, a 6D continuous orientation target instead of a quaternion. The 6D root velocity is local to each frame's current root full orientation: root linear velocity is rotated by the current root quaternion inverse, and MuJoCo free-joint angular qvel is kept in its root-local frame. For the G1 AMASS data this gives 73 input channels: 29 joint positions, 29 joint velocities, 9 local root-pose values, and 6 local root-velocity values.

The feature cache is built through `build_motion_loader` and `FKRunner`, following the FK cache pattern of lock/tmp/ready files. It stores final HME windows with shape `[num_windows, win_len, feature_dim]`; the window-relative root pose is materialized before writing `feature_cache/features/*.npy`. `normalization.npz` stores empirical per-dimension mean/std over those same cached final windows. Training and embedding both use `(feature - mean) / std`, and the checkpoint stores the same normalization statistics under `feature_normalization`.

For multi-GPU training:

```bash
cd /home/elijah/Documents/projects/simple-tracking/any4hdmi
torchrun --nproc_per_node <N> -m any4hdmi.limmt.hme.train \
  --project-path output/amass_limmt \
  --batch-size 256 \
  --epochs 30
```

Compute embeddings:

```bash
any4hdmi-limmt-embed \
  --project-path output/amass_limmt
```

Generate subsets:

```bash
any4hdmi-limmt-gqs \
  --project-path output/amass_limmt \
  --ratios 0.04 0.08 0.16 0.32
```

Visualize HME:

```bash
any4hdmi-limmt-visualize \
  --project-path output/amass_limmt
```

## Performance Target

`any4hdmi-limmt-score` supports CPU MuJoCo fallback when the optional Warp extra is not installed. The local AMASS acceptance target is a full `output/amass` run under 15 minutes with `uv sync --extra warp` and `--device cuda`.
