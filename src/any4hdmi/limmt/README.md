# LIMMT Pipeline

LIMMT tools select compact any4hdmi motion subsets from AMASS:

1. physical quality filtering
2. HME training
3. HME embedding and visualization
4. GQS weighted FPS subset generation

Default input:

```bash
output/amass
```

Default outputs:

```bash
output/limmt_amass/
```

## Commands

Score and filter AMASS with MuJoCo Warp:

```bash
any4hdmi-limmt-score \
  --input-root output/amass \
  --output-dir output/limmt_amass \
  --device cuda \
  --pass-threshold 90 \
  --batch-frames 131072 \
  --fk-batch-size 8192
```

Train HME:

```bash
cd /home/elijah/Documents/projects/simple-tracking/any4hdmi
any4hdmi-limmt-train-hme \
  --input-root output/limmt_amass/amass_limmt_pass \
  --output-dir output/limmt_amass/hme \
  --batch-size 256 \
  --epochs 30 \
  --win-sec 4.0 \
  --phase-dim 8 \
  --downsample-rate 5
```

HME windows are built from `joint_pos + joint_vel + root_pose6d_initial_heading + root_vel_current_root_local`. `root_pose6d_initial_heading` is computed as `T_heading0^-1 * T_root(t)`, where `T_heading0` uses the first frame's root position and yaw but removes first-frame roll/pitch from the reference frame. Root orientation is represented as the first two rotation-matrix columns, a 6D continuous orientation target instead of a quaternion. The 6D root velocity is local to each frame's current root full orientation: root linear velocity is rotated by the current root quaternion inverse, and MuJoCo free-joint angular qvel is kept in its root-local frame. For the G1 AMASS data this gives 73 input channels: 29 joint positions, 29 joint velocities, 9 local root-pose values, and 6 local root-velocity values.

The feature cache stores raw features plus empirical per-dimension mean/std in `normalization.npz`. Training and embedding both use `(feature - mean) / std`, and the checkpoint stores the same normalization statistics under `feature_normalization`.

For multi-GPU training:

```bash
cd /home/elijah/Documents/projects/simple-tracking/any4hdmi
torchrun --nproc_per_node <N> -m any4hdmi.limmt.train_hme \
  --input-root output/limmt_amass/amass_limmt_pass \
  --output-dir output/limmt_amass/hme \
  --batch-size 256 \
  --epochs 30
```

Compute embeddings:

```bash
any4hdmi-limmt-embed \
  --input-root output/limmt_amass/amass_limmt_pass \
  --checkpoint output/limmt_amass/hme/hme.pt \
  --output-dir output/limmt_amass/embeddings
```

Generate subsets:

```bash
any4hdmi-limmt-gqs \
  --input-root output/limmt_amass/amass_limmt_pass \
  --embeddings output/limmt_amass/embeddings/embeddings.npz \
  --scores-json output/limmt_amass/scores.json \
  --output-dir output/limmt_amass/subsets \
  --ratios 0.04 0.08 0.16 0.32
```

Visualize HME:

```bash
any4hdmi-limmt-visualize \
  --embeddings output/limmt_amass/embeddings/embeddings.npz \
  --scores-json output/limmt_amass/scores.json \
  --complexity-csv output/limmt_amass/subsets/complexity.csv \
  --output-dir output/limmt_amass/visualizations
```

## Performance Target

`any4hdmi-limmt-score` supports CPU MuJoCo fallback when the optional Warp extra is not installed. The local AMASS acceptance target is a full `output/amass` run under 15 minutes with `uv sync --extra warp` and `--device cuda`.
