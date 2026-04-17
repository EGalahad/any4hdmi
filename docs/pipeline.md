# Pipeline

This document explains how `any4hdmi` converts source motion datasets into the unified `qpos` format, how the viewer and filter load them, and how the MJCF is resolved.

For the final on-disk dataset format, including `manifest.json` and motion `.npz`, see [`dataset_format.md`](./dataset_format.md).
For runtime dataset loading and FK cache behavior, see [`dataset.md`](./dataset.md).

## Overview

`any4hdmi` normalizes different source datasets into one MuJoCo-friendly representation:

- one dataset-level `manifest.json`
- one Hugging Face MJCF reference per dataset
- one motion file per clip under `output/<dataset>/motions/**/*.npz`

The core motion payload is only:

- `qpos[T, nq]`

Everything else needed for replay is stored at dataset level in `manifest.json`.

## Repository Layout

```text
any4hdmi/
  docs/
    pipeline.md
  output/
    lafan/
      manifest.json
      motions/
    sonic/
      manifest.json
      motions/
    100style/
      manifest.json
      motions/
  src/any4hdmi/
```

This repo does not keep a checked-in `assets/` tree. MJCF XML and STL meshes are resolved from Hugging Face cache on demand.

Default MJCF reference:

- repo: `elijahgalahad/g1_xmls`
- path: `g1-mode_13_15.xml`
- revision: `main`

## Unified Motion Format

Each converted dataset root contains a `manifest.json` with dataset-level information:

- dataset name
- Hugging Face MJCF reference
- timestep
- qpos dimension
- qpos names
- number of motions
- source conversion settings

Each motion clip is saved as `motions/<...>.npz`.

The `.npz` stores:

- `qpos`: `float32`, shape `[num_frames, nq]`

## Dataset Conversion

### LAFAN

Source assumption:

- root position is already in meters
- root rotation is already quaternion `qx qy qz qw`
- joint values are already in radians

Default conversion settings:

- dataset: `lafan`
- FPS: `30`
- MJCF repo: `elijahgalahad/g1_xmls`
- MJCF path: `g1-mode_13_15.xml`
- output: `output/lafan`

Pipeline:

1. Resolve the MJCF repo snapshot into the Hugging Face cache.
2. Read each CSV into `float32`.
3. Slice frames with `start/end/stride`.
4. Reorder root quaternion from `xyzw` to MuJoCo `wxyz`.
5. Fill a `float32` `qpos` buffer.
6. Save `qpos` as compressed `.npz`.
7. Save/update dataset `manifest.json`.

### SONIC

Source assumption:

- `root_translateX/Y/Z` is in centimeters
- `root_rotateX/Y/Z` is in degrees
- `*_joint_dof` is in degrees

Default conversion settings:

- dataset: `sonic`
- FPS: `120`
- translation scale: `0.01`
- Euler order: `xyz`
- Euler frame: `extrinsic`
- MJCF repo: `elijahgalahad/g1_xmls`
- MJCF path: `g1-mode_13_15.xml`
- output: `output/sonic`

Pipeline:

1. Resolve the MJCF repo snapshot into the Hugging Face cache.
2. Read each CSV into `float32`.
3. Parse source columns by header name.
4. Convert root translation from centimeters to meters.
5. Convert root Euler angles and joint angles from degrees to radians.
6. Convert Euler angles into MuJoCo quaternions in `wxyz`.
7. Fill a `float32` `qpos` buffer.
8. Save `qpos` as compressed `.npz`.
9. Save/update dataset `manifest.json`.

### 100STYLE / Axellwppr MotionDataset

Source assumption:

- input is either `100style.tar` or an extracted MotionDataset directory
- root position, root quaternion, and joint positions already exist as TensorDict fields
- source export uses 50 FPS by default

Default conversion settings:

- dataset: `100style`
- FPS: `50`
- MJCF repo: `elijahgalahad/g1_xmls`
- MJCF path: `g1-mode_13_15.xml`
- output: `output/100style`

Pipeline:

1. Open the tarball or extracted directory.
2. Read `meta_motion.json`, `id_label.json`, and TensorDict memmaps.
3. Build one clip record per labeled segment.
4. Normalize root quaternions to MuJoCo `wxyz`.
5. Fill a `float32` `qpos` buffer for each segment.
6. Save one motion `.npz` per segment.
7. Save/update dataset `manifest.json`.

## Euler Convention In SONIC

The current default for SONIC is:

- `--euler-order xyz`
- `--euler-frame extrinsic`

This was chosen from visual validation on turning and side-walk motions. If the source export changes, override:

```bash
uv run any4hdmi-convert-sonic \
  --csv-dir ../g1_sonic/complete/g1/csv \
  --out-dir output/sonic \
  --euler-order xyz \
  --euler-frame extrinsic
```

## Parallel Conversion

For SONIC:

- `--workers 1` runs serial conversion
- `--workers > 1` uses a `ProcessPoolExecutor`
- on Linux, workers are automatically pinned to the first `N` available CPU cores using `os.sched_setaffinity()`

Example:

```bash
uv run any4hdmi-convert-sonic \
  --csv-dir ../g1_sonic/complete/g1/csv \
  --out-dir output/sonic \
  --workers 8
```

## Viewer Logic

The viewer takes a single converted motion file:

```bash
uv run any4hdmi-view --motion output/sonic/motions/220705/Sideway_Walk_Right_001__A017.npz
```

Viewer steps:

1. Find the nearest dataset `manifest.json`.
2. Resolve the referenced MJCF repo snapshot from Hugging Face cache.
3. Load the referenced MJCF.
4. Inject viewer-only XML for:
   - skybox
   - checker groundplane material
   - ground light
   - floor plane
5. Build a temporary MJCF next to the cached source XML so relative mesh paths still work.
6. Load the motion `qpos`.
7. For each frame:
   - assign `data.qpos[:]`
   - zero `data.qvel[:]`
   - call `mj_forward`
   - sync the viewer

The viewer does not simulate dynamics. It only replays kinematic `qpos` frames.

## Typical Commands

Create the environment:

```bash
cd any4hdmi
uv sync
```

Convert LAFAN:

```bash
uv run any4hdmi-convert-lafan \
  --csv-dir ../lafan-process/LAFAN1_Retargeting_Dataset/g1 \
  --out-dir output/lafan
```

Convert SONIC:

```bash
uv run any4hdmi-convert-sonic \
  --csv-dir ../g1_sonic/complete/g1/csv \
  --out-dir output/sonic \
  --workers 8
```

Convert 100STYLE from an Axellwppr tarball:

```bash
uv run any4hdmi-convert-axellwppr \
  --input /home/elijah/Downloads/100style.tar \
  --out-dir output/100style
```

Override the MJCF reference:

```bash
uv run any4hdmi-convert-lafan \
  --csv-dir ../lafan-process/LAFAN1_Retargeting_Dataset/g1 \
  --out-dir output/lafan \
  --mjcf-repo elijahgalahad/g1_xmls \
  --mjcf-path g1-mode_13_15.xml \
  --mjcf-revision main
```

Replay a motion:

```bash
uv run any4hdmi-view --motion output/lafan/motions/dance1_subject2.npz
```

Replay headless:

```bash
uv run any4hdmi-view \
  --motion output/sonic/motions/220705/Sideway_Walk_Right_001__A017.npz \
  --headless
```

Filter a converted dataset with FK checks:

```bash
uv run any4hdmi-filter-dataset \
  --input-root output/sonic \
  --output-root output/sonic_filtered
```

## Notes

- Converted `qpos` is saved as `float32`.
- Viewer loads motion as `float32` and assigns it into MuJoCo runtime state per frame.
- Existing output folders generated with older local-path manifests should be reconverted or have their manifests rewritten to Hugging Face MJCF references.
- The dataset filter reconstructs `qvel` from adjacent `qpos` frames and runs FK with `mujoco_warp` when available, otherwise it falls back to CPU MuJoCo.
