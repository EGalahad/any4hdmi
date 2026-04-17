# Dataset Format

This document defines the final on-disk dataset format used by `any4hdmi`.

## Overview

Each converted dataset root contains:

- one dataset-level `manifest.json`
- one `motions/` directory
- one motion `.npz` per clip

The final standard format does not include per-motion sidecar JSON files, and it does not require a checked-in local MJCF asset tree.

## Directory Layout

```text
output/<dataset>/
  manifest.json
  motions/
    <optional/subdirs>/
      clip_name.npz
```

Examples:

```text
output/lafan/
  manifest.json
  motions/
    dance1_subject2.npz

output/sonic/
  manifest.json
  motions/
    230322/
      jump_ff_180_R_002__A296.npz
```

## Dataset-Level File

### `manifest.json`

Stored at the dataset root.

Required fields:

- `format_version`: integer format version
- `dataset_name`: dataset identifier such as `lafan` or `sonic`
- `mjcf`: MJCF reference string
- `motions_subdir`: usually `"motions"`
- `timestep`: dataset timestep in seconds
- `qpos_dim`: MuJoCo `nq`
- `qpos_names`: ordered list of qpos names
- `num_motions`: number of converted clips
- `source`: dataset-level conversion settings and provenance

Example:

```json
{
  "format_version": 2,
  "dataset_name": "sonic",
  "mjcf": "hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml",
  "motions_subdir": "motions",
  "timestep": 0.008333333333333333,
  "qpos_dim": 36,
  "qpos_names": [
    "root_tx",
    "root_ty",
    "root_tz",
    "root_qw",
    "root_qx",
    "root_qy",
    "root_qz"
  ],
  "num_motions": 142220,
  "source": {
    "fps": 120.0,
    "translation_scale": 0.01,
    "angle_unit": "deg",
    "euler_order": "xyz",
    "euler_frame": "extrinsic"
  }
}
```

Current converter output writes `mjcf` as an `hf://...` string. The runtime also accepts
relative local XML paths for compatibility, but the canonical generated format is the
Hugging Face reference form shown above.

## Motion Files

### `motions/**/*.npz`

Required array:

- `qpos`: `float32`, shape `[num_frames, nq]`

Rules:

- `qpos` uses MuJoCo ordering
- the root free joint is stored as:
  - `root_tx`
  - `root_ty`
  - `root_tz`
  - `root_qw`
  - `root_qx`
  - `root_qy`
  - `root_qz`
- remaining entries follow the order in `manifest.json -> qpos_names`

Everything clip-specific that is needed for playback is derived from:

- `qpos.shape`
- the file path
- the dataset `manifest.json`

## Current Converter Behavior

The current converter entrypoints all write the same final structure:

- `src/any4hdmi/scripts/preprocess/lafan.py`
- `src/any4hdmi/scripts/preprocess/sonic.py`
- `src/any4hdmi/scripts/preprocess/axellwppr.py`

In all three cases:

1. One motion `.npz` is written per clip.
2. One dataset-level `manifest.json` is written at the output root.
3. `format_version` is `2`.
4. `mjcf` is emitted as an `hf://...` reference by default.

## Compatibility Notes

- Viewer and filtering code should rely on `manifest.json` plus motion `.npz`.
- Dataset-global timing should live in `manifest.json -> timestep`.
- Older manifests that stored a local string path for `mjcf` are still readable, but they are compatibility input, not the preferred emitted format.
- New manifests should use `format_version = 2` and the `hf://...` MJCF string shown above.
