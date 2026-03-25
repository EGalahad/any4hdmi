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
- `mjcf`: Hugging Face MJCF reference
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
  "mjcf": {
    "kind": "huggingface",
    "repo_id": "elijahgalahad/g1_xmls",
    "path": "g1.xml",
    "revision": "main"
  },
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

`mjcf.kind == "huggingface"` means the runtime should resolve the XML and meshes from the Hugging Face cache using `snapshot_download`.

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

The current converter code for both datasets writes the same final structure:

- `src/any4hdmi/datasets/lafan.py`
- `src/any4hdmi/datasets/sonic.py`

In both cases:

1. A motion `.npz` is saved.
2. A dataset `manifest.json` is written with a Hugging Face MJCF reference.

## Compatibility Notes

- Viewer and filtering code should rely on `manifest.json` plus motion `.npz`.
- Dataset-global settings such as `fps` should live in `manifest.json`.
- Older manifests that stored a local string path for `mjcf` are legacy format.
- New manifests should use `format_version = 2` and the Hugging Face MJCF object shown above.
