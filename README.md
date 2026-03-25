# any4hdmi

`any4hdmi` defines one simple `qpos`-based motion format for HDMI-related datasets.

- one dataset-level `manifest.json`
- one `motions/**/*.npz` file per motion
- each motion file stores only `qpos`
- each manifest stores an MJCF reference on Hugging Face, not a vendored local XML/STL tree

The motion plus the dataset timestep from `manifest.json` is enough to replay the clip in MuJoCo.

## MJCF Assets

This repo does not vendor `g1.xml` or STL meshes under `assets/`.

Instead, manifests and converters use the Hugging Face repo `elijahgalahad/g1_xmls` and resolve the MJCF from the standard Hugging Face cache at runtime. The default reference is:

- repo: `elijahgalahad/g1_xmls`
- path: `g1.xml`
- revision: `main`

The current XML/STL bundle can be refreshed from:

- `https://github.com/EGalahad/lafan-process`
- `https://huggingface.co/elijahgalahad/g1_xmls/tree/main`

## Layout

```text
any4hdmi/
  docs/
  output/
    <dataset>/
      manifest.json
      motions/
  src/any4hdmi/
```

## Environment

```bash
uv sync
```

## Commands

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
  --out-dir output/sonic
```

Override the MJCF reference if needed:

```bash
uv run any4hdmi-convert-sonic \
  --csv-dir ../g1_sonic/complete/g1/csv \
  --out-dir output/sonic \
  --mjcf-repo elijahgalahad/g1_xmls \
  --mjcf-path g1.xml \
  --mjcf-revision main
```

Replay a converted motion:

```bash
uv run any4hdmi-view --motion output/lafan/motions/dance1_subject2.npz
```

Headless check:

```bash
uv run any4hdmi-view \
  --motion output/sonic/motions/230322/reach_jump_R_001__A299_M.npz \
  --headless
```

Pipeline details live in [docs/pipeline.md](/home/elijah/Documents/projects/simple-tracking/any4hdmi/docs/pipeline.md).
Dataset format details live in [docs/dataset_format.md](/home/elijah/Documents/projects/simple-tracking/any4hdmi/docs/dataset_format.md).
