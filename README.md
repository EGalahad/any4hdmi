# any4hdmi

`any4hdmi` defines one simple `qpos`-based motion format for HDMI-related datasets.

- one dataset-level `manifest.json`
- one `motions/**/*.npz` file per motion
- each motion file stores only `qpos`
- each manifest stores an MJCF reference on Hugging Face, not a vendored local XML/STL tree

The motion plus the dataset timestep from `manifest.json` is enough to replay the clip in MuJoCo.

## Datasets

Clone the source datasets from Hugging Face:

```bash
git clone https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset data/LAFAN1_Retargeting_Dataset
git clone https://huggingface.co/datasets/bones-studio/seed data/seed

tar xzf data/seed/g1.tar -C data/seed/g1
```

## Commands

Convert LAFAN:

```bash
uv run any4hdmi-convert-lafan \
  --csv-dir data/LAFAN1_Retargeting_Dataset/g1 \
  --out-dir output/lafan
```

Convert SONIC:

```bash
uv run any4hdmi-convert-sonic \
  --csv-dir data/seed/g1/csv \
  --out-dir output/sonic
```

Override the MJCF reference if needed:

```bash
uv run any4hdmi-convert-sonic \
  --csv-dir data/seed/g1/csv \
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

Pipeline details live in [docs/pipeline.md](docs/pipeline.md).
Dataset format details live in [docs/dataset_format.md](docs/dataset_format.md).