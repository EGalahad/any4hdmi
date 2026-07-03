# Commands

```bash
cd /home/elijah/Documents/projects/simple-tracking/any4hdmi
```

Physical filter:

```bash
.venv/bin/any4hdmi-limmt-score --input-path output/amass --project-path output/limmt_amass --pass-dataset-name amass_limmt_pass --device cuda --pass-threshold 90 --batch-frames 131072 --fk-batch-size 8192
```

HME training:

```bash
.venv/bin/any4hdmi-limmt-train-hme --project-path output/limmt_amass --pass-dataset-name amass_limmt_pass --batch-size 256 --epochs 30 --win-sec 4.0 --phase-dim 8 --downsample-rate 5
```

HME multi-GPU training:

```bash
torchrun --nproc_per_node <N> -m any4hdmi.limmt.hme.train --project-path output/limmt_amass --pass-dataset-name amass_limmt_pass --batch-size 256 --epochs 30
```

Embeddings:

```bash
.venv/bin/any4hdmi-limmt-embed --project-path output/limmt_amass --pass-dataset-name amass_limmt_pass
```

GQS subsets:

```bash
.venv/bin/any4hdmi-limmt-gqs --project-path output/limmt_amass --pass-dataset-name amass_limmt_pass --ratios 0.04 0.08 0.16 0.32
```

Visualization:

```bash
.venv/bin/any4hdmi-limmt-visualize --project-path output/limmt_amass
```
