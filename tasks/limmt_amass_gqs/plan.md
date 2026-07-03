# LIMMT AMASS GQS Plan

Implement LIMMT-style selection in `src/any4hdmi/limmt` with full physical quality filtering, HME training, HME visualization, and GQS weighted FPS subsets.

Expected output roots:

- `output/limmt_amass/amass_limmt_pass`
- `output/limmt_amass/hme`
- `output/limmt_amass/embeddings`
- `output/limmt_amass/visualizations`
- `output/limmt_amass/subsets/amass_limmt_gqs_4`
- `output/limmt_amass/subsets/amass_limmt_gqs_8`
- `output/limmt_amass/subsets/amass_limmt_gqs_16`
- `output/limmt_amass/subsets/amass_limmt_gqs_32`

Acceptance:

- physical filter runs with MuJoCo Warp
- full AMASS physical filter wall time is under 15 minutes
- physical filter falls back to CPU MuJoCo when optional Warp packages are absent
- full HME training runs on rp SSH server
- unit tests pass
- generated subset manifests are loadable by `load_any4hdmi_dataset`
