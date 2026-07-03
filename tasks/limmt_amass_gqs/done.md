# Done

- Added LIMMT package implementation plan.
- Implemented `any4hdmi.limmt` modules and CLI entrypoints.
- Added focused LIMMT unit tests.
- Ran full local AMASS physical filtering with MuJoCo Warp.
- Validated one pass-dataset motion through `load_any4hdmi_dataset`.
- Rebuilt HME feature cache on `rp-4090-2` with heading-local root features.
- Started 8-GPU HME DDP training on `rp-4090-2`; first two epochs produced loss values.
- Recorded Humanoid-GPT HME implementation paths and method notes in `humanoid_gpt_hme_reference.md`.
- Trained the best `phase_dim=32`, `hidden_dims=(256,256)`, OneCycle HME configuration for 40 epochs; final loss `0.423773`, best epoch 37 loss `0.420378`.
- Generated HME t-SNE/UMAP visualizations and documented kept outputs in `output/limmt_amass/visualization_summary.md`.
- Generated GQS weighted-FPS subsets for 4%, 8%, 16%, and 32%.
- Generated selected-vs-not-selected scatter overlays on t-SNE and UMAP for each GQS ratio.
