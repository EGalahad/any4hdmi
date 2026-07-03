# Metrics

## Physical Filter

- Input: `output/amass`
- Output: `output/limmt_amass`
- Backend: `mujoco_warp`
- Device: `cuda:0`
- Wall time: `25.470482373000777s`
- Total motions: `10385`
- Kept motions: `10382`
- Rejected motions: `3`
- Pass dataset: `output/limmt_amass/amass_limmt_pass`
- Pass dataset total hours: `35.732683333333334`

## Tests

- `.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v`: passed, 11 tests.
- `rp-4090-2 .venv/bin/python -m unittest discover -s tests -p "test_limmt.py" -v`: passed, 11 tests.
- Pass dataset one-motion load via `load_any4hdmi_dataset`: passed.

## HME Training

- Host: `rp-4090-2`
- Session: `tmux attach -t limmt_hme_40ep`
- Log: `output/limmt_amass/hme_phase32_h256_onecycle_40ep/train.log`
- Cache log: `output/limmt_amass/hme/cache.log`
- Feature type: `joint_pos_joint_vel_root_pose6d_initial_heading_root_vel_current_root_local`
- Cache records: `10382`
- Cache windows: `4355483`
- State dim: `73`
- World size: `8`
- Best retained checkpoint: `output/limmt_amass/hme/experiments/hme_phase32_hidden256_onecycle_maxlr3e-3.pt`
- Retained checkpoint curve: epoch 20 `0.462068255702498`, epoch 21 `0.45780696192916626`
- Removed extra checkpoints: partial cosine, hidden512, phase64, canonical stale `hme.pt`, and rp-4090-1 max-lr trial checkpoint.
- 40 epoch training output: `output/limmt_amass/hme_phase32_h256_onecycle_40ep/`
- 40 epoch command: `HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -m torch.distributed.run --nproc_per_node 8 -m any4hdmi.limmt.hme.train --batch-size 256 --epochs 40 --phase-dim 32 --hidden-dims 256 256 --scheduler onecycle --lr 3e-4 --max-lr 3e-3 --onecycle-pct-start 0.15 --onecycle-div-factor 10 --onecycle-final-div-factor 100 ...`
- 40 epoch final loss: epoch 40 `0.42377325172206654`
- 40 epoch best loss: epoch 37 `0.42037769771086425`
- NCCL startup: clean after `init_process_group(..., device_id=torch.device(f"cuda:{local_rank}"))`; no barrier device warning in the restarted log.

## HME Visualization

- Checkpoint: `output/limmt_amass/hme/experiments/hme_phase32_hidden256_onecycle_maxlr3e-3.pt`
- Embeddings: `output/limmt_amass/embeddings_phase32_h256_onecycle_e20/embeddings.npz`
- Embeddings CSV: `output/limmt_amass/embeddings_phase32_h256_onecycle_e20/embeddings.csv`
- t-SNE length: `output/limmt_amass/visualizations_phase32_h256_onecycle_e20/hme_tsne_length.png`
- t-SNE physical score: `output/limmt_amass/visualizations_phase32_h256_onecycle_e20/hme_tsne_physical_score.png`
- UMAP kmeans30: `output/limmt_amass/visualizations_phase32_h256_onecycle_e20_umap_variants/hme_umap_spread04_kmeans30.png`
- UMAP length: `output/limmt_amass/visualizations_phase32_h256_onecycle_e20_umap_variants/hme_umap_spread04_length.png`
- UMAP physical score: `output/limmt_amass/visualizations_phase32_h256_onecycle_e20_umap_variants/hme_umap_spread04_physical_score.png`
- Visualization summary: `output/limmt_amass/visualization_summary.md`

## GQS Subsets

- Embeddings: `output/limmt_amass/embeddings_phase32_h256_onecycle_e20/embeddings.npz`
- Alpha: `0.6`
- Complexity CSV: `output/limmt_amass/subsets/complexity.csv`
- Report: `output/limmt_amass/subsets/gqs_report.json`
- 4% subset: `output/limmt_amass/subsets/amass_limmt_gqs_4`, selected/files `415`
- 8% subset: `output/limmt_amass/subsets/amass_limmt_gqs_8`, selected/files `831`
- 16% subset: `output/limmt_amass/subsets/amass_limmt_gqs_16`, selected/files `1661`
- 32% subset: `output/limmt_amass/subsets/amass_limmt_gqs_32`, selected/files `3322`
- Validation: subset manifests load and selected counts match copied `.npz` counts.

## GQS Selection Visualization

- Summary: `output/limmt_amass/gqs_selection_visualizations/selection_visualization_summary.md`
- Combined grid: `output/limmt_amass/gqs_selection_visualizations/gqs_selected_tsne_umap_grid.png`
- Per-ratio plots: `gqs_4_selected_{tsne,umap}.png`, `gqs_8_selected_{tsne,umap}.png`, `gqs_16_selected_{tsne,umap}.png`, `gqs_32_selected_{tsne,umap}.png`

## Pending

- Optional: embed/visualize the 40 epoch checkpoint if it replaces the retained 20/21 epoch checkpoint for downstream GQS.
