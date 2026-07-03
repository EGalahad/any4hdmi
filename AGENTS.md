# any4hdmi Notes

- Windowed datasets default to staging the next 512-frame window on the current runtime GPU (`windowed_next_window_device: current`) and using pinned CPU scratch for background window loads (`windowed_pin_window_load: true`).
- If a large run OOMs after enabling windowed datasets, first try moving the next window back to CPU with `windowed_next_window_device: cpu` or `ANY4HDMI_NEXT_WINDOW_DEVICE=cpu`. This saves the extra fp16 next-window GPU pool.
- If pinned host memory allocation fails or causes system memory pressure, disable it with `windowed_pin_window_load: false` or `ANY4HDMI_PIN_WINDOW_LOAD=0`.
- For HDMI-style tracking tasks, always prune unused motion fields before constructing full/windowed datasets. Keeping only task-used bodies plus required joints made GPU next-window staging feasible at 8192 envs on 24 GB GPUs; this should not be exposed as a normal config toggle.
- Do not assume `full_motion: false` is automatically faster. With CPU next-window staging it was measurably slower; the fast path needs pruned fields, GPU next-window staging, and pinned window loads.
- Keep `RUNTIME_MOTION_MAX_LEN` fixed at 512 unless there is a benchmark-backed reason to change it. Do not dynamically shorten sampled windows by motion length; clamp sampled starts instead.
- For pruned any4hdmi FK-cache VRAM estimates, count stored fields as:
  `motion_id:int64 + step:int64 + body_pos/body_lin_vel/body_quat/body_ang_vel:float32 + joint_pos/joint_vel:float32`.
  Formula: `bytes_per_frame = 16 + 52 * body_count + 8 * joint_count`.
  For G1 tracking-base pruning this is currently `body_count=14`, `joint_count=29`, so `976 bytes/frame`.
  At target FPS resampling, sum per-motion lengths with `floor((source_len - 1) / source_fps * target_fps) + 1`, then multiply by `bytes_per_frame`; do not just scale total frames when exact counts matter.
  Windowed mode keeps one fp32 current window plus, by default, one fp16 next window: approximately `num_envs * RUNTIME_MOTION_MAX_LEN * bytes_per_frame * 1.5`, plus small int metadata and overhead.
