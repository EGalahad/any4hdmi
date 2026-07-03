# LIMMT Contact-Based Physical Filter Plan

Date: 2026-06-17

## Goal

Make `any4hdmi-limmt-score` closer to the official LIMMT Stage I physical
filter by replacing the current height/body-distance proxies for floating,
penetration, and self-collision with MuJoCo contact-based metrics.

The change should stay local to the LIMMT scoring path. Do not change the
general FK cache or motion dataset runtime unless later evidence shows the
shared runner is the right ownership boundary.

## Current Local Behavior

`src/any4hdmi/limmt/physical_filter.py` currently uses `FKRunner` outputs:

- foot sliding: body height threshold plus body linear velocity
- velocity violation: root qvel, joint velocity, body linear velocity, body angular velocity
- self-collision: body-center pair distance below `self_collision_distance`
- penetration: minimum body z below `penetration_height`
- floating: minimum body z above `floating_height`
- jerk: second difference of qvel

This is fast and self-contained, but it is not equivalent to the official
contact-based feasibility score.

## Official LIMMT Reference

The official `projects/gqs/physics_filter.py` uses simulator contact data:

- add/use a floor in the MuJoCo scene
- run MJX forward/collision over each frame
- read active contact pairs and distances
- floating is based on floor-contact distance and a sustained air window
- penetration is based on floor contact distance
- self-collision counts active non-floor contacts with negative distance

The official code then computes `S_phy = 100 - sum(w_i * L_i)` and retains
motions with `S_phy >= 90`.

## Confirmed Local API Surface

Local environment checks show:

- `mjhub.temp_mjcf_with_floor(mjcf_path)` exists and is already used by
  `src/any4hdmi/scripts/viewer.py`.
- `mujoco_warp.make_data(mjm, nworld, nconmax=...)` supports an explicit
  contact buffer.
- `mujoco_warp` exposes `fwd_position`, `fwd_velocity`, `collision`, and
  `forward`.
- `mujoco_warp.Data.contact` has at least:
  - `geom`
  - `dist`
  - `worldid`
  - `dim`
- `wp.to_torch(d.contact.geom)` returns shape `[nworld * nconmax, 2]`.
- `wp.to_torch(d.contact.dist)` returns shape `[nworld * nconmax]`.
- `wp.to_torch(d.contact.worldid)` returns shape `[nworld * nconmax]`.

Avoid using `mujoco_warp.forward()` for this scoring path unless needed: it
enters the constraint solver and triggered heavy kernel compilation in a small
probe. The likely sufficient Warp sequence is:

```text
copy qpos/qvel
mjw.fwd_position(model, data)
mjw.collision(model, data)
mjw.fwd_velocity(model, data)
```

## Proposed Design

Add a LIMMT-local contact scoring helper, for example:

```text
ContactScoringRunner
```

or a LIMMT-local wrapper around `FKRunner` that returns the existing kinematic
outputs plus contact summaries.

Recommended returned summaries:

```text
floor_min_dist: [frames]
non_floor_contact_count: [frames]
left_foot_floor_contact: [frames]
right_foot_floor_contact: [frames]
contact_buffer_saturated: [frames] or scalar count
```

The scoring function should consume summaries instead of raw variable-length
contact arrays. This keeps `_score_motion` simple and avoids storing contact
lists in dataset caches.

## Implementation Steps

1. Add `temp_mjcf_with_floor(manifest.mjcf_path)` around the LIMMT scoring
   runner construction in `run_filter`.
2. Assert the floor-augmented model preserves the manifest qpos width.
3. Resolve the floor geom id by name `floor`; if missing, fall back to the
   first plane geom and report the selected id in the summary.
4. Keep existing FK outputs for velocity, jerk, and optional foot sliding.
5. Add CPU contact summary path using `mujoco.MjData.contact`.
6. Add Warp contact summary path using `mjw.fwd_position`,
   `mjw.collision`, and `mjw.fwd_velocity`.
7. Replace metrics in `_score_motion` when contact summaries are present:
   - `penetration = mean(max(0, -floor_min_dist - 0.01))`
   - `floating_frames_ratio = sustained_air_ratio(floor_min_dist > 0.05)`
   - `self_collision = mean(non_floor_contact_count > 0)` or official-style
     normalized contact count
8. Preserve fallback behavior for tests and for any future non-contact runner:
   if contact summaries are absent, use the current body-height/body-distance
   proxy logic.
9. Add summary metadata to `scores.json` / `summary.json`:
   - `contact_based: true`
   - `floor_geom_id`
   - `contact_backend`
   - `nconmax`
   - `contact_buffer_saturation_count`

## Open Choices

- Whether self-collision should be a binary per-frame ratio or the official
  clipped contact-count average.
- Whether foot sliding should immediately switch to floor-contact-gated feet
  or remain height-gated for the first pass.
- The right `nconmax` default. Start with `128` per world unless benchmarks
  show it is excessive.
- Whether to expose `--contact-nconmax`, `--contact-floating-distance`, and
  `--contact-penetration-margin` as CLI options or keep them internal.

## Validation

Required before trusting the output:

- Unit tests for CPU contact summaries on a tiny sphere/floor MJCF:
  - penetrating sphere produces negative floor distance and positive penetration
  - high sphere produces air/floating signal
  - two colliding non-floor geoms produce self-collision count
- Existing `test_limmt.py` still passes.
- Smoke run on a small AMASS subset with CPU and Warp backends.
- Compare old vs contact-based score distribution on the same subset:
  - kept/rejected count
  - bottom-ranked motions
  - floating/penetration/self-collision quantiles
- Check that full AMASS runtime remains acceptable with Warp.

## Risks

- Contact buffer saturation can silently undercount collisions if `nconmax` is
  too small.
- Floor insertion may fail for MJCFs without an `<asset>` or `<worldbody>` block,
  because `mjhub.temp_mjcf_with_floor` expects both.
- Warp contact fields are flattened by world id; implementation must group by
  `contact.worldid` and ignore inactive entries (`dim == 0` or invalid slots).
- Official LIMMT weights and documentation are not perfectly consistent, so the
  first contact-based pass should focus on metric correctness before retuning
  weights.

## Current Recommendation

Implement contact-based `floating`, `penetration`, and `self_collision` first.
Leave foot sliding height-gated for the first version unless a small validation
run shows it dominates remaining score error. After score distribution is
available, decide whether to retune weights toward the official code formula.
