# Dataset Runtime

`src/any4hdmi/dataset/` 负责把 any4hdmi 的 `manifest.json + motions/**/*.npz` 数据集加载成运行时可用的 motion dataset。

它的职责只包括：

- 解析输入路径和 dataset root
- 基于 `qpos` 构建或复用 FK cache
- 暴露统一的 `sample_motion()` / `get_slice()` 数据集接口

它不负责：

- 源数据集转换
- 上层任务逻辑或 reward / command 语义
- `docs/legacy/` 里那批已经归档的旧 `online.py` 设计

## Public API

当前对外导出的符号定义在 [src/any4hdmi/dataset/__init__.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/__init__.py)：

- `BaseDataset`
- `DatasetIndex`
- `MotionData`
- `MotionSample`
- `FullMotionDataset`
- `WindowedMotionDataset`
- `OnlineQposDataset`
- `load_any4hdmi_dataset`
- `resolve_input_paths`

其中 `OnlineQposDataset` 目前只是 `WindowedMotionDataset` 的别名。

## Module Layout

### `base.py`

[src/any4hdmi/dataset/base.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/base.py)

定义当前 runtime dataset 统一接口和基础数据结构：

- `MotionData`
  一个 `TensorClass`，字段包括 `motion_id`、`step`、`body_pos_w`、`body_lin_vel_w`、`body_quat_w`、`body_ang_vel_w`、`joint_pos`、`joint_vel`
- `DatasetIndex`
  只保存平铺后的 `motion_id` / `step` 索引
- `MotionSample`
  `sample_motion()` 的返回值，包含 `motion_id`、`motion_len`、`start_t`
- `BaseDataset`
  抽象基类，统一要求实现 `to()`、`get_slice()`、`sample_motion()`

### `loading.py`

[src/any4hdmi/dataset/loading.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/loading.py)

负责输入解析和 dataset root 解析：

- `resolve_input_paths(base_dir, root_path)`
- `find_any4hdmi_root(path)`
- `load_any4hdmi_manifest(dataset_root)`
- `resolve_any4hdmi_dataset_context(input_paths)`
- `resolve_any4hdmi_motion_paths(input_paths)`
- `resolve_source_fps(manifest)`

当前 canonical manifest 字段是 `timestep`。`resolve_source_fps()` 仍兼容读取 legacy 顶层 `fps`，但新格式默认从 `timestep` 反推。

### `fk_cache.py`

[src/any4hdmi/dataset/fk_cache.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/fk_cache.py)

负责 FK cache 的构建与加载。

当前 cache 根目录：

- `<base_dir>/.cache/motion/qpos_online_v2/`

核心流程：

1. 根据 `dataset_root`、`manifest.json`、MJCF 内容、`target_fps` 和 motion 文件 fingerprint 生成 cache key
2. 读取 motion `qpos`
3. 重建 `qvel`
4. 通过 `interpolate_qpos_qvel_batch_torch(...)` 对齐到 `target_fps`
5. 用 `FKRunner` 计算 FK
6. 把结果写入 memmap 存储，并保存 `motion_index.json` / `cache_meta.json`

`FKCacheEntry` 暴露的是已经 materialize 好的字段：

- `body_names`
- `joint_names`
- `motion_paths`
- `starts`
- `ends`
- `storage_fields`

### `interpolation.py`

[src/any4hdmi/dataset/interpolation.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/interpolation.py)

负责 cache build 过程里的时间重采样，包括：

- `interpolate_qpos_qvel_batch_torch(...)`
- `resampled_length(...)`

这里服务的是 FK cache 构建，不是 viewer 播放逻辑。

### `loaders.py`

[src/any4hdmi/dataset/loaders.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/loaders.py)

当前统一加载入口是：

```python
load_any4hdmi_dataset(
    *,
    input_paths: list[Path],
    target_fps: int,
    base_dir: Path,
    asset_joint_names: list[str] | None = None,
    num_envs: int,
    full_motion: bool = True,
) -> BaseDataset
```

行为：

- 总是先通过 `FKCache.from_inputs(...).get_or_build()` 拿到 cache
- `full_motion=True` 时返回 `FullMotionDataset`
- `full_motion=False` 时返回 `WindowedMotionDataset`

注意：

- `asset_joint_names` 目前被保留但未使用
- 加载器现在不再做旧版 joint remap / legacy dataset format 兼容层

### `full.py`

[src/any4hdmi/dataset/full.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/full.py)

`FullMotionDataset` 直接把整份 cache 当作可随机访问的数据集使用。

特点：

- `MotionSample.motion_id` 是真实的 dataset motion id
- `get_slice()` 通过 `starts[motion_id] + local_step` 直接 gather
- 适合整份数据都可直接放到运行设备上的场景

### `windowed.py`

[src/any4hdmi/dataset/windowed.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/windowed.py)

`WindowedMotionDataset` 是当前在线运行时实现。

当前语义不是历史 `docs/legacy/` 里那套 `online.py` 状态机，而是：

- 为每个 env 维护一份 `current` window 和一份 `next` window
- `RUNTIME_MOTION_MAX_LEN = 512`
- 用单线程 `ThreadPoolExecutor` 预取下一窗到 runtime pool
- `sample_motion()` 在 non-rewind 时等待并 promote `next -> current`，随后再次调度新的 `next`
- `get_slice()` 只从当前 window pool gather

这里有一个当前实现上的重要约定：

- `WindowedMotionDataset.sample_motion()` 返回的 `motion_id` 是运行时 handle，当前等于 `env_ids`
- 因此 `WindowedMotionDataset.get_slice(motion_ids=...)` 实际是在索引“env 对应的 current window”
- 这和 `FullMotionDataset` 使用真实 motion id 的语义不同

## Data Flow

当前 dataset runtime 流程可以概括为：

1. `resolve_input_paths(...)` 把 CLI / 上层传入路径规范化
2. `FKCache` 解析 dataset root、manifest、motion 列表并构建或命中磁盘 cache
3. `load_any4hdmi_dataset(...)` 选择返回 `FullMotionDataset` 或 `WindowedMotionDataset`
4. 上层通过 `sample_motion()` 和 `get_slice()` 读取 motion 数据

## Related Docs

- 数据集磁盘格式见 [dataset_format.md](/home/elijah/Documents/projects/simple-tracking/any4hdmi/docs/dataset_format.md)
- 转换、viewer、filter 流程见 [pipeline.md](/home/elijah/Documents/projects/simple-tracking/any4hdmi/docs/pipeline.md)
- 已归档的旧 online 设计文档见 [legacy/README.md](/home/elijah/Documents/projects/simple-tracking/any4hdmi/docs/legacy/README.md)
