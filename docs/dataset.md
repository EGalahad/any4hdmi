# Dataset Module

`src/any4hdmi/dataset` 现在只负责 any4hdmi 自己的数据集加载与 cache materialization。

它不负责：

- `MotionDataset`
- `MotionData` 的上层语义封装
- legacy dataset format 的加载

这些现在应该由上层项目自己持有，例如 `active-adaptation/projects/hdmi/hdmi/tasks/motion.py`。


## Public API

当前对外暴露的接口定义在 [src/any4hdmi/dataset/__init__.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/__init__.py)：

- `LoadedDatasetPayload`
- `resolve_input_paths`
- `load_cached_any4hdmi_dataset`

包根 [src/any4hdmi/__init__.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/__init__.py) 也只 re-export 这三个符号。


## Module Layout

### `loading.py`

文件位置：
[src/any4hdmi/dataset/loading.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/loading.py)

职责：

- 定义 `LoadedDatasetPayload`
- 解析输入路径
- 识别 any4hdmi dataset root
- 读取 `manifest.json`
- 收集 `.npz` motion 文件路径
- 构建 `LoadedDatasetPayload.data` 对应的 tensor container
- 在加载时按 `asset_joint_names` 做 joint remap

核心接口：

- `resolve_input_paths(base_dir, root_path) -> list[Path]`
- `find_any4hdmi_root(path) -> Path | None`
- `load_any4hdmi_manifest(dataset_root) -> dict`
- `resolve_any4hdmi_dataset_context(input_paths) -> tuple[Path, dict]`
- `resolve_any4hdmi_motion_paths(input_paths) -> tuple[Path, dict, list[Path]]`
- `build_motion_data(...) -> tuple[MotionData, list[int], list[int]]`
- `build_motion_data_from_fields(...) -> MotionData`
- `apply_joint_mapping(...) -> list[str]`
- `resolve_source_fps(manifest) -> float`


### `cache.py`

文件位置：
[src/any4hdmi/dataset/cache.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/cache.py)

职责：

- 根据 input paths 定位 any4hdmi dataset
- 基于 manifest、motion 文件、MJCF 和 `target_fps` 生成 cache key
- 首次加载时从 qpos motion materialize 出 FK 结果
- 把 materialized data 持久化到磁盘
- 下次命中 cache 时直接反序列化为 `LoadedDatasetPayload`

唯一对外公开的加载入口：

- `load_cached_any4hdmi_dataset(input_paths, asset_joint_names, target_fps, base_dir) -> LoadedDatasetPayload`

内部流程：

1. `resolve_any4hdmi_dataset_context()` 和 `resolve_any4hdmi_motion_paths()` 找到 dataset root、manifest、motion 列表
2. `_resolve_any4hdmi_mjcf_path()` 解析 MJCF
3. `_make_qpos_cache_key()` 计算缓存键
4. 若 cache 未命中，则 `_build_qpos_cache()`：
   - 读取每个 motion 的 `qpos`
   - 用 MuJoCo `mj_differentiatePos` 计算 `qvel`
   - 用 `any4hdmi.fk.FKRunner` 做 FK materialization
   - 用 `dataset.interpolation.interpolate()` 统一到 `target_fps`
   - 汇总成一份连续 tensor 数据
5. `_load_qpos_cache_entry()` 反序列化 cache，并在需要时做 joint remap


### `interpolation.py`

文件位置：
[src/any4hdmi/dataset/interpolation.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/interpolation.py)

职责：

- 实现 motion field 的时间插值

包含：

- `lerp()` / `_lerp_torch()`
- `slerp()` / `_slerp_torch()`
- `interpolate(motion, source_fps, target_fps)`

当前支持插值的 keys：

- `body_pos_w`
- `body_lin_vel_w`
- `body_quat_w`
- `body_ang_vel_w`
- `joint_pos`
- `joint_vel`

如果传入其他 key，会直接抛 `NotImplementedError`。


### `types.py`

文件位置：
[src/any4hdmi/dataset/types.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/types.py)

职责：

- 定义 `LoadedDatasetPayload.data` 使用的轻量 tensor container `MotionData`

字段：

- `motion_id`
- `step`
- `body_pos_w`
- `body_lin_vel_w`
- `body_quat_w`
- `body_ang_vel_w`
- `joint_pos`
- `joint_vel`

支持：

- `len(data)`
- `data.device`
- `data.to(device)`
- `data[idx]`

说明：

- 这个 `MotionData` 是 dataset loader 内部使用的数据载体
- 它不是上层任务语义里的 `MotionDataset`
- 如果上层项目需要自己的 `MotionData` / `MotionDataset` 类型，应该自己做转换


## LoadedDatasetPayload

`LoadedDatasetPayload` 定义在 [loading.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/loading.py)：

```python
@dataclass(frozen=True)
class LoadedDatasetPayload:
    body_names: list[str]
    joint_names: list[str]
    motion_paths: list[Path]
    starts: list[int]
    ends: list[int]
    data: MotionData
```

语义：

- `body_names`: FK 输出中的 body 名称顺序
- `joint_names`: `joint_pos` / `joint_vel` 的列顺序
- `motion_paths`: 对应的 motion `.npz` 文件列表
- `starts` / `ends`: 每条 motion 在扁平化 `data` 里的边界
- `data`: 按时间拼接后的 tensor 数据


## Cache Format

cache 根目录：

- `<base_dir>/.cache/motion/qpos_fk/`

每个 cache entry 的目录：

- `<base_dir>/.cache/motion/qpos_fk/<cache_key>/`

`cache_key` 由以下信息的哈希生成：

- `dataset_root`
- `manifest.json` 的 stat fingerprint
- `mjcf` 文件的 stat fingerprint
- `target_fps`
- 每个 motion `.npz` 文件的 stat fingerprint
- 若存在，同名 sidecar `.json` 的 stat fingerprint

cache 现在始终使用 `TensorDict.memmap` 持久化。

GPU promote 阈值定义在 [cache.py](/home/elijah/Documents/projects/simple-tracking/any4hdmi/src/any4hdmi/dataset/cache.py)：

```python
CACHE_GPU_PROMOTE_THRESHOLD_BYTES = 16 * 1024**3
```

构建时会先扫描每个 motion 的插值后长度，估算 materialized FK 数据总字节数。

读取时的逻辑是：

- 总是先从 memmap 读
- 如果当前有 CUDA
- 并且 `estimated_bytes <= 16 GiB`
- 并且当前 GPU 的 free memory 足够

则把整份 loaded payload 直接搬到 GPU。

一个 cache entry 当前包含：

### `motion_index.json`

JSON 元数据，字段包括：

- `body_names`
- `joint_names`
- `starts`
- `ends`
- `motion_paths`
- `source_fps`
- `target_fps`
- `total_length`


### `cache_meta.json`

JSON 元数据，字段包括：

- `cache_version`
- `dataset_root`
- `manifest_path`
- `mjcf_path`
- `target_fps`


### `td/`

`TensorDict.memmap(...)` 生成的磁盘映射张量目录。

存储的 tensor 字段：

- `motion_id`
- `step`
- `body_pos_w`
- `body_lin_vel_w`
- `body_quat_w`
- `body_ang_vel_w`
- `joint_pos`
- `joint_vel`

说明：

- 这不是原始 qpos cache
- 它存的是已经做完 FK 和插值后的 materialized motion tensors
- 默认先以 CPU memmap 方式读取
- 小 cache 会在 load 后被整体提升到 GPU


### `ready.flag`

纯文本标志文件，用于表示 cache entry 构建完成。


`motion_index.json` 和 `cache_meta.json` 都会记录：

- `storage: "memmap"`
- `estimated_bytes`

loader 读取时会调用：

- `TensorDict.load_memmap(cache_entry_dir / "td")`

然后按 `estimated_bytes` 和当前可用 GPU 内存决定是否调用 `.to("cuda")`。


## Locking Behavior

`cache.py` 使用目录锁避免并发重复构建：

- lock 路径：
  `<cache_root>/<cache_key>.lock`
- 如果某个进程持有锁，其他进程会等待
- 超时默认是 600 秒


## Joint Remap Behavior

`load_cached_any4hdmi_dataset()` 支持传入 `asset_joint_names`。

行为：

- 如果 `asset_joint_names is None`，保持 cache 中原始 joint 顺序
- 如果提供了 `asset_joint_names`，则：
  - 共有 joint 会重排到 `asset_joint_names` 指定顺序
  - cache 中存在但 asset 不存在的 joint 会追加到末尾

这个 remap 只作用于：

- `joint_names`
- `data.joint_pos`
- `data.joint_vel`


## Current Scope

`src/any4hdmi/dataset` 当前只面向 any4hdmi 自己的 dataset root layout：

- `manifest.json`
- `<motions_subdir>/**/*.npz`

它明确不处理：

- legacy `motion.npz + meta.json` layout
- 上层任务侧的 `MotionDataset` 封装
- 训练时的采样逻辑

如果上层项目需要兼容 legacy layout，应该在上层自己做。


## Typical Usage

上层项目的典型调用方式：

```python
from any4hdmi import load_cached_any4hdmi_dataset, resolve_input_paths

input_paths = resolve_input_paths(base_dir, root_path)
payload = load_cached_any4hdmi_dataset(
    input_paths=input_paths,
    asset_joint_names=asset_joint_names,
    target_fps=50,
    base_dir=base_dir,
)
```

然后由上层把 `payload` 转换成自己的 `MotionData` / `MotionDataset` 类型。


## Notes

- `dataset/types.py` 里的 `MotionData` 目前仍然存在，因为 `LoadedDatasetPayload` 需要一个统一的数据载体
- 但这个类型属于 loader 内部实现细节，不应再被当作 any4hdmi 的主对外抽象
- 如果后续需要进一步收缩 API，可以继续把 `MotionData` 变成纯内部类型，而不是在文档中强调直接使用它
- `memmap` 模式依赖 `tensordict`
