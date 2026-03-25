from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import mujoco
from mujoco import viewer
from tqdm import tqdm

from any4hdmi.format import find_dataset_root, load_manifest, load_motion


VIEWER_VISUAL_XML = """\
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
"""

VIEWER_ASSET_XML = """\
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
"""

VIEWER_WORLDBODY_XML = """\
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a qpos-only any4hdmi motion with MuJoCo."
    )
    parser.add_argument(
        "--motion",
        required=True,
        help="Path to a converted motion .npz file. The dataset root is inferred from manifest.json.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS override. Defaults to 1 / manifest.timestep.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a viewer window.",
    )
    return parser.parse_args()


def _iter_frame_indices(length: int, start: int, end: int, stride: int) -> range:
    resolved_end = end if end >= 0 else length
    return range(start, min(length, resolved_end), max(1, stride))


def _apply_qpos_frame(data: mujoco.MjData, qpos_frame) -> None:
    data.qpos[:] = qpos_frame
    data.qvel[:] = 0.0


def _inject_viewer_xml(xml_text: str) -> str:
    if "<visual>" not in xml_text:
        insertion_point = xml_text.find("<asset>")
        if insertion_point < 0:
            raise ValueError("Expected <asset> block in MJCF")
        xml_text = xml_text[:insertion_point] + VIEWER_VISUAL_XML + xml_text[insertion_point:]

    asset_close = xml_text.find("</asset>")
    if asset_close < 0:
        raise ValueError("Expected </asset> block in MJCF")
    xml_text = xml_text[:asset_close] + VIEWER_ASSET_XML + xml_text[asset_close:]

    worldbody_close = xml_text.find("</worldbody>")
    if worldbody_close < 0:
        raise ValueError("Expected </worldbody> block in MJCF")
    xml_text = (
        xml_text[:worldbody_close] + VIEWER_WORLDBODY_XML + xml_text[worldbody_close:]
    )
    return xml_text


def _load_model_with_floor(mjcf_path: Path) -> mujoco.MjModel:
    xml_text = mjcf_path.read_text(encoding="utf-8")
    viewer_xml = _inject_viewer_xml(xml_text)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=".any4hdmi_viewer_",
            dir=mjcf_path.parent,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(viewer_xml)
            tmp_path = tmp.name
        return mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass


def main() -> None:
    args = _parse_args()

    motion_path = Path(args.motion).expanduser().resolve()
    dataset_root = find_dataset_root(motion_path)
    manifest = load_manifest(dataset_root)
    qpos = load_motion(motion_path)

    model = _load_model_with_floor(manifest.mjcf_path)
    data = mujoco.MjData(model)

    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"Motion qpos width {qpos.shape[1]} does not match model.nq={model.nq}"
        )

    frame_indices = list(_iter_frame_indices(qpos.shape[0], args.start, args.end, args.stride))
    if not frame_indices:
        raise ValueError("No frames selected. Check --start/--end/--stride.")

    fps = float(args.fps) if args.fps is not None else 1.0 / manifest.timestep
    frame_dt = 1.0 / fps

    if args.headless:
        for frame_idx in tqdm(frame_indices, desc="Playing", unit="frame"):
            _apply_qpos_frame(data, qpos[frame_idx])
            mujoco.mj_forward(model, data)
        return

    next_time = time.time()
    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        while v.is_running():
            for frame_idx in frame_indices:
                if not v.is_running():
                    break
                _apply_qpos_frame(data, qpos[frame_idx])
                mujoco.mj_forward(model, data)
                v.sync()
                next_time += frame_dt
                sleep_for = next_time - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
            if not args.loop:
                break
            next_time = time.time()


if __name__ == "__main__":
    main()
