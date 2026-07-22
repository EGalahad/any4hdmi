from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import tyro

from any4hdmi.core.format import load_manifest, load_motion
from any4hdmi.limmt.common import (
    project_embeddings_root,
    project_pass_root,
    project_subsets_root,
    read_json,
    resolve_project_root,
    write_json,
)
from any4hdmi.limmt.embedding_viewer import make_embedding_projection, run_embedding_motion_viewer
from any4hdmi.limmt.score_browser import ScoreBin, build_score_bins, load_score_rows, plot_score_histogram


@dataclass(frozen=True)
class VisualizeArgs:
    """Visualize LIMMT score reports and HME embeddings."""

    project_path: str
    pass_dataset_name: str = "passed"
    embeddings_folder: str = "embeddings"
    subsets_folder: str = "subsets"
    visualizations_folder: str = "visualizations"
    perplexity: float = 30.0
    random_state: int = 42
    score_samples_per_bin: int = 10
    interactive: bool = False
    interactive_method: str = "tsne"
    interactive_sample_fraction: float = 1.0
    interactive_max_points: int | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    point_size: float = 0.035
    pick_radius: float = 0.08
    loop: bool = True


def _load_score_map(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    score_report = read_json(path)
    return {str(score_row["motion"]): float(score_row.get("physical_score", np.nan)) for score_row in score_report.get("details", [])}


def _load_complexity_map(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    import csv

    complexity_by_name = {}
    with path.open("r", encoding="utf-8") as f:
        for complexity_row in csv.DictReader(f):
            complexity_by_name[complexity_row["motion"]] = float(complexity_row["complexity_raw"])
    return complexity_by_name


def _plot(points: np.ndarray, color_values: np.ndarray, names: list[str], out_path: Path, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=color_values, s=8, cmap="viridis", alpha=0.75)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _write_score_visualizations(
    project_root: Path,
    output_dir: Path,
    *,
    samples_per_bin: int,
    seed: int,
) -> list[ScoreBin]:
    scores_path = project_root / "scores.json"
    if not scores_path.is_file():
        return []
    score_rows = load_score_rows(scores_path)
    score_bins = build_score_bins(score_rows, samples_per_bin=int(samples_per_bin), seed=int(seed))
    plot_score_histogram(score_rows, output_dir / "score_histogram.png")
    write_json(
        output_dir / "score_bins.json",
        {
            "scores": str(scores_path),
            "samples_per_bin": int(samples_per_bin),
            "bins": [
                {
                    "label": score_bin.label,
                    "lower": score_bin.lower,
                    "upper": score_bin.upper,
                    "motions": list(score_bin.motions),
                }
                for score_bin in score_bins
            ],
        },
    )
    return score_bins


def _load_embeddings(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    embedding_archive = np.load(path, allow_pickle=False)
    return (
        [str(name) for name in embedding_archive["names"].tolist()],
        np.asarray(embedding_archive["embeddings"], dtype=np.float32),
        np.asarray(embedding_archive["lengths"], dtype=np.float32),
    )


def _resolve_pass_dataset_root(project_root: Path, pass_dataset_name: str) -> Path:
    preferred = project_pass_root(project_root, pass_dataset_name)
    if (preferred / "manifest.json").is_file():
        return preferred

    manifest_dirs = sorted(path for path in project_root.iterdir() if path.is_dir() and (path / "manifest.json").is_file())
    pass_dirs = [path for path in manifest_dirs if "pass" in path.name]
    if len(pass_dirs) == 1:
        return pass_dirs[0]
    if len(manifest_dirs) == 1:
        return manifest_dirs[0]
    candidates = ", ".join(path.name for path in manifest_dirs) or "<none>"
    raise FileNotFoundError(
        f"Could not resolve pass dataset {pass_dataset_name!r} under {project_root}. "
        f"Use --pass-dataset-name. Candidate manifest dirs: {candidates}"
    )


def _frame_indices(length: int) -> list[int]:
    return list(range(max(0, int(length))))


def _apply_qpos_frame(model, data, scene, qpos: np.ndarray, frame_idx: int) -> None:
    data.qpos[:] = qpos[int(frame_idx)]
    data.qvel[:] = 0.0
    import mujoco

    mujoco.mj_forward(model, data)
    scene.update_from_mjdata(data)


def _run_interactive_viewer(
    *,
    args: VisualizeArgs,
    project_root: Path,
    embeddings_path: Path,
    score_bins: list[ScoreBin],
) -> None:
    import mujoco
    import viser
    from mjhub import temp_mjcf_with_floor
    from mjviser import ViserMujocoScene

    motion_names, embeddings, _ = _load_embeddings(embeddings_path)
    projection = make_embedding_projection(
        motion_names,
        embeddings,
        fraction=float(args.interactive_sample_fraction),
        max_points=args.interactive_max_points,
        method=args.interactive_method,  # type: ignore[arg-type]
        perplexity=float(args.perplexity),
        seed=int(args.random_state),
    )

    input_root = _resolve_pass_dataset_root(project_root, args.pass_dataset_name)
    manifest = load_manifest(input_root)
    with temp_mjcf_with_floor(manifest.mjcf_path) as viewer_mjcf_path:
        model = mujoco.MjModel.from_xml_path(str(viewer_mjcf_path))
    data = mujoco.MjData(model)
    server = viser.ViserServer(host=args.host, port=args.port, label="any4hdmi-limmt-visualize")
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = True
    scene.create_scene_gui()

    score_by_name = _load_score_map(project_root / "scores.json")
    state = {
        "motion": "",
        "qpos": None,
        "frame_indices": [0],
        "cursor": 0,
        "playing": True,
        "pending_seek": None,
        "ignore_slider_value": None,
        "syncing_slider": False,
        "speed": 1.0,
        "next_time": time.monotonic(),
    }
    base_frame_dt = float(manifest.timestep)

    tab_group = server.gui.add_tab_group()
    with tab_group.add_tab("Motion"):
        selected_motion = server.gui.add_text("Selected motion", initial_value="", disabled=True, order=0)
        selected_score = server.gui.add_text("Physical score", initial_value="", disabled=True, order=1)
        play_pause_button = server.gui.add_button("Start / Pause", color="green", order=2)
        status_indicator = server.gui.add_text("Status", initial_value="Playing", disabled=True, order=3)
        frame_slider = server.gui.add_slider("Frame", min=0, max=0, step=1, initial_value=0, order=4)
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0, order=5)

    score_bin_labels = [score_bin.label for score_bin in score_bins if score_bin.motions]
    motion_dropdown = None
    bin_dropdown = None
    with tab_group.add_tab("Score bins"):
        if score_bin_labels:
            bin_dropdown = server.gui.add_dropdown("Score bin", score_bin_labels, initial_value=score_bin_labels[0], order=0)
            first_bin = next(score_bin for score_bin in score_bins if score_bin.label == score_bin_labels[0])
            motion_dropdown = server.gui.add_dropdown("Motion", first_bin.motions, initial_value=first_bin.motions[0], order=1)
        else:
            server.gui.add_text("Sampled motions", initial_value="No score bins available", disabled=True, order=0)

    def effective_frame_dt() -> float:
        return base_frame_dt / max(float(state["speed"]), 1e-6)

    def update_indicator() -> None:
        status_indicator.value = "Playing" if state["playing"] else "Paused"

    def select_motion(motion_name: str) -> None:
        qpos = load_motion(input_root / motion_name)
        if qpos.shape[1] != model.nq:
            raise ValueError(f"Motion {motion_name} qpos width {qpos.shape[1]} does not match model.nq={model.nq}")
        state["motion"] = motion_name
        state["qpos"] = qpos
        state["frame_indices"] = _frame_indices(qpos.shape[0])
        state["cursor"] = 0
        state["playing"] = True
        state["pending_seek"] = None
        state["next_time"] = time.monotonic()
        selected_motion.value = motion_name
        selected_score.value = f"{score_by_name[motion_name]:.3f}" if motion_name in score_by_name else "nan"
        frame_slider.max = max(0, len(state["frame_indices"]) - 1)
        state["syncing_slider"] = True
        state["ignore_slider_value"] = 0
        frame_slider.value = 0
        state["syncing_slider"] = False
        update_indicator()
        _apply_qpos_frame(model, data, scene, qpos, 0)

    def select_cursor(cursor: int, *, sync_slider: bool) -> None:
        qpos = state["qpos"]
        if qpos is None:
            return
        frame_indices = state["frame_indices"]
        cursor = max(0, min(int(cursor), len(frame_indices) - 1))
        state["cursor"] = cursor
        if sync_slider:
            state["syncing_slider"] = True
            state["ignore_slider_value"] = cursor
            frame_slider.value = cursor
            state["syncing_slider"] = False
        _apply_qpos_frame(model, data, scene, qpos, frame_indices[cursor])

    if bin_dropdown is not None and motion_dropdown is not None:

        @bin_dropdown.on_update
        def _(_) -> None:
            selected_label = str(bin_dropdown.value)
            selected_bin = next(score_bin for score_bin in score_bins if score_bin.label == selected_label)
            motion_dropdown.options = selected_bin.motions
            if selected_bin.motions:
                motion_dropdown.value = selected_bin.motions[0]
                select_motion(selected_bin.motions[0])

        @motion_dropdown.on_update
        def _(_) -> None:
            select_motion(str(motion_dropdown.value))

    @play_pause_button.on_click
    def _(_) -> None:
        should_play = not state["playing"]
        if should_play and state["cursor"] >= len(state["frame_indices"]) - 1 and not args.loop:
            state["cursor"] = 0
        state["playing"] = should_play
        if should_play:
            state["next_time"] = time.monotonic()
        update_indicator()

    @frame_slider.on_update
    def _(_) -> None:
        slider_value = int(frame_slider.value)
        if state["syncing_slider"] or state["ignore_slider_value"] == slider_value:
            state["ignore_slider_value"] = None
            return
        state["pending_seek"] = slider_value
        state["playing"] = False
        update_indicator()

    @speed_slider.on_update
    def _(_) -> None:
        state["speed"] = float(speed_slider.value)
        if state["playing"]:
            state["next_time"] = time.monotonic()

    def on_embedding_select(motion_name: str, _: int) -> None:
        select_motion(motion_name)

    run_embedding_motion_viewer(
        projection,
        server=server,
        on_select=on_embedding_select,
        host=args.host,
        port=args.port,
        point_size=float(args.point_size),
        pick_radius=float(args.pick_radius),
        position=(-3.0, 0.0, 1.0),
        block=False,
    )

    initial_motion = projection.names[0] if projection.names else (score_bins[0].motions[0] if score_bins and score_bins[0].motions else None)
    if initial_motion is not None:
        select_motion(initial_motion)

    print(f"[any4hdmi-limmt-visualize] Viser server: http://{args.host}:{args.port}")
    print("[any4hdmi-limmt-visualize] Open the URL in a browser. Press Ctrl+C to quit.")
    while True:
        pending_seek = state["pending_seek"]
        if pending_seek is not None:
            state["pending_seek"] = None
            select_cursor(int(pending_seek), sync_slider=False)
            state["next_time"] = time.monotonic() + effective_frame_dt()

        if state["playing"] and time.monotonic() >= state["next_time"]:
            next_cursor = int(state["cursor"]) + 1
            if next_cursor >= len(state["frame_indices"]):
                if args.loop:
                    next_cursor = 0
                else:
                    next_cursor = len(state["frame_indices"]) - 1
                    state["playing"] = False
                    update_indicator()
            select_cursor(next_cursor, sync_slider=True)
            state["next_time"] = time.monotonic() + effective_frame_dt()

        time.sleep(min(0.01, max(effective_frame_dt() / 4.0, 0.001)))


def main() -> None:
    args = tyro.cli(VisualizeArgs)
    project_root = resolve_project_root(args.project_path)
    output_dir = project_root / args.visualizations_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    score_bins = _write_score_visualizations(
        project_root,
        output_dir,
        samples_per_bin=args.score_samples_per_bin,
        seed=args.random_state,
    )
    embeddings_path = project_embeddings_root(project_root, args.embeddings_folder) / "embeddings.npz"
    if args.interactive:
        _run_interactive_viewer(
            args=args,
            project_root=project_root,
            embeddings_path=embeddings_path,
            score_bins=score_bins,
        )
        return
    if not embeddings_path.is_file():
        if score_bins:
            print(f"Saved score visualizations to {output_dir}")
            return
        raise FileNotFoundError(f"Embeddings archive not found: {embeddings_path}")
    names, embeddings, lengths = _load_embeddings(embeddings_path)
    score_map = _load_score_map(project_root / "scores.json")
    complexity_map = _load_complexity_map(project_subsets_root(project_root, args.subsets_folder) / "complexity.csv")

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca = PCA(n_components=2, random_state=args.random_state).fit_transform(embeddings)
    perplexity = min(float(args.perplexity), max(1.0, (len(names) - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=args.random_state).fit_transform(embeddings)

    score_values = np.asarray([score_map.get(name, np.nan) for name in names], dtype=np.float32)
    complexity_values = np.asarray([complexity_map.get(name, np.nan) for name in names], dtype=np.float32)
    for method_name, points in (("pca", pca), ("tsne", tsne)):
        _plot(points, lengths, names, output_dir / f"hme_{method_name}_length.png", title=f"HME {method_name.upper()} by clip length")
        if np.isfinite(score_values).any():
            _plot(points, score_values, names, output_dir / f"hme_{method_name}_physical_score.png", title=f"HME {method_name.upper()} by physical score")
        if np.isfinite(complexity_values).any():
            _plot(points, complexity_values, names, output_dir / f"hme_{method_name}_complexity.png", title=f"HME {method_name.upper()} by complexity")
    np.savez_compressed(output_dir / "projection_points.npz", names=np.asarray(names), pca=pca, tsne=tsne)
    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
