from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

from any4hdmi.core.format import load_manifest, load_motion
from any4hdmi.limmt.common import project_pass_root, resolve_project_root, write_json


ScoreRow = dict[str, Any]
DEFAULT_SCORE_BIN_RANGES: tuple[tuple[float, float], ...] = tuple(
    (lower, lower + 5.0) for lower in range(95, 55, -5)
)


@dataclass(frozen=True)
class ScoreBin:
    label: str
    lower: float
    upper: float
    motions: tuple[str, ...]


@dataclass(frozen=True)
class ScoreBrowserArgs:
    """Open a score-bin motion browser for a LIMMT score report."""

    project_path: str
    pass_dataset_name: str = "passed"
    visualizations_folder: str = "score_browser_visualizations"
    samples_per_bin: int = 10
    random_state: int = 42
    host: str = "127.0.0.1"
    port: int = 8080
    loop: bool = True


def load_score_rows(path: Path) -> list[ScoreRow]:
    """Load and lightly normalize rows from a LIMMT scores.json report."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {path} to contain a JSON object")

    details = payload.get("details", [])
    if not isinstance(details, list):
        raise ValueError(f"Expected {path} details to be a list")

    rows: list[ScoreRow] = []
    for row_idx, row in enumerate(details):
        if not isinstance(row, dict):
            raise ValueError(f"Expected score row {row_idx} in {path} to be an object")
        if "motion" not in row:
            raise ValueError(f"Score row {row_idx} in {path} is missing motion")
        if "physical_score" not in row:
            raise ValueError(f"Score row {row_idx} in {path} is missing physical_score")

        normalized_row = dict(row)
        normalized_row["motion"] = str(row["motion"])
        try:
            normalized_row["physical_score"] = float(row["physical_score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Score row {row_idx} in {path} has a non-numeric physical_score") from exc
        rows.append(normalized_row)
    return rows


def score_values_for_names(score_path: Path, names: Iterable[str] | None = None) -> dict[str, float]:
    """Return physical scores by motion name, optionally aligned to the requested names."""

    score_by_name = _score_map_from_rows(load_score_rows(score_path))
    if names is None:
        return score_by_name
    return {str(name): score_by_name.get(str(name), math.nan) for name in names}


def build_score_bins(
    score_rows: Sequence[ScoreRow],
    bins: Sequence[tuple[float, float]] | None = None,
    *,
    samples_per_bin: int = 10,
    seed: int = 42,
) -> list[ScoreBin]:
    """Sample motions into descending LIMMT physical-score bins.

    Scores on a shared boundary are assigned to the higher scoring range: 95.0
    is in the 100-95 bin, 90.0 is in the 95-90 bin, and so on.
    """

    if samples_per_bin < 0:
        raise ValueError("samples_per_bin must be non-negative")

    normalized_bins = _normalize_bins(DEFAULT_SCORE_BIN_RANGES if bins is None else bins)
    motions_by_bin: list[list[str]] = [[] for _ in normalized_bins]
    for row in score_rows:
        motion, score = _motion_and_score(row)
        if not math.isfinite(score):
            continue
        for bin_idx, (lower, upper) in enumerate(normalized_bins):
            if _score_in_bin(score, lower=lower, upper=upper, is_first_bin=bin_idx == 0):
                motions_by_bin[bin_idx].append(motion)
                break

    score_bins: list[ScoreBin] = []
    for (lower, upper), motions in zip(normalized_bins, motions_by_bin):
        candidates = sorted(motions)
        sampled = _sample_stable(
            candidates,
            samples_per_bin=samples_per_bin,
            seed=seed,
            lower=lower,
            upper=upper,
        )
        score_bins.append(
            ScoreBin(
                label=_score_bin_label(lower, upper),
                lower=lower,
                upper=upper,
                motions=sampled,
            )
        )
    return score_bins


def plot_score_histogram(
    score_rows: Sequence[ScoreRow],
    out_path: Path,
    bins: int | Sequence[float] | None = None,
) -> Path:
    """Write a LIMMT physical-score histogram image and return its path."""

    score_values = [score for _, score in (_motion_and_score(row) for row in score_rows) if math.isfinite(score)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist_bins = list(range(0, 105, 5)) if bins is None else bins

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(score_values, bins=hist_bins, color="#4C78A8", edgecolor="#222222", linewidth=0.7)
    ax.set_title("LIMMT Physical Score Histogram")
    ax.set_xlabel("Physical score")
    ax.set_ylabel("Motion count")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def flatten_bin_motion_choices(score_bins: Sequence[ScoreBin]) -> list[str]:
    """Return stable dropdown labels for sampled score-bin motions."""

    return [f"{score_bin.label} | {motion}" for score_bin in score_bins for motion in score_bin.motions]


def write_score_browser_artifacts(
    *,
    project_root: Path,
    output_dir: Path,
    samples_per_bin: int = 10,
    seed: int = 42,
) -> list[ScoreBin]:
    """Write the histogram and sampled score-bin manifest for a LIMMT project."""

    scores_path = project_root / "scores.json"
    score_rows = load_score_rows(scores_path)
    score_bins = build_score_bins(score_rows, samples_per_bin=samples_per_bin, seed=seed)
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


def run_score_browser(
    *,
    project_root: Path,
    pass_dataset_name: str = "passed",
    visualizations_folder: str = "score_browser_visualizations",
    samples_per_bin: int = 10,
    seed: int = 42,
    host: str = "127.0.0.1",
    port: int = 8080,
    loop: bool = True,
) -> None:
    """Open a Viser browser for sampled motions grouped by physical-score bins."""

    import mujoco
    import viser
    from mjhub import temp_mjcf_with_floor
    from mjviser import ViserMujocoScene

    output_dir = project_root / visualizations_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    score_bins = write_score_browser_artifacts(
        project_root=project_root,
        output_dir=output_dir,
        samples_per_bin=int(samples_per_bin),
        seed=int(seed),
    )
    nonempty_bins = [score_bin for score_bin in score_bins if score_bin.motions]
    if not nonempty_bins:
        raise ValueError(f"No sampled score-bin motions found in {project_root / 'scores.json'}")

    score_by_name = _score_map_from_rows(load_score_rows(project_root / "scores.json"))
    input_root = _resolve_pass_dataset_root(project_root, pass_dataset_name)
    manifest = load_manifest(input_root)
    with temp_mjcf_with_floor(manifest.mjcf_path) as viewer_mjcf_path:
        model = mujoco.MjModel.from_xml_path(str(viewer_mjcf_path))
    data = mujoco.MjData(model)

    server = viser.ViserServer(host=host, port=port, label="any4hdmi-limmt-score-browser")
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = True
    scene.create_scene_gui()

    state: dict[str, Any] = {
        "qpos": None,
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
    with tab_group.add_tab("Score browser"):
        bin_dropdown = server.gui.add_dropdown(
            "Score bin",
            [score_bin.label for score_bin in nonempty_bins],
            initial_value=nonempty_bins[0].label,
            order=0,
        )
        motion_dropdown = server.gui.add_dropdown(
            "Motion",
            nonempty_bins[0].motions,
            initial_value=nonempty_bins[0].motions[0],
            order=1,
        )
        selected_score = server.gui.add_text("Physical score", initial_value="", disabled=True, order=2)
        selected_motion = server.gui.add_text("Selected motion", initial_value="", disabled=True, order=3)
        play_pause_button = server.gui.add_button("Start / Pause", color="green", order=4)
        status_indicator = server.gui.add_text("Status", initial_value="Playing", disabled=True, order=5)
        frame_slider = server.gui.add_slider("Frame", min=0, max=0, step=1, initial_value=0, order=6)
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0, order=7)

    def effective_frame_dt() -> float:
        return base_frame_dt / max(float(state["speed"]), 1e-6)

    def update_indicator() -> None:
        status_indicator.value = "Playing" if state["playing"] else "Paused"

    def apply_cursor(cursor: int, *, sync_slider: bool) -> None:
        qpos = state["qpos"]
        if qpos is None:
            return
        cursor = max(0, min(int(cursor), int(qpos.shape[0]) - 1))
        state["cursor"] = cursor
        data.qpos[:] = qpos[cursor]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        scene.update_from_mjdata(data)
        if sync_slider:
            state["syncing_slider"] = True
            state["ignore_slider_value"] = cursor
            frame_slider.value = cursor
            state["syncing_slider"] = False

    def select_motion(motion_name: str) -> None:
        qpos = load_motion(input_root / motion_name)
        if qpos.shape[1] != model.nq:
            raise ValueError(f"Motion {motion_name} qpos width {qpos.shape[1]} does not match model.nq={model.nq}")
        state["qpos"] = qpos
        state["cursor"] = 0
        state["playing"] = True
        state["pending_seek"] = None
        state["next_time"] = time.monotonic()
        selected_motion.value = motion_name
        selected_score.value = f"{score_by_name[motion_name]:.3f}" if motion_name in score_by_name else "nan"
        frame_slider.max = max(0, int(qpos.shape[0]) - 1)
        state["syncing_slider"] = True
        state["ignore_slider_value"] = 0
        frame_slider.value = 0
        state["syncing_slider"] = False
        update_indicator()
        apply_cursor(0, sync_slider=False)

    @bin_dropdown.on_update
    def _(_) -> None:
        selected_bin = next(score_bin for score_bin in nonempty_bins if score_bin.label == str(bin_dropdown.value))
        motion_dropdown.options = selected_bin.motions
        motion_dropdown.value = selected_bin.motions[0]
        select_motion(selected_bin.motions[0])

    @motion_dropdown.on_update
    def _(_) -> None:
        select_motion(str(motion_dropdown.value))

    @play_pause_button.on_click
    def _(_) -> None:
        should_play = not state["playing"]
        qpos = state["qpos"]
        if qpos is not None and should_play and state["cursor"] >= int(qpos.shape[0]) - 1 and not loop:
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

    select_motion(nonempty_bins[0].motions[0])
    print(f"[any4hdmi-limmt-score-browser] Viser server: http://{host}:{port}")
    print(f"[any4hdmi-limmt-score-browser] Score artifacts: {output_dir}")
    print("[any4hdmi-limmt-score-browser] Open the URL in a browser. Press Ctrl+C to quit.")

    while True:
        pending_seek = state["pending_seek"]
        if pending_seek is not None:
            state["pending_seek"] = None
            apply_cursor(int(pending_seek), sync_slider=False)
            state["next_time"] = time.monotonic() + effective_frame_dt()

        qpos = state["qpos"]
        if qpos is not None and state["playing"] and time.monotonic() >= state["next_time"]:
            next_cursor = int(state["cursor"]) + 1
            if next_cursor >= int(qpos.shape[0]):
                if loop:
                    next_cursor = 0
                else:
                    next_cursor = int(qpos.shape[0]) - 1
                    state["playing"] = False
                    update_indicator()
            apply_cursor(next_cursor, sync_slider=True)
            state["next_time"] = time.monotonic() + effective_frame_dt()

        time.sleep(min(0.01, max(effective_frame_dt() / 4.0, 0.001)))


def main() -> None:
    import tyro

    args = tyro.cli(ScoreBrowserArgs)
    run_score_browser(
        project_root=resolve_project_root(args.project_path),
        pass_dataset_name=args.pass_dataset_name,
        visualizations_folder=args.visualizations_folder,
        samples_per_bin=args.samples_per_bin,
        seed=args.random_state,
        host=args.host,
        port=args.port,
        loop=args.loop,
    )


def _score_map_from_rows(score_rows: Sequence[ScoreRow]) -> dict[str, float]:
    score_by_name: dict[str, float] = {}
    for row in score_rows:
        motion, score = _motion_and_score(row)
        if motion in score_by_name:
            raise ValueError(f"Duplicate physical score for motion {motion!r}")
        score_by_name[motion] = score
    return score_by_name


def _motion_and_score(row: ScoreRow) -> tuple[str, float]:
    try:
        motion = str(row["motion"])
        score = float(row["physical_score"])
    except KeyError as exc:
        raise ValueError(f"Score row is missing {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("Score row physical_score must be numeric") from exc
    return motion, score


def _normalize_bins(bins: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    normalized_bins = tuple(
        (min(float(left), float(right)), max(float(left), float(right)))
        for left, right in bins
    )
    if not normalized_bins:
        raise ValueError("At least one score bin is required")
    for lower, upper in normalized_bins:
        if lower >= upper:
            raise ValueError(f"Invalid score bin ({lower}, {upper})")
    return normalized_bins


def _score_in_bin(score: float, *, lower: float, upper: float, is_first_bin: bool) -> bool:
    if is_first_bin and math.isclose(score, upper):
        return True
    return lower <= score < upper


def _sample_stable(
    candidates: Sequence[str],
    *,
    samples_per_bin: int,
    seed: int,
    lower: float,
    upper: float,
) -> tuple[str, ...]:
    if len(candidates) <= samples_per_bin:
        return tuple(candidates)
    bin_seed_payload = f"{seed}:{lower:g}:{upper:g}".encode("utf-8")
    bin_seed = int.from_bytes(hashlib.sha256(bin_seed_payload).digest()[:8], byteorder="big")
    rng = random.Random(bin_seed)
    sample_ids = sorted(rng.sample(range(len(candidates)), samples_per_bin))
    return tuple(candidates[sample_id] for sample_id in sample_ids)


def _score_bin_label(lower: float, upper: float) -> str:
    return f"{_format_score_endpoint(upper)}-{_format_score_endpoint(lower)}"


def _format_score_endpoint(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


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


if __name__ == "__main__":
    main()
