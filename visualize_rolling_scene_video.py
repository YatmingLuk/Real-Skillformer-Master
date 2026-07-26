"""Animate rolling Wayformer inference over one continuous NuPlan scenario.

At every selected NuPlan iteration this script:

1. rebuilds the scene input using the current ego pose, two-second history,
   nearby agents, and local map;
2. infers ``z_pred(t)`` from the current scene;
3. encodes the current three-second expert future as ``z_gt(t)``;
4. decodes both skills into trajectories;
5. renders the moving ego, current forecasts, and all eight continuous
   ``z_gt(t)``/``z_pred(t)`` time series.

Unlike ``visualize_scene_skill_video.py``, no artificial interpolation is used:
every frame is a real, newly constructed scene input from the same scenario.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skillformer-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from tqdm import tqdm

from Wayformer.wf_dataset import NuplanDataset, data_processor
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioMapping,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from utils.common_utils import get_scenario_map
from visualize_skills import SKILL_NAMES, load_model, resolve_device


@dataclass
class RollingFrame:
    iteration: int
    timestamp_us: int
    time_s: float
    mapping: Dict
    ego_global_pose: np.ndarray
    ego_display_pose: np.ndarray
    gt_skill: np.ndarray
    pred_skill: np.ndarray
    normalized_skill_error: np.ndarray
    expert_local: np.ndarray
    oracle_local: np.ndarray
    pred_local: np.ndarray
    expert_display: np.ndarray
    oracle_display: np.ndarray
    pred_display: np.ndarray
    map_center_segments: List[np.ndarray]
    map_center_colors: List[str]
    map_boundary_segments: List[np.ndarray]
    agent_segments: List[np.ndarray]
    agent_colors: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Wayformer repeatedly over continuous iterations of one "
            "NuPlan scenario and animate skills/trajectories."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Single-mode skill-only Wayformer checkpoint.",
    )
    parser.add_argument("--mode", choices=("train", "val"), default="val")
    parser.add_argument(
        "--scene-index",
        type=int,
        default=13,
        help="Index in the cached dataset, used to obtain a scenario token.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Exact scenario token; bypasses --scene-index when provided.",
    )
    parser.add_argument("--start-iteration", type=int, default=0)
    parser.add_argument(
        "--end-iteration",
        type=int,
        default=None,
        help=(
            "Inclusive final iteration. By default, uses all iterations that "
            "still contain the full 3-second GT future."
        ),
    )
    parser.add_argument(
        "--iteration-stride",
        type=int,
        default=1,
        help="One means every 10-Hz scenario input; two means every other input.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=8,
        help="Number of rolling scene mappings inferred together.",
    )
    parser.add_argument(
        "--forecast-trail",
        type=int,
        default=5,
        help="Number of recent predicted future trajectories kept as faint trails.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS. Defaults to the actual sampled scenario frequency.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.8,
        help="Hold the last frame for this many seconds.",
    )
    parser.add_argument(
        "--view-margin",
        type=float,
        default=15.0,
        help="Fixed-frame margin around ego motion and forecasts in meters.",
    )
    parser.add_argument(
        "--min-view-span",
        type=float,
        default=60.0,
        help="Minimum span of the fixed scene view in meters.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rolling_scene_video"),
    )
    return parser.parse_args()


def cached_token(
    config,
    mode: str,
    scene_index: int,
    val_ratio: float,
) -> str:
    dataset = NuplanDataset(
        config=config,
        mode=mode,
        reuse_cache=True,
        num_workers=1,
        val_ratio=val_ratio,
    )
    if not 0 <= scene_index < len(dataset):
        raise IndexError(
            f"--scene-index must be in [0, {len(dataset) - 1}], "
            f"got {scene_index}"
        )
    return str(dataset[scene_index]["token"])


def load_raw_scenario(config, token: str):
    """Load exactly one full NuPlan scenario by its initial lidar token."""
    builder = NuPlanScenarioBuilder(
        data_root=config.DATA_PATH,
        map_root=config.MAP_PATH,
        sensor_root=None,
        db_files=None,
        map_version=config.MAP_VERSION,
        scenario_mapping=ScenarioMapping(
            scenario_map=get_scenario_map(),
            subsample_ratio_override=0.5,
        ),
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=[token],
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
    )
    scenarios = builder.get_scenarios(scenario_filter, Sequential())
    if len(scenarios) != 1:
        raise RuntimeError(
            f"Expected exactly one scenario for token {token}, got {len(scenarios)}"
        )
    return scenarios[0]


def choose_iterations(
    scenario,
    config,
    start_iteration: int,
    end_iteration: Optional[int],
    stride: int,
) -> List[int]:
    if stride <= 0:
        raise ValueError("--iteration-stride must be positive")

    count = scenario.get_number_of_iterations()
    future_steps = int(
        math.ceil(config.FUTURE_TIME_HORIZON / scenario.database_interval)
    )
    safe_end = count - future_steps - 1
    if safe_end < 0:
        raise RuntimeError(
            f"Scenario has {count} iterations, fewer than the required "
            f"{future_steps + 1} for a full future target"
        )
    if not 0 <= start_iteration <= safe_end:
        raise ValueError(
            f"--start-iteration must be in [0, {safe_end}], "
            f"got {start_iteration}"
        )

    final_iteration = safe_end if end_iteration is None else end_iteration
    if final_iteration > safe_end:
        print(
            f"Requested end iteration {final_iteration} exceeds the last "
            f"full-future iteration {safe_end}; clipping to {safe_end}."
        )
        final_iteration = safe_end
    if final_iteration < start_iteration:
        raise ValueError("--end-iteration precedes --start-iteration")

    iterations = list(range(start_iteration, final_iteration + 1, stride))
    if not iterations:
        raise RuntimeError("No rolling iterations selected")
    return iterations


def build_mappings(scenario, iterations: Sequence[int]) -> List[Dict]:
    mappings: List[Dict] = []
    for iteration in tqdm(iterations, desc="Building rolling scene inputs"):
        mapping = data_processor(
            scenario,
            iteration=iteration,
        ).build_mapping()
        mappings.append(mapping)
    return mappings


@torch.no_grad()
def infer_all_skills(
    model,
    mappings: Sequence[Dict],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")

    gt_chunks: List[np.ndarray] = []
    pred_chunks: List[np.ndarray] = []
    oracle_chunks: List[np.ndarray] = []
    trajectory_chunks: List[np.ndarray] = []

    for offset in tqdm(
        range(0, len(mappings), batch_size),
        desc="Rolling Wayformer inference",
    ):
        batch = list(mappings[offset : offset + batch_size])
        gt_skill = model.model.encode_gt_skill(batch, device)
        pred_skill = model.model.predict_skill(batch, device)
        oracle = model.model.decoder.decode(gt_skill)
        prediction = model.model.decoder.decode(pred_skill)

        gt_chunks.append(gt_skill.cpu().numpy())
        pred_chunks.append(pred_skill.cpu().numpy())
        oracle_chunks.append(oracle.cpu().numpy())
        trajectory_chunks.append(prediction.cpu().numpy())

    return (
        np.concatenate(gt_chunks, axis=0),
        np.concatenate(pred_chunks, axis=0),
        np.concatenate(oracle_chunks, axis=0),
        np.concatenate(trajectory_chunks, axis=0),
    )


def rotation_matrix(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )


def local_to_global(local_xy: np.ndarray, ego_global_pose: np.ndarray) -> np.ndarray:
    """Invert the project's local transform rot=-heading+pi/2."""
    to_global = rotation_matrix(float(ego_global_pose[2]) - math.pi / 2.0)
    return (
        np.asarray(local_xy, dtype=np.float64) @ to_global.T
        + ego_global_pose[:2]
    )


def global_to_display(
    global_xy: np.ndarray,
    display_origin: np.ndarray,
    display_rotation: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(global_xy, dtype=np.float64) - display_origin
    ) @ display_rotation.T


def local_to_display(
    local_xy: np.ndarray,
    ego_global_pose: np.ndarray,
    display_origin: np.ndarray,
    display_rotation: np.ndarray,
) -> np.ndarray:
    return global_to_display(
        local_to_global(local_xy, ego_global_pose),
        display_origin,
        display_rotation,
    )


def lane_segments(
    matrix: np.ndarray,
    ego_global_pose: np.ndarray,
    display_origin: np.ndarray,
    display_rotation: np.ndarray,
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray]]:
    centers: List[np.ndarray] = []
    colors: List[str] = []
    boundaries: List[np.ndarray] = []

    for lane in np.asarray(matrix):
        if lane.ndim != 2 or lane.shape[1] < 18:
            continue
        center_local = np.stack(
            (
                np.concatenate(([lane[0, -1]], lane[:, -3])),
                np.concatenate(([lane[0, -2]], lane[:, -4])),
            ),
            axis=-1,
        )
        left_local = np.stack((lane[:, -12], lane[:, -13]), axis=-1)
        right_local = np.stack((lane[:, -14], lane[:, -15]), axis=-1)
        centers.append(
            local_to_display(
                center_local,
                ego_global_pose,
                display_origin,
                display_rotation,
            )
        )
        boundaries.extend(
            (
                local_to_display(
                    left_local,
                    ego_global_pose,
                    display_origin,
                    display_rotation,
                ),
                local_to_display(
                    right_local,
                    ego_global_pose,
                    display_origin,
                    display_rotation,
                ),
            )
        )

        green = lane[0, -8] > 0
        yellow = lane[0, -9] > 0
        red = lane[0, -10] > 0
        if red:
            colors.append("#E63946")
        elif yellow:
            colors.append("#F4A261")
        elif green:
            colors.append("#2A9D8F")
        else:
            colors.append("#8D99AE")
    return centers, colors, boundaries


def agent_segments(
    agents: np.ndarray,
    ego_global_pose: np.ndarray,
    display_origin: np.ndarray,
    display_rotation: np.ndarray,
) -> Tuple[List[np.ndarray], List[str]]:
    segments: List[np.ndarray] = []
    colors: List[str] = []
    type_colors = ("#FF9F1C", "#277DA1", "#7B2CBF")

    for index, agent in enumerate(np.asarray(agents)):
        xy = agent[:, :2]
        if not np.isfinite(xy).all() or np.allclose(xy, 0.0):
            continue
        segments.append(
            local_to_display(
                xy,
                ego_global_pose,
                display_origin,
                display_rotation,
            )
        )
        if index == 0:
            colors.append("#D00000")
        else:
            type_index = (
                int(np.argmax(agent[0, 5:8]))
                if agent.shape[1] >= 8
                else 1
            )
            colors.append(type_colors[type_index])
    return segments, colors


def assemble_frames(
    mappings: Sequence[Dict],
    gt_skills: np.ndarray,
    pred_skills: np.ndarray,
    oracle_local: np.ndarray,
    pred_local: np.ndarray,
    z_std: np.ndarray,
) -> List[RollingFrame]:
    first_pose = np.asarray(mappings[0]["ego_global_pose"], dtype=np.float64)
    display_origin = first_pose[:2].copy()
    display_rotation = rotation_matrix(-float(first_pose[2]) + math.pi / 2.0)
    first_timestamp = int(mappings[0]["timestamp_us"])
    frames: List[RollingFrame] = []

    for index, mapping in enumerate(mappings):
        pose = np.asarray(mapping["ego_global_pose"], dtype=np.float64)
        display_xy = global_to_display(
            pose[:2][None],
            display_origin,
            display_rotation,
        )[0]
        display_heading = (
            float(pose[2])
            - float(first_pose[2])
            + math.pi / 2.0
        )
        display_pose = np.asarray(
            [display_xy[0], display_xy[1], display_heading],
            dtype=np.float64,
        )
        expert_local = np.asarray(mapping["labels"], dtype=np.float32)[:, :2]
        centers, center_colors, boundaries = lane_segments(
            mapping["matrix"],
            pose,
            display_origin,
            display_rotation,
        )
        agents, colors = agent_segments(
            mapping["agents"],
            pose,
            display_origin,
            display_rotation,
        )

        frames.append(
            RollingFrame(
                iteration=int(mapping["iteration"]),
                timestamp_us=int(mapping["timestamp_us"]),
                time_s=(
                    int(mapping["timestamp_us"]) - first_timestamp
                )
                / 1e6,
                mapping=mapping,
                ego_global_pose=pose,
                ego_display_pose=display_pose,
                gt_skill=gt_skills[index],
                pred_skill=pred_skills[index],
                normalized_skill_error=(
                    pred_skills[index] - gt_skills[index]
                )
                / z_std,
                expert_local=expert_local,
                oracle_local=oracle_local[index],
                pred_local=pred_local[index],
                expert_display=local_to_display(
                    expert_local,
                    pose,
                    display_origin,
                    display_rotation,
                ),
                oracle_display=local_to_display(
                    oracle_local[index],
                    pose,
                    display_origin,
                    display_rotation,
                ),
                pred_display=local_to_display(
                    pred_local[index],
                    pose,
                    display_origin,
                    display_rotation,
                ),
                map_center_segments=centers,
                map_center_colors=center_colors,
                map_boundary_segments=boundaries,
                agent_segments=agents,
                agent_colors=colors,
            )
        )
    return frames


def trajectory_metrics(
    trajectory: np.ndarray,
    expert: np.ndarray,
) -> Tuple[float, float]:
    distances = np.linalg.norm(trajectory - expert, axis=-1)
    return float(distances.mean()), float(distances[-1])


def fixed_bounds(
    frames: Sequence[RollingFrame],
    margin: float,
    min_span: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    points = []
    for frame in frames:
        points.extend(
            (
                frame.ego_display_pose[None, :2],
                frame.expert_display,
                frame.oracle_display,
                frame.pred_display,
            )
        )
    merged = np.concatenate(points, axis=0)
    lower = merged.min(axis=0) - margin
    upper = merged.max(axis=0) + margin
    for axis in range(2):
        span = upper[axis] - lower[axis]
        if span < min_span:
            center = 0.5 * (lower[axis] + upper[axis])
            lower[axis] = center - min_span / 2.0
            upper[axis] = center + min_span / 2.0
    return (float(lower[0]), float(upper[0])), (
        float(lower[1]),
        float(upper[1]),
    )


def skill_limits(
    frames: Sequence[RollingFrame],
    dimension: int,
) -> Tuple[float, float]:
    values = np.concatenate(
        (
            np.asarray([frame.gt_skill[dimension] for frame in frames]),
            np.asarray([frame.pred_skill[dimension] for frame in frames]),
            np.asarray([0.0]),
        )
    )
    value_range = max(float(values.max() - values.min()), 0.5)
    return (
        float(values.min() - 0.12 * value_range),
        float(values.max() + 0.15 * value_range),
    )


def create_rolling_animation(
    frames: Sequence[RollingFrame],
    output_path: Path,
    fps: float,
    hold_seconds: float,
    forecast_trail: int,
    view_margin: float,
    min_view_span: float,
) -> None:
    times = np.asarray([frame.time_s for frame in frames])
    gt_skills = np.stack([frame.gt_skill for frame in frames])
    pred_skills = np.stack([frame.pred_skill for frame in frames])
    x_limits, y_limits = fixed_bounds(frames, view_margin, min_view_span)

    figure = plt.figure(figsize=(18, 10))
    grid = figure.add_gridspec(
        4,
        4,
        width_ratios=(1.35, 1.35, 1.0, 1.0),
        wspace=0.34,
        hspace=0.48,
    )
    scene_ax = figure.add_subplot(grid[:, :2])
    skill_axes = [
        figure.add_subplot(grid[dimension // 2, 2 + dimension % 2])
        for dimension in range(8)
    ]
    figure.suptitle(
        "Continuous rolling scene input: ego motion, skills, and forecasts",
        fontsize=16,
    )

    scene_ax.set_xlim(x_limits)
    scene_ax.set_ylim(y_limits)
    scene_ax.set_aspect("equal", adjustable="box")
    scene_ax.set_xlabel("fixed scenario-frame x [m]")
    scene_ax.set_ylabel("fixed scenario-frame y [m]")
    scene_ax.grid(alpha=0.18)

    map_collection = LineCollection([], linewidths=1.15, alpha=0.58)
    boundary_collection = LineCollection(
        [],
        colors="#ADB5BD",
        linewidths=0.55,
        alpha=0.32,
    )
    agent_collection = LineCollection([], linewidths=1.4, alpha=0.68)
    forecast_trails = LineCollection([], linewidths=1.2)
    scene_ax.add_collection(map_collection)
    scene_ax.add_collection(boundary_collection)
    scene_ax.add_collection(agent_collection)
    scene_ax.add_collection(forecast_trails)

    ego_path_line, = scene_ax.plot(
        [],
        [],
        color="#D00000",
        linewidth=2.8,
        label="rolling ego path",
    )
    ego_marker, = scene_ax.plot(
        [],
        [],
        marker="*",
        markersize=13,
        color="#D00000",
        markeredgecolor="white",
        markeredgewidth=0.7,
    )
    ego_heading_line, = scene_ax.plot(
        [],
        [],
        color="#D00000",
        linewidth=2.2,
    )
    expert_line, = scene_ax.plot(
        [],
        [],
        color="#111111",
        linewidth=3.0,
        label="current expert future",
    )
    oracle_line, = scene_ax.plot(
        [],
        [],
        color="#2A9D8F",
        linewidth=2.4,
        linestyle="--",
        label="current VAE oracle",
    )
    pred_line, = scene_ax.plot(
        [],
        [],
        color="#F72585",
        linewidth=2.9,
        label="current predicted future",
    )
    expert_endpoint, = scene_ax.plot(
        [],
        [],
        marker="o",
        markersize=6,
        color="#111111",
    )
    pred_endpoint, = scene_ax.plot(
        [],
        [],
        marker="o",
        markersize=6,
        color="#F72585",
    )
    info_text = scene_ax.text(
        0.02,
        0.98,
        "",
        transform=scene_ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#CED4DA"},
    )
    scene_ax.legend(
        handles=[
            Line2D([], [], color="#D00000", linewidth=2.8, label="ego motion"),
            Line2D([], [], color="#111111", linewidth=3.0, label="expert future"),
            Line2D(
                [],
                [],
                color="#2A9D8F",
                linewidth=2.4,
                linestyle="--",
                label="VAE oracle",
            ),
            Line2D(
                [],
                [],
                color="#F72585",
                linewidth=2.9,
                label="predicted future",
            ),
            Line2D(
                [],
                [],
                color="#F72585",
                linewidth=1.2,
                alpha=0.25,
                label="recent forecast trail",
            ),
        ],
        loc="best",
        fontsize=9,
    )

    gt_lines = []
    pred_lines = []
    gt_markers = []
    pred_markers = []
    cursor_lines = []
    value_texts = []
    time_max = max(float(times[-1]), 1e-3)

    for dimension, ax in enumerate(skill_axes):
        # Faint full curves provide context; bold curves reveal rolling history.
        ax.plot(
            times,
            gt_skills[:, dimension],
            color="#212529",
            linewidth=0.9,
            alpha=0.13,
        )
        ax.plot(
            times,
            pred_skills[:, dimension],
            color="#F72585",
            linewidth=0.9,
            alpha=0.13,
        )
        gt_line, = ax.plot(
            [],
            [],
            color="#212529",
            linewidth=1.8,
            label="GT $\\mu(t)$",
        )
        pred_skill_line, = ax.plot(
            [],
            [],
            color="#F72585",
            linewidth=1.8,
            label="pred $z(t)$",
        )
        gt_marker, = ax.plot(
            [],
            [],
            marker="o",
            markersize=4,
            color="#212529",
        )
        pred_marker, = ax.plot(
            [],
            [],
            marker="o",
            markersize=4,
            color="#F72585",
        )
        cursor = ax.axvline(
            times[0],
            color="#4361EE",
            linewidth=0.8,
            alpha=0.65,
        )
        value_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )

        ax.set_xlim(float(times[0]), time_max)
        ax.set_ylim(skill_limits(frames, dimension))
        ax.set_title(SKILL_NAMES[dimension], fontsize=10)
        ax.grid(alpha=0.2)
        if dimension >= 6:
            ax.set_xlabel("scenario time [s]", fontsize=8)
        if dimension % 2 == 0:
            ax.set_ylabel("raw z", fontsize=8)
        ax.tick_params(labelsize=8)
        if dimension == 0:
            ax.legend(fontsize=7.5, loc="lower right")

        gt_lines.append(gt_line)
        pred_lines.append(pred_skill_line)
        gt_markers.append(gt_marker)
        pred_markers.append(pred_marker)
        cursor_lines.append(cursor)
        value_texts.append(value_text)

    ego_positions = np.stack([frame.ego_display_pose[:2] for frame in frames])
    hold_frames = max(0, int(round(hold_seconds * fps)))
    animation_indices = list(range(len(frames))) + [len(frames) - 1] * hold_frames

    def update(animation_index: int):
        frame_index = animation_indices[animation_index]
        frame = frames[frame_index]

        map_collection.set_segments(frame.map_center_segments)
        map_collection.set_color(frame.map_center_colors)
        boundary_collection.set_segments(frame.map_boundary_segments)
        agent_collection.set_segments(frame.agent_segments)
        agent_collection.set_color(frame.agent_colors)

        trail_start = max(0, frame_index - max(0, forecast_trail))
        trail_segments = [
            frames[index].pred_display
            for index in range(trail_start, frame_index)
        ]
        forecast_trails.set_segments(trail_segments)
        if trail_segments:
            alphas = np.linspace(0.05, 0.32, len(trail_segments))
            forecast_trails.set_color(
                [(0.969, 0.145, 0.522, float(alpha)) for alpha in alphas]
            )

        ego_path_line.set_data(
            ego_positions[: frame_index + 1, 0],
            ego_positions[: frame_index + 1, 1],
        )
        ego_x, ego_y, ego_heading = frame.ego_display_pose
        ego_marker.set_data([ego_x], [ego_y])
        heading_length = 3.0
        ego_heading_line.set_data(
            [ego_x, ego_x + heading_length * math.cos(ego_heading)],
            [ego_y, ego_y + heading_length * math.sin(ego_heading)],
        )

        expert_line.set_data(
            frame.expert_display[:, 0],
            frame.expert_display[:, 1],
        )
        oracle_line.set_data(
            frame.oracle_display[:, 0],
            frame.oracle_display[:, 1],
        )
        pred_line.set_data(
            frame.pred_display[:, 0],
            frame.pred_display[:, 1],
        )
        expert_endpoint.set_data(
            [frame.expert_display[-1, 0]],
            [frame.expert_display[-1, 1]],
        )
        pred_endpoint.set_data(
            [frame.pred_display[-1, 0]],
            [frame.pred_display[-1, 1]],
        )

        oracle_ade, oracle_fde = trajectory_metrics(
            frame.oracle_local,
            frame.expert_local,
        )
        pred_ade, pred_fde = trajectory_metrics(
            frame.pred_local,
            frame.expert_local,
        )
        skill_rmse = float(
            np.sqrt(np.mean(frame.normalized_skill_error**2))
        )
        info_text.set_text(
            f"time: {frame.time_s:.1f} s | iteration: {frame.iteration}\n"
            f"normalized skill RMSE: {skill_rmse:.3f}\n"
            f"prediction ADE/FDE: {pred_ade:.3f} / {pred_fde:.3f} m\n"
            f"VAE oracle ADE/FDE: {oracle_ade:.3f} / {oracle_fde:.3f} m"
        )
        scene_ax.set_title(
            f"Rolling input at t={frame.time_s:.1f}s: newly inferred 3-second future"
        )

        for dimension in range(8):
            prefix = slice(0, frame_index + 1)
            gt_lines[dimension].set_data(
                times[prefix],
                gt_skills[prefix, dimension],
            )
            pred_lines[dimension].set_data(
                times[prefix],
                pred_skills[prefix, dimension],
            )
            gt_markers[dimension].set_data(
                [frame.time_s],
                [frame.gt_skill[dimension]],
            )
            pred_markers[dimension].set_data(
                [frame.time_s],
                [frame.pred_skill[dimension]],
            )
            cursor_lines[dimension].set_xdata(
                [frame.time_s, frame.time_s]
            )
            value_texts[dimension].set_text(
                f"GT {frame.gt_skill[dimension]:+.2f} | "
                f"pred {frame.pred_skill[dimension]:+.2f}"
            )

        return (
            map_collection,
            boundary_collection,
            agent_collection,
            forecast_trails,
            ego_path_line,
            ego_marker,
            ego_heading_line,
            expert_line,
            oracle_line,
            pred_line,
            expert_endpoint,
            pred_endpoint,
            info_text,
            *gt_lines,
            *pred_lines,
            *gt_markers,
            *pred_markers,
            *cursor_lines,
            *value_texts,
        )

    clip = animation.FuncAnimation(
        figure,
        update,
        frames=len(animation_indices),
        interval=1000.0 / fps,
        blit=False,
    )
    save_video(clip, output_path, fps)
    plt.close(figure)


def save_video(
    clip: animation.FuncAnimation,
    output_path: Path,
    fps: float,
) -> None:
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required to create the rolling MP4")
    print(f"Rendering rolling video: {output_path}")
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=4500,
        metadata={"artist": "SkillFormer"},
        extra_args=["-pix_fmt", "yuv420p"],
    )
    clip.save(str(output_path), writer=writer, dpi=120)


def write_timeseries(
    frames: Sequence[RollingFrame],
    output_path: Path,
) -> None:
    fieldnames = ["iteration", "time_s", "timestamp_us"]
    for dimension in range(8):
        fieldnames.extend(
            (
                f"gt_z{dimension}",
                f"pred_z{dimension}",
                f"normalized_error_z{dimension}",
            )
        )
    fieldnames.extend(
        (
            "skill_rmse",
            "oracle_ADE",
            "oracle_FDE",
            "prediction_ADE",
            "prediction_FDE",
        )
    )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            oracle_ade, oracle_fde = trajectory_metrics(
                frame.oracle_local,
                frame.expert_local,
            )
            pred_ade, pred_fde = trajectory_metrics(
                frame.pred_local,
                frame.expert_local,
            )
            row = {
                "iteration": frame.iteration,
                "time_s": frame.time_s,
                "timestamp_us": frame.timestamp_us,
                "skill_rmse": float(
                    np.sqrt(np.mean(frame.normalized_skill_error**2))
                ),
                "oracle_ADE": oracle_ade,
                "oracle_FDE": oracle_fde,
                "prediction_ADE": pred_ade,
                "prediction_FDE": pred_fde,
            }
            for dimension in range(8):
                row[f"gt_z{dimension}"] = frame.gt_skill[dimension]
                row[f"pred_z{dimension}"] = frame.pred_skill[dimension]
                row[
                    f"normalized_error_z{dimension}"
                ] = frame.normalized_skill_error[dimension]
            writer.writerow(row)


def save_skill_timeseries_plot(
    times: np.ndarray,
    gt_skills: np.ndarray,
    pred_skills: np.ndarray,
    normalized_errors: np.ndarray,
    output_path: Path,
) -> None:
    """Save the complete continuous GT/pred skill comparison as a static plot."""
    figure, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    for dimension, ax in enumerate(axes.flat):
        ax.plot(
            times,
            gt_skills[:, dimension],
            color="#212529",
            linewidth=2.0,
            label="GT $\\mu(t)$",
        )
        ax.plot(
            times,
            pred_skills[:, dimension],
            color="#F72585",
            linewidth=2.0,
            label="pred $z(t)$",
        )
        ax.fill_between(
            times,
            gt_skills[:, dimension],
            pred_skills[:, dimension],
            color="#F72585",
            alpha=0.12,
        )
        normalized_rmse = float(
            np.sqrt(np.mean(normalized_errors[:, dimension] ** 2))
        )
        if (
            np.std(gt_skills[:, dimension]) > 1e-12
            and np.std(pred_skills[:, dimension]) > 1e-12
        ):
            correlation = float(
                np.corrcoef(
                    gt_skills[:, dimension],
                    pred_skills[:, dimension],
                )[0, 1]
            )
        else:
            correlation = float("nan")
        ax.text(
            0.02,
            0.96,
            f"normalized RMSE={normalized_rmse:.3f} | r={correlation:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
        ax.set_title(SKILL_NAMES[dimension])
        ax.set_ylabel("raw latent")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("scenario time [s]")
    axes[-1, 1].set_xlabel("scenario time [s]")
    figure.suptitle(
        "Continuous rolling inputs: GT and inferred 8-D skills",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.forecast_trail < 0:
        raise ValueError("--forecast-trail cannot be negative")
    if args.hold_seconds < 0:
        raise ValueError("--hold-seconds cannot be negative")

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading checkpoint: {args.ckpt}")
    print(f"Using device: {device}")
    model = load_model(args.ckpt, device)

    token = args.token or cached_token(
        model.config,
        args.mode,
        args.scene_index,
        args.val_ratio,
    )
    print(f"Loading full scenario token: {token}")
    scenario = load_raw_scenario(model.config, token)
    iterations = choose_iterations(
        scenario,
        model.config,
        args.start_iteration,
        args.end_iteration,
        args.iteration_stride,
    )
    print(
        f"Scenario type={scenario.scenario_type}, "
        f"iterations={scenario.get_number_of_iterations()}, "
        f"database_interval={scenario.database_interval:.3f}s"
    )
    print(
        f"Rolling over {len(iterations)} inputs: "
        f"{iterations[0]}..{iterations[-1]} "
        f"(stride={args.iteration_stride})"
    )

    mappings = build_mappings(scenario, iterations)
    gt_skills, pred_skills, oracle_local, pred_local = infer_all_skills(
        model,
        mappings,
        args.inference_batch_size,
        device,
    )
    z_std = np.clip(
        model.model.z_std.detach().cpu().numpy(),
        1e-4,
        None,
    )
    frames = assemble_frames(
        mappings,
        gt_skills,
        pred_skills,
        oracle_local,
        pred_local,
        z_std,
    )

    sampled_interval = (
        scenario.database_interval * args.iteration_stride
    )
    fps = args.fps or (1.0 / sampled_interval)
    if fps <= 0:
        raise ValueError("--fps must be positive")

    token_short = token[:12]
    prefix = (
        f"rolling_{token_short}_"
        f"iter{iterations[0]:03d}-{iterations[-1]:03d}"
    )
    video_path = args.output_dir / f"{prefix}.mp4"
    times = np.asarray([frame.time_s for frame in frames])
    gt_skill_values = np.stack([frame.gt_skill for frame in frames])
    pred_skill_values = np.stack([frame.pred_skill for frame in frames])
    normalized_errors = np.stack(
        [frame.normalized_skill_error for frame in frames]
    )
    save_skill_timeseries_plot(
        times,
        gt_skill_values,
        pred_skill_values,
        normalized_errors,
        args.output_dir / f"{prefix}_skill_timeseries.png",
    )
    create_rolling_animation(
        frames,
        video_path,
        fps=fps,
        hold_seconds=args.hold_seconds,
        forecast_trail=args.forecast_trail,
        view_margin=args.view_margin,
        min_view_span=args.min_view_span,
    )
    write_timeseries(
        frames,
        args.output_dir / f"{prefix}_skills.csv",
    )
    np.savez_compressed(
        args.output_dir / f"{prefix}_values.npz",
        token=token,
        iterations=np.asarray([frame.iteration for frame in frames]),
        timestamps_us=np.asarray([frame.timestamp_us for frame in frames]),
        times_s=np.asarray([frame.time_s for frame in frames]),
        ego_global_pose=np.stack(
            [frame.ego_global_pose for frame in frames]
        ),
        ego_display_pose=np.stack(
            [frame.ego_display_pose for frame in frames]
        ),
        gt_skill=gt_skill_values,
        pred_skill=pred_skill_values,
        normalized_skill_error=normalized_errors,
        expert_local=np.stack([frame.expert_local for frame in frames]),
        oracle_local=np.stack([frame.oracle_local for frame in frames]),
        pred_local=np.stack([frame.pred_local for frame in frames]),
        expert_display=np.stack(
            [frame.expert_display for frame in frames]
        ),
        oracle_display=np.stack(
            [frame.oracle_display for frame in frames]
        ),
        pred_display=np.stack(
            [frame.pred_display for frame in frames]
        ),
    )

    skill_rmse = np.sqrt(
        np.mean(normalized_errors**2, axis=1)
    )
    pred_ades = np.asarray(
        [
            trajectory_metrics(frame.pred_local, frame.expert_local)[0]
            for frame in frames
        ]
    )
    print("\nRolling scenario summary")
    print(f"  token: {token}")
    print(f"  input frames: {len(frames)}")
    print(f"  covered time: {frames[-1].time_s:.1f}s")
    print(f"  mean normalized skill RMSE: {skill_rmse.mean():.4f}")
    print(f"  mean prediction ADE: {pred_ades.mean():.4f}m")
    print(f"  video: {video_path.resolve()}")


if __name__ == "__main__":
    main()
