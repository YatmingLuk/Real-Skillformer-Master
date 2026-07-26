"""Create same-scene skill comparisons and trajectory-change videos.

Two complementary animations are supported:

``latent``
    Keep the scene fixed and interpolate from the GT VAE latent ``mu_gt`` to
    the Wayformer prediction ``z_pred``. The decoded trajectory morphs as the
    eight latent dimensions change.

``rollout``
    Keep both latents fixed and progressively draw the expert, VAE-oracle, and
    predicted trajectories over the 30 future time steps.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skillformer-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from Wayformer.wf_dataset import NuplanDataset
from visualize_skills import SKILL_NAMES, load_model, resolve_device


@dataclass
class SceneInference:
    scene_index: int
    token: str
    sample: Dict
    gt_skill: np.ndarray
    pred_skill: np.ndarray
    normalized_error: np.ndarray
    expert_trajectory: np.ndarray
    oracle_trajectory: np.ndarray
    pred_trajectory: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize GT/predicted skills and their decoded trajectories "
            "for one fixed NuPlan scene."
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
        default=0,
        help="Dataset index. Ignored when --token is provided.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional exact scenario token. Finding it may scan the cache.",
    )
    parser.add_argument(
        "--animation",
        choices=("latent", "rollout", "both"),
        default="both",
        help="Video type to generate.",
    )
    parser.add_argument(
        "--format",
        choices=("mp4", "gif"),
        default="mp4",
        dest="video_format",
    )
    parser.add_argument(
        "--morph-frames",
        type=int,
        default=60,
        help="Number of transition frames for GT-to-pred latent morphing.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.7,
        help="Seconds to hold the first and last animation states.",
    )
    parser.add_argument(
        "--view-margin",
        type=float,
        default=12.0,
        help="Plot margin around expert/oracle/predicted trajectories.",
    )
    parser.add_argument(
        "--min-view-span",
        type=float,
        default=40.0,
        help="Minimum x/y span of the scene plot in meters.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/scene_skill_video"),
    )
    return parser.parse_args()


def select_scene(
    dataset: NuplanDataset,
    scene_index: int,
    token: Optional[str],
) -> Tuple[int, Dict]:
    if token is None:
        if not 0 <= scene_index < len(dataset):
            raise IndexError(
                f"--scene-index must be in [0, {len(dataset) - 1}], "
                f"got {scene_index}"
            )
        return scene_index, dataset[scene_index]

    print(f"Searching {len(dataset)} cached scenes for token {token} ...")
    for index in range(len(dataset)):
        sample = dataset[index]
        if str(sample.get("token", "")) == token:
            return index, sample
    raise ValueError(f"Scenario token was not found in the {dataset} cache: {token}")


@torch.no_grad()
def infer_scene(
    model,
    scene_index: int,
    sample: Dict,
    device: torch.device,
) -> SceneInference:
    batch = [sample]
    gt_skill_tensor = model.model.encode_gt_skill(batch, device)
    pred_skill_tensor = model.model.predict_skill(batch, device)

    oracle_tensor = model.model.decoder.decode(gt_skill_tensor)
    pred_trajectory_tensor = model.model.decoder.decode(pred_skill_tensor)

    gt_skill = gt_skill_tensor[0].cpu().numpy()
    pred_skill = pred_skill_tensor[0].cpu().numpy()
    z_std = np.clip(model.model.z_std.detach().cpu().numpy(), 1e-4, None)

    result = SceneInference(
        scene_index=scene_index,
        token=str(sample.get("token", "")),
        sample=sample,
        gt_skill=gt_skill,
        pred_skill=pred_skill,
        normalized_error=(pred_skill - gt_skill) / z_std,
        expert_trajectory=np.asarray(sample["labels"], dtype=np.float32)[:, :2],
        oracle_trajectory=oracle_tensor[0].cpu().numpy(),
        pred_trajectory=pred_trajectory_tensor[0].cpu().numpy(),
    )
    arrays = (
        result.gt_skill,
        result.pred_skill,
        result.expert_trajectory,
        result.oracle_trajectory,
        result.pred_trajectory,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("NaN or Inf found in the selected scene inference")
    return result


def trajectory_metrics(
    trajectory: np.ndarray,
    expert: np.ndarray,
) -> Tuple[float, float]:
    distances = np.linalg.norm(trajectory[:, :2] - expert[:, :2], axis=-1)
    return float(np.mean(distances)), float(distances[-1])


def smoothstep(values: np.ndarray) -> np.ndarray:
    return values * values * (3.0 - 2.0 * values)


def center_bounds(
    trajectories: Sequence[np.ndarray],
    margin: float,
    min_span: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    all_points = np.concatenate(
        [np.asarray([[0.0, 0.0]])] + [traj[:, :2] for traj in trajectories],
        axis=0,
    )
    lower = all_points.min(axis=0) - margin
    upper = all_points.max(axis=0) + margin

    for axis in range(2):
        span = upper[axis] - lower[axis]
        if span < min_span:
            center = 0.5 * (lower[axis] + upper[axis])
            lower[axis] = center - 0.5 * min_span
            upper[axis] = center + 0.5 * min_span
    return (float(lower[0]), float(upper[0])), (
        float(lower[1]),
        float(upper[1]),
    )


def plot_map(ax: plt.Axes, matrix: np.ndarray) -> None:
    """Draw lane centerlines, boundaries, and traffic-light state."""
    for lane in np.asarray(matrix):
        if lane.ndim != 2 or lane.shape[1] < 18:
            continue

        center_x = np.concatenate(([lane[0, -1]], lane[:, -3]))
        center_y = np.concatenate(([lane[0, -2]], lane[:, -4]))
        green = lane[0, -8] > 0
        yellow = lane[0, -9] > 0
        red = lane[0, -10] > 0
        if red:
            color = "#E63946"
        elif yellow:
            color = "#F4A261"
        elif green:
            color = "#2A9D8F"
        else:
            color = "#8D99AE"

        ax.plot(center_x, center_y, color=color, linewidth=1.2, alpha=0.6)
        ax.plot(
            lane[:, -12],
            lane[:, -13],
            color="#ADB5BD",
            linewidth=0.65,
            alpha=0.4,
        )
        ax.plot(
            lane[:, -14],
            lane[:, -15],
            color="#ADB5BD",
            linewidth=0.65,
            alpha=0.4,
        )


def plot_agents(ax: plt.Axes, agents: np.ndarray) -> None:
    agents = np.asarray(agents)
    if len(agents) == 0:
        return

    ego = agents[0]
    ax.plot(
        ego[:, 0],
        ego[:, 1],
        color="#D00000",
        linewidth=2.2,
        label="ego history",
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=130,
        color="#D00000",
        edgecolor="white",
        linewidth=0.7,
        zorder=8,
        label="current ego",
    )

    type_colors = ("#FF9F1C", "#277DA1", "#7B2CBF")
    for agent in agents[1:]:
        xy = agent[:, :2]
        if not np.isfinite(xy).all() or np.allclose(xy, 0.0):
            continue
        type_index = (
            int(np.argmax(agent[0, 5:8]))
            if agent.shape[1] >= 8
            else 1
        )
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=type_colors[type_index],
            linewidth=1.0,
            alpha=0.55,
        )
        ax.scatter(
            [xy[-1, 0]],
            [xy[-1, 1]],
            s=18,
            color=type_colors[type_index],
            alpha=0.75,
        )


def setup_scene_axis(
    ax: plt.Axes,
    result: SceneInference,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
) -> None:
    plot_map(ax, result.sample["matrix"])
    plot_agents(ax, result.sample["agents"])
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ego-local x [m]")
    ax.set_ylabel("ego-local y [m]")
    ax.grid(alpha=0.18)


def plot_skill_bars(
    ax: plt.Axes,
    gt_skill: np.ndarray,
    pred_skill: np.ndarray,
    include_current: bool = False,
):
    x = np.arange(len(SKILL_NAMES))
    if include_current:
        width = 0.24
        ax.bar(
            x - width,
            gt_skill,
            width,
            color="#212529",
            alpha=0.82,
            label="GT $\\mu$",
        )
        ax.bar(
            x,
            pred_skill,
            width,
            color="#F72585",
            alpha=0.72,
            label="pred $z$",
        )
        current_bars = ax.bar(
            x + width,
            gt_skill,
            width,
            color="#4361EE",
            alpha=0.85,
            label="current $z(\\alpha)$",
        )
    else:
        width = 0.36
        ax.bar(
            x - width / 2,
            gt_skill,
            width,
            color="#212529",
            alpha=0.82,
            label="GT $\\mu$",
        )
        ax.bar(
            x + width / 2,
            pred_skill,
            width,
            color="#F72585",
            alpha=0.78,
            label="pred $z$",
        )
        current_bars = None

    values = np.concatenate((gt_skill, pred_skill, np.asarray([0.0])))
    value_range = max(float(values.max() - values.min()), 1.0)
    ax.set_ylim(
        float(values.min() - 0.15 * value_range),
        float(values.max() + 0.20 * value_range),
    )
    ax.axhline(0.0, color="#6C757D", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in x])
    ax.set_ylabel("Raw latent value")
    ax.set_title("8-D GT and inferred skills")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3 if include_current else 2, fontsize=9)
    return current_bars


def plot_error_panel(
    ax: plt.Axes,
    result: SceneInference,
) -> None:
    x = np.arange(len(SKILL_NAMES))
    colors = np.where(
        result.normalized_error >= 0,
        "#E63946",
        "#457B9D",
    )
    ax.bar(x, result.normalized_error, color=colors, alpha=0.82)
    ax.axhline(0.0, color="#212529", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in x])
    ax.set_ylabel("Normalized pred - GT")
    ax.set_title("Per-skill normalized error")
    ax.grid(axis="y", alpha=0.2)


def add_metric_text(
    ax: plt.Axes,
    result: SceneInference,
    extra_text: str = "",
):
    oracle_ade, oracle_fde = trajectory_metrics(
        result.oracle_trajectory,
        result.expert_trajectory,
    )
    pred_ade, pred_fde = trajectory_metrics(
        result.pred_trajectory,
        result.expert_trajectory,
    )
    latent_rmse = float(np.sqrt(np.mean(result.normalized_error**2)))
    text = (
        f"normalized latent RMSE: {latent_rmse:.4f}\n"
        f"VAE oracle   ADE/FDE: {oracle_ade:.3f} / {oracle_fde:.3f} m\n"
        f"prediction   ADE/FDE: {pred_ade:.3f} / {pred_fde:.3f} m"
    )
    if extra_text:
        text += f"\n{extra_text}"
    return ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#CED4DA"},
    )


def create_layout(title: str):
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.65, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.22,
        hspace=0.28,
    )
    scene_ax = fig.add_subplot(grid[:, 0])
    skill_ax = fig.add_subplot(grid[0, 1])
    error_ax = fig.add_subplot(grid[1, 1])
    fig.suptitle(title, fontsize=15)
    return fig, scene_ax, skill_ax, error_ax


def save_static_comparison(
    result: SceneInference,
    output_path: Path,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
) -> None:
    title = (
        f"Scene {result.scene_index} | token={result.token} | "
        "skill and trajectory comparison"
    )
    fig, scene_ax, skill_ax, error_ax = create_layout(title)
    setup_scene_axis(scene_ax, result, x_limits, y_limits)
    scene_ax.plot(
        result.expert_trajectory[:, 0],
        result.expert_trajectory[:, 1],
        color="#111111",
        linewidth=3.0,
        label="expert GT",
    )
    scene_ax.plot(
        result.oracle_trajectory[:, 0],
        result.oracle_trajectory[:, 1],
        color="#2A9D8F",
        linewidth=2.4,
        linestyle="--",
        label="VAE decode($\\mu_{gt}$)",
    )
    scene_ax.plot(
        result.pred_trajectory[:, 0],
        result.pred_trajectory[:, 1],
        color="#F72585",
        linewidth=2.7,
        label="decode($z_{pred}$)",
    )
    scene_ax.set_title("Fixed scene and decoded trajectories")
    scene_ax.legend(loc="best", fontsize=9)

    plot_skill_bars(skill_ax, result.gt_skill, result.pred_skill)
    plot_error_panel(error_ax, result)
    add_metric_text(error_ax, result)

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def decoded_latent_morph(
    model,
    result: SceneInference,
    alphas: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    gt = torch.as_tensor(result.gt_skill, device=device, dtype=torch.float32)
    pred = torch.as_tensor(result.pred_skill, device=device, dtype=torch.float32)
    alpha_tensor = torch.as_tensor(
        alphas,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(-1)
    latents = (1.0 - alpha_tensor) * gt + alpha_tensor * pred
    return model.model.decoder.decode(latents).cpu().numpy()


def create_latent_animation(
    model,
    result: SceneInference,
    output_path: Path,
    morph_frames: int,
    fps: int,
    hold_seconds: float,
    device: torch.device,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
) -> None:
    if morph_frames < 2:
        raise ValueError("--morph-frames must be at least 2")

    hold_frames = max(0, int(round(hold_seconds * fps)))
    transition = smoothstep(np.linspace(0.0, 1.0, morph_frames))
    alphas = np.concatenate(
        (
            np.zeros(hold_frames),
            transition,
            np.ones(hold_frames),
        )
    )
    trajectories = decoded_latent_morph(
        model,
        result,
        alphas,
        device,
    )

    title = (
        f"Fixed scene {result.scene_index} | latent interpolation "
        "$\\mu_{gt} \\rightarrow z_{pred}$"
    )
    fig, scene_ax, skill_ax, error_ax = create_layout(title)
    setup_scene_axis(scene_ax, result, x_limits, y_limits)
    scene_ax.plot(
        result.expert_trajectory[:, 0],
        result.expert_trajectory[:, 1],
        color="#111111",
        linewidth=3.0,
        label="expert GT",
    )
    scene_ax.plot(
        result.oracle_trajectory[:, 0],
        result.oracle_trajectory[:, 1],
        color="#2A9D8F",
        linewidth=2.0,
        linestyle="--",
        alpha=0.9,
        label="decode($\\mu_{gt}$)",
    )
    scene_ax.plot(
        result.pred_trajectory[:, 0],
        result.pred_trajectory[:, 1],
        color="#F72585",
        linewidth=2.0,
        linestyle="--",
        alpha=0.85,
        label="decode($z_{pred}$)",
    )
    current_line, = scene_ax.plot(
        [],
        [],
        color="#4361EE",
        linewidth=3.2,
        label="decode($z(\\alpha)$)",
    )
    current_endpoint, = scene_ax.plot(
        [],
        [],
        marker="o",
        markersize=8,
        color="#4361EE",
    )
    scene_ax.set_title("Trajectory change while the scene stays fixed")
    scene_ax.legend(loc="best", fontsize=9)

    current_bars = plot_skill_bars(
        skill_ax,
        result.gt_skill,
        result.pred_skill,
        include_current=True,
    )
    plot_error_panel(error_ax, result)
    metric_text = add_metric_text(error_ax, result)

    def update(frame: int):
        alpha = float(alphas[frame])
        trajectory = trajectories[frame]
        current_skill = (
            (1.0 - alpha) * result.gt_skill
            + alpha * result.pred_skill
        )
        current_line.set_data(trajectory[:, 0], trajectory[:, 1])
        current_endpoint.set_data(
            [trajectory[-1, 0]],
            [trajectory[-1, 1]],
        )
        for bar, value in zip(current_bars, current_skill):
            bar.set_height(float(value))

        ade, fde = trajectory_metrics(
            trajectory,
            result.expert_trajectory,
        )
        latent_distance = float(
            np.linalg.norm(current_skill - result.gt_skill)
        )
        metric_text.set_text(
            f"$\\alpha$ (GT→pred): {alpha:.3f}\n"
            f"$||z(\\alpha)-\\mu_{{gt}}||_2$: {latent_distance:.3f}\n"
            f"current trajectory ADE/FDE: {ade:.3f} / {fde:.3f} m"
        )
        return current_line, current_endpoint, metric_text, *current_bars

    clip = animation.FuncAnimation(
        fig,
        update,
        frames=len(alphas),
        interval=1000.0 / fps,
        blit=False,
    )
    save_video(clip, output_path, fps)
    plt.close(fig)


def create_rollout_animation(
    result: SceneInference,
    output_path: Path,
    fps: int,
    hold_seconds: float,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
) -> None:
    horizon = len(result.expert_trajectory)
    hold_frames = max(0, int(round(hold_seconds * fps)))
    time_steps = np.concatenate(
        (
            np.ones(hold_frames, dtype=int),
            np.arange(1, horizon + 1),
            np.full(hold_frames, horizon, dtype=int),
        )
    )

    title = (
        f"Fixed scene {result.scene_index} | 3-second trajectory rollout"
    )
    fig, scene_ax, skill_ax, error_ax = create_layout(title)
    setup_scene_axis(scene_ax, result, x_limits, y_limits)

    # Faint complete paths make the progressive rollout easy to interpret.
    scene_ax.plot(
        result.expert_trajectory[:, 0],
        result.expert_trajectory[:, 1],
        color="#111111",
        linewidth=1.0,
        alpha=0.18,
    )
    scene_ax.plot(
        result.oracle_trajectory[:, 0],
        result.oracle_trajectory[:, 1],
        color="#2A9D8F",
        linewidth=1.0,
        alpha=0.18,
    )
    scene_ax.plot(
        result.pred_trajectory[:, 0],
        result.pred_trajectory[:, 1],
        color="#F72585",
        linewidth=1.0,
        alpha=0.18,
    )
    gt_line, = scene_ax.plot(
        [],
        [],
        color="#111111",
        linewidth=3.0,
        label="expert GT",
    )
    oracle_line, = scene_ax.plot(
        [],
        [],
        color="#2A9D8F",
        linewidth=2.5,
        linestyle="--",
        label="VAE decode($\\mu_{gt}$)",
    )
    pred_line, = scene_ax.plot(
        [],
        [],
        color="#F72585",
        linewidth=2.8,
        label="decode($z_{pred}$)",
    )
    scene_ax.set_title("Future trajectory revealed at 10 Hz")
    scene_ax.legend(loc="best", fontsize=9)

    plot_skill_bars(skill_ax, result.gt_skill, result.pred_skill)
    plot_error_panel(error_ax, result)
    metric_text = add_metric_text(error_ax, result)

    def update(frame: int):
        step = int(time_steps[frame])
        gt_prefix = result.expert_trajectory[:step]
        oracle_prefix = result.oracle_trajectory[:step]
        pred_prefix = result.pred_trajectory[:step]
        gt_line.set_data(gt_prefix[:, 0], gt_prefix[:, 1])
        oracle_line.set_data(oracle_prefix[:, 0], oracle_prefix[:, 1])
        pred_line.set_data(pred_prefix[:, 0], pred_prefix[:, 1])

        pred_distances = np.linalg.norm(
            pred_prefix - gt_prefix,
            axis=-1,
        )
        oracle_distances = np.linalg.norm(
            oracle_prefix - gt_prefix,
            axis=-1,
        )
        metric_text.set_text(
            f"time: {step / 10.0:.1f} / {horizon / 10.0:.1f} s\n"
            f"pred prefix ADE/current error: "
            f"{pred_distances.mean():.3f} / {pred_distances[-1]:.3f} m\n"
            f"oracle prefix ADE/current error: "
            f"{oracle_distances.mean():.3f} / {oracle_distances[-1]:.3f} m"
        )
        return gt_line, oracle_line, pred_line, metric_text

    clip = animation.FuncAnimation(
        fig,
        update,
        frames=len(time_steps),
        interval=1000.0 / fps,
        blit=False,
    )
    save_video(clip, output_path, fps)
    plt.close(fig)


def save_video(
    clip: animation.FuncAnimation,
    output_path: Path,
    fps: int,
) -> None:
    print(f"Rendering video: {output_path}")
    if output_path.suffix.lower() == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "Matplotlib cannot find ffmpeg; use --format gif or install ffmpeg"
            )
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=3500,
            metadata={"artist": "SkillFormer"},
            extra_args=["-pix_fmt", "yuv420p"],
        )
    else:
        if not animation.writers.is_available("pillow"):
            raise RuntimeError("Pillow animation writer is unavailable")
        writer = animation.PillowWriter(fps=fps)
    clip.save(str(output_path), writer=writer, dpi=120)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.hold_seconds < 0:
        raise ValueError("--hold-seconds cannot be negative")

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading checkpoint: {args.ckpt}")
    print(f"Using device: {device}")
    model = load_model(args.ckpt, device)

    dataset = NuplanDataset(
        config=model.config,
        mode=args.mode,
        reuse_cache=True,
        num_workers=1,
        val_ratio=args.val_ratio,
    )
    scene_index, sample = select_scene(
        dataset,
        args.scene_index,
        args.token,
    )
    result = infer_scene(model, scene_index, sample, device)

    x_limits, y_limits = center_bounds(
        (
            result.expert_trajectory,
            result.oracle_trajectory,
            result.pred_trajectory,
        ),
        margin=args.view_margin,
        min_span=args.min_view_span,
    )
    token_short = result.token[:12] if result.token else "no_token"
    prefix = f"scene_{scene_index:04d}_{token_short}"

    static_path = args.output_dir / f"{prefix}_comparison.png"
    save_static_comparison(
        result,
        static_path,
        x_limits,
        y_limits,
    )

    suffix = f".{args.video_format}"
    if args.animation in {"latent", "both"}:
        create_latent_animation(
            model=model,
            result=result,
            output_path=args.output_dir / f"{prefix}_latent_morph{suffix}",
            morph_frames=args.morph_frames,
            fps=args.fps,
            hold_seconds=args.hold_seconds,
            device=device,
            x_limits=x_limits,
            y_limits=y_limits,
        )
    if args.animation in {"rollout", "both"}:
        create_rollout_animation(
            result=result,
            output_path=args.output_dir / f"{prefix}_rollout{suffix}",
            fps=args.fps,
            hold_seconds=args.hold_seconds,
            x_limits=x_limits,
            y_limits=y_limits,
        )

    np.savez_compressed(
        args.output_dir / f"{prefix}_values.npz",
        token=result.token,
        scene_index=result.scene_index,
        gt_skill=result.gt_skill,
        pred_skill=result.pred_skill,
        normalized_error=result.normalized_error,
        expert_trajectory=result.expert_trajectory,
        oracle_trajectory=result.oracle_trajectory,
        pred_trajectory=result.pred_trajectory,
    )

    oracle_ade, oracle_fde = trajectory_metrics(
        result.oracle_trajectory,
        result.expert_trajectory,
    )
    pred_ade, pred_fde = trajectory_metrics(
        result.pred_trajectory,
        result.expert_trajectory,
    )
    print("\nScene summary")
    print(f"  index: {scene_index}")
    print(f"  token: {result.token}")
    print(
        "  normalized skill RMSE: "
        f"{np.sqrt(np.mean(result.normalized_error**2)):.4f}"
    )
    print(f"  VAE oracle ADE/FDE: {oracle_ade:.3f} / {oracle_fde:.3f} m")
    print(f"  prediction ADE/FDE: {pred_ade:.3f} / {pred_fde:.3f} m")
    print(f"  outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
