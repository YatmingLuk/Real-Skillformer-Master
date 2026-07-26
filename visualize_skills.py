"""Visualize the eight ground-truth and inferred VAE skills.

The script compares:
  - GT skill: VAE posterior mean ``mu_gt = VAE.encode(labels[..., :2])``
  - Predicted skill: ``z_pred = Wayformer(scene)``

Both raw latent values and the normalized values used by the training loss are
saved. A checkpoint trained with the single-mode skill-only architecture is
required.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skillformer-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from Wayformer.wayformer import WayformerPL
from Wayformer.wayformer_config import batch_list_to_batch_tensors
from Wayformer.wf_dataset import NuplanDataset


SKILL_NAMES = [
    "skill 0 (speed anchor)",
    "skill 1 (lateral anchor)",
    "skill 2",
    "skill 3",
    "skill 4",
    "skill 5",
    "skill 6",
    "skill 7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the eight GT VAE skills with Wayformer inference."
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint trained by the single-mode skill-only Wayformer.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "val"),
        default="val",
        help="Dataset cache to visualize.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
        help="Maximum number of samples to run through the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size; defaults to the checkpoint config.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Zero is the safest option for visualization.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Used only if the requested dataset cache has to be rebuilt.",
    )
    parser.add_argument(
        "--trace-samples",
        type=int,
        default=50,
        help="Number of samples shown in the GT/prediction line plot.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or a value such as cuda:1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/skill_visualization"),
        help="Directory for figures, statistics, and latent arrays.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def load_model(checkpoint: Path, device: torch.device) -> WayformerPL:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    try:
        model = WayformerPL.load_from_checkpoint(
            checkpoint_path=str(checkpoint),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with Lightning/PyTorch versions without weights_only.
        model = WayformerPL.load_from_checkpoint(
            checkpoint_path=str(checkpoint),
            map_location="cpu",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load the checkpoint. This script requires a checkpoint "
            "retrained with the single-mode skill-only architecture."
        ) from exc

    model.to(device)
    model.eval()
    return model


def collect_skills(
    model: WayformerPL,
    mode: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    dataset = NuplanDataset(
        config=model.config,
        mode=mode,
        reuse_cache=True,
        num_workers=max(1, num_workers),
        val_ratio=val_ratio,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"The {mode} dataset is empty")

    sample_count = min(max_samples, len(dataset))
    dataset = Subset(dataset, range(sample_count))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=batch_list_to_batch_tensors,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    pred_chunks: List[np.ndarray] = []
    gt_chunks: List[np.ndarray] = []
    tokens: List[str] = []

    for batch in tqdm(loader, desc=f"Inferring {mode} skills"):
        z_pred = model.model.predict_skill(batch, device)
        mu_gt = model.model.encode_gt_skill(batch, device)

        pred_chunks.append(z_pred.cpu().numpy())
        gt_chunks.append(mu_gt.cpu().numpy())
        tokens.extend(str(sample.get("token", "")) for sample in batch)

    predicted = np.concatenate(pred_chunks, axis=0)
    ground_truth = np.concatenate(gt_chunks, axis=0)
    token_array = np.asarray(tokens, dtype=str)

    expected_dim = int(model.config.vae_latent_dim)
    if predicted.shape != ground_truth.shape:
        raise RuntimeError(
            f"Prediction/GT shape mismatch: {predicted.shape} vs {ground_truth.shape}"
        )
    if predicted.ndim != 2 or predicted.shape[1] != expected_dim:
        raise RuntimeError(
            f"Expected [N, {expected_dim}] skills, got {predicted.shape}"
        )
    if expected_dim != len(SKILL_NAMES):
        raise RuntimeError(
            f"This visualization expects 8 skills, checkpoint has {expected_dim}"
        )
    if not np.isfinite(predicted).all() or not np.isfinite(ground_truth).all():
        raise RuntimeError("NaN or Inf found in collected skill values")

    return ground_truth, predicted, token_array


def dimension_stats(gt: np.ndarray, pred: np.ndarray) -> List[Dict[str, float]]:
    error = pred - gt
    rows: List[Dict[str, float]] = []
    for dim, name in enumerate(SKILL_NAMES):
        gt_dim = gt[:, dim]
        pred_dim = pred[:, dim]
        err_dim = error[:, dim]
        gt_std = float(np.std(gt_dim))
        pred_std = float(np.std(pred_dim))
        if gt_std > 1e-12 and pred_std > 1e-12:
            corr = float(np.corrcoef(gt_dim, pred_dim)[0, 1])
        else:
            corr = float("nan")

        denominator = float(np.sum((gt_dim - np.mean(gt_dim)) ** 2))
        r2 = (
            1.0 - float(np.sum(err_dim**2)) / denominator
            if denominator > 1e-12
            else float("nan")
        )
        rows.append(
            {
                "dimension": dim,
                "name": name,
                "mae": float(np.mean(np.abs(err_dim))),
                "rmse": float(np.sqrt(np.mean(err_dim**2))),
                "bias": float(np.mean(err_dim)),
                "error_std": float(np.std(err_dim)),
                "p95_abs_error": float(np.percentile(np.abs(err_dim), 95)),
                "correlation": corr,
                "r2": r2,
            }
        )
    return rows


def plot_dashboard(
    gt: np.ndarray,
    pred: np.ndarray,
    stats: Sequence[Dict[str, float]],
    title: str,
    output_path: Path,
) -> None:
    error = pred - gt
    fig, axes = plt.subplots(4, 4, figsize=(19, 17))

    for dim, name in enumerate(SKILL_NAMES):
        row = dim // 2
        column = (dim % 2) * 2
        scatter_ax = axes[row, column]
        error_ax = axes[row, column + 1]

        scatter_ax.scatter(
            gt[:, dim],
            pred[:, dim],
            s=18,
            alpha=0.55,
            color="#4361EE",
            edgecolors="none",
        )
        low = float(min(gt[:, dim].min(), pred[:, dim].min()))
        high = float(max(gt[:, dim].max(), pred[:, dim].max()))
        padding = max((high - low) * 0.08, 1e-3)
        limits = (low - padding, high + padding)
        scatter_ax.plot(limits, limits, "--", color="#D00000", linewidth=1.5)
        scatter_ax.set_xlim(limits)
        scatter_ax.set_ylim(limits)
        scatter_ax.set_aspect("equal", adjustable="box")
        scatter_ax.set_title(f"{name}: GT vs prediction")
        scatter_ax.set_xlabel("GT $\\mu$")
        scatter_ax.set_ylabel("Predicted $z$")
        scatter_ax.grid(alpha=0.2)

        row_stats = stats[dim]
        annotation = (
            f"MAE={row_stats['mae']:.4f}\n"
            f"RMSE={row_stats['rmse']:.4f}\n"
            f"bias={row_stats['bias']:+.4f}\n"
            f"r={row_stats['correlation']:.3f}"
        )
        scatter_ax.text(
            0.03,
            0.97,
            annotation,
            transform=scatter_ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )

        error_ax.hist(
            error[:, dim],
            bins=min(35, max(10, int(np.sqrt(len(error))))),
            color="#F72585",
            alpha=0.75,
        )
        error_ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        error_ax.axvline(
            row_stats["bias"],
            color="#3A0CA3",
            linewidth=1.5,
            label=f"bias={row_stats['bias']:+.4f}",
        )
        error_ax.set_title(f"{name}: prediction - GT")
        error_ax.set_xlabel("Error")
        error_ax.set_ylabel("Count")
        error_ax.legend(fontsize=8)
        error_ax.grid(axis="y", alpha=0.2)

    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmap(
    normalized_error: np.ndarray,
    normalized_stats: Sequence[Dict[str, float]],
    output_path: Path,
) -> None:
    abs_error = np.abs(normalized_error)
    color_limit = float(np.percentile(abs_error, 98))
    color_limit = max(color_limit, 1e-3)

    fig, (heat_ax, bar_ax) = plt.subplots(
        1,
        2,
        figsize=(18, 6),
        gridspec_kw={"width_ratios": (5, 1.4)},
    )
    image = heat_ax.imshow(
        normalized_error.T,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
    )
    heat_ax.set_yticks(np.arange(len(SKILL_NAMES)))
    heat_ax.set_yticklabels(SKILL_NAMES)
    heat_ax.set_xlabel("Sample index")
    heat_ax.set_title("Normalized error: prediction - GT")
    fig.colorbar(image, ax=heat_ax, label="Normalized latent error")

    y = np.arange(len(SKILL_NAMES))
    mae = np.asarray([row["mae"] for row in normalized_stats])
    rmse = np.asarray([row["rmse"] for row in normalized_stats])
    bar_ax.barh(y, rmse, color="#B8C0FF", label="RMSE")
    bar_ax.barh(y, mae, color="#4361EE", label="MAE")
    bar_ax.set_yticks(y)
    bar_ax.set_yticklabels([f"skill {i}" for i in y])
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("Error")
    bar_ax.set_title("Per-skill error")
    bar_ax.grid(axis="x", alpha=0.2)
    bar_ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sample_trace(
    gt: np.ndarray,
    pred: np.ndarray,
    trace_samples: int,
    output_path: Path,
) -> None:
    count = min(max(1, trace_samples), len(gt))
    indices = np.arange(count)
    fig, axes = plt.subplots(4, 2, figsize=(17, 15), sharex=True)

    for dim, ax in enumerate(axes.flat):
        ax.plot(
            indices,
            gt[:count, dim],
            color="#222222",
            linewidth=1.8,
            marker="o",
            markersize=3,
            label="GT",
        )
        ax.plot(
            indices,
            pred[:count, dim],
            color="#F72585",
            linewidth=1.5,
            marker="x",
            markersize=4,
            label="prediction",
        )
        ax.fill_between(
            indices,
            gt[:count, dim],
            pred[:count, dim],
            color="#F72585",
            alpha=0.12,
        )
        ax.set_title(SKILL_NAMES[dim])
        ax.set_ylabel("Raw latent value")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Sample index")
    axes[-1, 1].set_xlabel("Sample index")
    fig.suptitle(f"Raw GT and predicted skills for the first {count} samples")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_stats(
    output_path: Path,
    raw_stats: Sequence[Dict[str, float]],
    normalized_stats: Sequence[Dict[str, float]],
) -> None:
    rows = []
    for raw, normalized in zip(raw_stats, normalized_stats):
        row = {
            "dimension": raw["dimension"],
            "name": raw["name"],
        }
        row.update(
            {
                f"raw_{key}": value
                for key, value in raw.items()
                if key not in {"dimension", "name"}
            }
        )
        row.update(
            {
                f"normalized_{key}": value
                for key, value in normalized.items()
                if key not in {"dimension", "name"}
            }
        )
        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    raw_stats: Sequence[Dict[str, float]],
    normalized_stats: Sequence[Dict[str, float]],
) -> None:
    print("\nPer-skill error summary")
    print(
        f"{'skill':<8} {'raw MAE':>10} {'raw RMSE':>10} "
        f"{'norm MAE':>10} {'norm RMSE':>11} {'corr':>9}"
    )
    for raw, normalized in zip(raw_stats, normalized_stats):
        print(
            f"{raw['dimension']:<8d} "
            f"{raw['mae']:>10.4f} "
            f"{raw['rmse']:>10.4f} "
            f"{normalized['mae']:>10.4f} "
            f"{normalized['rmse']:>11.4f} "
            f"{raw['correlation']:>9.3f}"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    print(f"Using device: {device}")
    model = load_model(args.ckpt, device)
    batch_size = args.batch_size or int(model.config.batch_size)

    gt, pred, tokens = collect_skills(
        model=model,
        mode=args.mode,
        max_samples=args.max_samples,
        batch_size=batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        device=device,
    )

    z_mean = model.model.z_mean.detach().cpu().numpy()
    z_std = np.clip(model.model.z_std.detach().cpu().numpy(), 1e-4, None)
    gt_normalized = (gt - z_mean) / z_std
    pred_normalized = (pred - z_mean) / z_std

    raw_stats = dimension_stats(gt, pred)
    normalized_stats = dimension_stats(gt_normalized, pred_normalized)

    checkpoint_name = args.ckpt.stem
    plot_dashboard(
        gt,
        pred,
        raw_stats,
        title=f"Raw skill GT vs prediction | N={len(gt)} | {checkpoint_name}",
        output_path=output_dir / "skill_raw_dashboard.png",
    )
    plot_dashboard(
        gt_normalized,
        pred_normalized,
        normalized_stats,
        title=(
            f"Normalized skill GT vs prediction | N={len(gt)} | "
            f"{checkpoint_name}"
        ),
        output_path=output_dir / "skill_normalized_dashboard.png",
    )
    plot_error_heatmap(
        pred_normalized - gt_normalized,
        normalized_stats,
        output_dir / "skill_normalized_error_heatmap.png",
    )
    plot_sample_trace(
        gt,
        pred,
        args.trace_samples,
        output_dir / "skill_sample_trace.png",
    )
    write_stats(
        output_dir / "skill_stats.csv",
        raw_stats,
        normalized_stats,
    )
    np.savez_compressed(
        output_dir / "skill_values.npz",
        tokens=tokens,
        gt=gt,
        prediction=pred,
        error=pred - gt,
        normalized_gt=gt_normalized,
        normalized_prediction=pred_normalized,
        normalized_error=pred_normalized - gt_normalized,
        z_mean=z_mean,
        z_std=z_std,
    )

    print_summary(raw_stats, normalized_stats)
    print(f"\nSaved visualization outputs to: {output_dir.resolve()}")
    for path in sorted(output_dir.iterdir()):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
