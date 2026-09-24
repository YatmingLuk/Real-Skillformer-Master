"""
Baseline Audit Script for SkillFormer Input Representation (Phase 1 / Step 1)

Research Question (RQ1):
Does current SkillFormer sufficiently model interaction-dependent behaviors?

Audits:
1. Agent Capacity, Occupancy, Unused Slot Waste Ratio (NUM_AGENTS = 19), and Max Observed Neighbors
2. Effective Retained Neighbor Range & Individual Distance Histogram (matching sort_agents_hist_by_distance physics)
3. Nearest Neighbor Distance
4. Agent Dynamic Properties (Historically Active vs. Static Neighbors over 2s window)

Includes full-feature zero-padding mask defense for robust cache parsing.
"""

from __future__ import annotations

import pickle
import zlib
import numpy as np
from pathlib import Path


def load_cached_scenarios(cache_path: Path):
    """Safely loads compressed scenarios from the validation cache file."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Validation cache file not found: {cache_path}")

    with cache_path.open("rb") as f:
        ex_list = pickle.load(f)

    if not ex_list:
        print(f"[WARNING] Cache file is empty: {cache_path}")
        return

    for idx, data_compress in enumerate(ex_list):
        try:
            mapping = pickle.loads(zlib.decompress(data_compress))
            yield idx, mapping
        except Exception as exc:
            print(f"[WARNING] Scenario {idx} decode failed: {exc}")
            continue


def get_neighbor_bucket(count: int) -> str:
    """Categorizes neighbor count into specified scene-level distribution buckets."""
    if count == 0:
        return "0"
    elif 1 <= count <= 5:
        return "1-5"
    elif 6 <= count <= 10:
        return "6-10"
    elif 11 <= count <= 19:
        return "11-19"
    else:
        return ">19"


def analyze_scene(mapping: dict):
    """
    Performs baseline audit on a single scene.

    Expected shape of mapping["agents"]: [1 + N, 20, C]
      - agents[0]: Ego vehicle
      - agents[1:]: Candidate neighbor vehicles
    """
    agents = mapping["agents"]

    if agents.ndim != 3:
        raise ValueError(f"Unexpected agents dim: {agents.ndim}, expected 3D array")

    total_agents = agents.shape[0]
    if total_agents < 1:
        raise ValueError(f"Invalid agents shape: {agents.shape}, missing Ego vehicle")

    raw_neighbors = agents[1:]  # [N_raw, 20, C]

    # -----------------------------------------------------------------
    # Optimization 1: Full-Feature Zero-Padding Mask Filter
    # Check across ALL 20 timesteps and ALL feature channels (axis=(1,2))
    # to safely handle edge cases where a neighbor temporarily passes (0,0).
    # -----------------------------------------------------------------
    if raw_neighbors.shape[0] > 0:
        valid_mask = ~(np.all(raw_neighbors == 0, axis=(1, 2)))
        neighbors = raw_neighbors[valid_mask]
    else:
        neighbors = raw_neighbors

    num_neighbors = len(neighbors)
    max_capacity = 19
    occupancy_ratio = num_neighbors / max_capacity

    scene_retained_dist = None
    nearest_neighbor_dist = None
    individual_min_dists = []
    num_dynamic = 0
    num_static = 0

    if num_neighbors > 0:
        # -----------------------------------------------------------------
        # Alignment with SkillFormer sort_agents_hist_by_distance():
        # dist2 = (agents_hist[:,:,0:2] ** 2).sum(dim=-1)
        # min_dist, _ = dist2.min(dim=1)
        # order = min_dist.argsort()
        # -----------------------------------------------------------------
        # Calculate squared distance to Ego (0,0) across all 20 timesteps: [N_valid, 20]
        hist_dist2 = neighbors[:, :, 0] ** 2 + neighbors[:, :, 1] ** 2

        # Minimum historical distance over 20 timesteps for each valid neighbor: [N_valid]
        min_hist_dists = np.sqrt(np.min(hist_dist2, axis=1))
        individual_min_dists = min_hist_dists.tolist()

        # Effective Retained Neighbor Range = max(min_t(d_t))
        scene_retained_dist = float(np.max(min_hist_dists))

        # Nearest Neighbor Distance = min(min_t(d_t))
        nearest_neighbor_dist = float(np.min(min_hist_dists))

        # -----------------------------------------------------------------
        # Dynamic Neighbor Instance Analysis
        # heuristic threshold: an agent moving >1.0m over 2s history window (20 steps)
        # is considered historically dynamic
        # -----------------------------------------------------------------
        start_pos = neighbors[:, 0, 0:2]  # timestep 0
        end_pos = neighbors[:, 19, 0:2]  # timestep 19
        displacements = np.linalg.norm(end_pos - start_pos, axis=1)  # [N_valid]

        is_dynamic = displacements > 1.0
        num_dynamic = int(np.sum(is_dynamic))
        num_static = num_neighbors - num_dynamic

    return (num_neighbors, occupancy_ratio, scene_retained_dist,
            nearest_neighbor_dist, individual_min_dists, num_dynamic, num_static)


def main():
    project_root = Path(__file__).resolve().parent.parent
    cache_path = project_root / "temp_config" / "ex_list_nuplan_val.pkl"
    # 定义输出文本文件路径: analysis/result.txt
    output_txt_path = project_root / "analysis" / "result.txt"

    all_neighbors = []
    all_occupancies = []
    retained_distances = []
    nearest_distances = []
    all_individual_distances = []

    total_dynamic = 0
    total_static = 0

    bucket_counts = {"0": 0, "1-5": 0, "6-10": 0, "11-19": 0, ">19": 0}

    try:
        scenarios = list(load_cached_scenarios(cache_path))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    if not scenarios:
        print("[ERROR] No valid scenarios were loaded from cache.")
        return

    for idx, mapping in scenarios:
        if "agents" not in mapping:
            continue

        try:
            (num_neighbors, occ_ratio, retained_dist, nearest_dist,
             ind_dists, num_dyn, num_stat) = analyze_scene(mapping)
        except Exception as exc:
            print(f"[WARNING] Scenario {idx} processing failed: {exc}")
            continue

        all_neighbors.append(num_neighbors)
        all_occupancies.append(occ_ratio)

        # Scene-level Neighbor Count Bucket
        bucket = get_neighbor_bucket(num_neighbors)
        bucket_counts[bucket] += 1

        if num_neighbors > 0:
            retained_distances.append(retained_dist)
            nearest_distances.append(nearest_dist)
            all_individual_distances.extend(ind_dists)
            total_dynamic += num_dyn
            total_static += num_stat

    total_scenes = len(all_neighbors)
    if total_scenes == 0:
        print("[ERROR] No valid agent tensors found in dataset.")
        return

    total_neighbor_instances = sum(all_neighbors)

    # Compute Summary Metrics
    mean_neighbors = np.mean(all_neighbors)
    median_neighbors = float(np.median(all_neighbors))
    max_neighbors_observed = int(np.max(all_neighbors))
    mean_slot_occupancy = np.mean(all_occupancies) * 100.0
    mean_unused_ratio = 100.0 - mean_slot_occupancy

    if retained_distances:
        mean_retained = np.mean(retained_distances)
        median_retained = float(np.median(retained_distances))
        max_retained = np.max(retained_distances)
        min_retained = np.min(retained_distances)
    else:
        mean_retained = median_retained = max_retained = min_retained = 0.0

    if nearest_distances:
        mean_nearest = np.mean(nearest_distances)
        median_nearest = float(np.median(nearest_distances))
    else:
        mean_nearest = median_nearest = 0.0

    if total_neighbor_instances > 0:
        dynamic_instance_ratio = (total_dynamic / total_neighbor_instances) * 100.0
        static_instance_ratio = (total_static / total_neighbor_instances) * 100.0
    else:
        dynamic_instance_ratio = 0.0
        static_instance_ratio = 0.0

    avg_dynamic_per_scene = total_dynamic / total_scenes

    # Individual Neighbor Distance Histogram
    dist_bins = [0.0, 10.0, 20.0, 30.0, 50.0, np.inf]
    dist_bin_labels = ["0-10m", "10-20m", "20-30m", "30-50m", "50m+"]
    hist_counts, _ = np.histogram(all_individual_distances, bins=dist_bins)

    # -----------------------------------------------------------------
    # Build Formal Audit Report String
    # -----------------------------------------------------------------
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("STEP 1: BASELINE INPUT REPRESENTATION AUDIT")
    report_lines.append("==================================================")
    report_lines.append("")
    report_lines.append("1. Agent Capacity and Occupancy")
    report_lines.append("--------------------------------")
    report_lines.append(f"Total Scenes: {total_scenes}")
    report_lines.append(f"Mean Neighbors: {mean_neighbors:.2f}")
    report_lines.append(f"Median Neighbors: {median_neighbors:.1f}")
    report_lines.append(f"Max Neighbors Observed: {max_neighbors_observed}")
    report_lines.append(f"Mean Slot Occupancy: {mean_slot_occupancy:.2f}%")
    report_lines.append(f"Unused Neighbor Slots: {mean_unused_ratio:.2f}%")
    report_lines.append("")
    report_lines.append("Neighbor Count Distribution (per scene):")
    report_lines.append(f"0: {bucket_counts['0']}")
    report_lines.append(f"1-5: {bucket_counts['1-5']}")
    report_lines.append(f"6-10: {bucket_counts['6-10']}")
    report_lines.append(f"11-19: {bucket_counts['11-19']}")
    if bucket_counts[">19"] > 0:
        report_lines.append(f">19: {bucket_counts['>19']}")
    report_lines.append("")
    report_lines.append("2. Effective Retained Neighbor Range & Distance Distribution")
    report_lines.append("--------------------------------")
    report_lines.append(f"Mean Retained Range: {mean_retained:.2f} m")
    report_lines.append(f"Median Retained Range: {median_retained:.2f} m")
    report_lines.append(f"Max Retained Range: {max_retained:.2f} m")
    report_lines.append(f"Min Retained Range: {min_retained:.2f} m")
    report_lines.append("")
    report_lines.append("Nearest Neighbor Distance:")
    report_lines.append(f"Mean: {mean_nearest:.2f} m")
    report_lines.append(f"Median: {median_nearest:.2f} m")
    report_lines.append("")
    report_lines.append("Individual Neighbor Distance Distribution (all valid neighbors):")
    if total_neighbor_instances > 0:
        for label, count in zip(dist_bin_labels, hist_counts):
            percentage = (count / total_neighbor_instances) * 100.0
            report_lines.append(f"  {label:<8s}: {count:6d} ({percentage:5.2f}%)")
    else:
        report_lines.append("  No valid neighbor instances found.")
    report_lines.append("")
    report_lines.append("3. Agent Dynamic Properties")
    report_lines.append("--------------------------------")
    report_lines.append(f"Total Neighbor Instances: {total_neighbor_instances}")
    report_lines.append(f"Historically Active Neighbors (>1m over 2s): {total_dynamic}")
    report_lines.append(f"Dynamic Neighbor Instance Ratio: {dynamic_instance_ratio:.2f}%")
    report_lines.append(f"Historically Static Instance Ratio: {static_instance_ratio:.2f}%")
    report_lines.append(f"Average Dynamic Neighbors / Scene: {avg_dynamic_per_scene:.2f}")
    report_lines.append("==================================================")

    full_report = "\n".join(report_lines)

    # 1. 屏幕打印输出
    print(full_report)

    # 2. 写入/保存为文本文件
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.write_text(full_report, encoding="utf-8")
    print(f"\n[INFO] Audit report successfully saved to: {output_txt_path}")


if __name__ == "__main__":
    main()