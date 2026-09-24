"""
Appendix Threshold Sensitivity Analysis
Path: analysis/appendix_threshold_sensitivity.py

Purpose:
Proves that the diagnostic probe findings (historical candidate truncation bias)
are robust across varying parameter thresholds for emerging interaction detection.

Outputs:
- analysis/results/dynamic_threshold_sensitivity.csv
- analysis/results/dynamic_threshold_sensitivity.png (Figure A1: 3-Panel Plot)
"""

import os
import sys
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from Wayformer.utils import extract_agent_feature, wrap_angle
try:
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
except ImportError:
    from nuplan.planning.scenario_builder.abstract_tracked_objects import TrackedObjectType


def extract_scenario_cache(scenario, iteration=0):
    ego_state_0 = scenario.get_ego_state_at_iteration(iteration)
    ego_x0, ego_y0 = ego_state_0.rear_axle.x, ego_state_0.rear_axle.y
    angle0 = ego_state_0.rear_axle.heading

    rot = -angle0 + np.pi / 2.0
    c, s = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[c, -s], [s, c]])

    past_objs = list(scenario.get_past_tracked_objects(iteration=iteration, time_horizon=2.0, num_samples=20))
    past_ts = list(scenario.get_past_timestamps(iteration=iteration, time_horizon=2.0, num_samples=20))

    track_token_ids = {}
    agents_per_ts = []
    flat_ts = [t.time_s - past_ts[0].time_s for t in past_ts]
    types = [TrackedObjectType.PEDESTRIAN, TrackedObjectType.VEHICLE, TrackedObjectType.BICYCLE]

    for i, tracked_objs in enumerate(past_objs):
        tensorized, track_token_ids = extract_agent_feature(types, tracked_objs, track_token_ids, flat_ts[i])
        agents_per_ts.append(tensorized)

    id_to_token = {v: k for k, v in track_token_ids.items()}
    num_agents = len(track_token_ids)
    T = len(past_objs)

    output_agents = torch.ones((num_agents, T, 9), dtype=torch.float32)
    for t_idx, features in enumerate(agents_per_ts):
        for row in features:
            agent_idx = int(row[4].item())
            if agent_idx < num_agents:
                dx, dy = row[0] - ego_x0, row[1] - ego_y0
                row[0], row[1] = rot_mat @ np.array([dx, dy])
                row[2] = wrap_angle(row[2] + rot)
                output_agents[agent_idx, t_idx, :] = row

    mask = (output_agents[:, -1, 8] >= 1.8) & (output_agents[:, 0, 8] < 0.5)
    candidate_indices = torch.where(mask)[0].tolist()
    candidate_tokens = [id_to_token[idx] for idx in candidate_indices]
    candidates = output_agents[mask]

    if len(candidate_tokens) == 0:
        return None

    dist2_hist = (candidates[:, :, 0:2] ** 2).sum(dim=-1)
    min_d_hist = torch.sqrt(dist2_hist.min(dim=1)[0]).numpy()
    hist_dist_map = {token: min_d_hist[i] for i, token in enumerate(candidate_tokens)}

    fut_objs_list = list(scenario.get_future_tracked_objects(iteration=iteration, time_horizon=6.0, num_samples=60))
    d_fut_map = {}
    traveled_dist_map = {}

    for token in candidate_tokens:
        fut_dists = []
        fut_pts_global = []
        for t_idx in range(len(fut_objs_list)):
            f_objs = fut_objs_list[t_idx]
            match = next((obj for obj in f_objs.tracked_objects if obj.track_token == token), None)
            if match is not None:
                dx, dy = match.center.x - ego_x0, match.center.y - ego_y0
                fut_dists.append(np.linalg.norm(rot_mat @ np.array([dx, dy])))
                fut_pts_global.append([match.center.x, match.center.y])

        v_rate = len(fut_dists) / 60.0
        if v_rate >= 0.8:
            d_fut_map[token] = float(np.min(fut_dists))
            pts = np.array(fut_pts_global)
            traveled_dist_map[token] = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0

    valid_oracle_tokens = [t for t in candidate_tokens if t in d_fut_map]
    if len(valid_oracle_tokens) == 0:
        return None

    K = min(19, len(valid_oracle_tokens))

    d_hist_subset = [hist_dist_map[t] for t in valid_oracle_tokens]
    hist_sort_idx = np.argsort(d_hist_subset)
    topK_hist_tokens = set(valid_oracle_tokens[idx] for idx in hist_sort_idx[:K])

    return {
        "hist_dist_map": hist_dist_map,
        "d_fut_map": d_fut_map,
        "traveled_dist_map": traveled_dist_map,
        "valid_oracle_tokens": valid_oracle_tokens,
        "topK_hist_tokens": topK_hist_tokens,
    }


def identify_emerging_interaction_parameterized(cache, d_fut_th, delta_d_th, travel_dist_th):
    emerging_tokens = set()
    for token in cache["valid_oracle_tokens"]:
        d_h = cache["hist_dist_map"][token]
        d_f = cache["d_fut_map"][token]
        L_i = cache["traveled_dist_map"][token]

        if d_f < d_fut_th and (d_h - d_f) > delta_d_th and L_i > travel_dist_th:
            emerging_tokens.add(token)
    return emerging_tokens


def run_appendix_sensitivity_analysis(scenarios):
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"\nCaching {len(scenarios)} Scenarios for Sensitivity Sweep...")
    caches = [c for sc in tqdm(scenarios) if (c := extract_scenario_cache(sc)) is not None]

    d_fut_thresholds = [5.0, 10.0, 15.0]
    delta_d_thresholds = [0.5, 1.0, 2.0]
    travel_dist_thresholds = [3.0, 5.0, 8.0]

    results_grid = []

    print("\nRunning Parameter Grid Search (27 Combinations)...")
    for travel_th in travel_dist_thresholds:
        for d_fut_th in d_fut_thresholds:
            for delta_d_th in delta_d_thresholds:
                tot_crit = 0
                tot_missed = 0
                tot_covered = 0

                for c in caches:
                    crit_tokens = identify_emerging_interaction_parameterized(c, d_fut_th, delta_d_th, travel_th)
                    missed = crit_tokens - c["topK_hist_tokens"]
                    covered = crit_tokens & c["topK_hist_tokens"]

                    tot_crit += len(crit_tokens)
                    tot_missed += len(missed)
                    tot_covered += len(covered)

                miss_rate = (tot_missed / max(1, tot_crit)) * 100.0
                coverage = (tot_covered / max(1, tot_crit)) * 100.0 if tot_crit > 0 else 100.0

                results_grid.append({
                    "d_future": d_fut_th,
                    "delta_d": delta_d_th,
                    "travel_dist": travel_th,
                    "n_critical": tot_crit,
                    "coverage": coverage,
                    "miss_rate": miss_rate
                })

    # Save CSV
    os.makedirs("analysis/results", exist_ok=True)
    csv_path = "analysis/results/dynamic_threshold_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["d_future", "delta_d", "travel_dist", "n_critical", "coverage", "miss_rate"])
        writer.writeheader()
        writer.writerows(results_grid)

    print(f"Saved sensitivity CSV table to: {csv_path}")

    # Generate Figure A1: 3-Panel Plot
    default_travel_th = 5.0
    sub_results = [r for r in results_grid if r["travel_dist"] == default_travel_th]

    grid_count = np.zeros((len(d_fut_thresholds), len(delta_d_thresholds)))
    grid_cov = np.zeros((len(d_fut_thresholds), len(delta_d_thresholds)))
    grid_miss = np.zeros((len(d_fut_thresholds), len(delta_d_thresholds)))

    for r in sub_results:
        i = d_fut_thresholds.index(r["d_future"])
        j = delta_d_thresholds.index(r["delta_d"])
        grid_count[i, j] = r["n_critical"]
        grid_cov[i, j] = r["coverage"]
        grid_miss[i, j] = r["miss_rate"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Threshold Sensitivity of Emerging Interaction Detection (Travel Dist = {default_travel_th}m)", fontsize=13, fontweight="bold")

    # Panel A: Emerging Agent Count
    im0 = axes[0].imshow(grid_count, cmap="Blues")
    axes[0].set_title("(a) Emerging Agent Count ($N_{emerging}$)")
    axes[0].set_xlabel(r"$\Delta d$ Threshold ($d_{hist} - d_{fut}$)")
    axes[0].set_ylabel(r"$d_{future}$ Threshold")
    axes[0].set_xticks(range(len(delta_d_thresholds)))
    axes[0].set_xticklabels([f"{v}m" for v in delta_d_thresholds])
    axes[0].set_yticks(range(len(d_fut_thresholds)))
    axes[0].set_yticklabels([f"{v}m" for v in d_fut_thresholds])
    for i in range(len(d_fut_thresholds)):
        for j in range(len(delta_d_thresholds)):
            axes[0].text(j, i, f"{int(grid_count[i, j])}", ha="center", va="center", color="black", fontweight="bold")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Panel B: Historical Future-Proxy Coverage@K
    im1 = axes[1].imshow(grid_cov, cmap="Greens", vmin=0, vmax=100)
    axes[1].set_title("(b) Historical Future-Proxy Coverage@K (%)")
    axes[1].set_xlabel(r"$\Delta d$ Threshold ($d_{hist} - d_{fut}$)")
    axes[1].set_xticks(range(len(delta_d_thresholds)))
    axes[1].set_xticklabels([f"{v}m" for v in delta_d_thresholds])
    axes[1].set_yticks(range(len(d_fut_thresholds)))
    axes[1].set_yticklabels([f"{v}m" for v in d_fut_thresholds])
    for i in range(len(d_fut_thresholds)):
        for j in range(len(delta_d_thresholds)):
            axes[1].text(j, i, f"{grid_cov[i, j]:.1f}%", ha="center", va="center", color="black", fontweight="bold")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # Panel C: Historical Selector Miss Rate
    im2 = axes[2].imshow(grid_miss, cmap="YlOrRd", vmin=0, vmax=100)
    axes[2].set_title("(c) Historical Selector Miss Rate (%)")
    axes[2].set_xlabel(r"$\Delta d$ Threshold ($d_{hist} - d_{fut}$)")
    axes[2].set_xticks(range(len(delta_d_thresholds)))
    axes[2].set_xticklabels([f"{v}m" for v in delta_d_thresholds])
    axes[2].set_yticks(range(len(d_fut_thresholds)))
    axes[2].set_yticklabels([f"{v}m" for v in d_fut_thresholds])
    for i in range(len(d_fut_thresholds)):
        for j in range(len(delta_d_thresholds)):
            axes[2].text(j, i, f"{grid_miss[i, j]:.1f}%", ha="center", va="center", color="black", fontweight="bold")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    fig_path = "analysis/results/dynamic_threshold_sensitivity.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 3-Panel Figure A1 plot to: {fig_path}\n")


if __name__ == "__main__":
    pass