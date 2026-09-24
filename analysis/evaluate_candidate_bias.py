"""
Batch Candidate Selection Bias Evaluator (Publication-Ready & Defensible)
Path: analysis/evaluate_candidate_bias.py

Purpose:
Diagnostic probe evaluating historical candidate truncation bias in autonomous driving datasets.
Extracts critical rank positions AND detailed agent physical kinematics for failure mechanism analysis.

Metrics:
- Coverage@K: Recall of temporally emerging interaction partners
- Overlap@K:  Ranking Consistency with Future Proximity Reference (Top-K spatial overlap)
- Agent Details: Kinematic features (d_hist, d_future_min, speeds, yaw changes) for physics diagnosis.

Outputs:
- Export: analysis/results/candidate_bias_per_scene.json
"""

import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import spearmanr

sys.path.append(os.path.abspath("."))

from Wayformer.utils import extract_agent_feature, wrap_angle
try:
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
except ImportError:
    from nuplan.planning.scenario_builder.abstract_tracked_objects import TrackedObjectType


def identify_emerging_interaction_agents(hist_dist_map, d_fut_map, traveled_dist_map, valid_oracle_tokens):
    """
    Constructs a diagnostic set of temporally emerging interaction partners.
    Rule: d_future < 10m, d_hist - d_future > 1m, travel_distance > 5m.
    """
    emerging_interaction_tokens = set()
    for token in valid_oracle_tokens:
        d_h = hist_dist_map[token]
        d_f = d_fut_map[token]
        L_i = traveled_dist_map[token]

        if d_f < 10.0 and (d_h - d_f) > 1.0 and L_i > 5.0:
            emerging_interaction_tokens.add(token)

    return emerging_interaction_tokens


def evaluate_scenario_candidate_bias(scenario, iteration=0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    # 1. Coordinate Anchor (t=0 Ego Local Frame)
    ego_state_0 = scenario.get_ego_state_at_iteration(iteration)
    ego_x0, ego_y0 = ego_state_0.rear_axle.x, ego_state_0.rear_axle.y
    angle0 = ego_state_0.rear_axle.heading

    rot = -angle0 + np.pi / 2.0
    c, s = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[c, -s], [s, c]])

    # 2. Extract Past 20 Frames & Candidate Pool M
    past_objs = list(scenario.get_past_tracked_objects(iteration=iteration, time_horizon=2.0, num_samples=20))
    past_ts = list(scenario.get_past_timestamps(iteration=iteration, time_horizon=2.0, num_samples=20))

    track_token_ids = {}
    agents_per_ts = []
    flat_ts = [t.time_s - past_ts[0].time_s for t in past_ts]
    types = [TrackedObjectType.PEDESTRIAN, TrackedObjectType.VEHICLE, TrackedObjectType.BICYCLE]
    agent_type_map = {}

    for i, tracked_objs in enumerate(past_objs):
        tensorized, track_token_ids = extract_agent_feature(types, tracked_objs, track_token_ids, flat_ts[i])
        agents_per_ts.append(tensorized)

        for obj in tracked_objs.tracked_objects:
            agent_type_map[obj.track_token] = obj.tracked_object_type

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
    M = len(candidate_tokens)

    if M == 0:
        return None

    dist2_hist = (candidates[:, :, 0:2] ** 2).sum(dim=-1)
    min_d_hist = torch.sqrt(dist2_hist.min(dim=1)[0]).numpy()
    hist_dist_map = {token: min_d_hist[i] for i, token in enumerate(candidate_tokens)}

    # 3. Extract Future 60 Frames & Construct Shared Universe M_oracle
    fut_objs_list = list(scenario.get_future_tracked_objects(iteration=iteration, time_horizon=6.0, num_samples=60))
    d_fut_map = {}
    traveled_dist_map = {}
    agent_details = {}

    for idx, token in enumerate(candidate_tokens):
        fut_dists = []
        fut_pts_global = []
        fut_headings = []

        for t_idx in range(len(fut_objs_list)):
            f_objs = fut_objs_list[t_idx]
            match = next((obj for obj in f_objs.tracked_objects if obj.track_token == token), None)
            if match is not None:
                dx = match.center.x - ego_x0
                dy = match.center.y - ego_y0
                local_xy = rot_mat @ np.array([dx, dy])
                fut_dists.append(np.linalg.norm(local_xy))
                fut_pts_global.append([match.center.x, match.center.y])
                fut_headings.append(match.center.heading)

        v_rate = len(fut_dists) / 60.0

        if v_rate >= 0.8:
            d_fut_map[token] = float(np.min(fut_dists))
            pts = np.array(fut_pts_global)
            traveled_dist_map[token] = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0

            # ===============================
            # Physical feature extraction
            # ===============================
            future_displacement = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 0 else 0.0
            future_avg_speed = float(traveled_dist_map[token] / 6.0)

            if len(fut_headings) > 1:
                heading_change = float(abs(wrap_angle(fut_headings[-1] - fut_headings[0])))
            else:
                heading_change = 0.0

            # Historical speed
            hist_positions = candidates[idx, :, 0:2].numpy()
            hist_disp = np.linalg.norm(hist_positions[-1] - hist_positions[0])
            hist_speed = float(hist_disp / 2.0)

            agent_type_enum = agent_type_map.get(token)
            agent_type_str = "vehicle" if agent_type_enum == TrackedObjectType.VEHICLE else "pedestrian_bicycle"

            agent_details[token] = {
                "type": agent_type_str,
                "d_hist": float(hist_dist_map[token]),
                "d_future_min": float(d_fut_map[token]),
                "future_displacement": future_displacement,
                "hist_speed": hist_speed,
                "future_avg_speed": future_avg_speed,
                "heading_change": heading_change
            }

    valid_oracle_tokens = [t for t in candidate_tokens if t in d_fut_map]
    M_oracle_count = len(valid_oracle_tokens)

    if M_oracle_count == 0:
        return None

    K = min(19, M_oracle_count)

    # 4. Diagnostic Probe Analysis: Temporally Emerging Interaction Partners
    emerging_interaction_tokens = identify_emerging_interaction_agents(hist_dist_map, d_fut_map, traveled_dist_map, valid_oracle_tokens)
    critical_veh = {t for t in emerging_interaction_tokens if agent_type_map.get(t) == TrackedObjectType.VEHICLE}
    critical_ped_bic = {
        t for t in emerging_interaction_tokens
        if agent_type_map.get(t) in [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    }

    # 5. Selector Rankings inside M_oracle Universe
    # 1) Historical Selector
    d_hist_oracle_subset = [hist_dist_map[t] for t in valid_oracle_tokens]
    hist_sort_idx = np.argsort(d_hist_oracle_subset)
    hist_ranked_tokens = [valid_oracle_tokens[idx] for idx in hist_sort_idx]

    critical_rank_positions = {}
    for rank, token in enumerate(hist_ranked_tokens):
        if token in emerging_interaction_tokens:
            critical_rank_positions[token] = rank + 1  # 1-indexed

    topK_hist_tokens = set(hist_ranked_tokens[:K])

    # 2) Future Proximity Reference Selector
    d_fut_oracle_subset = [d_fut_map[t] for t in valid_oracle_tokens]
    oracle_sort_idx = np.argsort(d_fut_oracle_subset)
    topK_future_proxy_tokens = set(valid_oracle_tokens[idx] for idx in oracle_sort_idx[:K])

    # 3) Random Baseline Selector (N_MC = 1000)
    N_MC = 1000
    rand_overlaps = []
    rand_coverages = []

    for _ in range(N_MC):
        rand_sort_idx = rng.choice(M_oracle_count, size=K, replace=False)
        topK_rand_tokens = set(valid_oracle_tokens[idx] for idx in rand_sort_idx)

        rand_overlaps.append((len(topK_rand_tokens & topK_future_proxy_tokens) / float(K)) * 100.0)
        if len(emerging_interaction_tokens) > 0:
            rand_coverages.append((len(topK_rand_tokens & emerging_interaction_tokens) / len(emerging_interaction_tokens)) * 100.0)

    mean_random_overlap = np.mean(rand_overlaps)
    std_random_overlap = np.std(rand_overlaps)

    if len(emerging_interaction_tokens) > 0:
        mean_random_coverage = np.mean(rand_coverages)
        std_random_coverage = np.std(rand_coverages)
    else:
        mean_random_coverage = np.nan
        std_random_coverage = np.nan

    # 6. Metric Computation & Direct Rates
    missed_emerging_hist = emerging_interaction_tokens - topK_hist_tokens
    missed_veh_hist = critical_veh - topK_hist_tokens
    missed_ped_bic_hist = critical_ped_bic - topK_hist_tokens

    hist_cov_count = len(topK_hist_tokens & emerging_interaction_tokens)
    total_emerging_count = len(emerging_interaction_tokens)

    if total_emerging_count > 0:
        critical_coverage_rate = (hist_cov_count / float(total_emerging_count)) * 100.0
        critical_miss_rate = 100.0 - critical_coverage_rate
    else:
        critical_coverage_rate = np.nan
        critical_miss_rate = np.nan

    overlap_at_k = (len(topK_hist_tokens & topK_future_proxy_tokens) / float(K)) * 100.0

    if M_oracle_count > 1:
        spearman_rho = float(spearmanr(d_hist_oracle_subset, d_fut_oracle_subset).statistic)
    else:
        spearman_rho = 1.0

    return {
        "M": M,
        "M_oracle": M_oracle_count,
        "K": K,
        "critical_rank_positions": critical_rank_positions,
        "agent_details": agent_details,
        "num_critical_with_rank": len(critical_rank_positions),
        "total_critical": total_emerging_count,
        "critical_veh": len(critical_veh),
        "critical_ped_bic": len(critical_ped_bic),
        "missed_critical_hist": len(missed_emerging_hist),
        "missed_veh_hist": len(missed_veh_hist),
        "missed_ped_bic_hist": len(missed_ped_bic_hist),
        "hist_cov_count": hist_cov_count,
        "hist_veh_cov_count": len(topK_hist_tokens & critical_veh),
        "hist_ped_bic_cov_count": len(topK_hist_tokens & critical_ped_bic),
        "critical_coverage_rate": critical_coverage_rate,
        "critical_miss_rate": critical_miss_rate,
        "mean_random_overlap": mean_random_overlap,
        "std_random_overlap": std_random_overlap,
        "mean_random_coverage": mean_random_coverage,
        "std_random_coverage": std_random_coverage,
        "overlap_at_k": overlap_at_k,
        "spearman_rho": spearman_rho,
        "scenario_token": scenario.token,
        "missed_tokens": list(missed_emerging_hist),
        "critical_tokens": list(emerging_interaction_tokens),
    }


def run_batch_evaluation(scenarios, save_json_path="analysis/results/candidate_bias_per_scene.json"):
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(42)

    results = []
    print(f"\nEvaluating Candidate Bias across {len(scenarios)} Scenarios...")

    for sc in tqdm(scenarios):
        res = evaluate_scenario_candidate_bias(sc, rng=rng)
        if res is not None:
            results.append(res)

    if not results:
        print("No valid scenarios evaluated.")
        return

    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        export_results = []
        for r in results:
            clean_dict = {
                k: v.item() if isinstance(v, np.generic) else v
                for k, v in r.items()
            }
            export_results.append(clean_dict)

        with open(save_json_path, "w") as f:
            json.dump(export_results, f, indent=2)
        print(f"\nSaved per-scenario results to: {save_json_path}")


if __name__ == "__main__":
    import utils.config as config

    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    from Wayformer.wf_dataset import (
        get_scenario_map,
        get_filter_parameters
    )

    print("Building scenarios...")

    scenario_mapping = ScenarioMapping(
        scenario_map=get_scenario_map(),
        subsample_ratio_override=0.5
    )

    builder = NuPlanScenarioBuilder(
        data_root=config.DATA_PATH,
        map_root=config.MAP_PATH,
        sensor_root=None,
        db_files=None,
        map_version=config.MAP_VERSION,
        scenario_mapping=scenario_mapping
    )

    print("Filtering scenarios...")

    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            config.SCENARIOS_PER_TYPE,
            None,
            False
        )
    )

    worker = Sequential()

    scenarios = builder.get_scenarios(
        scenario_filter,
        worker
    )

    print(f"Loaded {len(scenarios)} scenarios")

    run_batch_evaluation(scenarios)