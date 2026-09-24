"""
Failure Case Selector & Mining Framework for Qualitative Visualization

Input:
    analysis/results/agent_physics_features.json

Output:
    analysis/results/selected_failure_cases.json
    analysis/results/selected_failure_cases_summary.txt

Key Refinement Highlights:
    1. Physics-Driven Weighting: S = 0.55 * Δd + 0.35 * Displacement + 0.10 * HeadingChange.
    2. Explicit Category Terminology: Refined taxonomy to 'future_interaction_emerging',
       'maneuvering_failure', and 'complex_interaction'.
    3. Safe Type Parsing: Defensive safe_float conversion with try-except for robust JSON handling.
    4. Unambiguous Percentile Benchmark: Clarified percentile metric semantics (Lower = Higher Severity).
    5. Scenario Diversity & Rank Bucket: Limits to MAX_PER_SCENARIO = 2, attaches 'rank_bucket' metadata.
"""

import os
import json
import numpy as np

INPUT_JSON = "analysis/results/agent_physics_features.json"
OUTPUT_JSON = "analysis/results/selected_failure_cases.json"
OUTPUT_TXT = "analysis/results/selected_failure_cases_summary.txt"

PRIMARY_RANK_MIN = 20
PRIMARY_RANK_MAX = 30

MIN_DELTA_DISTANCE = 10.0  # meters
MIN_FUTURE_DISPLACEMENT = 15.0  # meters
MIN_HEADING_CHANGE = 0.20  # radians (~11.5 degrees)

TOP_K_CASES = 25
MAX_PER_SCENARIO = 2


def safe_float(val, default=0.0):
    """Robust conversion to float with fallback for None, NaN, or string 'nan'."""
    try:
        if val is None:
            return default
        val_float = float(val)
        if np.isnan(val_float):
            return default
        return val_float
    except (ValueError, TypeError):
        return default


def normalize_array(arr):
    clean_vals = [safe_float(x, 0.0) for x in arr]
    arr_np = np.array(clean_vals, dtype=np.float64)
    if len(arr_np) == 0:
        return np.array([], dtype=np.float64)
    min_val, max_val = np.min(arr_np), np.max(arr_np)
    if max_val - min_val == 0:
        return np.zeros_like(arr_np)
    return (arr_np - min_val) / (max_val - min_val)


def compute_severity_score(delta_d_norm, disp_norm, heading_norm):
    # Weight formula: 55% Δd + 35% Displacement + 10% Heading Change
    return (0.55 * delta_d_norm) + (0.35 * disp_norm) + (0.10 * heading_norm)


def compute_percentile_stats(boundary_scores, selected_scores):
    if not boundary_scores or not selected_scores:
        return {"mean_percentile": 0.0, "best_percentile": 0.0, "worst_percentile": 0.0}

    sorted_scores = sorted(boundary_scores, reverse=True)
    total_cnt = len(sorted_scores)

    percentiles = []
    for s in selected_scores:
        # Rank 1 = highest severity -> rank_pos / total_cnt gives the top percentile
        rank_pos = sum(1 for x in sorted_scores if x >= s)
        pct = (rank_pos / total_cnt) * 100.0
        percentiles.append(pct)

    return {
        "mean_percentile": float(np.mean(percentiles)),
        "best_percentile": float(np.min(percentiles)),
        "worst_percentile": float(np.max(percentiles))
    }


def run_random_baseline_benchmark(boundary_missed, selected_scores, sample_size=25, num_trials=1000):
    if len(boundary_missed) < sample_size:
        return {"random_mean_severity": 0.0, "mining_gain_multiplier": 0.0}, {}

    delta_ds = [a.get("delta_distance") for a in boundary_missed]
    disps = [a.get("future_displacement") for a in boundary_missed]
    headings = [a.get("heading_change") for a in boundary_missed]

    norm_delta = normalize_array(delta_ds)
    norm_disp = normalize_array(disps)
    norm_heading = normalize_array(headings)

    all_boundary_scores = [
        compute_severity_score(norm_delta[i], norm_disp[i], norm_heading[i])
        for i in range(len(boundary_missed))
    ]

    np.random.seed(42)
    random_means = []
    for _ in range(num_trials):
        rand_idx = np.random.choice(len(all_boundary_scores), size=sample_size, replace=False)
        random_means.append(np.mean([all_boundary_scores[i] for i in rand_idx]))

    avg_random_severity = float(np.mean(random_means))
    avg_selected_severity = float(np.mean(selected_scores)) if len(selected_scores) > 0 else 0.0
    multiplier = (avg_selected_severity / avg_random_severity) if avg_random_severity > 0 else 0.0

    percentile_stats = compute_percentile_stats(all_boundary_scores, selected_scores)

    benchmark_results = {
        "random_mean_severity": avg_random_severity,
        "selected_mean_severity": avg_selected_severity,
        "mining_gain_multiplier": multiplier,
        "num_trials": num_trials,
        "benchmark_type": "Random Boundary Sampling (Rank 20-30)",
        "percentile_stats": percentile_stats
    }

    return benchmark_results, percentile_stats


def select_cases(data):
    missed_agents = data.get("missed_critical_agents", [])
    used_fallback = False

    # Primary pool: Rank 20-30
    boundary_missed = [
        a for a in missed_agents
        if a.get("rank") is not None and PRIMARY_RANK_MIN <= a.get("rank") <= PRIMARY_RANK_MAX
    ]

    candidates = []
    for agent in boundary_missed:
        delta_d = safe_float(agent.get("delta_distance"))
        disp = safe_float(agent.get("future_displacement"))
        heading = safe_float(agent.get("heading_change"))

        is_emerging = (delta_d >= MIN_DELTA_DISTANCE) and (disp >= MIN_FUTURE_DISPLACEMENT)
        is_maneuvering = (heading >= MIN_HEADING_CHANGE)

        if is_emerging or is_maneuvering:
            if is_maneuvering and is_emerging:
                reason = "complex_interaction"
            elif is_maneuvering:
                reason = "maneuvering_failure"
            else:
                reason = "future_interaction_emerging"

            agent_copy = dict(agent)
            agent_copy["failure_reason"] = reason
            agent_copy["rank_bucket"] = f"{PRIMARY_RANK_MIN}-{PRIMARY_RANK_MAX}"
            candidates.append(agent_copy)

    print(f"Filtered {len(candidates)} candidates in Rank [{PRIMARY_RANK_MIN}-{PRIMARY_RANK_MAX}] matching criteria.")

    # Fallback pool: Rank 20-50 (if primary pool produces insufficient candidates)
    if len(candidates) < TOP_K_CASES:
        print("Warning: Candidates below Top-K budget. Activating fallback (Rank 20-50)...")
        used_fallback = True
        candidates = []
        for agent in missed_agents:
            r = agent.get("rank")
            if r is not None and 20 <= r <= 50:
                delta_d = safe_float(agent.get("delta_distance"))
                disp = safe_float(agent.get("future_displacement"))
                heading = safe_float(agent.get("heading_change"))

                is_emerging = (delta_d >= MIN_DELTA_DISTANCE) and (disp >= MIN_FUTURE_DISPLACEMENT)
                is_maneuvering = (heading >= MIN_HEADING_CHANGE)

                if is_emerging or is_maneuvering:
                    if is_maneuvering and is_emerging:
                        reason = "complex_interaction"
                    elif is_maneuvering:
                        reason = "maneuvering_failure"
                    else:
                        reason = "future_interaction_emerging"

                    agent_copy = dict(agent)
                    agent_copy["failure_reason"] = reason
                    agent_copy["rank_bucket"] = "20-50"
                    candidates.append(agent_copy)

    # Calculate normalized severity score
    delta_ds = [a.get("delta_distance") for a in candidates]
    disps = [a.get("future_displacement") for a in candidates]
    headings = [a.get("heading_change") for a in candidates]

    norm_delta = normalize_array(delta_ds)
    norm_disp = normalize_array(disps)
    norm_heading = normalize_array(headings)

    for idx, agent in enumerate(candidates):
        score = compute_severity_score(norm_delta[idx], norm_disp[idx], norm_heading[idx])
        agent["severity_score"] = float(score)

    candidates.sort(key=lambda x: x["severity_score"], reverse=True)

    # Apply scenario diversity limit (MAX_PER_SCENARIO = 2)
    selected = []
    scenario_counts = {}

    for agent in candidates:
        scen_id = agent.get("scenario_token") or agent.get("scene_id", "unknown")
        cnt = scenario_counts.get(scen_id, 0)

        if cnt < MAX_PER_SCENARIO:
            selected.append(agent)
            scenario_counts[scen_id] = cnt + 1

        if len(selected) >= TOP_K_CASES:
            break

    # Calculate category coverage distribution
    category_counts = {}
    for case in selected:
        cat = case.get("failure_reason", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    selected_scores = [a["severity_score"] for a in selected]
    random_benchmark, percentile_stats = run_random_baseline_benchmark(
        boundary_missed, selected_scores, sample_size=len(selected)
    )

    return selected, category_counts, random_benchmark, percentile_stats, used_fallback, len(boundary_missed)


def generate_summary_text(selected_cases, category_counts, benchmark, percentile_stats, used_fallback,
                          total_boundary_cnt):
    lines = []
    lines.append("==========================================================================================")
    lines.append("             FAILURE CASE MINING & SEVERITY BENCHMARK REPORT                              ")
    lines.append("==========================================================================================\n")
    lines.append(f"Target Boundary Pool : Rank {PRIMARY_RANK_MIN}-{PRIMARY_RANK_MAX} ({total_boundary_cnt} agents)")
    lines.append(f"Fallback Used        : {used_fallback}")
    lines.append(f"Mining Selection Filter: Future Interaction Emerging OR Maneuvering Failure (>= 0.20 rad)")
    lines.append(f"Severity Score Formula : 0.55 * Δd + 0.35 * Displacement + 0.10 * HeadingChange\n")

    lines.append("--- CATEGORY COVERAGE DISTRIBUTION (Top-25 Selected Cases) ---")
    for cat, count in category_counts.items():
        pct = (count / len(selected_cases) * 100.0) if len(selected_cases) > 0 else 0.0
        lines.append(f"  * {cat:<30s}: {count:<3d} ({pct:.1f}%)")

    lines.append("\n--- MINING EFFICIENCY & PERCENTILE BENCHMARK ---")
    lines.append(f"Selected Top-{len(selected_cases)} Mean Severity Score : {benchmark['selected_mean_severity']:.4f}")
    lines.append(f"Random Boundary Sampling Mean Severity : {benchmark['random_mean_severity']:.4f}")
    lines.append(
        f"🔥 MINING GAIN MULTIPLIER                      : {benchmark['mining_gain_multiplier']:.2f}x Higher Severity")
    lines.append(
        f"💡 SEVERITY PERCENTILE (Lower = Higher Severity): Mean Percentile = {percentile_stats.get('mean_percentile', 0.0):.2f}% "
        f"[Worst Case: {percentile_stats.get('worst_percentile', 0.0):.2f}%]\n")

    lines.append("--- SELECTED REPRESENTATIVE CASES FOR VISUALIZATION ---")
    header = f"{'Idx':<4s} | {'Scenario Token':<18s} | {'Rank':<5s} | {'Bucket':<7s} | {'Failure Category':<28s} | {'Δd(m)':<7s} | {'Disp(m)':<7s} | {'Severity':<8s}"
    lines.append(header)
    lines.append("-" * len(header))

    for idx, case in enumerate(selected_cases, 1):
        scen = str(case.get("scenario_token", "unknown"))[:16]
        rank = case.get("rank", -1)
        bucket = case.get("rank_bucket", f"{PRIMARY_RANK_MIN}-{PRIMARY_RANK_MAX}")
        reason = case.get("failure_reason", "unknown")
        dd = safe_float(case.get("delta_distance"))
        disp = safe_float(case.get("future_displacement"))
        score = case.get("severity_score", 0.0)

        lines.append(
            f"{idx:<4d} | {scen:<18s} | {rank:<5d} | {bucket:<7s} | {reason:<28s} | {dd:<7.1f} | {disp:<7.1f} | {score:<8.3f}"
        )

    lines.append("\n==========================================================================================")
    return "\n".join(lines)


def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input file not found: {INPUT_JSON}")

    print(f"Loading agent physics features from {INPUT_JSON}...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    print("Executing failure case mining and random benchmark evaluation...")
    selected_cases, category_counts, benchmark, percentile_stats, used_fallback, total_boundary_cnt = select_cases(data)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    output_payload = {
        "mining_metadata": {
            "total_selected": len(selected_cases),
            "total_boundary_agents_analyzed": total_boundary_cnt,
            "target_rank_range": f"{PRIMARY_RANK_MIN}-{PRIMARY_RANK_MAX}",
            "fallback_used": used_fallback,
            "max_per_scenario": MAX_PER_SCENARIO,
            "severity_weights": {"delta_distance": 0.55, "displacement": 0.35, "heading_change": 0.10},
            "selected_category_distribution": category_counts,
            "mining_benchmark": benchmark
        },
        "selected_cases": selected_cases
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_payload, f, indent=2)

    summary_text = generate_summary_text(selected_cases, category_counts, benchmark, percentile_stats, used_fallback,
                                         total_boundary_cnt)
    with open(OUTPUT_TXT, "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f"\n✅ Failure case mining completed successfully!")
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved TXT : {OUTPUT_TXT}")


if __name__ == "__main__":
    main()