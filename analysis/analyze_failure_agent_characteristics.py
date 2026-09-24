"""
Comprehensive Failure Mechanism Diagnosis, Hypothesis Testing, Taxonomy & Correlation Pipeline

Input:
    analysis/results/agent_physics_features.json

Output:
    analysis/results/failure_agent_characteristics.json
    analysis/results/failure_agent_characteristics.txt

Key Refinements:
    1. Noise-filtered Taxonomy: Distinguishes 'maneuvering_complex' (Cut-in/Turning + Motion)
       from 'maneuvering_only' (Low-motion heading noise).
    2. Rank Semantic Alignment: Explicitly maps Spearman rho (+) to 'Lower Priority Rank -> Higher Future Dynamics'.
    3. Cohen's d Effect Magnitude: Annotates Effect Size (Large >= 0.8, Medium >= 0.5, Small >= 0.2).
    4. Safe Formatting: Robust None-type checking for statistical outputs.
"""

import os
import json
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

INPUT_JSON = "analysis/results/agent_physics_features.json"
OUTPUT_JSON = "analysis/results/failure_agent_characteristics.json"
OUTPUT_TXT = "analysis/results/failure_agent_characteristics.txt"

FEATURE_KEYS = [
    "d_hist",
    "d_future_min",
    "delta_distance",
    "future_displacement",
    "hist_speed",
    "future_avg_speed",
    "heading_change"
]

TAXONOMY_THRESHOLDS = {
    "emerging_delta_d": 10.0,  # meters (Significant approach)
    "high_displacement": 15.0,  # meters (High movement span)
    "maneuvering_heading": 0.15  # radians (~8.6 degrees heading change)
}

RANK_BUCKETS = [
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-19", 11, 19),  # Captured budget limit
    ("20-30", 20, 30),  # Marginal boundary (Crucial)
    ("31-50", 31, 50),  # Far boundary
    ("50+", 51, 999999)  # Deeply missed
]


def safe_array(arr):
    return np.array([x for x in arr if x is not None and not np.isnan(x)], dtype=np.float64)


def compute_stats(arr):
    clean_arr = safe_array(arr)
    if len(clean_arr) == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "iqr": 0.0}
    q25, median, q75 = np.percentile(clean_arr, [25, 50, 75])
    return {
        "count": int(len(clean_arr)),
        "mean": float(np.mean(clean_arr)),
        "std": float(np.std(clean_arr)),
        "median": float(median),
        "iqr": float(q75 - q25)
    }


def get_effect_size_magnitude(d_val):
    """Categorizes Cohen's d into standard effect size thresholds."""
    if d_val is None:
        return "N/A"
    abs_d = abs(d_val)
    if abs_d >= 0.8:
        return "Large"
    elif abs_d >= 0.5:
        return "Medium"
    elif abs_d >= 0.2:
        return "Small"
    else:
        return "Negligible"


def compute_cohens_d_missed_first(missed_arr, other_arr):
    x, y = safe_array(missed_arr), safe_array(other_arr)
    if len(x) < 2 or len(y) < 2:
        return 0.0, "neutral"
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))

    if pooled_std == 0:
        return 0.0, "neutral"

    d_val = float((np.mean(x) - np.mean(y)) / pooled_std)
    direction = "missed_higher" if d_val > 0 else "missed_lower"
    return d_val, direction


def run_test_pair(missed_vals, comparison_vals):
    if len(missed_vals) == 0 or len(comparison_vals) == 0:
        return {"u_stat": None, "p_value": None, "cohens_d": 0.0, "direction": "none", "is_significant_001": False}

    stat, p_val = mannwhitneyu(missed_vals, comparison_vals, alternative='two-sided')
    d_val, direction = compute_cohens_d_missed_first(missed_vals, comparison_vals)

    return {
        "u_stat": float(stat),
        "p_value": float(p_val),
        "cohens_d": d_val,
        "effect_magnitude": get_effect_size_magnitude(d_val),
        "direction": direction,
        "is_significant_001": bool(p_val < 0.001)
    }


# -------------------------------------------------------------------------
# Analysis 1: Group Feature Stats & Dual Hypothesis Tests
# -------------------------------------------------------------------------
def run_group_comparison(missed, captured, background):
    summary = {
        "missed": {}, "captured": {}, "background": {},
        "tests_missed_vs_captured": {},
        "tests_missed_vs_background": {}
    }

    for key in FEATURE_KEYS:
        val_m = [a[key] for a in missed if a.get(key) is not None]
        val_c = [a[key] for a in captured if a.get(key) is not None]
        val_b = [a[key] for a in background if a.get(key) is not None]

        summary["missed"][key] = compute_stats(val_m)
        summary["captured"][key] = compute_stats(val_c)
        summary["background"][key] = compute_stats(val_b)

        summary["tests_missed_vs_captured"][key] = run_test_pair(val_m, val_c)
        summary["tests_missed_vs_background"][key] = run_test_pair(val_m, val_b)

    return summary


# -------------------------------------------------------------------------
# Analysis 2: Rank vs Physics Spearman Correlation Analysis
# -------------------------------------------------------------------------
def run_rank_correlation_analysis(missed, captured):
    all_critical = missed + captured
    correlation_results = {}

    target_features = ["delta_distance", "future_displacement", "heading_change", "d_future_min", "future_avg_speed"]

    for key in target_features:
        ranks = []
        feature_vals = []
        for agent in all_critical:
            r = agent.get("rank")
            v = agent.get(key)
            if r is not None and v is not None and not np.isnan(v):
                ranks.append(r)
                feature_vals.append(v)

        if len(ranks) > 2:
            corr, p_val = spearmanr(ranks, feature_vals)
            correlation_results[key] = {
                "sample_size": len(ranks),
                "spearman_rho": float(corr),
                "p_value": float(p_val),
                "is_significant_001": bool(p_val < 0.001)
            }

    return correlation_results


# -------------------------------------------------------------------------
# Analysis 3: Rank Boundary Analysis
# -------------------------------------------------------------------------
def run_rank_boundary_analysis(missed, captured):
    all_critical = missed + captured
    total_critical = len(all_critical)
    total_missed = len(missed)

    bucket_distribution = {}
    for name, low, high in RANK_BUCKETS:
        cnt = sum(1 for a in all_critical if a.get("rank") is not None and low <= a["rank"] <= high)
        bucket_distribution[name] = {
            "count": cnt,
            "percentage_of_all_critical": float(cnt / total_critical * 100.0) if total_critical > 0 else 0.0
        }

    missed_20_30_cnt = sum(1 for a in missed if a.get("rank") is not None and 20 <= a["rank"] <= 30)
    pct_of_missed_in_boundary = (missed_20_30_cnt / total_missed * 100.0) if total_missed > 0 else 0.0

    return {
        "total_critical_agents": total_critical,
        "total_missed_agents": total_missed,
        "bucket_distribution": bucket_distribution,
        "marginal_boundary_miss_analysis": {
            "missed_rank_20_30_count": missed_20_30_cnt,
            "percentage_of_total_missed": float(pct_of_missed_in_boundary)
        }
    }


# -------------------------------------------------------------------------
# Analysis 4: Refined Domain-Aware Failure Taxonomy
# -------------------------------------------------------------------------
def run_failure_taxonomy(missed):
    total_missed = len(missed)
    if total_missed == 0:
        return {}

    prevalence_counts = {"emerging_approaching": 0, "high_motion": 0, "maneuvering": 0}

    dominant_counts = {
        "maneuvering_complex": 0,  # Turning/Cut-in WITH active motion/approach (High Priority Interaction)
        "maneuvering_only": 0,  # Pure heading change without strong motion (Noise/Drift filter)
        "dynamic_emerging": 0,  # Both Emerging & High Motion
        "emerging_only": 0,  # Emerging only
        "high_motion_only": 0,  # High Motion only
        "low_dynamic_static": 0  # Static or low dynamic
    }

    for agent in missed:
        delta_d = agent.get("delta_distance") or 0.0
        disp = agent.get("future_displacement") or 0.0
        heading = agent.get("heading_change") or 0.0

        is_emerging = delta_d >= TAXONOMY_THRESHOLDS["emerging_delta_d"]
        is_high_motion = disp >= TAXONOMY_THRESHOLDS["high_displacement"]
        is_maneuvering = heading >= TAXONOMY_THRESHOLDS["maneuvering_heading"]

        # Track multi-label prevalence
        if is_emerging: prevalence_counts["emerging_approaching"] += 1
        if is_high_motion: prevalence_counts["high_motion"] += 1
        if is_maneuvering: prevalence_counts["maneuvering"] += 1

        # Classify dominant category with Noise-filtered Maneuvering priority
        if is_maneuvering and (is_emerging or is_high_motion):
            dominant_counts["maneuvering_complex"] += 1
        elif is_maneuvering:
            dominant_counts["maneuvering_only"] += 1
        elif is_emerging and is_high_motion:
            dominant_counts["dynamic_emerging"] += 1
        elif is_emerging:
            dominant_counts["emerging_only"] += 1
        elif is_high_motion:
            dominant_counts["high_motion_only"] += 1
        else:
            dominant_counts["low_dynamic_static"] += 1

    multi_label_res = {
        k: {"count": v, "percentage": float(v / total_missed * 100.0)}
        for k, v in prevalence_counts.items()
    }
    dominant_res = {
        k: {"count": v, "percentage": float(v / total_missed * 100.0)}
        for k, v in dominant_counts.items()
    }

    return {
        "total_missed_analyzed": total_missed,
        "thresholds": TAXONOMY_THRESHOLDS,
        "multi_label_prevalence": multi_label_res,
        "dominant_mutually_exclusive_type": dominant_res
    }


# -------------------------------------------------------------------------
# Analysis 5: Agent Type Breakdown
# -------------------------------------------------------------------------
def run_agent_type_analysis(missed, captured):
    type_stats = {}

    for agent in captured:
        a_type = agent.get("type", "unknown")
        if a_type not in type_stats:
            type_stats[a_type] = {"captured": 0, "missed": 0}
        type_stats[a_type]["captured"] += 1

    for agent in missed:
        a_type = agent.get("type", "unknown")
        if a_type not in type_stats:
            type_stats[a_type] = {"captured": 0, "missed": 0}
        type_stats[a_type]["missed"] += 1

    summary_type = {}
    for a_type, counts in type_stats.items():
        cap = counts["captured"]
        mis = counts["missed"]
        tot = cap + mis
        miss_rate = (mis / tot * 100.0) if tot > 0 else 0.0
        summary_type[a_type] = {
            "captured_count": cap,
            "missed_count": mis,
            "total_critical_count": tot,
            "miss_rate_percentage": float(miss_rate)
        }

    return summary_type


# -------------------------------------------------------------------------
# Safe String Formatting Helper
# -------------------------------------------------------------------------
def format_p_val_str(test_dict):
    p_val = test_dict.get("p_value")
    if p_val is None:
        return "N/A"
    return "< 0.001 ***" if test_dict.get("is_significant_001", False) else f"{p_val:.3f}"


# -------------------------------------------------------------------------
# TXT Report Generator
# -------------------------------------------------------------------------
def generate_report(summary, rank_corr, rank_analysis, taxonomy, agent_types):
    lines = []
    lines.append("==========================================================================================")
    lines.append("             FAILURE MECHANISM DIAGNOSIS & STATISTICAL ANALYSIS REPORT                    ")
    lines.append("==========================================================================================\n")

    # Analysis 1: Group Feature Stats
    lines.append("--- ANALYSIS 1: PHYSICAL FEATURE DISTRIBUTION (Mean ± Std) ---")
    lines.append(f"{'Feature':<20s} | {'Captured':<18s} | {'Missed':<18s} | {'Background':<18s}")
    lines.append("-" * 82)
    for key in FEATURE_KEYS:
        cap_str = f"{summary['captured'][key]['mean']:.2f} ± {summary['captured'][key]['std']:.2f}"
        mis_str = f"{summary['missed'][key]['mean']:.2f} ± {summary['missed'][key]['std']:.2f}"
        bg_str = f"{summary['background'][key]['mean']:.2f} ± {summary['background'][key]['std']:.2f}"
        lines.append(f"{key:<20s} | {cap_str:<18s} | {mis_str:<18s} | {bg_str:<18s}")

    lines.append("\n--- DUAL HYPOTHESIS TESTS & EFFECT SIZE (Cohen's d = Missed - Other) ---")
    lines.append(f"{'Feature':<20s} | {'Missed vs Captured (d)':<25s} | {'Missed vs Background (d)':<25s}")
    lines.append("-" * 76)
    for key in FEATURE_KEYS:
        tc = summary["tests_missed_vs_captured"][key]
        tb = summary["tests_missed_vs_background"][key]

        tc_p = format_p_val_str(tc)
        tb_p = format_p_val_str(tb)

        str_c = f"{tc['cohens_d']:+.3f} ({tc['effect_magnitude']}, p={tc_p})"
        str_b = f"{tb['cohens_d']:+.3f} ({tb['effect_magnitude']}, p={tb_p})"
        lines.append(f"{key:<20s} | {str_c:<25s} | {str_b:<25s}")

    lines.append("\n" + "=" * 90 + "\n")

    # Analysis 2: Spearman Correlation
    lines.append("--- ANALYSIS 2: RANK vs PHYSICS SPEARMAN CORRELATION ---")
    lines.append(
        f"{'Physics Feature':<22s} | {'Spearman Rho (ρ)':<18s} | {'p-value':<15s} | {'Semantic Interpretation':<35s}")
    lines.append("-" * 95)
    for feat, res in rank_corr.items():
        p_val = res.get("p_value")
        p_str = "< 0.001 ***" if res.get("is_significant_001", False) else (
            f"{p_val:.4f}" if p_val is not None else "N/A")
        rho = res["spearman_rho"]

        # Explicit alignment with selection priority semantics
        if rho > 0:
            interp = "Lower priority rank -> Higher future dynamics"
        else:
            interp = "Higher priority rank -> Higher future dynamics"

        lines.append(f"{feat:<22s} | {rho:<+18.4f} | {p_str:<15s} | {interp:<35s}")

    lines.append("\n" + "=" * 90 + "\n")

    # Analysis 3: Rank Boundary
    lines.append("--- ANALYSIS 3: RANK BOUNDARY ANALYSIS ---")
    lines.append(
        f"Total Critical Agents: {rank_analysis['total_critical_agents']} | Total Missed: {rank_analysis['total_missed_agents']}")
    lines.append(f"{'Rank Bucket':<15s} | {'Count':<10s} | {'% of All Critical':<20s} | {'Note':<20s}")
    lines.append("-" * 70)
    for name, _, _ in RANK_BUCKETS:
        b_data = rank_analysis["bucket_distribution"][name]
        note = "<-- K Budget Boundary" if name == "11-19" else ("<-- Marginal Boundary!" if name == "20-30" else "")
        lines.append(
            f"{name:<15s} | {b_data['count']:<10d} | {b_data['percentage_of_all_critical']:<20.2f} | {note:<20s}")

    mb = rank_analysis["marginal_boundary_miss_analysis"]
    lines.append(
        f"\n💡 KEY INSIGHT: {mb['percentage_of_total_missed']:.2f}% of all Missed Critical Agents fall in Rank 20-30!")

    lines.append("\n" + "=" * 90 + "\n")

    # Analysis 4: Failure Taxonomy
    lines.append("--- ANALYSIS 4: REFINED DOMAIN-AWARE FAILURE TAXONOMY ---")
    lines.append("[A] Multi-Label Factor Prevalence:")
    for k, v in taxonomy["multi_label_prevalence"].items():
        lines.append(f"  * {k:<25s}: {v['count']:<5d} ({v['percentage']:.2f}%)")

    lines.append("\n[B] Dominant Category (Mutually Exclusive - Noise Filtered):")
    lines.append(f"{'Category':<25s} | {'Count':<10s} | {'Percentage (%)':<15s}")
    lines.append("-" * 55)
    for k, v in taxonomy["dominant_mutually_exclusive_type"].items():
        lines.append(f"{k:<25s} | {v['count']:<10d} | {v['percentage']:.2f}")

    lines.append("\n" + "=" * 90 + "\n")

    # Analysis 5: Agent Type Breakdown
    lines.append("--- ANALYSIS 5: AGENT TYPE FAILURE BREAKDOWN ---")
    lines.append(f"{'Agent Type':<15s} | {'Captured':<10s} | {'Missed':<10s} | {'Total':<10s} | {'Miss Rate (%)':<15s}")
    lines.append("-" * 68)
    for a_type, data in agent_types.items():
        lines.append(
            f"{a_type:<15s} | {data['captured_count']:<10d} | {data['missed_count']:<10d} | "
            f"{data['total_critical_count']:<10d} | {data['miss_rate_percentage']:.2f}"
        )

    lines.append("\n==========================================================================================")
    return "\n".join(lines)


def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input file not found: {INPUT_JSON}")

    print(f"Loading agent physics features from {INPUT_JSON}...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    missed = data.get("missed_critical_agents", [])
    captured = data.get("captured_critical_agents", [])
    background = data.get("background_agents", [])

    print("Executing Analysis 1: Group Feature Stats & Dual Hypothesis Tests (with Effect Magnitudes)...")
    group_summary = run_group_comparison(missed, captured, background)

    print("Executing Analysis 2: Spearman Rank Correlation Analysis (Semantically Aligned)...")
    rank_corr = run_rank_correlation_analysis(missed, captured)

    print("Executing Analysis 3: Bug-fixed Rank Boundary Analysis...")
    rank_analysis = run_rank_boundary_analysis(missed, captured)

    print("Executing Analysis 4: Refined Noise-Filtered Failure Taxonomy...")
    taxonomy = run_failure_taxonomy(missed)

    print("Executing Analysis 5: Agent Type Breakdown...")
    agent_types = run_agent_type_analysis(missed, captured)

    full_output = {
        "group_comparison": group_summary,
        "spearman_rank_correlation": rank_corr,
        "rank_boundary_analysis": rank_analysis,
        "failure_taxonomy": taxonomy,
        "agent_type_breakdown": agent_types
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(full_output, f, indent=2)

    report_text = generate_report(group_summary, rank_corr, rank_analysis, taxonomy, agent_types)
    with open(OUTPUT_TXT, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✅ Diagnostic analysis completed successfully!")
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved TXT : {OUTPUT_TXT}")


if __name__ == "__main__":
    main()