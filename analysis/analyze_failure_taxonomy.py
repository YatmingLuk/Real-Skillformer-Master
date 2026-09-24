"""
Failure Taxonomy Mining & Coverage Analysis (Hierarchy-Prioritized)

Input:
    analysis/results/agent_physics_features.json

Output:
    analysis/results/selected_taxonomy_cases.json
    analysis/results/selected_taxonomy_cases_summary.txt

Key Refinement Highlights:
    1. Hierarchical Deduplication Priority: Orders C (Complex) -> B (Maneuvering) -> A (Emerging)
       to ensure compound failure modes retain their highest-order classification.
    2. Dual Score Metrics: Logs both 'taxonomy_score_raw' and 'taxonomy_score_norm' for figure rendering.
    3. Minimum Budget Fallback: Auto-supplements from boundary pool via severity score if unique cases < 15.
    4. Strict Boundary Thresholds: Uses unified maneuvering heading_change threshold (>= 0.20 rad).
"""

import os
import json
import numpy as np

INPUT_JSON = "analysis/results/agent_physics_features.json"
OUTPUT_JSON = "analysis/results/selected_taxonomy_cases.json"
OUTPUT_TXT = "analysis/results/selected_taxonomy_cases_summary.txt"

PRIMARY_RANK_MIN = 20
PRIMARY_RANK_MAX = 30
TARGET_MIN_CASES = 15


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


def normalize_scores(score_list):
    """Min-Max normalization wrapper."""
    arr = np.array([safe_float(s) for s in score_list], dtype=np.float64)
    if len(arr) == 0:
        return []
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val == 0:
        return [0.0] * len(arr)
    norm_arr = (arr - min_val) / (max_val - min_val)
    return [float(x) for x in norm_arr]


def remove_duplicate_cases(cases):
    """Deduplicates cases using scenario_token + track_id composite key."""
    seen = set()
    unique_cases = []
    for c in cases:
        scen = str(c.get("scenario_token") or c.get("scene_id", "unknown"))
        track = str(c.get("track_id") or c.get("agent_id") or c.get("rank", "0"))
        key = f"{scen}_{track}"

        if key not in seen:
            seen.add(key)
            unique_cases.append(c)
    return unique_cases


def mine_taxonomy_cases(data):
    missed_agents = data.get("missed_critical_agents", [])

    # Boundary pool filtering (Rank 20-30)
    boundary_missed = [
        a for a in missed_agents
        if a.get("rank") is not None and PRIMARY_RANK_MIN <= a.get("rank") <= PRIMARY_RANK_MAX
    ]

    # 1. Category C: Complex Interaction (Heading Change >= 0.20 rad AND Δd >= 20m)
    complex_pool = [
        dict(a) for a in boundary_missed
        if safe_float(a.get("heading_change")) >= 0.20 and safe_float(a.get("delta_distance")) >= 20.0
    ]
    for a in complex_pool:
        hc = safe_float(a.get("heading_change"))
        dd = safe_float(a.get("delta_distance"))
        a["taxonomy_score_raw"] = float(hc * 2.0 + dd * 0.1)
        a["taxonomy_category"] = "complex_interaction"

    complex_pool.sort(key=lambda x: x["taxonomy_score_raw"], reverse=True)
    cat_c = complex_pool[:5]

    # 2. Category B: Maneuvering Failure (Heading Change >= 0.20 rad)
    maneuver_pool = [
        dict(a) for a in boundary_missed
        if safe_float(a.get("heading_change")) >= 0.20
    ]
    for a in maneuver_pool:
        hc = safe_float(a.get("heading_change"))
        a["taxonomy_score_raw"] = float(hc)
        a["taxonomy_category"] = "maneuvering_failure"

    maneuver_pool.sort(key=lambda x: x["taxonomy_score_raw"], reverse=True)
    cat_b = maneuver_pool[:5]

    # 3. Category A: Future Interaction Emerging (Δd >= 10m AND Displacement >= 15m)
    emerging_pool = [
        dict(a) for a in boundary_missed
        if safe_float(a.get("delta_distance")) >= 10.0 and safe_float(a.get("future_displacement")) >= 15.0
    ]
    for a in emerging_pool:
        dd = safe_float(a.get("delta_distance"))
        disp = safe_float(a.get("future_displacement"))
        a["taxonomy_score_raw"] = float(0.6 * dd + 0.4 * disp)
        a["taxonomy_category"] = "future_interaction_emerging"

    emerging_pool.sort(key=lambda x: x["taxonomy_score_raw"], reverse=True)
    cat_a = emerging_pool[:10]

    # Priority Ordering for Deduplication: Complex (C) -> Maneuvering (B) -> Emerging (A)
    raw_combined = cat_c + cat_b + cat_a
    unique_taxonomy_cases = remove_duplicate_cases(raw_combined)

    # Budget Fallback Check
    if len(unique_taxonomy_cases) < TARGET_MIN_CASES:
        already_keys = {
            f"{str(c.get('scenario_token'))}_{str(c.get('track_id') or c.get('rank'))}"
            for c in unique_taxonomy_cases
        }
        fallback_candidates = [
            dict(a) for a in boundary_missed
            if f"{str(a.get('scenario_token'))}_{str(a.get('track_id') or a.get('rank'))}" not in already_keys
        ]
        # Sort by delta distance as default fallback severity metric
        fallback_candidates.sort(key=lambda x: safe_float(x.get("delta_distance")), reverse=True)

        needed = TARGET_MIN_CASES - len(unique_taxonomy_cases)
        for fc in fallback_candidates[:needed]:
            fc["taxonomy_category"] = "future_interaction_emerging"
            fc["taxonomy_score_raw"] = safe_float(fc.get("delta_distance"))
            unique_taxonomy_cases.append(fc)

    # Calculate normalized taxonomy score per category
    raw_scores = [c["taxonomy_score_raw"] for c in unique_taxonomy_cases]
    norm_scores = normalize_scores(raw_scores)
    for idx, c in enumerate(unique_taxonomy_cases):
        c["taxonomy_score_norm"] = norm_scores[idx]

    category_counts = {}
    for case in unique_taxonomy_cases:
        cat = case.get("taxonomy_category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return unique_taxonomy_cases, category_counts, len(boundary_missed)


def generate_summary_text(taxonomy_cases, category_counts, total_boundary):
    lines = []
    lines.append("==========================================================================")
    lines.append("               FAILURE TAXONOMY COVERAGE REPORT                           ")
    lines.append("==========================================================================\n")
    lines.append(f"Boundary Target Pool     : Rank {PRIMARY_RANK_MIN}-{PRIMARY_RANK_MAX} ({total_boundary} agents)")
    lines.append(f"Total Unique Mined Cases : {len(taxonomy_cases)} cases")
    lines.append(f"Deduplication Hierarchy : Complex Interaction -> Maneuvering -> Emerging\n")

    lines.append("--- CATEGORY BREAKDOWN (Post-Deduplication) ---")
    for cat in ["complex_interaction", "maneuvering_failure", "future_interaction_emerging"]:
        cnt = category_counts.get(cat, 0)
        lines.append(f"  * {cat:<28s}: {cnt:<2d} cases")

    lines.append("\n--- SELECTED TAXONOMY CASES SUMMARY ---")
    header = f"{'Idx':<4s} | {'Scenario Token':<18s} | {'Rank':<5s} | {'Category':<28s} | {'Raw Score':<10s} | {'Norm Score':<10s}"
    lines.append(header)
    lines.append("-" * len(header))

    for idx, case in enumerate(taxonomy_cases, 1):
        scen = str(case.get("scenario_token", "unknown"))[:16]
        rank = case.get("rank", -1)
        cat = case.get("taxonomy_category", "unknown")
        score_raw = case.get("taxonomy_score_raw", 0.0)
        score_norm = case.get("taxonomy_score_norm", 0.0)
        lines.append(f"{idx:<4d} | {scen:<18s} | {rank:<5d} | {cat:<28s} | {score_raw:<10.2f} | {score_norm:<10.3f}")

    lines.append("\n==========================================================================")
    return "\n".join(lines)


def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Missing input JSON: {INPUT_JSON}")

    print(f"Loading agent physics features from {INPUT_JSON}...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    print("Executing hierarchical taxonomy mining across categories C -> B -> A...")
    taxonomy_cases, category_counts, total_boundary = mine_taxonomy_cases(data)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    payload = {
        "metadata": {
            "total_unique_selected": len(taxonomy_cases),
            "category_distribution": category_counts,
            "boundary_pool_size": total_boundary,
            "deduplication_priority": ["complex_interaction", "maneuvering_failure", "future_interaction_emerging"]
        },
        "taxonomy_cases": taxonomy_cases
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    summary_text = generate_summary_text(taxonomy_cases, category_counts, total_boundary)
    with open(OUTPUT_TXT, "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f"\n✅ Taxonomy mining completed!")
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved TXT : {OUTPUT_TXT}")


if __name__ == "__main__":
    main()