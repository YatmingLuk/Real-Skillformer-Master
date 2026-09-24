"""
Qualitative Case Inspector & Story Finder (NeurIPS/ICLR Qualitative Ready)

Inputs:
    analysis/results/selected_taxonomy_cases.json
    analysis/results/selected_failure_cases.json

Outputs:
    analysis/results/inspected_taxonomy_cases.json  (Full inspected dataset)
    analysis/results/main_paper_cases.json         (Top 9 story-rich cases for Main Text)
    analysis/results/appendix_cases.json           (Extended pool for Appendix)
"""

import os
import json
import numpy as np

TAXONOMY_JSON = "analysis/results/selected_taxonomy_cases.json"
SEVERITY_JSON = "analysis/results/selected_failure_cases.json"

OUTPUT_INSPECTED_JSON = "analysis/results/inspected_taxonomy_cases.json"
OUTPUT_MAIN_JSON = "analysis/results/main_paper_cases.json"
OUTPUT_APPENDIX_JSON = "analysis/results/appendix_cases.json"

MAIN_CASES_PER_CATEGORY = 3  # Top 3 per category reserved for Main Paper Qualitative Figures


def safe_float(val, default=0.0):
    """Robust float conversion handling None, NaN, string 'nan'."""
    try:
        if val is None:
            return default
        val_float = float(val)
        if np.isnan(val_float):
            return default
        return val_float
    except (ValueError, TypeError):
        return default


def extract_track_key(c):
    """Robust track identifier extraction with multi-level fallbacks."""
    for key in ["track_id", "agent_id", "instance_id"]:
        val = c.get(key)
        if val is not None and str(val).strip() != "":
            return str(val)
    return "unknown_agent"


def load_severity_map(severity_path):
    severity_map = {}
    if not os.path.exists(severity_path):
        print(f"⚠️ Warning: Severity JSON not found at {severity_path}. Skipping global severity fusion.")
        return severity_map

    try:
        with open(severity_path, "r") as f:
            data = json.load(f)

        cases = data.get("selected_cases", []) or data.get("failure_cases", []) or data.get("missed_critical_agents",
                                                                                            [])
        for c in cases:
            scen = str(c.get("scenario_token") or c.get("scene_id", "unknown"))
            track = extract_track_key(c)
            key = f"{scen}_{track}"
            severity_map[key] = {
                "global_severity_score": safe_float(c.get("severity_score")),
                "global_severity_rank": c.get("rank", -1),
                "is_in_worst_severity_pool": True
            }
        print(f"🔍 [Debug] Loaded {len(severity_map)} reference severity cases from {severity_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to parse severity JSON ({e}).")

    return severity_map


def normalize_per_category(cases):
    categories = set(c.get("taxonomy_category", "unknown") for c in cases)

    for cat in categories:
        cat_cases = [c for c in cases if c.get("taxonomy_category") == cat]
        raw_scores = [
            safe_float(c.get("taxonomy_score_raw", c.get("taxonomy_score")))
            for c in cat_cases
        ]

        if not raw_scores:
            continue

        min_s, max_s = min(raw_scores), max(raw_scores)
        denom = max_s - min_s if max_s - min_s > 1e-6 else 1.0

        for c in cat_cases:
            raw = safe_float(c.get("taxonomy_score_raw", c.get("taxonomy_score")))
            c["taxonomy_score_raw"] = raw
            c["cat_norm_score"] = float((raw - min_s) / denom)

    return cases


def inspect_and_tag_cases(cases, severity_map):
    matched_count = 0
    for c in cases:
        scen = str(c.get("scenario_token") or c.get("scene_id", "unknown"))
        track = extract_track_key(c)
        key = f"{scen}_{track}"

        if key in severity_map:
            sev_info = severity_map[key]
            c["global_severity_score"] = sev_info["global_severity_score"]
            c["is_in_worst_severity_pool"] = True
            matched_count += 1
        else:
            c["global_severity_score"] = safe_float(c.get("severity_score", 0.0))
            c["is_in_worst_severity_pool"] = False

        d_fut_min = safe_float(c.get("d_future_min"), 999.0)
        dd = safe_float(c.get("delta_distance"))
        hc = safe_float(c.get("heading_change"))
        num_neighbors = int(safe_float(c.get("num_neighbors", c.get("neighbor_count", 0))))
        fde = safe_float(c.get("fde", c.get("prediction_error", c.get("ade", 0.0))))

        flags = []
        if d_fut_min <= 15.0:
            flags.append("🔥 High-Proximity Conflict")
        if dd >= 50.0:
            flags.append("🚀 Large Dynamic Escalation")
        if hc >= 0.35:
            flags.append("🔄 Sharp Maneuver")
        if num_neighbors >= 2:
            flags.append("👥 Multi-Agent Interaction")
        if fde >= 3.0:
            flags.append("⚠️ Large Prediction Error")
        if c["is_in_worst_severity_pool"]:
            flags.append("🏆 Top-Severity Case")

        c["visual_tags"] = flags

    total = len(cases)
    match_pct = (matched_count / total * 100) if total > 0 else 0.0
    print(f"🎯 [Debug] Severity fusion matched: {matched_count}/{total} cases ({match_pct:.1f}%)")
    return cases


def calculate_story_score(c):
    """
    Blends physical severity (0.7) and visual story richness (0.3).
    Ensures selected cases are both physically critical and visually illustrative for paper figures.
    """
    norm_score = safe_float(c.get("cat_norm_score"))
    tag_count = len(c.get("visual_tags", []))
    tag_richness = min(tag_count / 5.0, 1.0)
    return 0.7 * norm_score + 0.3 * tag_richness


def organize_and_prioritize_cases(cases):
    cats = ["future_interaction_emerging", "complex_interaction", "maneuvering_failure"]
    prioritized_cases = []

    for cat in cats:
        cat_cases = [c for c in cases if c.get("taxonomy_category") == cat]

        for c in cat_cases:
            c["story_score"] = float(calculate_story_score(c))

        # Sort by story_score (descending)
        cat_cases.sort(key=lambda x: safe_float(x.get("story_score")), reverse=True)

        for idx, c in enumerate(cat_cases, 1):
            c["paper_priority"] = "main" if idx <= MAIN_CASES_PER_CATEGORY else "appendix"
            prioritized_cases.append(c)

    return prioritized_cases


def print_inspection_report(taxonomy_cases):
    print("=" * 125)
    print("                      QUALITATIVE CASE INSPECTION & STORY-SELECTION REPORT                       ")
    print("=" * 125)

    cats = ["future_interaction_emerging", "complex_interaction", "maneuvering_failure"]
    main_recommendations = []

    for cat in cats:
        cat_cases = [c for c in taxonomy_cases if c.get("taxonomy_category") == cat]

        print(f"\n📂 CATEGORY: {cat.upper()} (Total: {len(cat_cases)} cases)")
        print("-" * 125)
        header = (
            f"{'Idx':<3s} | {'Scenario Token':<16s} | {'Rank':<5s} | {'Priority':<8s} | "
            f"{'StoryScore':<10s} | {'NormScore':<10s} | {'d_fut_min':<10s} | {'Visual Tags'}"
        )
        print(header)
        print("-" * 125)

        for idx, c in enumerate(cat_cases, 1):
            scen = str(c.get("scenario_token", "unknown"))[:16]
            rank = c.get("rank", -1)
            prio = c.get("paper_priority", "appendix").upper()
            story_score = safe_float(c.get("story_score"))
            norm_score = safe_float(c.get("cat_norm_score"))
            d_min = safe_float(c.get("d_future_min"))
            tags = ", ".join(c.get("visual_tags", [])) if c.get("visual_tags") else "Standard"

            print(
                f"{idx:<3d} | {scen:<16s} | {rank:<5d} | {prio:<8s} | "
                f"{story_score:<10.2f} | {norm_score:<10.2f} | {d_min:<10.1f} | {tags}"
            )

            if c.get("paper_priority") == "main":
                main_recommendations.append(c)

    print("\n" + "=" * 125)
    print(f"🌟 MAIN PAPER SELECTION ({len(main_recommendations)} Cases Selected)")
    print("=" * 125)
    for idx, c in enumerate(main_recommendations, 1):
        scen = str(c.get("scenario_token"))[:16]
        cat = c.get("taxonomy_category")
        s_score = safe_float(c.get("story_score"))
        tags = ", ".join(c.get("visual_tags", []))
        print(
            f"  {idx:02d}. [{cat:<28s}] Scenario: {scen} (Rank {c.get('rank')}) | StoryScore: {s_score:.2f} | Tags: {tags}")
    print("=" * 125 + "\n")


def main():
    if not os.path.exists(TAXONOMY_JSON):
        raise FileNotFoundError(f"Missing required input: {TAXONOMY_JSON}")

    print(f"Loading taxonomy cases from {TAXONOMY_JSON}...")
    with open(TAXONOMY_JSON, "r") as f:
        tax_data = json.load(f)

    taxonomy_cases = tax_data.get("taxonomy_cases", [])

    severity_map = load_severity_map(SEVERITY_JSON)
    taxonomy_cases = normalize_per_category(taxonomy_cases)
    taxonomy_cases = inspect_and_tag_cases(taxonomy_cases, severity_map)
    taxonomy_cases = organize_and_prioritize_cases(taxonomy_cases)

    print_inspection_report(taxonomy_cases)

    main_cases = [c for c in taxonomy_cases if c.get("paper_priority") == "main"]
    appendix_cases = [c for c in taxonomy_cases if c.get("paper_priority") == "appendix"]

    meta = tax_data.get("metadata", {})
    meta["inspected_total"] = len(taxonomy_cases)
    meta["main_paper_count"] = len(main_cases)
    meta["appendix_count"] = len(appendix_cases)

    os.makedirs(os.path.dirname(OUTPUT_INSPECTED_JSON), exist_ok=True)

    with open(OUTPUT_INSPECTED_JSON, "w") as f:
        json.dump({"metadata": meta, "taxonomy_cases": taxonomy_cases}, f, indent=2)

    with open(OUTPUT_MAIN_JSON, "w") as f:
        json.dump({"metadata": meta, "main_cases": main_cases}, f, indent=2)

    with open(OUTPUT_APPENDIX_JSON, "w") as f:
        json.dump({"metadata": meta, "appendix_cases": appendix_cases}, f, indent=2)

    print(f"✅ Inspection complete! File Outputs:")
    print(f"  * Full Set     : {OUTPUT_INSPECTED_JSON}")
    print(f"  * Main Paper   : {OUTPUT_MAIN_JSON} ({len(main_cases)} cases)")
    print(f"  * Appendix Set : {OUTPUT_APPENDIX_JSON} ({len(appendix_cases)} cases)")


if __name__ == "__main__":
    main()