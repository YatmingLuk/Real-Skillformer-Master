"""
Compute Agent Physics Features for Candidate Selection Failure Diagnosis

Input:
    analysis/results/candidate_bias_per_scene.json

Output:
    analysis/results/agent_physics_features.json
    analysis/results/agent_physics_features.txt

Purpose:
    Convert scene-level candidate bias evaluation results into an
    agent-level physics feature dataset.

    Agents are categorized into:
        1. Missed Critical Agents  (is_critical=True, rank > K)
        2. Captured Critical Agents (is_critical=True, rank <= K)
        3. Background Agents        (is_critical=False)
"""

import os
import json
import numpy as np


INPUT_JSON = "analysis/results/candidate_bias_per_scene.json"
OUTPUT_JSON = "analysis/results/agent_physics_features.json"
OUTPUT_TXT = "analysis/results/agent_physics_features.txt"


def safe_value(x):
    """
    Convert numpy values and NaN safely for JSON serialization.
    """
    if x is None:
        return None
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return None
    return float(x)


def compute_agent_feature(token, scenario_token, detail, rank=None, category=None, is_critical=False):
    """
    Build unified agent feature representation.
    Includes scenario_token for downstream visualization mining and explicit is_critical flag.
    """
    d_hist = detail.get("d_hist", None)
    d_future = detail.get("d_future_min", None)

    delta_distance = None
    if d_hist is not None and d_future is not None:
        delta_distance = d_hist - d_future

    return {
        "token": token,
        "scenario_token": scenario_token,
        "is_critical": is_critical,  # 显式布尔标记，方便后续 ML analysis
        "category": category,
        "type": detail.get("type", "unknown"),
        "rank": rank,

        # Distance features
        "d_hist": safe_value(d_hist),
        "d_future_min": safe_value(d_future),
        "delta_distance": safe_value(delta_distance),

        # Kinematic features
        "future_displacement": safe_value(detail.get("future_displacement", None)),
        "hist_speed": safe_value(detail.get("hist_speed", None)),
        "future_avg_speed": safe_value(detail.get("future_avg_speed", None)),

        # Heading change
        "heading_change": safe_value(detail.get("heading_change", None))
    }


def load_data():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Cannot find {INPUT_JSON}")
    with open(INPUT_JSON, "r") as f:
        return json.load(f)


def compute_features(results):
    missed_agents = []
    captured_agents = []
    background_agents = []

    statistics = {
        "total_scenes": len(results),
        "critical_scenes": 0,
        "missed_agents": 0,
        "captured_agents": 0,
        "background_agents": 0
    }

    for idx, scene in enumerate(results):
        K = scene.get("K", 19)
        scenario_token = scene.get("scenario_token", f"scene_{idx}")
        critical_rank_positions = scene.get("critical_rank_positions", {})
        agent_details = scene.get("agent_details", {})

        if len(critical_rank_positions) > 0:
            statistics["critical_scenes"] += 1

        # ---------------------------------------------------------
        # Sanity Check: 校验是否存在关键 Agent 丢失详细特征的情况
        # ---------------------------------------------------------
        missing_critical_tokens = set(critical_rank_positions.keys()) - set(agent_details.keys())
        if missing_critical_tokens:
            print(
                f"[Warning] Scene {scenario_token}: "
                f"Missing {len(missing_critical_tokens)} critical agents in details!"
            )

        # 单次遍历 agent_details 完成分类与特征构建
        for token, detail in agent_details.items():
            if token in critical_rank_positions:
                rank = critical_rank_positions[token]
                if rank <= K:
                    feature = compute_agent_feature(
                        token, scenario_token, detail, rank,
                        category="captured", is_critical=True
                    )
                    captured_agents.append(feature)
                    statistics["captured_agents"] += 1
                else:
                    feature = compute_agent_feature(
                        token, scenario_token, detail, rank,
                        category="missed", is_critical=True
                    )
                    missed_agents.append(feature)
                    statistics["missed_agents"] += 1
            else:
                feature = compute_agent_feature(
                    token, scenario_token, detail, rank=None,
                    category="background", is_critical=False
                )
                background_agents.append(feature)
                statistics["background_agents"] += 1

    return {
        "statistics": statistics,
        "missed_critical_agents": missed_agents,
        "captured_critical_agents": captured_agents,
        "background_agents": background_agents
    }


def save_txt(summary):
    stats = summary["statistics"]

    with open(OUTPUT_TXT, "w") as f:
        f.write("========== Agent Physics Feature Extraction ==========\n\n")
        f.write(f"Total scenes: {stats['total_scenes']}\n")
        f.write(f"Critical scenes: {stats['critical_scenes']}\n\n")

        f.write("Agent Statistics:\n")
        f.write(f"Captured critical agents : {stats['captured_agents']}\n")
        f.write(f"Missed critical agents   : {stats['missed_agents']}\n")
        f.write(f"Background agents        : {stats['background_agents']}\n")

        f.write("\n\nExample Missed Agent:\n")
        if len(summary["missed_critical_agents"]) > 0:
            f.write(json.dumps(summary["missed_critical_agents"][0], indent=2))
        else:
            f.write("None")


def main():
    print("Loading candidate bias results...")
    results = load_data()
    print(f"Loaded {len(results)} scenes")

    print("Extracting agent-level physics features...")
    summary = compute_features(results)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    save_txt(summary)

    print("\n✅ Agent physics feature extraction finished.")
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved TXT : {OUTPUT_TXT}")


if __name__ == "__main__":
    main()