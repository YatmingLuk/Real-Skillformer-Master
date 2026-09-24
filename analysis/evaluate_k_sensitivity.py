"""
Candidate Budget (K) Sensitivity Analysis

Input:
    analysis/results/candidate_bias_per_scene.json

Outputs:
    analysis/results/k_sensitivity_analysis.json
    analysis/results/k_sensitivity_analysis.txt

Purpose:
    Offline evaluation of Scene Mean Coverage, Agent Miss Rate, Scene Failure Rate,
    and Critical Agent Rank Distributions across multiple K budgets [5, 10, 15, 19, 25, 30, 40, 50, 60]
    using exact critical rank positions and strict sanity checks.
"""

import os
import json
import numpy as np

INPUT_JSON = "analysis/results/candidate_bias_per_scene.json"
OUTPUT_JSON = "analysis/results/k_sensitivity_analysis.json"
OUTPUT_TXT = "analysis/results/k_sensitivity_analysis.txt"

# 评估的 K 值序列
K_VALUES = [5, 10, 15, 19, 25, 30, 40, 50, 60]


def safe_rate(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b) * 100.0


def load_results():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Cannot find input JSON file: {INPUT_JSON}")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    return data


def evaluate_single_K(results, K):
    """
    基于每个 scene 中的 critical_rank_positions 字典，精准计算给定 K 下的各项指标。
    包含 strict sanity check 和 critical rank 统计。
    """
    total_critical_scenes = 0
    total_critical_agents = 0
    missed_critical_agents = 0

    scene_coverages = []
    failed_scenes = 0
    severe_failed_scenes = 0
    all_critical_ranks = []

    for idx, r in enumerate(results):
        total_crit = r.get("total_critical", 0)
        # 跳过不存在关键交互目标的普通场景
        if total_crit == 0:
            continue

        rank_positions = r.get("critical_rank_positions", {})

        # 1. 严谨性校验：缺失排名数据直接报错
        if not rank_positions:
            raise ValueError(
                f"Missing critical_rank_positions in scenario index {idx} "
                f"(total_critical={total_crit}). Please ensure dataset is generated correctly."
            )

        # 2. 严谨性校验：关键目标总数与记录的 rank 数量必须完全严格一致
        if len(rank_positions) != total_crit:
            raise ValueError(
                f"Inconsistent critical agent count at scenario index {idx}: "
                f"total_critical={total_crit}, "
                f"rank_positions_count={len(rank_positions)}"
            )

        # 收集所有关键目标的真实历史排名
        all_critical_ranks.extend(rank_positions.values())

        # 解决 Dict 遍历 bug：精准遍历 .values() 提取 integer rank
        hits = sum(1 for rank in rank_positions.values() if rank <= K)

        misses = total_crit - hits
        cov_rate = safe_rate(hits, total_crit)

        total_critical_scenes += 1
        total_critical_agents += total_crit
        missed_critical_agents += misses

        scene_coverages.append(cov_rate)

        # Scene-level 失败判定
        if cov_rate < 100.0:
            failed_scenes += 1
        if cov_rate < 80.0:
            severe_failed_scenes += 1

    return {
        "K": K,
        "scene_mean_coverage": float(np.mean(scene_coverages)) if scene_coverages else 0.0,
        "median_scene_coverage": float(np.median(scene_coverages)) if scene_coverages else 0.0,
        "agent_miss_rate": safe_rate(missed_critical_agents, total_critical_agents),
        "scene_failure_rate": safe_rate(failed_scenes, total_critical_scenes),
        "severe_scene_failure_rate": safe_rate(severe_failed_scenes, total_critical_scenes),
        "max_critical_rank": int(np.max(all_critical_ranks)) if all_critical_ranks else 0,
        "p95_critical_rank": float(np.percentile(all_critical_ranks, 95)) if all_critical_ranks else 0.0,
        "missed_agents_count": missed_critical_agents,
        "total_agents_count": total_critical_agents,
        "failed_scenes_count": failed_scenes,
        "total_critical_scenes": total_critical_scenes
    }


def compute_marginal_gain(curves):
    """
    计算相较于上一阶 K 值的边际收益（Marginal Gain）
    """
    sorted_k = sorted([int(k) for k in curves.keys()])
    marginal_gains = {}

    for i in range(1, len(sorted_k)):
        prev_k = sorted_k[i - 1]
        curr_k = sorted_k[i]

        prev_res = curves[str(prev_k)]
        curr_res = curves[str(curr_k)]

        delta_k = curr_k - prev_k
        delta_cov = curr_res["scene_mean_coverage"] - prev_res["scene_mean_coverage"]
        delta_scene_fail = prev_res["scene_failure_rate"] - curr_res["scene_failure_rate"]

        marginal_gains[f"{prev_k}->{curr_k}"] = {
            "delta_K": delta_k,
            "coverage_gain": float(delta_cov),
            "coverage_gain_per_K": float(delta_cov / delta_k) if delta_k > 0 else 0.0,
            "scene_failure_reduction": float(delta_scene_fail),
            "scene_failure_reduction_per_K": float(delta_scene_fail / delta_k) if delta_k > 0 else 0.0
        }

    return marginal_gains


def save_txt(summary):
    curves = summary["K_curve"]
    marginal = summary["marginal_gain"]

    with open(OUTPUT_TXT, "w") as f:
        f.write("========== Candidate Budget (K) Sensitivity Analysis ==========\n\n")
        f.write(
            f"{'K':<6} | {'Scene Mean Cov (%)':<18} | {'Median Cov (%)':<14} | {'Agent Miss (%)':<15} | {'Scene Fail (%)':<15} | {'Severe Fail (%)':<15}\n")
        f.write("-" * 100 + "\n")

        for k_str, res in curves.items():
            f.write(
                f"{res['K']:<6} | "
                f"{res['scene_mean_coverage']:<18.2f} | "
                f"{res['median_scene_coverage']:<14.2f} | "
                f"{res['agent_miss_rate']:<15.2f} | "
                f"{res['scene_failure_rate']:<15.2f} | "
                f"{res['severe_scene_failure_rate']:<15.2f}\n"
            )

        # 打印 Critical Rank 的全局极值信息
        first_key = list(curves.keys())[0]
        f.write(f"\n[Critical Rank Statistics Across All Critical Agents]\n")
        f.write(f"  - Max Critical Agent Rank: {curves[first_key]['max_critical_rank']}\n")
        f.write(f"  - 95th Percentile Rank   : {curves[first_key]['p95_critical_rank']:.2f}\n")

        f.write("\n\n========== Marginal Gain Analysis ==========\n\n")
        f.write(
            f"{'K Transition':<14} | {'Cov Gain (%)':<14} | {'Gain / ΔK':<12} | {'Fail Reduction (%)':<20} | {'Reduction / ΔK':<15}\n")
        f.write("-" * 80 + "\n")

        for trans, m_res in marginal.items():
            f.write(
                f"{trans:<14} | "
                f"{m_res['coverage_gain']:<14.2f} | "
                f"{m_res['coverage_gain_per_K']:<12.3f} | "
                f"{m_res['scene_failure_reduction']:<20.2f} | "
                f"{m_res['scene_failure_reduction_per_K']:<15.3f}\n"
            )


def main():
    print("🚀 Loading scenario results...")
    results = load_results()
    print(f"Loaded {len(results)} scenarios")

    curves = {}
    for K in K_VALUES:
        print(f"Evaluating K={K}")
        curves[str(K)] = evaluate_single_K(results, K)

    summary = {
        "K_curve": curves,
        "marginal_gain": compute_marginal_gain(curves)
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    save_txt(summary)

    print("\n✅ K sensitivity analysis finished successfully with full Sanity Checks.")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_TXT}")

    print("\n" + "=" * 80)
    with open(OUTPUT_TXT, "r") as f:
        print(f.read())
    print("=" * 80)


if __name__ == "__main__":
    main()