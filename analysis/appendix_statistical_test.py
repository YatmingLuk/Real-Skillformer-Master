"""
Appendix Statistical Significance & Gap Test Suite
Path: analysis/appendix_statistical_test.py

Evaluates 4 Statistical Comparison Pairs:
1. Coverage@K: Historical vs Random (Directional Test: H1: Historical > Random)
2. Coverage@K: Historical vs Future Proximity Reference (Gap Analysis: Historical < Reference)
3. Overlap@K: Historical vs Random (Directional Test: H1: Historical > Random)
4. Overlap@K: Historical vs Future Proximity Reference (Gap Analysis: Historical < Reference)

Outputs:
- analysis/results/statistical_significance_report.txt
"""

import os
import json
import numpy as np


def paired_permutation_test_directional(a, b, alternative="greater", num_permutations=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = a - b
    obs_diff = np.mean(diffs)
    n = len(diffs)
    count_extreme = 0

    for _ in range(num_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_diff = np.mean(diffs * signs)

        if alternative == "greater" and perm_diff >= obs_diff:
            count_extreme += 1
        elif alternative == "less" and perm_diff <= obs_diff:
            count_extreme += 1

    p_value = count_extreme / float(num_permutations)
    return obs_diff, p_value


def compute_cohens_d(a, b):
    diffs = a - b
    std_diff = np.std(diffs, ddof=1)
    if std_diff == 0:
        return 0.0
    return float(np.mean(diffs) / std_diff)


def bootstrap_ci_diff(a, b, num_bootstraps=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = a - b
    n = len(diffs)
    boot_means = []

    for _ in range(num_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        boot_means.append(np.mean(diffs[boot_idx]))

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return ci_low, ci_high


def run_appendix_statistical_tests(json_path="analysis/results/candidate_bias_per_scene.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing input file: {json_path}. Run evaluate_candidate_bias.py first.")

    with open(json_path, "r") as f:
        data = json.load(f)

    hist_overlaps = np.array([r["overlap_at_k"] for r in data])
    rand_overlaps = np.array([r["mean_random_overlap"] for r in data])
    oracle_overlaps = np.ones_like(hist_overlaps) * 100.0

    valid_cov_scenes = [r for r in data if r["total_critical"] > 0 and not np.isnan(r["mean_random_coverage"])]

    hist_coverages = np.array([
        r["critical_coverage_rate"] if "critical_coverage_rate" in r and not np.isnan(r["critical_coverage_rate"])
        else (r["hist_cov_count"] / float(r["total_critical"])) * 100.0
        for r in valid_cov_scenes
    ])
    rand_coverages = np.array([r["mean_random_coverage"] for r in valid_cov_scenes])
    oracle_coverages = np.ones_like(hist_coverages) * 100.0

    report_lines = [
        "=" * 80,
        " APPENDIX STATISTICAL SIGNIFICANCE & GAP ANALYSIS REPORT",
        "=" * 80,
        f" Total Scenarios Evaluated           : {len(data)}",
        f" Scenarios with Emerging Partners    : {len(valid_cov_scenes)}",
        "=" * 80,
    ]

    tests_config = [
        ("1. Coverage@K: Historical vs Random (H1: Historical > Random)",
         hist_coverages, rand_coverages, "greater"),

        ("2. Coverage@K: Historical vs Future Proximity Reference (H1: Historical < Reference)",
         hist_coverages, oracle_coverages, "less"),

        ("3. Overlap@K: Historical vs Random (H1: Historical > Random)",
         hist_overlaps, rand_overlaps, "greater"),

        ("4. Overlap@K: Historical vs Future Proximity Reference (H1: Historical < Reference)",
         hist_overlaps, oracle_overlaps, "less"),
    ]

    for title, vec_a, vec_b, alt_perm in tests_config:
        obs_diff, p_perm = paired_permutation_test_directional(vec_a, vec_b, alternative=alt_perm)
        ci_low, ci_high = bootstrap_ci_diff(vec_a, vec_b)
        effect_size = compute_cohens_d(vec_a, vec_b)

        p_perm_str = "< 0.0001" if p_perm < 1e-4 else f"{p_perm:.4f}"

        report_lines.extend([
            f"\n--- {title} ---",
            f"  Mean Difference (A - B)         : {obs_diff:+.2f}%",
            f"  Bootstrap 95% CI of Difference  : [{ci_low:+.2f}%, {ci_high:+.2f}%]",
            f"  Effect Size (Cohen's d)         : {effect_size:.4f}",
            f"  Paired Permutation p-value      : {p_perm_str}"
        ])

    report_lines.append("\n" + "=" * 80)
    report_text = "\n".join(report_lines)
    print(report_text)

    os.makedirs("analysis/results", exist_ok=True)
    report_path = "analysis/results/statistical_significance_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nSaved statistical report to: {report_path}\n")


if __name__ == "__main__":
    run_appendix_statistical_tests()