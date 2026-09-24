"""
Candidate Bias Statistical Summarizer

Input:
    analysis/results/candidate_bias_per_scene.json

Output:
    analysis/results/candidate_bias_summary.json
    analysis/results/candidate_bias_summary.txt


Purpose:
    Summarize historical candidate truncation bias statistics
    for publication-level analysis.

Metrics:
    - Critical coverage statistics
    - Failure rate
    - Vehicle / pedestrian-bicycle miss statistics
    - Overlap@K distribution
    - Spearman ranking consistency
    - Worst-case scenarios
"""


import os
import json
import numpy as np


INPUT_PATH = (
    "analysis/results/candidate_bias_per_scene.json"
)

OUTPUT_JSON = (
    "analysis/results/candidate_bias_summary.json"
)

OUTPUT_TXT = (
    "analysis/results/candidate_bias_summary.txt"
)



def safe_mean(values):
    values = [
        v for v in values
        if not np.isnan(v)
    ]

    if len(values) == 0:
        return None

    return float(np.mean(values))



def safe_std(values):
    values = [
        v for v in values
        if not np.isnan(v)
    ]

    if len(values) == 0:
        return None

    return float(np.std(values))



def percentile(values, p):

    values = [
        v for v in values
        if not np.isnan(v)
    ]

    if len(values) == 0:
        return None

    return float(
        np.percentile(values, p)
    )



def main():

    print(
        "Loading:",
        INPUT_PATH
    )


    with open(INPUT_PATH, "r") as f:
        results = json.load(f)



    total_scene = len(results)


    # -----------------------------
    # Basic statistics
    # -----------------------------

    critical_scene_results = [
        r for r in results
        if r["total_critical"] > 0
    ]


    critical_scene_num = len(
        critical_scene_results
    )


    # -----------------------------
    # Coverage
    # -----------------------------

    coverage = [
        r["critical_coverage_rate"]
        for r in critical_scene_results
    ]


    miss_rate = [
        r["critical_miss_rate"]
        for r in critical_scene_results
    ]



    failure_cases = [
        r for r in critical_scene_results
        if r["critical_coverage_rate"] < 100
    ]


    severe_failure_cases = [
        r for r in critical_scene_results
        if r["critical_coverage_rate"] < 80
    ]



    # -----------------------------
    # Vehicle / Pedestrian
    # -----------------------------

    vehicle_cases = [
        r for r in critical_scene_results
        if r["critical_veh"] > 0
    ]


    pedbic_cases = [
        r for r in critical_scene_results
        if r["critical_ped_bic"] > 0
    ]


    vehicle_miss = sum(
        r["missed_veh_hist"]
        for r in results
    )


    pedbic_miss = sum(
        r["missed_ped_bic_hist"]
        for r in results
    )



    # -----------------------------
    # Ranking consistency
    # -----------------------------

    overlap = [
        r["overlap_at_k"]
        for r in results
    ]


    rho = [
        r["spearman_rho"]
        for r in results
    ]



    # -----------------------------
    # Worst cases
    # -----------------------------

    worst_cases = sorted(
        critical_scene_results,
        key=lambda x:
            x["critical_coverage_rate"]
    )


    worst_20 = worst_cases[:20]



    # -----------------------------
    # Summary dictionary
    # -----------------------------

    summary = {

        "dataset_statistics": {

            "total_scenes":
                total_scene,

            "scenes_with_critical_interaction":
                critical_scene_num,

            "critical_scene_ratio":
                critical_scene_num /
                max(total_scene,1)

        },


        "critical_coverage": {

            "mean":
                safe_mean(coverage),

            "std":
                safe_std(coverage),

            "median":
                percentile(
                    coverage,
                    50
                ),

            "p10":
                percentile(
                    coverage,
                    10
                ),

            "worst":
                float(
                    np.min(coverage)
                )
                if len(coverage)
                else None,


            "failure_rate":
                len(failure_cases) /
                max(critical_scene_num,1),


            "severe_failure_rate":
                len(severe_failure_cases) /
                max(critical_scene_num,1)

        },


        "failure_statistics": {

            "num_failure_cases":
                len(failure_cases),

            "num_severe_failure_cases":
                len(severe_failure_cases),

            "total_missed_critical_agents":
                sum(
                    r["missed_critical_hist"]
                    for r in results
                )

        },


        "agent_type_analysis": {


            "vehicle_cases":
                len(vehicle_cases),


            "pedestrian_bicycle_cases":
                len(pedbic_cases),


            "missed_vehicle_agents":
                vehicle_miss,


            "missed_ped_bicycle_agents":
                pedbic_miss

        },


        "ranking_statistics": {


            "mean_overlap_at_k":
                safe_mean(overlap),


            "std_overlap_at_k":
                safe_std(overlap),


            "mean_spearman_rho":
                safe_mean(rho),


            "std_spearman_rho":
                safe_std(rho)

        },


        "worst_20_scenarios":
            worst_20

    }



    os.makedirs(
        os.path.dirname(OUTPUT_JSON),
        exist_ok=True
    )


    with open(
        OUTPUT_JSON,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )



    # -----------------------------
    # Human readable report
    # -----------------------------

    with open(
        OUTPUT_TXT,
        "w"
    ) as f:


        f.write(
            "\n========== Candidate Bias Summary ==========\n\n"
        )


        f.write(
            f"Total scenarios: {total_scene}\n"
        )

        f.write(
            f"Critical interaction scenes: "
            f"{critical_scene_num}\n\n"
        )


        f.write(
            "------ Critical Coverage ------\n"
        )


        f.write(
            f"Mean coverage: "
            f"{summary['critical_coverage']['mean']:.2f}%\n"
        )

        f.write(
            f"Failure rate (<100%): "
            f"{summary['critical_coverage']['failure_rate']*100:.2f}%\n"
        )


        f.write(
            f"Severe failure (<80%): "
            f"{summary['critical_coverage']['severe_failure_rate']*100:.2f}%\n\n"
        )


        f.write(
            "------ Ranking Consistency ------\n"
        )


        f.write(
            f"Mean Overlap@K: "
            f"{summary['ranking_statistics']['mean_overlap_at_k']:.2f}%\n"
        )


        f.write(
            f"Mean Spearman rho: "
            f"{summary['ranking_statistics']['mean_spearman_rho']:.3f}\n\n"
        )


        f.write(
            "------ Worst Cases ------\n"
        )


        for i,case in enumerate(worst_20):

            f.write(
                f"{i+1}: "
                f"coverage="
                f"{case['critical_coverage_rate']:.2f}% "
                f"miss="
                f"{case['missed_critical_hist']}\n"
            )


    print(
        "\nFinished!"
    )

    print(
        "Saved:",
        OUTPUT_JSON
    )

    print(
        "Saved:",
        OUTPUT_TXT
    )



if __name__ == "__main__":

    main()