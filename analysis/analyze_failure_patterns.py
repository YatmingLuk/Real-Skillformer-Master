"""
Failure Pattern Analysis for Candidate Selection Bias

Input:
    analysis/results/candidate_bias_per_scene.json

Outputs:
    analysis/results/candidate_bias_failure_analysis.json
    analysis/results/candidate_bias_failure_analysis.txt


Purpose:
    Analyze where and why historical candidate selection fails,
    using rigorous dual-level metrics (Scene-level vs. Agent-level).

Metrics:
    - Coverage distribution & Scene-level failure rate
    - Critical interaction complexity
    - Candidate pool size sensitivity
    - Vehicle / Pedestrian agent-level miss rates
    - Ranking consistency vs coverage relationship
"""

import os
import json
import numpy as np


INPUT_JSON = "analysis/results/candidate_bias_per_scene.json"
OUTPUT_JSON = "analysis/results/candidate_bias_failure_analysis.json"
OUTPUT_TXT = "analysis/results/candidate_bias_failure_analysis.txt"



def safe_rate(a, b):
    if b == 0:
        return 0.0

    return float(a) / float(b) * 100.0



def load_results():

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    return data



def analyze_coverage(results):

    coverage = [
        r["critical_coverage_rate"]
        for r in results
        if not np.isnan(r["critical_coverage_rate"])
    ]


    bins = {
        "0-20%": 0,
        "20-40%": 0,
        "40-60%": 0,
        "60-80%": 0,
        "80-100%": 0,
        "100%": 0
    }


    for c in coverage:

        if c == 100:
            bins["100%"] += 1

        elif c < 20:
            bins["0-20%"] += 1

        elif c < 40:
            bins["20-40%"] += 1

        elif c < 60:
            bins["40-60%"] += 1

        elif c < 80:
            bins["60-80%"] += 1

        else:
            bins["80-100%"] += 1



    if len(coverage) == 0:

        return {
            "mean": 0.0,
            "median": 0.0,
            "distribution": bins
        }



    return {

        "mean": float(np.mean(coverage)),

        "median": float(np.median(coverage)),

        "distribution": bins

    }





def analyze_failure(results):

    valid_critical_results = [

        r for r in results

        if not np.isnan(
            r.get("critical_coverage_rate", np.nan)
        )

    ]


    total = len(valid_critical_results)


    if total == 0:

        return {
            "total_critical_scenes": 0
        }



    failure = [

        r for r in valid_critical_results

        if r["critical_coverage_rate"] < 100

    ]



    severe = [

        r for r in valid_critical_results

        if r["critical_coverage_rate"] < 80

    ]



    return {

        "total_critical_scenes": total,

        "scene_failure_cases": len(failure),

        "scene_failure_rate":
            safe_rate(len(failure), total),

        "severe_scene_failure_cases":
            len(severe),

        "severe_scene_failure_rate":
            safe_rate(len(severe), total)

    }





def analyze_critical_complexity(results):

    stats = {

        "1_agent": {
            "count":0,
            "failure":0
        },

        "2_agents":{
            "count":0,
            "failure":0
        },

        "3+_agents":{
            "count":0,
            "failure":0
        }

    }



    for r in results:


        if np.isnan(
            r.get("critical_coverage_rate", np.nan)
        ):
            continue



        n = r["total_critical"]



        if n <= 1:

            key = "1_agent"

        elif n == 2:

            key = "2_agents"

        else:

            key = "3+_agents"



        stats[key]["count"] += 1



        if r["critical_coverage_rate"] < 100:

            stats[key]["failure"] += 1



    for k,v in stats.items():

        v["scene_failure_rate"] = safe_rate(
            v["failure"],
            v["count"]
        )



    return stats





def analyze_candidate_pool(results):


    groups = {

        "<20":{
            "count":0,
            "failure":0
        },

        "20-40":{
            "count":0,
            "failure":0
        },

        "40-60":{
            "count":0,
            "failure":0
        },

        "60+":{
            "count":0,
            "failure":0
        }

    }



    for r in results:


        if np.isnan(
            r.get("critical_coverage_rate", np.nan)
        ):
            continue



        m = r["M_oracle"]



        if m < 20:

            key="<20"

        elif m < 40:

            key="20-40"

        elif m < 60:

            key="40-60"

        else:

            key="60+"



        groups[key]["count"] += 1



        if r["critical_coverage_rate"] < 100:

            groups[key]["failure"] += 1



    for k,v in groups.items():

        v["scene_failure_rate"] = safe_rate(
            v["failure"],
            v["count"]
        )



    return groups





def analyze_agent_type(results):

    """
    Agent-level Miss Rate.

    Only evaluate scenes containing
    valid critical interactions.
    """


    total_vehicle = 0
    missed_vehicle = 0


    total_ped = 0
    missed_ped = 0



    for r in results:


        # keep the same evaluation scope
        # as other failure analysis functions

        if np.isnan(
            r.get("critical_coverage_rate", np.nan)
        ):
            continue



        # Vehicle

        critical_veh = r.get(
            "critical_veh",
            0
        )

        missed_veh = r.get(
            "missed_veh_hist",
            0
        )


        total_vehicle += critical_veh

        missed_vehicle += missed_veh




        # Pedestrian / Bicycle

        critical_ped = r.get(
            "critical_ped_bic",
            0
        )


        missed_ped_count = r.get(
            "missed_ped_bic_hist",
            0
        )


        total_ped += critical_ped

        missed_ped += missed_ped_count




    return {


        "vehicle":{

            "critical_agents":
                total_vehicle,

            "missed_agents":
                missed_vehicle,

            "agent_miss_rate":
                safe_rate(
                    missed_vehicle,
                    total_vehicle
                )

        },


        "pedestrian_bicycle":{


            "critical_agents":
                total_ped,


            "missed_agents":
                missed_ped,


            "agent_miss_rate":
                safe_rate(
                    missed_ped,
                    total_ped
                )

        }

    }





def analyze_ranking_relation(results):


    coverage=[]
    overlap=[]
    rho=[]



    for r in results:


        if np.isnan(
            r.get("critical_coverage_rate", np.nan)
        ):
            continue



        coverage.append(
            r["critical_coverage_rate"]
        )


        overlap.append(
            r["overlap_at_k"]
        )



        if not np.isnan(
            r["spearman_rho"]
        ):

            rho.append(
                r["spearman_rho"]
            )



    result = {

        "coverage_overlap_corr":0.0,

        "coverage_rho_corr":0.0

    }



    if len(coverage) >= 2:

        result["coverage_overlap_corr"] = float(
            np.corrcoef(
                coverage,
                overlap
            )[0,1]
        )



    if len(rho) >= 2:

        result["coverage_rho_corr"] = float(
            np.corrcoef(
                coverage[:len(rho)],
                rho
            )[0,1]
        )


    return result





def save_txt(summary):


    with open(OUTPUT_TXT,"w") as f:


        f.write(
            "========== Candidate Bias Failure Pattern Analysis ==========\n\n"
        )


        for k,v in summary.items():

            f.write(
                f"\n[{k}]\n"
            )


            f.write(
                json.dumps(
                    v,
                    indent=2
                )
            )


            f.write("\n")





def main():


    results = load_results()


    summary={}



    summary["failure_statistics"] = analyze_failure(results)

    summary["coverage_distribution"] = analyze_coverage(results)

    summary["critical_complexity"] = analyze_critical_complexity(results)

    summary["candidate_pool_size"] = analyze_candidate_pool(results)

    summary["agent_type_analysis"] = analyze_agent_type(results)

    summary["ranking_relation"] = analyze_ranking_relation(results)



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



    save_txt(summary)



    print("\nFailure pattern analysis finished.")

    print(
        f"Saved: {OUTPUT_JSON}"
    )

    print(
        f"Saved: {OUTPUT_TXT}"
    )





if __name__=="__main__":

    main()