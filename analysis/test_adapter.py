"""
Test Scenario Adapter

Purpose:
    Verify that a NuPlan scenario_token from analysis results
    can be dynamically converted into the same sample format
    used by NuplanDataset cache.

Pipeline:

    scenario_token
          |
          v
    NuPlanScenarioBuilder
          |
          v
    Scenario
          |
          v
    data_processor(sc).build_mapping()
          |
          v
    sample(dict)

Expected:
    sample keys should match cached mapping:

    [
        'agents',
        'matrix',
        'labels',
        'token',
        'map_api',
        'iteration',
        'timestamp_us',
        'ego_global_pose'
    ]

"""

import os
import sys

# ----------------------------------------------------
# Project root
# ----------------------------------------------------

sys.path.append(
    os.path.abspath(".")
)

# ----------------------------------------------------
# NuPlan imports
# ----------------------------------------------------

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)

from nuplan.planning.scenario_builder.scenario_filter import (
    ScenarioFilter,
)

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioMapping,
)

from nuplan.planning.utils.multithreading.worker_sequential import (
    Sequential,
)

# ----------------------------------------------------
# Wayformer imports
# ----------------------------------------------------

from Wayformer.wf_dataset import (
    data_processor,
    get_scenario_map,
)

import utils.config as config


# ====================================================
# Adapter
# ====================================================

def get_sample_by_scenario_token(
        scenario_token: str
):
    """
    Convert NuPlan scenario_token
    into Wayformer sample dict.

    This should reproduce:

        mapping = data_processor(sc).build_mapping()

    inside NuplanDataset.
    """

    print("\n==============================")
    print("Adapter Test")
    print("==============================")

    print(
        f"Searching scenario token:\n{scenario_token}"
    )

    # --------------------------------------------
    # 1. Scenario Mapping
    # --------------------------------------------

    scenario_mapping = ScenarioMapping(
        scenario_map=get_scenario_map(),
        subsample_ratio_override=0.5
    )

    # --------------------------------------------
    # 2. Exact scenario filter
    # --------------------------------------------

    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=[scenario_token],
        log_names=None,
        map_names=None,

        num_scenarios_per_type=None,
        limit_total_scenarios=1,

        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,

        expand_scenarios=False,
        remove_invalid_goals=False,

        shuffle=False,
    )

    # --------------------------------------------
    # 3. Build NuPlan Scenario
    # --------------------------------------------

    print("\nBuilding NuPlanScenarioBuilder...")

    builder = NuPlanScenarioBuilder(
        data_root=config.DATA_PATH,
        map_root=config.MAP_PATH,
        sensor_root=None,
        db_files=None,
        map_version=config.MAP_VERSION,
        scenario_mapping=scenario_mapping,
    )

    worker = Sequential()

    scenarios = builder.get_scenarios(
        scenario_filter,
        worker
    )

    print(
        f"Found scenarios: {len(scenarios)}"
    )

    if len(scenarios) == 0:
        raise RuntimeError(
            "Scenario token not found!"
        )

    sc = scenarios[0]

    print(
        "\nScenario information:"
    )

    print(
        "scenario.token =",
        sc.token
    )

    # --------------------------------------------
    # 4. Convert to sample
    # --------------------------------------------

    print(
        "\nRunning data_processor..."
    )

    sample = data_processor(sc).build_mapping()

    print(
        "Adapter conversion success!"
    )

    return sample


# ====================================================
# Main
# ====================================================


if __name__ == "__main__":
    # 修改成你的目标 token
    TEST_TOKEN = (
        "005fd0f78d2c5056"
    )

    sample = get_sample_by_scenario_token(
        TEST_TOKEN
    )

    print("\n==============================")
    print("Sample Check")
    print("==============================")

    print(
        "\nSample keys:"
    )

    print(
        sample.keys()
    )

    print(
        "\nSample token:"
    )

    print(
        sample["token"]
    )

    print(
        "\nAgents:"
    )

    print(
        sample["agents"].shape
    )

    print(
        "\nMatrix:"
    )

    print(
        sample["matrix"].shape
    )

    print(
        "\nLabels:"
    )

    print(
        sample["labels"].shape
    )

    print(
        "\nIteration:"
    )

    print(
        sample["iteration"]
    )

    print(
        "\nTimestamp:"
    )

    print(
        sample["timestamp_us"]
    )

    print(
        "\n=============================="
    )

    print(
        "Adapter test finished."
    )
