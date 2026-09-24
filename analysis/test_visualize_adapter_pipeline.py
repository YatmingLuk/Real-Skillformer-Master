"""
End-to-End Adapter + SkillFormer Inference Verification

Purpose:
Verify that a dynamically constructed NuPlan sample
(from scenario_token -> Scenario -> data_processor -> sample)
can be directly consumed by SkillFormer inference pipeline.

Pipeline:

scenario_token
        |
        v
NuPlanScenarioBuilder
        |
        v
NuPlan Scenario
        |
        v
data_processor(sc).build_mapping()
        |
        v
sample
        |
        v
SkillFormer checkpoint
        |
        v
encode_gt_skill()
predict_skill()
decoder.decode()

This script does NOT generate visualization.
It only validates the complete inference compatibility.
"""


import os
import sys
from pathlib import Path

import numpy as np
import torch


sys.path.append(os.path.abspath("."))


# -----------------------------
# NuPlan / Wayformer imports
# -----------------------------

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)

from nuplan.planning.scenario_builder.scenario_filter import (
    ScenarioFilter,
)

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioMapping,
)

from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)


import utils.config as config


from Wayformer.wf_dataset import (
    data_processor,
    get_scenario_map,
)


from visualize_skills import (
    load_model,
    resolve_device,
)



# ============================================================
# Adapter
# ============================================================

def get_sample_by_scenario_token(
    scenario_token: str,
):
    """
    Scenario token -> NuPlan Scenario -> SkillFormer sample
    """

    print("\n==============================")
    print("Adapter Stage")
    print("==============================")

    print(
        f"Searching scenario token:\n{scenario_token}"
    )


    scenario_mapping = ScenarioMapping(
        scenario_map=get_scenario_map(),
        subsample_ratio_override=0.5,
    )


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


    print("\nBuilding NuPlanScenarioBuilder...")


    builder = NuPlanScenarioBuilder(
        data_root=config.DATA_PATH,
        map_root=config.MAP_PATH,
        sensor_root=None,
        db_files=None,
        map_version=config.MAP_VERSION,
        scenario_mapping=scenario_mapping,
    )


    worker = SingleMachineParallelExecutor(
        use_process_pool=False
    )


    scenarios = builder.get_scenarios(
        scenario_filter,
        worker,
    )


    print(
        f"Found scenarios: {len(scenarios)}"
    )


    if len(scenarios) == 0:
        raise RuntimeError(
            "Scenario token not found!"
        )


    scenario = scenarios[0]


    print("\nScenario information:")
    print(
        "scenario.token =",
        scenario.token
    )


    print("\nRunning data_processor...")


    sample = data_processor(
        scenario
    ).build_mapping()


    print(
        "Adapter conversion success!"
    )


    return sample



# ============================================================
# Inference verification
# ============================================================


@torch.no_grad()
def test_skillformer_inference(
    model,
    sample,
    device,
):

    print("\n==============================")
    print("SkillFormer Inference Stage")
    print("==============================")


    batch = [
        sample
    ]


    print("\nEncoding GT skill...")


    gt_skill = (
        model.model
        .encode_gt_skill(
            batch,
            device,
        )
    )


    print(
        "GT skill shape:",
        gt_skill.shape
    )


    print("\nPredicting skill...")


    pred_skill = (
        model.model
        .predict_skill(
            batch,
            device,
        )
    )


    print(
        "Pred skill shape:",
        pred_skill.shape
    )


    print("\nDecoding trajectory...")


    oracle_traj = (
        model.model.decoder
        .decode(gt_skill)
    )


    pred_traj = (
        model.model.decoder
        .decode(pred_skill)
    )


    print(
        "Oracle trajectory:",
        oracle_traj.shape
    )

    print(
        "Pred trajectory:",
        pred_traj.shape
    )


    arrays = [
        gt_skill.cpu().numpy(),
        pred_skill.cpu().numpy(),
        oracle_traj.cpu().numpy(),
        pred_traj.cpu().numpy(),
    ]


    for name, arr in zip(
        [
            "gt_skill",
            "pred_skill",
            "oracle",
            "pred",
        ],
        arrays,
    ):

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"{name} contains NaN/Inf"
            )


    print(
        "\nInference verification SUCCESS!"
    )



# ============================================================
# Main
# ============================================================


def main():


    scenario_token = (
        "005fd0f78d2c5056"
    )


    ckpt = (
        "output/wayformer.skill_only/"
        "version_3/checkpoints/"
        "epoch=64-val_loss=0.1399.ckpt"
    )


    print("\n==============================")
    print("Adapter + Visualization Pipeline Test")
    print("==============================")


    device = resolve_device(
        "auto"
    )


    print(
        "Device:",
        device
    )


    # 1. Adapter

    sample = get_sample_by_scenario_token(
        scenario_token
    )


    print("\n==============================")
    print("Sample Check")
    print("==============================")


    print(
        "Sample keys:"
    )

    print(
        sample.keys()
    )


    print(
        "\nAgents:",
        sample["agents"].shape
    )

    print(
        "Matrix:",
        sample["matrix"].shape
    )

    print(
        "Labels:",
        sample["labels"].shape
    )


    print(
        "Token:",
        sample["token"]
    )


    # 2. Load model

    print("\nLoading checkpoint...")


    model = load_model(
        Path(ckpt),
        device,
    )


    model.eval()


    print(
        "Checkpoint loaded!"
    )


    # 3. Inference

    test_skillformer_inference(
        model,
        sample,
        device,
    )


    print(
        "\n=============================="
    )

    print(
        "ALL TESTS PASSED"
    )

    print(
        "=============================="
    )



if __name__ == "__main__":
    main()