import json


TARGET = "005fd0f78d2c5056"

PATH = "analysis/results/candidate_bias_per_scene.json"


with open(PATH, "r") as f:
    data = json.load(f)


for item in data:
    if item.get("scenario_token") == TARGET:
        print("=" * 50)
        print("Found scenario")
        print("=" * 50)

        for k, v in item.items():
            print(f"\n{k}:")
            print(v)

        break
else:
    print("Scenario not found")