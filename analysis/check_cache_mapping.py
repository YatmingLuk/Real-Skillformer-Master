import pickle
import zlib


path="temp_config/ex_list_nuplan_val.pkl"


with open(path,"rb") as f:
    data=pickle.load(f)


print("num:",len(data))


sample=pickle.loads(
    zlib.decompress(data[0])
)


print(sample.keys())


if "token" in sample:
    print("token:",sample["token"])

if "scenario_token" in sample:
    print("scenario_token:",sample["scenario_token"])