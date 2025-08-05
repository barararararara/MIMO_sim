import numpy as np
import os

# 設定
channel_type = "InH"
NF_setting = "Near_Unif"
d = 5

Scatter1 = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/{channel_type}/{NF_setting}/d={d}/Scatter1.npy", allow_pickle=True)
print(Scatter1[:1])

def save_channel_data(data_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for key in data_list[0].keys():
        values = [entry[key] for entry in data_list]
        np.save(os.path.join(save_dir, f"{key}.npy"), np.array(values, dtype=object))

save_channel_data(Scatter1, f"C:/Users/tai20/Downloads/研究データ/Data/{channel_type}/{NF_setting}/d={d}")


def load_all_variables(load_dir):
    from glob import glob
    files = glob(os.path.join(load_dir, "*_shape.npy"))
    keys = [os.path.basename(f).replace("_shape.npy", "") for f in files]
    result_list = []

    for i in range(1000):  # チャネル数に合わせて調整
        entry = {}
        for key in keys:
            try:
                full_data = load_and_unpad_variable(key, load_dir)
                entry[key] = full_data[i]
            except:
                continue
        result_list.append(entry)

    return result_list
