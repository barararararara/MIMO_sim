import numpy as np
import os
from glob import glob

# 設定
channel_type = "InF"
NF_setting = "Near"
d = 30

Scatter1 = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Scatter1/d={d}/Scatter1.npy", allow_pickle=True)

N = []
M = []
M_flat = []
idx=0
multipath = 0
for i in range(1000):
    N.append(Scatter1[i]['N'])
    M.append(Scatter1[i]['M'])
    for n in range(N[i]):
        multipath += Scatter1[i]['M'][n]
        M_flat.append(Scatter1[i]['M'][n])

# 保存先ディレクトリ作成
save_dir = f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Scatter1/d={d}/Flattened"
os.makedirs(save_dir, exist_ok=True)

def flatten(multipath, flatten_data, N, M, keys):
    save_data = np.zeros((len(keys),multipath), dtype=float)
    key_idx = 0
    for key in keys:
        multipath_idx = 0
        for i in range(1000):
            for n in range(N[i]):
                for m in range(M[i][n]):
                    try:
                        save_data[key_idx][multipath_idx] = flatten_data[i][key][n][m]
                    except Exception as e:
                        print(f"[ERROR] i={i}, key={key}, n={n}, m={m}, error={e}")
                    multipath_idx += 1
        key_idx += 1
    return save_data

keys = [
    'beta', 'phi_deg', 'varphi_deg',
    'theta_deg', 'eta_deg', 't_nm', 'R', 'Pi_each_career'
]

flatten_data = flatten(multipath, Scatter1, N, M, keys)

key_idx = 0
for key in keys:
    multipath_idx = 0
    for i in range(1000):
        for n in range(N[i]):
            for m in range(M[i][n]):
                if Scatter1[i][key][n][m] - flatten_data[key_idx][multipath_idx] != 0:
                    print(Scatter1[i][key][n][m] , flatten_data[key_idx][multipath_idx])
                multipath_idx += 1
    if key == 'R':
        np.save(f'{save_dir}/largeR_flat.npy', flatten_data[key_idx])
    elif key == 'r':
        flatten_data[key_idx].fill(3)
        print(flatten_data[key_idx], key)
        np.save(f'{save_dir}/{key}_flat.npy', flatten_data[key_idx])
    else:
        np.save(f'{save_dir}/{key}_flat.npy', flatten_data[key_idx])
    key_idx += 1

np.save(f'{save_dir}/N.npy', N)
np.save(f'{save_dir}/M_flat.npy', M_flat)