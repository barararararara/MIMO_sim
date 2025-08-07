import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import math
import Channel_functions as channel
import os
import logging

def Beam_Allocation_function(channel_type, d, NF_setting, W, Power):
    beam_allocation = np.full((1000, 3, 12, 2, 4), -100, dtype=int)
    
    for Channel_number in range(1000):
        for w in range(W):
            # ビーム候補の抽出
            v_indices = np.array([1, 5, 9])  # v のインデックスを限定
            pa_range = np.arange(64)
            pe_range = np.arange(64)

            v_grid, pa_grid, pe_grid = np.meshgrid(v_indices, pa_range, pe_range, indexing="ij")
            valid_mask = Power[Channel_number, v_grid, pa_grid, pe_grid] >= threshold
            valid_values = Power[Channel_number, v_grid, pa_grid, pe_grid][valid_mask]
            valid_indices = np.array((v_grid[valid_mask], pa_grid[valid_mask], pe_grid[valid_mask])).T

            candidates = np.hstack((valid_values[:, None], valid_indices))
            sorted_candidates = candidates[np.argsort(-candidates[:, 0])][:w+1]
            v_values = sorted_candidates[:, 1]
            missing_values = [i for i, val in enumerate([1, 5, 9]) if val not in v_values]

            # ログ出力：候補数などを確認
            logging.debug(f"Channel {Channel_number}, Allocate_max w={w+1}: missing_face: {missing_values}")
            
            face_candidates = {i: [] for i in range(3)}
            
            while any(len(face_candidates[i]) < 4 for i in range(3)):
                for candidate in sorted_candidates:
                    value, v, pa, pe = candidate
                    if v == 1:
                        face_index = 0
                    elif v == 5:
                        face_index = 1
                    else:
                        face_index = 2
                    face_candidates[face_index].append(candidate)
                if all(len(face_candidates[i]) >= 4 for i in range(3) if i not in missing_values):
                    break

            for face_index, candidates in face_candidates.items():
                beam_count = len(candidates)
                if beam_count == 1:
                    allocation_pattern = [1] * 4
                elif beam_count == 2:
                    allocation_pattern = [1, 2] * 2
                elif beam_count == 3:
                    allocation_pattern = [1, 2, 3, 1]
                elif beam_count >= 4:
                    allocation_pattern = [1, 2, 3, 4]
                
                for k, candidate in enumerate(candidates):
                    value, v, pa, pe = candidate
                    pattern = allocation_pattern[k]
                    beam_allocation[Channel_number, face_index, w, 0, pattern-1] = pa
                    beam_allocation[Channel_number, face_index, w, 1, pattern-1] = pe
                    if k == 3:
                        break
    save_dir = f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Beamallocation/{NF_setting}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/d={d}.npy', beam_allocation)

# パラメータ設定
Q = 64
V = 12
lam = (3*1e8) / (142*1e9)
Pt = 10
Pu = -23
Pu_dBm_per_carrer = -23.0
g_dB = 0
f = np.linspace(141.50025*1e9, 142.49975*1e9, 2000) #2000個の周波数

# 設定
channel_type = "InH"
NF_setting = "Near"
# スレッショルド値
threshold = -73
# 割り当て数の最大
W=12



# シミュレーション実行
for d in range(5, 31, 5):  # d = 5, 10, ..., 30
    load_dir = f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Power"
    Power_data = np.load(f"{load_dir}/d={d}/Power_{NF_setting}.npy", allow_pickle=True)
    Beam_Allocation_function(channel_type, d, NF_setting, W, Power_data)
