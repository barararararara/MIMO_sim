import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import math
import Channel_functions as channel
import os
import logging
import pandas as pd

def Beam_Allocation_function(channel_type, d, NF_setting, W, Power):
    beam_allocation = np.full((1000, 3, 12, 2, 4), -100, dtype=int)
    
    rows_sc = []   # sorted_candidates
    rows_mv = []   # missing_values
    rows_top = []  # topN
    rows_per_v = []  # 各vごとの上位候補

    for Channel_number in range(1000):
        for w in range(W):
            v_indices = np.array([1, 5, 9])
            pa_range = np.arange(64)
            pe_range = np.arange(64)

            v_grid, pa_grid, pe_grid = np.meshgrid(v_indices, pa_range, pe_range, indexing="ij")
            valid_mask = Power[Channel_number, v_grid, pa_grid, pe_grid] >= threshold
            valid_values = Power[Channel_number, v_grid, pa_grid, pe_grid][valid_mask]
            valid_indices = np.array((v_grid[valid_mask], pa_grid[valid_mask], pe_grid[valid_mask])).T

            candidates = np.hstack((valid_values[:, None], valid_indices))
            sorted_candidates = candidates[np.argsort(-candidates[:, 0])][:w+1]

            if w == W - 1:
                order = np.argsort(-candidates[:, 0])
                topN = candidates[order][:12]
                df_top = pd.DataFrame(topN, columns=["Power_dBm", "v", "pa", "pe"])
                df_top.insert(0, "rank", np.arange(1, len(df_top)+1))
                df_top.insert(0, "Channel", Channel_number)
                df_top[["v","pa","pe"]] = df_top[["v","pa","pe"]].round().astype("Int64")
                rows_top.append(df_top)

            if sorted_candidates.size > 0:
                df_sc = pd.DataFrame(sorted_candidates, columns=["Power_dBm", "v", "pa", "pe"])
                df_sc.insert(0, "w", w+1)
                df_sc.insert(0, "Channel", Channel_number)
                df_sc.insert(0, "rank", np.arange(1, len(df_sc)+1))
                df_sc[["v","pa","pe"]] = df_sc[["v","pa","pe"]].astype("int64")
                rows_sc.append(df_sc)
            else:
                rows_sc.append(pd.DataFrame([{
                    "rank": None, "Channel": Channel_number, "w": w+1,
                    "Power_dBm": None, "v": None, "pa": None, "pe": None
                }]))

            v_values = sorted_candidates[:, 1] if sorted_candidates.size > 0 else np.array([])
            missing_values = [val for val in [1, 5, 9] if val not in v_values]
            rows_mv.append({
                "Channel": Channel_number,
                "w": w+1,
                "missing_v_list": missing_values,
                "missing_v_csv": ",".join(map(str, missing_values)) if missing_values else ""
            })

            logging.debug(f"Channel {Channel_number}, Allocate_max w={w+1}: missing_face: {missing_values}")
            face_candidates = {i: [] for i in range(3)}
            v_to_face = {1: 0, 5: 1, 9: 2}
            missing_face_idx = [v_to_face[v] for v in missing_values]

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
                if all(len(face_candidates[i]) >= 4 for i in range(3) if i not in missing_face_idx):
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

        # === 追加：全vの上位4候補を記録 ===
        for v in range(12):
            valid_mask_v = Power[Channel_number, v, :, :] >= threshold
            valid_values_v = Power[Channel_number, v, :, :][valid_mask_v]
            if valid_values_v.size == 0:
                # 候補がない場合 → None埋め
                df_v = pd.DataFrame([{
                    "Channel": Channel_number,
                    "v": v,
                    "rank": None,
                    "Power_dBm": None,
                    "pa": None,
                    "pe": None
                }])
                rows_per_v.append(df_v)
                continue

            # 候補がある場合
            pa_indices, pe_indices = np.where(valid_mask_v)
            candidates_v = np.stack([valid_values_v, np.full_like(pa_indices, v), pa_indices, pe_indices], axis=1)
            top4_v = candidates_v[np.argsort(-candidates_v[:, 0])][:4]

            df_v = pd.DataFrame(top4_v, columns=["Power_dBm", "v", "pa", "pe"])
            df_v.insert(0, "rank", np.arange(1, len(df_v)+1))
            df_v.insert(0, "Channel", Channel_number)
            df_v["Power_dBm"] = df_v["Power_dBm"].round(3)
            df_v[["v", "pa", "pe"]] = df_v[["v", "pa", "pe"]].astype("Int64")
            df_v = df_v[["Channel", "v", "rank", "Power_dBm", "pa", "pe"]]
            rows_per_v.append(df_v)


    df_sorted_all = pd.concat(rows_sc, ignore_index=True)
    df_missing_all = pd.DataFrame(rows_mv)
    df_top_all = pd.concat(rows_top, ignore_index=True) if rows_top else pd.DataFrame(columns=["rank", "Channel", "Power_dBm", "v", "pa", "pe"])
    df_top_per_v = pd.concat(rows_per_v, ignore_index=True)

    return beam_allocation, df_sorted_all, df_missing_all, df_top_all, df_top_per_v
    

# パラメータ設定
Q = 64
V = 12
lam = (3*1e8) / (142*1e9)
Pt = 10
Pu = -23
Pu_dBm_per_carrer = -23.0
g_dB = 0
f = np.linspace(141.50025*1e9, 142.49975*1e9, 2000) #2000個の周波数


# スレッショルド値
threshold = -73
# 割り当て数の最大
W=12

# 設定
channel_type = "InH"
NF_setting = "Far"
Method = "Mirror"
S_sub = 0

# シミュレーション実行
for d in range(5, 31, 5):
    data_dir = f"C:/Users/tai20/Downloads/sim_data/Data"
    Power_data = np.load(f"{data_dir}/{Method}/{channel_type}/Ssub={S_sub}lam/Power/d={d}/Power_{NF_setting}.npy", allow_pickle=True)
    beam_allocation, df_sc, df_mv, df_top, df_top_per_v = Beam_Allocation_function(channel_type, d, NF_setting, W, Power_data)
    # 結果保存
    save_dir = f"{data_dir}/{Method}/{channel_type}/Ssub={S_sub}lam/Beamallocation/{NF_setting}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/d={d}.npy', beam_allocation)

# vごとの上位4候補を表示
# print(df_top_per_v.to_string(index=False))