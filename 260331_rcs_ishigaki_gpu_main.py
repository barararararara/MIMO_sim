# 260331_rcs_ishigaki_gpu_main.py
import numpy as np
import Channel_function_gpu as ch_func
import Channel_functions as channel
import pandas as pd
import torch
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#ベースデータからバッチサイズ分データ取得し、GPUへ転送する関数
def get_batch_data(base_all, start_idx, b_size, device):
    end_idx = start_idx + b_size
    batch = {}
    
    # 辞書のキー名に合わせて修正
    batch['chi'] = torch.tensor(base_all['chi'][start_idx:end_idx], device=device, dtype=torch.float32)
    batch['N']   = torch.tensor(base_all['N_actual'][start_idx:end_idx], device=device, dtype=torch.long) # N -> N_actual
    batch['Z']   = torch.tensor(base_all['Z'][start_idx:end_idx], device=device, dtype=torch.float32)
    
    # パディング用マスクを取得 (B, N, M)
    batch['mask'] = torch.tensor(base_all['mask'][start_idx:end_idx], device=device, dtype=torch.float32)

    # 角度データ (B, N, M)
    batch['theta_nd_deg'] = torch.tensor(base_all['theta_ND_deg'][start_idx:end_idx], device=device, dtype=torch.float32)
    batch['eta_nd_deg']   = torch.tensor(base_all['eta_ND_deg'][start_idx:end_idx], device=device, dtype=torch.float32)
    batch['phi_deg']      = torch.tensor(base_all['phi_deg'][start_idx:end_idx], device=device, dtype=torch.float32)
    batch['varphi_deg']   = torch.tensor(base_all['varphi_deg'][start_idx:end_idx], device=device, dtype=torch.float32)
    
    batch['beta_rad']     = torch.tensor(base_all['beta'][start_idx:end_idx], device=device, dtype=torch.float32) # beta -> beta_rad
    batch['tau']          = torch.tensor(base_all['tau'][start_idx:end_idx], device=device, dtype=torch.float32)

    # 基準角度
    batch['theta_nd_00'] = batch['theta_nd_deg'][:, 0, 0]
    batch['eta_nd_00']   = batch['eta_nd_deg'][:, 0, 0]
    
    return batch

def run_data_acquisition(scenario, d_values, Ssub_list, total_trials, B):
    # デバイスとシステム設定の初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ch_func.SystemConfig(device=device)

    for scenario in scenarios:
        print(f"=== Starting Scenario: {scenario} ===")
        # ベースデータのロード (あらかじめ用意されたNYUSIM出力)
        base_all = np.load(f"Base_{scenario}_rect.npy", allow_pickle=True).item()
        
        # 結果格納用配列: (d, Ssub, Trial, Type) 
        # Type: 0=真のチャネル(理想), 1=推定チャネル(現実)
        all_cap = np.zeros((len(d_values), len(Ssub_list), total_trials, 2))
        all_ly  = np.zeros((len(d_values), len(Ssub_list), total_trials, 2))

        start_time = time.time()

        for d_idx, d in enumerate(d_values):
            for ssub_idx, Ssub in enumerate(Ssub_list):
                print(f"Running: d={d}m, Ssub={Ssub}λ ...", end=" ", flush=True)
                
                # 1000例を B 個ずつのバッチで回す
                for s_idx in range(0, total_trials, B):
                    # データ抽出
                    batch = get_batch_data(base_all, s_idx, B, device)
                    
                    # 1. GPUでチャネル行列計算 (真のチャネル / 推定・デノイズ後チャネル)
                    h_tru, h_est = ch_func.simulation_core_channelcalculation_gpu(
                        batch, d, Ssub, scenario, B, config
                    )
                    
                    # 2. チャネル容量計算 (ハイブリッド方式)
                    # Case A: 真のチャネルでの理想性能
                    cap_tru, ly_tru = ch_func.calc_channel_capacity_hybrid_all_data(
                        h_tru, h_tru, config, ch_func.water_filling_ratio
                    )
                    # Case B: 推定チャネルでの実力値
                    cap_est, ly_est = ch_func.calc_channel_capacity_hybrid_all_data(
                        h_est, h_tru, config, ch_func.water_filling_ratio
                    )
                    
                    # 結果を格納 (B個分を一気に入れる)
                    all_cap[d_idx, ssub_idx, s_idx:s_idx+B, 0] = cap_tru
                    all_cap[d_idx, ssub_idx, s_idx:s_idx+B, 1] = cap_est
                    all_ly[d_idx, ssub_idx, s_idx:s_idx+B, 0] = ly_tru
                    all_ly[d_idx, ssub_idx, s_idx:s_idx+B, 1] = ly_est

                print("Done.")

        # データの保存 (np.savez で圧縮保存)
        filename = f"Results_{scenario}_1000trials.npz"
        np.savez(filename, 
                capacity=all_cap, 
                layers=all_ly, 
                d=d_values, 
                Ssub=Ssub_list)
        
        end_time = time.time()
        print(f"=== {scenario} Finished. Total Time: {end_time - start_time:.2f}s ===")
        print(f"Saved to {filename}")


###############################################################
scenario = "InH"
Base_data = np.load(f"Base_{scenario}_rect.npy", allow_pickle=True)
base_seed = 9  # 今までの固定seed

# シミュレーション条件
scenarios = ['InH']  # 'InF' なども追加可能
d_values = [10, 20, 30, 40, 50]  # 通信距離 (m)
Ssub_list = [0, 50, 100]         # サブアレー間隔 (λ)
total_trials = 1              # 総試行数
B = 1                          # バッチサイズ (GPUメモリに合わせて調整)

if __name__ == "__main__":
    run_data_acquisition(scenario, d_values, Ssub_list, total_trials, B)
################################################################





