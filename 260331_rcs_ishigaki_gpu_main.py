# 260331_rcs_ishigaki_gpu_main.py
import os
import numpy as np
import Channel_function_gpu as ch_func
import Channel_functions as channel
import pandas as pd
import torch
import time
from pathlib import Path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 元データ・rectデータの置き場所 (260401_convert_rect.py と同じ場所を使うこと)。
# 環境変数 MIMO_SIM_DATA_DIR が設定されていればそこを、無ければこのスクリプトと
# 同じディレクトリ(リポジトリ直下)を使う。Windows開発機とLinux実行機など、
# マシンによってデータの置き場所が違っても環境変数だけで切り替えられるようにするため。
DATA_DIR = Path(os.environ.get("MIMO_SIM_DATA_DIR", Path(__file__).resolve().parent))

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
    batch['rho']          = torch.tensor(base_all['rho'][start_idx:end_idx], device=device, dtype=torch.float32)
    batch['U_nm']         = torch.tensor(base_all['U_nm'][start_idx:end_idx], device=device, dtype=torch.float32)

    # 基準角度
    batch['theta_nd_00'] = batch['theta_nd_deg'][:, 0, 0]
    batch['eta_nd_00']   = batch['eta_nd_deg'][:, 0, 0]

    # --- マスク整合性チェック ---
    # mask==0(パディング)の場所は必ず0埋めされている、という前提に
    # calc_Pi_mW_batched 等の下流の計算が依存している。
    # データ形式が変わるなどして前提が崩れていたら、ここで早期にエラーにする。
    subpath_keys = ['theta_nd_deg', 'eta_nd_deg', 'phi_deg', 'varphi_deg', 'beta_rad', 'rho', 'U_nm']
    invalid_path = batch['mask'] == 0
    for key in subpath_keys:
        if torch.any(batch[key][invalid_path] != 0):
            raise ValueError(
                f"batch['{key}'] はマスク外(パディング)のはずの箇所に非ゼロの値を含んでいます。"
                f"base_all のデータ形式を確認してください。"
            )

    invalid_cluster = batch['mask'].sum(dim=2) == 0 # (B, N): 有効なサブパスが1つも無いクラスタ
    for key in ['tau', 'Z']:
        if torch.any(batch[key][invalid_cluster] != 0):
            raise ValueError(
                f"batch['{key}'] はマスク外(パディング)のはずのクラスタに非ゼロの値を含んでいます。"
                f"base_all のデータ形式を確認してください。"
            )

    return batch

def run_data_acquisition(scenario, d_values, Ssub_list, total_trials, B):
    # デバイスとシステム設定の初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ch_func.SystemConfig(device=device)

    # DFTウェイトは d/Ssub/batch に依存しないため、全ループの外で1回だけ計算して使い回す
    DFT_weights = ch_func.DFT_weight_calc_gpu(config.Q, device=device)

    for scenario in scenarios:
        print(f"=== Starting Scenario: {scenario} ===")
        # ベースデータのロード (あらかじめ用意されたNYUSIM出力)
        source_file = DATA_DIR / f"Base_{scenario}.npy"
        rect_file = DATA_DIR / f"Base_{scenario}_rect.npy"
        # rectデータが元データより古い(=reshape_to_rectの再実行忘れ)場合はここで気づけるようにする
        channel.ensure_rect_data_fresh(source_file, rect_file)
        base_all = np.load(rect_file, allow_pickle=True).item()

        # 結果格納用配列: (d, Ssub, Trial, Type)
        # Type: 0=真のチャネル(理想), 1=推定チャネル(現実)
        all_cap = np.zeros((len(d_values), len(Ssub_list), total_trials, 2))
        all_ly  = np.zeros((len(d_values), len(Ssub_list), total_trials, 2))

        start_time = time.time()

        for d_idx, d in enumerate(d_values):
            for ssub_idx, Ssub in enumerate(Ssub_list):
                print(f"Running: d={d}m, Ssub={Ssub}λ ...", end=" ", flush=True)

                # サブアレー座標は Ssub にのみ依存するため、Ssub ごとに1回だけ計算して使い回す
                subarray_v_qy_qz = ch_func.calc_anntena_xyz_Ssub_gpu(
                    config.lam_cen, config.V, config.Q, Ssub, device=device
                )

                # total_trials 例を B 個ずつのバッチで回す
                for s_idx in range(0, total_trials, B):
                    # データ抽出
                    batch = get_batch_data(base_all, s_idx, B, device)
                    # total_trials が B で割り切れない場合、最後のバッチは B 個に満たない。
                    # その実際のサイズを使う (固定の B を使うと形状不一致でクラッシュする)
                    actual_b = batch['chi'].shape[0]

                    # 1. GPUでチャネル行列計算 (真のチャネル / 推定・デノイズ後チャネル)
                    h_tru, h_est = ch_func.simulation_core_channelcalculation_gpu(
                        batch, d, Ssub, scenario, actual_b, config,
                        subarray_v_qy_qz=subarray_v_qy_qz, DFT_weights=DFT_weights
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

                    # 結果を格納 (actual_b 個分を一気に入れる)
                    all_cap[d_idx, ssub_idx, s_idx:s_idx+actual_b, 0] = cap_tru
                    all_cap[d_idx, ssub_idx, s_idx:s_idx+actual_b, 1] = cap_est
                    all_ly[d_idx, ssub_idx, s_idx:s_idx+actual_b, 0] = ly_tru
                    all_ly[d_idx, ssub_idx, s_idx:s_idx+actual_b, 1] = ly_est

                print("Done.")

        # データの保存 (np.savez で圧縮保存)
        # ファイル名に試行数とタイムスタンプを含め、設定を変えて再実行した際に
        # 過去の結果を気づかず上書きしてしまわないようにする
        timestamp = time.strftime("%y%m%d_%H%M%S")
        filename = f"Results_{scenario}_{total_trials}trials_{timestamp}.npz"
        np.savez(filename,
                capacity=all_cap,
                layers=all_ly,
                d=d_values,
                Ssub=Ssub_list,
                total_trials=total_trials,
                B=B)

        end_time = time.time()
        print(f"=== {scenario} Finished. Total Time: {end_time - start_time:.2f}s ===")
        print(f"Saved to {filename}")


###############################################################
scenario = "InH"

# シミュレーション条件
scenarios = ['InH']  # 'InF' なども追加可能
d_values = [10, 20, 30, 40, 50]  # 通信距離 (m)
Ssub_list = [0, 50, 100]         # サブアレー間隔 (λ)
total_trials = 1              # 総試行数
B = 1                          # バッチサイズ (GPUメモリに合わせて調整)

if __name__ == "__main__":
    run_data_acquisition(scenario, d_values, Ssub_list, total_trials, B)
################################################################





