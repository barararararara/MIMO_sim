import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import Channel_functions as channel
from joblib import Parallel, delayed
import os

# 警告の設定を変更（divide や invalid に関する警告を抑制）
old_settings = np.seterr(divide='ignore', invalid='ignore')

# 各角度をnumpyに統一する関数
def transform_angle_to_numpy(phi_deg, theta_deg, varphi_deg, eta_deg, N, M):
    theta_rad = [np.zeros(M[i]) for i in range(N)]
    phi_rad = [np.zeros(M[i]) for i in range(N)]
    varphi_rad = [np.zeros(M[i]) for i in range(N)]
    eta_rad = [np.zeros(M[i]) for i in range(N)]

    for n in range(N):
        for m in  range(M[n]):
            theta_rad[n][m] = np.radians(theta_deg[n][m])
            phi_rad[n][m] = np.radians(phi_deg[n][m])
            varphi_rad[n][m] = np.radians(varphi_deg[n][m])
            eta_rad[n][m] = np.radians(eta_deg[n][m])

    theta_rad_tmp = np.zeros((N, max(M)))
    phi_rad_tmp = np.zeros((N, max(M)))
    eta_rad_tmp = np.zeros((N, max(M)))
    varphi_rad_tmp = np.zeros((N, max(M)))
    # theta の値を埋める
    for i, values in enumerate(theta_rad):
        theta_rad_tmp[i, :len(values)] = values  # 長さが足りない部分はそのままゼロ
    for i, values in enumerate(phi_rad):
        phi_rad_tmp[i, :len(values)] = values
    for i, values in enumerate(eta_rad):
        eta_rad_tmp[i, :len(values)] = values
    for i, values in enumerate(varphi_rad):
        varphi_rad_tmp[i, :len(values)] = values
        
    return phi_rad_tmp, theta_rad_tmp, varphi_rad_tmp, eta_rad_tmp

# チャネル行列計算関数
def Channel_Matrix_Calculation(channel_type, d, NF_setting):
    h_tru_list = []
    h_est_list = []
    V = 12
    channel_data = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Scatter1/d={d}/Scatter1.npy", allow_pickle=True)
    
    load_dir = f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Beamallocation/{NF_setting}"
    beam_allocation  = np.load(f'{load_dir}/d={d}.npy', allow_pickle=True)
    
    # サブアレーの各素子の座標を取得
    subarray_coordinates_v_qy_qz = np.load('C:/Users/tai20/Downloads/NYUSIMchannel_shelter/subarray_coordinates_Q64.npy')
    # # サブアレー間隔を調整したアンテナの座標を計算
    # S_sub = 0.5  # サブアレー間隔（波長単位）
    # channel.calc_anntena_xyz_wide(lam, V, Q, S_sub)
    x = subarray_coordinates_v_qy_qz[:,0,0,0]
    y = subarray_coordinates_v_qy_qz[:,0,0,1]
    for Channel_number in range(1000):
        N = channel_data[Channel_number]["N"]
        M = channel_data[Channel_number]["M"]
        phi_deg = channel_data[Channel_number]["phi_deg"]
        theta_deg = channel_data[Channel_number]["theta_deg"]
        eta_deg = channel_data[Channel_number]["eta_deg"]
        varphi_deg = channel_data[Channel_number]["varphi_deg"]
        Pi = channel_data[Channel_number]["Pi"]
        beta = channel_data[Channel_number]["beta"]
        t_nm = channel_data[Channel_number]["t_nm"]
        R = channel_data[Channel_number]["R"]
        dis_sca1_to_anntena = channel_data[Channel_number]["dis_sca1_to_anntena"]
        
        phi_rad, theta_rad, varphi_rad, eta_rad = transform_angle_to_numpy(phi_deg, theta_deg, varphi_deg, eta_deg, N, M)
        
        # ビーム割当に必要な関数
        a_phi_theta = channel.define_a(V, N, M, phi_rad, theta_rad)
        b_varphi_eta = channel.define_b_verphi_eta(N, M, eta_rad)
        zeta = channel.define_zeta(V)
        Amp_per_career_desital = np.sqrt(Pi) / np.sqrt(Pu)
        
        # UEのアンテナ素子#0から送信されたSPm,nの，原点Oでの，周波数fkにおける複素振幅 ㊺
        a_O_m_n_u_k = np.zeros((N, max(M), U), dtype=np.complex64)
        u = np.arange(U)  # shape: (U,)
        
        beta_tmp = np.zeros((N,max(M)))
        for i, values in enumerate(beta):
            beta_tmp[i, :len(values)] = values  # 長さが足りない部分はそのままゼロ
        beta = np.array(beta_tmp)
        t_nm_tmp = np.zeros((N,max(M)))
        for i, values in enumerate(t_nm):
            t_nm_tmp[i, :len(values)] = values  # 長さが足りない部分はそのままゼロ
        t_nm = np.array(t_nm_tmp)
        
        b_varphi_eta = np.array(b_varphi_eta)
        
        f_common = Amp_per_career_desital * np.exp(1j * beta) * np.exp(-1j * 2 * np.pi * f_GHz_val * t_nm) * b_varphi_eta
        # eta_rad と varphi_rad の組み合わせから位相計算
        c = np.cos(eta_rad) * np.sin(varphi_rad)  # shape: (N, M)
        # 各 u に対してブロードキャストで位相成分を計算！
        phase = np.exp(-1j * np.pi * u[:, None, None] * c[None, :, :])  # shape: (U, N, M)
        # 共通因子と位相成分を掛け合わせる
        a_temp = phase * f_common[None, :, :]  # shape: (U, N, M)
        # 最終的な出力を (N, M, U) に転置
        a_O_m_n_u_k = np.transpose(a_temp, (1, 2, 0))

        # 第1散乱体と原点O間で電波が伝搬に要する時間 ㉒
        R_tmp = np.zeros((N,max(M)))
        for i, values in enumerate(R):
            R_tmp[i, :len(values)] = values  # 長さが足りない部分はそのままゼロ
        R = R_tmp
        tau_1_ns = R / c_ns

        # 第1散乱体での複素振幅 ㊻
        tau_exp = np.exp(1j * 2 * np.pi * f_GHz_val * tau_1_ns)[:, :, None]  # shape: (N, M, 1)
        # 要素ごとの積で計算完了！
        a_1_m_n_u_k = a_O_m_n_u_k * tau_exp  # shape: (N, M, U)

        # 第1散乱体とアンテナ間の伝搬時間 ㉓
        tau_nmvqyqz = dis_sca1_to_anntena / c_ns

        a = np.zeros((N, max(M), V, U, Q, Q), dtype=np.complex64)
        
        # tau_nmvqyqz は (N, max(M), V, Q, Q) で与えられてる前提
        exp_term = np.exp(-2j * np.pi * f_GHz_val * tau_nmvqyqz)  # shape: (N, max(M), V, Q, Q)
        a = np.einsum('nmu,vnm,vnmyz->nmvuyz', a_1_m_n_u_k, a_phi_theta, exp_term, optimize=True)
        a = np.sum(a, (0,1))
        
        w_DD_pape = np.zeros((V, Q, Q), dtype=np.complex64)
        g_DD_pape = np.zeros((V, N, max(M)), dtype=np.complex64)
        invalid_indices = []  # vの削除対象インデックスを格納するリスト

        h_u_v_k = np.zeros((W,U,V), dtype=complex)
        h_u_v_k_est = np.zeros((W,U,V), dtype=complex)
        n_dash = channel.noise_u_v_k(U,V)
        for w in range(W):            
            invalid_indices = []  # 各wごとに初期化
            v = 0  # v をループ内で一意に管理
            for face in range(3):
                for array in range(4):
                    pa = beam_allocation[Channel_number,face, w, 0, array]
                    pe = beam_allocation[Channel_number,face, w, 1, array]
                    if pa == -1 and pe == -1:
                        invalid_indices.append(v)  # 削除対象のvを記録
                        # print(f"v={v} は削除されます。")
                    else:
                        w_DD_pape[v,:,:] = channel.DFT_weight_calc_pape(Q, pa, pe)
                        g_DD_pape[v,:,:] = channel.g_dd_depend_on_pape(N, M, phi_rad, theta_rad, zeta, Q, v, pa, pe)
                        # print('v, pa, pe', v, pa, pe)
                    
                    v += 1  # ループごとに v を増加

            # vの次元を削除して V' にする
            valid_indices = np.setdiff1d(np.arange(V), invalid_indices)  # 残すvのインデックス
            w_DD_pape_reduced = w_DD_pape[valid_indices]  # w_DD_pape の v 次元を縮小
            n_dash = n_dash[:,valid_indices,:]

###########################################################################################################################
# 近傍界チャネル
            if NF_setting == 'Near':
                a = a[valid_indices]
                h_u_v_k[w] = np.einsum('vyz,vuyz->uv', w_DD_pape_reduced, a, optimize=True)
                h_u_v_k_est[w] = h_u_v_k[w] + n_dash[:,:,0] / np.sqrt(Pu/2000)
                
############################################################################################################################################
# 遠方界チャネル
            elif NF_setting == 'Far':
                h_dash_far = np.zeros((U, V, N, max(M)))
                # 必要に応じて次元を追加 (ブロードキャスト用)
                for i in range(N):
                    t_nm[i] = np.array(t_nm[i])[np.newaxis, :]

                # 位置成分を事前計算
                position_term = np.exp(1j * (2 * np.pi / lam_val) * (
                    x[np.newaxis, :,np.newaxis,np.newaxis] * np.cos(theta_rad[np.newaxis,np.newaxis,:,:]) * np.cos(phi_rad[np.newaxis,np.newaxis,:,:]) +
                    y[np.newaxis, :,np.newaxis,np.newaxis] * np.cos(theta_rad[np.newaxis,np.newaxis,:,:]) * np.sin(phi_rad[np.newaxis,np.newaxis,:,:])
                ))

                # 2. 矩形配列を作成（ここでは0でパディングしていますが、必要に応じてnp.nanなどを使えます）
                t_nm_rect = np.zeros((N, max(M)), dtype=t_nm[0].dtype)
                beta_rect = np.zeros((N, max(M)), dtype=beta[0].dtype)
                b_varphi_eta_rect = np.zeros((N, max(M)), dtype=b_varphi_eta[0].dtype)
                # 3. 各行にデータをコピー
                for i, row in enumerate(t_nm):
                    t_nm_rect[i, :len(row)] = row
                for i, row in enumerate(beta):
                    beta_rect[i, :len(row)] = row
                for i, row in enumerate(b_varphi_eta):
                    b_varphi_eta_rect[i, :len(row)] = row
                # 4. 次元を拡張して目的の形状にする
                t_nm_expanded = t_nm_rect[np.newaxis, np.newaxis, :, :]  # (1, 1, N, M)
                beta_expanded = beta_rect[np.newaxis, np.newaxis, :, :]  # (1, 1, N, M)
                b_varphi_eta_expanded = b_varphi_eta_rect[np.newaxis, np.newaxis, :, :]  # (1, 1, N, M)

                # `h_dash_far` の計算を一括処理
                h_dash_far = (Amp_per_career_desital[np.newaxis, np.newaxis, :, :]  # (1, 1, N, M)
                            * np.exp(1j * beta_expanded[np.newaxis, np.newaxis, :, :])  # (1, 1, N, M)
                            * np.exp(-1j * 2 * np.pi * f_GHz_val * t_nm_expanded)  # (1, 1, N, M)
                            * a_phi_theta[np.newaxis, :, :, :]  # (U, V, N, M)
                            * g_DD_pape[np.newaxis, :, :, :]  # (U, V, N, M)
                            * position_term  # (U, V, N, M, 1)
                            * b_varphi_eta_expanded[np.newaxis, np.newaxis, :, :]  # (1,1, N, M)
                            * np.exp(-1j * np.pi * np.arange(U)[:, np.newaxis, np.newaxis, np.newaxis]  # (U, 1, 1, 1)
                            * np.cos(eta_rad[np.newaxis, np.newaxis, :, :])  # (1, 1, N, M)
                            * np.sin(varphi_rad[np.newaxis, np.newaxis, :, :]))  # (1, 1, N, M)
                            )
                h_dash_far = np.squeeze(h_dash_far, axis=(0, 1))
                # M[n] を超える部分を 0 にする
                for n in range(N):
                    h_dash_far[:, :, n, M[n]:] = 0  # M[n] 以降の次元を 0 にする
                    
                # 全てのマルチパスの寄与
                h_u_v_k[w] = np.sum(h_dash_far, (2,3))
                h_u_v_k_est[w] = h_u_v_k[w] + n_dash[:,:,0] / np.sqrt(Pu / 2000)

        h_tru_list.append(h_u_v_k)
        h_est_list.append(h_u_v_k_est)
    
    save_dir =  f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Channel_Matrix/{NF_setting}/"
    
    os.makedirs(save_dir + "/H_tru", exist_ok = True)
    np.save(f"{save_dir}/H_tru/d={d}.npy", h_tru_list)
    
    os.makedirs(save_dir + "/H_est", exist_ok = True)
    np.save(f"{save_dir}/H_est/d={d}.npy", h_est_list) 
    print(f'{NF_setting}:{d} has done')

# パラメータ設定
Pu = 10
f_GHz = np.linspace(141.50025, 142.49975, 2000)
f_GHz_val = f_GHz[1000]
lam = 0.3 / f_GHz #2000個の波長
lam_val = 0.3 /f_GHz_val
V=12
U=8
c_ns = 0.3
Q=64
W =12

# 実行部分
channel_type = 'InF'
NF_setting = 'Near'

# シミュレーション実行
d=30
Channel_Matrix_Calculation(channel_type, d, NF_setting)
