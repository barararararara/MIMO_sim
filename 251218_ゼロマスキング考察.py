# 251218_「ゼロマスキングについてのお尋ね」考察

import numpy as np
import Channel_functions as channel
import math
import pandas as pd
import matplotlib.pyplot as plt

# 必要なパラメータ設定
lam = (3.0 * 1e8) / (142 * 1e9)
Pt = 10
Pu_dBm_per_carrer = -23
g_dB = 0
Q = 16
V = 12
f = np.linspace(141.50025*1e9, 142.49975*1e9, 2000) #2000個の周波数



def Add_scatter1(channel_base, d, Ssub_lam, channel_type):
    L_AOD = channel_base['L_AOD']
    L_AOA = channel_base['L_AOA']
    chi = channel_base['chi']
    N = channel_base['N']
    M = channel_base['M']
    Z = channel_base['Z']
    U = channel_base['U']
    rho = channel_base['rho']
    tau = channel_base['tau']
    beta = channel_base['beta']
    phi_mean_AOD = channel_base['phi_mean_AOD']
    varphi_mean_AOA = channel_base['varphi_mean_AOA']
    theta_mean_AOD = channel_base['theta_mean_AOD']
    eta_mean_AOA = channel_base['eta_mean_AOA']
    phi_deg = channel_base['phi_deg']
    varphi_deg = channel_base['varphi_deg']
    theta_ND_deg = channel_base['theta_ND_deg']
    eta_ND_deg = channel_base['eta_ND_deg']

    eta_dir = np.degrees(np.arcsin(1 / d))
    theta_dir = -eta_dir
    theta_0_0 = theta_ND_deg[0][0] if len(theta_ND_deg[0]) > 0 else theta_ND_deg[0]
    eta_0_0 = eta_ND_deg[0][0] if len(eta_ND_deg[0]) > 0 else eta_ND_deg[0]

    Delta_EOD = theta_dir - theta_0_0
    Delta_EOA = eta_dir - eta_0_0

    theta_deg = [theta_ND_deg[n] + Delta_EOD for n in range(N)]
    eta_deg = [eta_ND_deg[n] + Delta_EOA for n in range(N)]

    Pr = channel.calc_Pr(lam, d, chi, Pt, g_dB, channel_type, do=1)
    Pr_each_career = channel.calc_Pr_each_career(lam, d, chi, Pu_dBm_per_carrer, g_dB, channel_type, do=1)
    t_nm = channel.abs_timedelays(d, rho, tau, N, M)
    P = channel.cluster_power(Pr, N, tau, Z, channel_type)
    Pi = channel.SP_power(N, M, P, rho, U, channel_type)
    P_each_career = channel.cluster_Power_each_career(Pr_each_career, Z, N, tau, channel_type)
    Pi_each_career = channel.SP_Power_each_career(N, M, P_each_career, rho, U, channel_type)
    r_dash = lam * (Q + 1) * V / (6 * math.sqrt(3))
    r, R = channel.scatter1_distance(d, Q, V, N, M, rho, tau)
    phi_rad = [np.radians(phi_deg[n]) for n in range(N)]
    theta_rad = [np.radians(theta_deg[n]) for n in range(N)]
    sca1_xyz_co = channel.scatter1_xyz_cordinate(N, M, R, phi_rad, theta_rad)
    # サブアレー各素子の座標wideversion
    subarray_v_qy_qz = channel.calc_anntena_xyz_Ssub(lam, V, Q, Ssub_lam)
    dis_sca1_to_anntena = channel.distance_scatter1_to_eachanntena(sca1_xyz_co, subarray_v_qy_qz, N, M, V, Q)
    phi_deg_v, theta_deg_v = channel.calc_each_subarray_AOD_EOD(N, M, sca1_xyz_co, subarray_v_qy_qz, V)
    return {
        "L_AOD": L_AOD, "L_AOA": L_AOA, "chi": chi, "N": N, "M": M, "Z": Z, "U": U,
        "rho": rho, "tau": tau, "beta": beta, "phi_mean_AOD": phi_mean_AOD, "varphi_mean_AOA": varphi_mean_AOA,
        "theta_mean_AOD": theta_mean_AOD, "eta_mean_AOA": eta_mean_AOA,
        "phi_deg": phi_deg, "varphi_deg": varphi_deg, "theta_deg": theta_deg, "eta_deg": eta_deg,
        "Pr": Pr, "Pr_each_career": Pr_each_career, "t_nm": t_nm, "P": P, "Pi": Pi,
        "P_each_career": P_each_career, "Pi_each_career": Pi_each_career,
        "r_dash": r_dash, "r": r, "R": R, "sca1_xyz_co": sca1_xyz_co,
        "subarray_v_qy_qz": subarray_v_qy_qz, "dis_sca1_to_anntena": dis_sca1_to_anntena,
        "phi_deg_v": phi_deg_v, "theta_deg_v": theta_deg_v
    }

def Power_calculation(Channel_scatter1, channel_type, NF_setting, Method, d, f, start_idx, end_idx):
    DFT_weights = channel.DFT_weight_calc(Q)
    zeta = channel.define_zeta(V)
    for number in range(start_idx, end_idx):
        data = Channel_scatter1
        N, M = data['N'], data['M']
        # r, r_dash = data[''], data['r_dash']
        R = data['R']
        if Method == "Mirror":
            r=R
        elif Method == "Scatter1":
            r=data['r']
        t_nm = data['t_nm']
        Pi_each_career = data['Pi_each_career']
        beta = data['beta']
        phi_deg, varphi_deg = data['phi_deg'], data['varphi_deg']
        theta_deg, eta_deg = data['theta_deg'], data['eta_deg']
        dis_sca1_to_anntena = data['dis_sca1_to_anntena']
        phi_deg_v = data['phi_deg_v']
        theta_deg_v = data['theta_deg_v']
        
        theta_rad = np.zeros((N,max(M)))
        phi_rad = np.zeros((N,max(M)))
        varphi_rad = np.zeros((N,max(M)))
        eta_rad = np.zeros((N,max(M)))
        
        phi_rad_v = np.zeros((V,N,max(M)))
        theta_rad_v = np.zeros((V,N,max(M)))

        for n in range(N):
            for m in  range(M[n]):
                theta_rad[n][m] = np.radians(theta_deg[n][m])
                phi_rad[n][m] = np.radians(phi_deg[n][m])
                varphi_rad[n][m] = np.radians(varphi_deg[n][m])
                eta_rad[n][m] = np.radians(eta_deg[n][m])
                
        phi_rad_v = np.zeros((V, N, max(M)))
        theta_rad_v = np.zeros((V, N, max(M)))

        for v in range(V):
            for n in range(N):
                for m in range(M[n]):
                    phi_rad_v[v, n, m]   = np.radians(phi_deg_v[v][n][m])
                    theta_rad_v[v, n, m] = np.radians(theta_deg_v[v][n][m])

        b_varphi_eta = channel.define_b_verphi_eta(N, M, eta_rad)
        # 指向性を計算（近傍界の場合サブアレー毎）
        if NF_setting == "Near":
            a_phi_theta = channel.define_a(V, N, M, phi_rad_v, theta_rad_v, NF_setting)
        elif NF_setting == "Far":
            a_phi_theta = channel.define_a(V, N, M, phi_rad, theta_rad, NF_setting)

        Amp_per_career = np.sqrt(Pi_each_career)
        complex_Amp_at_O = channel.calc_complex_Amp_at_O(f, V, N, M, Amp_per_career, beta, t_nm, b_varphi_eta)
        complex_Amp_at_scatter1 = channel.complex_Amp_at_scatter1(N, M, f, r, complex_Amp_at_O)
        
        complex_Amp_at_each_antena = channel.complex_Amp_at_each_antena(f, V, Q, N, M, dis_sca1_to_anntena, complex_Amp_at_scatter1, a_phi_theta)
        
        g_DD = channel.g_DD_pa_pe(N, M, phi_rad, theta_rad, zeta, Q, V)
        n_k_v = channel.noise_n_k_v(V)

        K = len(f)
        P_sub_dash_dBm = channel.near_Power_inc_noise_optimized(V, Q, f, DFT_weights, complex_Amp_at_each_antena, n_k_v)
        P_far_sub_dash_dBm = channel.far_Power_inc_noise(V,Q,f,complex_Amp_at_O,g_DD,n_k_v,a_phi_theta)
        
    return P_sub_dash_dBm, P_far_sub_dash_dBm

def Beam_allocation_method_1(Power, threshold, test_num):
    """
    ビーム割当法１：各サブアレーの中で最大受信電力を与えるビームを選択
    """
    beam_allocation = np.full((3, 2, 4), -1, dtype=int)  # w削除済み
    rows_per_v = []

    for v in range(V):
        valid_mask = Power[v] >= threshold
        valid_values = Power[v][valid_mask]

        if valid_values.size == 0:
            df_v = pd.DataFrame([{
                "Channel": test_num,
                "v": v,
                "rank": None,
                "Power_dBm": None,
                "pa": None,
                "pe": None
            }])
            rows_per_v.append(df_v)
            continue

        pa_indices, pe_indices = np.where(valid_mask)
        candidates = np.stack([valid_values, np.full_like(pa_indices, v), pa_indices, pe_indices], axis=1)
        top4 = candidates[np.argsort(-candidates[:, 0])][:4]

        df_v = pd.DataFrame(top4, columns=["Power_dBm", "v", "pa", "pe"])
        df_v.insert(0, "rank", np.arange(1, len(df_v)+1))
        df_v.insert(0, "Channel", test_num)
        df_v[["v", "pa", "pe"]] = df_v[["v", "pa", "pe"]].astype("Int64")
        rows_per_v.append(df_v)
        

    # 割り当て処理
    for face in range(3):
        for k in range(4):
            v = face * 4 + k
            df_v = rows_per_v[v]

            if isinstance(df_v, pd.DataFrame) and df_v["rank"].notnull().any():
                # 常に rank=1（最大電力）を取る
                row = df_v.iloc[0]
                beam_allocation[face, 0, k] = row["pa"]
                beam_allocation[face, 1, k] = row["pe"]

    
    return beam_allocation, rows_per_v

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

def Channel_Calculation(beam_allocation, Channel_scatter1, NS, Ssub_lam, NF_setting, th_dB):
    W = 12
    U=8
    f_GHz = np.linspace(141.50025, 142.49975, 2000)
    f_GHz_val = f_GHz[1000]
    lam = 0.3 / f_GHz #2000個の波長
    lam_val = 0.3 /f_GHz_val
    c_ns = 0.3
    Pu = 10
        
        
    N = Channel_scatter1["N"]
    M = Channel_scatter1["M"]
    phi_deg = Channel_scatter1["phi_deg"]
    theta_deg = Channel_scatter1["theta_deg"]
    eta_deg = Channel_scatter1["eta_deg"]
    varphi_deg = Channel_scatter1["varphi_deg"]
    Pi = Channel_scatter1["Pi"]
    beta = Channel_scatter1["beta"]
    t_nm = Channel_scatter1["t_nm"]
    R = Channel_scatter1["R"]
    dis_sca1_to_anntena = Channel_scatter1["dis_sca1_to_anntena"]
    phi_deg_v = Channel_scatter1["phi_deg_v"]
    theta_deg_v = Channel_scatter1["theta_deg_v"]

    phi_rad, theta_rad, varphi_rad, eta_rad = transform_angle_to_numpy(phi_deg, theta_deg, varphi_deg, eta_deg, N, M)
    phi_rad_v = np.zeros((V,N,max(M)))
    theta_rad_v = np.zeros((V,N,max(M)))

    phi_rad_v = np.zeros((V, N, max(M)))
    theta_rad_v = np.zeros((V, N, max(M)))

    print("Pi:", Pi)

    for v in range(V):
        for n in range(N):
            for m in range(M[n]):
                phi_rad_v[v, n, m]   = np.radians(phi_deg_v[v][n][m])
                theta_rad_v[v, n, m] = np.radians(theta_deg_v[v][n][m])

    print("phi_deg", phi_deg)
    print("theta_deg", theta_deg)
    # 近傍界の場合、指向性をサブアレー毎に計算する
    if NF_setting == "Near":
        a_phi_theta = channel.define_a(V, N, M, phi_rad_v, theta_rad_v,NF_setting)
    elif NF_setting == "Far":
        a_phi_theta = channel.define_a(V, N, M, phi_rad, theta_rad, NF_setting)
    
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
    
    print("t_nm - t0:", t_nm-t_nm[0,0])

    b_varphi_eta = np.array(b_varphi_eta)

    f_common = np.zeros((N, max(M), len(f_GHz)), dtype=np.complex64)
    f_common[None, None, :] = Amp_per_career_desital[:, :, None] * np.exp(1j * beta[:, :, None]) * np.exp(-1j * 2 * np.pi * f_GHz[None, None, :] * t_nm[:, :, None]) * b_varphi_eta[:, :, None]
    print("f_common shape : ", np.shape(f_common))
    # eta_rad と varphi_rad の組み合わせから位相計算
    c = np.cos(eta_rad) * np.sin(varphi_rad)  # shape: (N, M)
    # 各 u に対してブロードキャストで位相成分を計算！
    phase = np.exp(-1j * np.pi * u[:, None, None] * c[None, :, :])  # shape: (U, N, M)
    # 共通因子と位相成分を掛け合わせる
    a_temp = phase[:, :, :, None] * f_common[None, :, :, :]  # shape: (U, N, M, K)
    print("a_temp shape : ", np.shape(a_temp))
    # 最終的な出力を (N, M, U, K) に転置
    a_O_m_n_u_k = np.transpose(a_temp, (1, 2, 0, 3))
    print("a_O_m_n_u_k shape : ", np.shape(a_O_m_n_u_k))

    # 第1散乱体と原点O間で電波が伝搬に要する時間 ㉒
    R_tmp = np.zeros((N,max(M)))
    for i, values in enumerate(R):
        R_tmp[i, :len(values)] = values  # 長さが足りない部分はそのままゼロ
    R = R_tmp
    tau_1_ns = R / c_ns

    # 第1散乱体での複素振幅 ㊻
    tau_exp = np.exp(1j * 2 * np.pi * f_GHz[None, None, :] * tau_1_ns[:, :, None])[:,:,None,:]  # shape: (N, M, 1, K)
    # 要素ごとの積で計算完了！
    a_1_m_n_u_k = a_O_m_n_u_k * tau_exp  # shape: (N, M, U, K)
    print("a_1_m_n_u_k shape : ", np.shape(a_1_m_n_u_k))

    # 第1散乱体とアンテナ間の伝搬時間 ㉓
    tau_nmvqyqz = dis_sca1_to_anntena / c_ns

    a = np.zeros((N, max(M), V, U, Q, Q, len(f_GHz)), dtype=np.complex64)
    print("a_phi_theta shape : ", np.shape(a_phi_theta))
    # tau_nmvqyqz は (N, max(M), V, Q, Q) で与えられてる前提
    exp_term = np.exp(-2j * np.pi * f_GHz[None, None, None, None, None, :] * tau_nmvqyqz[:,:,:,:,:,None])  # shape: (V, N, max(M), Q, Q, K)
    print("exp_term shape : ", np.shape(exp_term))
    a = np.einsum('nmuk,vnm,vnmyzk->nmvuyzk', a_1_m_n_u_k, a_phi_theta, exp_term, optimize=True)
    a = np.sum(a, (0,1))
    print("a shape : ", np.shape(a))  # (V, U, Q, Q, K)
    
    w_DD_pape = np.zeros((V, Q, Q), dtype=np.complex64)
    g_DD_pape = np.zeros((V, N, max(M)), dtype=np.complex64)
    invalid_indices = []  # vの削除対象インデックスを格納するリスト

    n_dash_full = channel.noise_u_v_k(U, V)  # フルサイズを保持（後で毎回切る）
    score_list = []   # wごとの「代表周波数k0での大きさ」
    mag_list = []     # wごとの「全周波数での大きさ」(K,)
    # for w in range(W):            
    invalid_indices = []  # 各wごとに初期化
    v = 0  # v をループ内で一意に管理
    for face in range(3):
        for array in range(4):
            pa = beam_allocation[face, 0, array]
            pe = beam_allocation[face, 1, array]
            if pa == -100 and pe == -100:
                invalid_indices.append(v)  # 削除対象のvを記録
                # print(f"v={v} は削除されます。")
            elif pa == -1 and pe == -1:
                invalid_indices.append(v)
            else:
                w_DD_pape[v,:,:] = channel.DFT_weight_calc_pape(Q, pa, pe)
                g_DD_pape[v,:,:] = channel.g_dd_depend_on_pape(N, M, phi_rad, theta_rad, zeta, Q, v, pa, pe)
            v += 1  # ループごとに v を増加

    # vの次元を削除して V' にする
    valid = np.setdiff1d(np.arange(V), invalid_indices)  # 残すvのインデックス
    w_DD_pape_red = w_DD_pape[valid]      # (V′,Q,Q)
    a_red = a[valid]                      # (V′,U,Q,Q,K)
    n_dash_w = n_dash_full[:, valid, :]

###########################################################################################################################.
    if NF_setting == 'Near':
        H_w = np.einsum('vyz,vuyzk->uvk', w_DD_pape_red, a_red, optimize=True)  # (U, V′, K)
        H_w *= np.exp(1j * 2 * np.pi * f_GHz[None, None, :]*t_nm[0, 0])

        # 周波数軸
        f_Hz = f_GHz * 1e9
        
        # # ==================================================
        # # 疑似的に帯域拡張して IDFT（裾がどう変わるか確認用）
        # # ==================================================

        # extend_factor = 16              # 8倍くらいで十分
        # K = H_w.shape[-1]
        # K_ext = K * extend_factor

        # pad_total = K_ext - K
        # pad_left  = pad_total // 2
        # pad_right = pad_total - pad_left

        # # --- 端の値でフラットに延長（疑似無限帯域仮定） ---
        # H_left  = np.repeat(H_w[..., :1],  pad_left,  axis=-1)
        # H_right = np.repeat(H_w[..., -1:], pad_right, axis=-1)

        # H_ext = np.concatenate([H_left, H_w, H_right], axis=-1)

        # # --- IDFT ---
        # h_tau_ext = np.fft.ifft(H_ext, axis=-1)
        # h_mag_ext = np.linalg.norm(h_tau_ext, axis=(0,1))

        # # --- 遅延軸 ---
        # df = f_Hz[1] - f_Hz[0]
        # tau_ext = np.arange(K_ext) / (K_ext * df)
        # tau_ext_ns = tau_ext * 1e9


        # # ==================================================
        # # プロット
        # # ==================================================
        # plt.figure(figsize=(8,5))
        # plt.plot(tau_ext_ns, h_mag_ext)
        # plt.xlabel("Delay [ns]")
        # plt.ylabel("|h(τ)|")
        # plt.title("Pseudo-wideband IDFT (flat extension)")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        f_Hz  = f_GHz * 1e9
        K = f_Hz.size
        df = f_Hz[1] - f_Hz[0]

        # 遅延軸（IFFTの遅延サンプル位置）
        tau = np.arange(K) / (K * df)      # [s]
        tau_ns = tau * 1e9                 # [ns]

        H_mag_org = np.linalg.norm(H_w, axis=(0, 1))  # (K,)

        plt.figure(figsize=(8, 5))
        plt.plot(H_mag_org)
        plt.xlabel("Frequency index k")
        plt.ylabel("|H_bb,k|")
        title = "Channel amplitude in frequency domain"
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        # channel.save_current_fig_pdf(title)
        plt.show()

        # ==================================================
        # ② IFFT → 遅延領域（マスキング前）
        # ==================================================
        h_tau = np.fft.ifft(H_w, axis=-1)          # (U, V', K)
        h_mag = np.linalg.norm(h_tau, axis=(0, 1))    # (K,)

        h = h_mag  # (K,)
        # --- dB 化（ピーク基準） ---
        h_db = 20 * np.log10(h / np.max(h) + 1e-30)

        # --- 2000ns側（周期端）から判定 ---
        # 「影響ない」＝ th_db 未満
        under = (h_db < th_dB)

        # 末尾から見て、最初に「ほぼ0」になる点
        rev_idx = np.where(under[::-1])[0]

        if len(rev_idx) == 0:
            print("No region below", th_dB, "dB (sidelobe everywhere)")
        else:
            k = rev_idx[0]                  # 末尾からの距離
            idx_start = len(h) - 1 - k      # 元のインデックス

            print("below", th_dB, "dB starts at [ns]:", tau_ns[idx_start])
            print("wrap length judged negligible [ns]:", tau_ns[-1] - tau_ns[idx_start])

        
        plt.figure(figsize=(8, 5))
        plt.plot(tau_ns, h_mag)
        plt.xlabel("Delay [ns]")
        plt.ylabel("|h(τ)| (before masking)")
        title = "Delay response (IFFT, baseband)"
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        # channel.save_current_fig_pdf(title)
        plt.show()

        # ==================================================
        # ③ 遅延マスキング
        # ==================================================
        mask = (tau_ns >= 100.0) & (tau_ns <= tau_ns[idx_start])
        h_tau_masked = h_tau.copy()
        h_tau_masked[..., mask] = 0

        # ---- マスキング後の確認用グラフ ----
        h_mag_masked = np.linalg.norm(h_tau_masked, axis=(0, 1))

        plt.figure(figsize=(8, 5))
        plt.plot(tau_ns, h_mag_masked)
        plt.xlabel("Delay [ns]")
        plt.ylabel("|h(τ)| (after masking)")
        title = "Delay response after zero-masking"
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        # channel.save_current_fig_pdf(title)
        plt.show()

        # ==================================================
        # ④ FFTで周波数へ戻す（DC先頭のままFFT）
        # ==================================================
        H_back = np.fft.fft(h_tau_masked, axis=-1)

        H_mag_org  = np.linalg.norm(H_w,      axis=(0, 1))
        H_mag_back = np.linalg.norm(H_back, axis=(0, 1))

        plt.figure(figsize=(9, 5))
        plt.plot(H_mag_org,  label="Original")
        plt.plot(H_mag_back, "--", label="After masking+FFT")

        plt.xlabel("Frequency index k")
        plt.ylabel("|H|")
        title = "Channel amplitude before and after delay masking"
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # channel.save_current_fig_pdf(title)
        plt.show()

        # （必要なら）RFへ戻した後の比較も追加で見れる
        plt.figure(figsize=(9, 5))
        plt.plot(np.linalg.norm(H_w, axis=(0, 1)), label="Original H_w,k (as-is)")
        plt.plot(H_mag_back, "--", label="After masking+FFT (RF restored)")
        plt.xlabel("Frequency index k")
        plt.ylabel("|H|")
        title = "Channel amplitude (RF) before and after delay masking"
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # channel.save_current_fig_pdf(title)
        plt.show()

    """
# 近傍界チャネル
    if NF_setting == 'Near':
        H_w *= np.exp(1j * 2 * np.pi * f_GHz[None, None, :] * t_nm[0,0])
        # --- ① 中心周波数を引いてベースバンド化 ---
        fc = 0.5 * (f_GHz[0] + f_GHz[-1])      # GHz
        f_bb = (f_GHz - fc) * 1e9              # Hz に直すの忘れない

        # --- ② ベースバンド周波数で遅延戻し ---
        H_bb = H_w * np.exp(1j * 2 * np.pi * f_bb[None, None, :] * t_nm[0,0])

        # --- ③ IFFT が期待する並び（DC先頭）に変換 ---
        H_ifft = np.fft.ifftshift(H_bb, axes=-1)

        # --- ④ IFFT ---
        h_tau = np.fft.ifft(H_ifft, axis=-1)

        # --- ⑤ 振幅確認用 ---
        H_mag_org = np.linalg.norm(H_bb, axis=(0,1))


        plt.figure(figsize=(8,5))
        plt.plot(H_mag_org)
        plt.xlabel("Frequency index k")
        plt.ylabel("|H_tru,k|")
        title = "Channel amplitude in frequency domain"
        plt.title(title)
        
        plt.grid(True)
        plt.tight_layout()
        channel.save_current_fig_pdf(title)
        plt.show()

        # --- 周波数軸（H_w と必ず対応している前提） ---
        f_GHz = np.linspace(141.50025, 142.49975, 2000)
        f_Hz  = f_GHz * 1e9
        K = f_Hz.size
        df = f_Hz[1] - f_Hz[0]

        # 遅延軸
        tau = np.arange(K) / (K * df)      # [s]
        tau_ns = tau * 1e9                 # [ns]

        # 中心周波数
        f0 = np.mean(f_Hz)
        # ==================================================
        # ② IFFT → 遅延領域（マスキング前）
        # ==================================================
        h_tau = np.fft.ifft(H_w, axis=-1)          # (U, V, K)
        h_mag = np.linalg.norm(h_tau, axis=(0,1))  # (K,)

        plt.figure(figsize=(8,5))
        plt.plot(tau_ns, h_mag)
        plt.xlabel("Delay [ns]")
        plt.ylabel("|h(τ)| (before masking)")
        title = "Delay response (IFFT)"
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        channel.save_current_fig_pdf(title)
        plt.show()

        # ==================================================
        # ③ 遅延マスキング
        # ==================================================
        mask = (tau_ns >= 100.0) & (tau_ns <= 2000.0)
        h_tau_masked = h_tau.copy()
        h_tau_masked[..., mask] = 0

        # ---- マスキング後の確認用グラフ ----
        h_mag_masked = np.linalg.norm(h_tau_masked, axis=(0,1))

        plt.figure(figsize=(8,5))
        plt.plot(tau_ns, h_mag_masked)
        plt.xlabel("Delay [ns]")
        plt.ylabel("|h(τ)| (after masking)")
        title = "Delay response after zero-masking"
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # ==================================================
        # ④ FFTで周波数へ戻す
        # ==================================================
        H_bb_back = np.fft.fft(h_tau_masked, axis=-1)

        # ==================================================
        # ⑤ 元の 142GHz 帯に戻す
        # ==================================================
        H_back = H_bb_back * np.exp(-1j * 2*np.pi * f0 * tau)[None, None, :]

        # ==================================================
        # 周波数領域での比較
        # ==================================================
        H_mag_org  = np.linalg.norm(H_w,    axis=(0,1))
        H_mag_back = np.linalg.norm(H_back, axis=(0,1))

        plt.figure(figsize=(9,5))
        plt.plot(H_mag_org,  label="Original H_tru,k")
        plt.plot(H_mag_back, "--", label="After delay masking and FFT")
        plt.xlabel("Frequency index k")
        plt.ylabel("|H|")
        title = "Channel amplitude before and after delay masking"
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        channel.save_current_fig_pdf(title)
        plt.show()
    """



            # if NS == 'Wo_NS':
            #     H_w_est = H_w + n_dash_w[:, :, 0] / np.sqrt(Pu/2000)
            # elif NS == 'W_NS':
            #     H_w_est = H_w + (n_dash_w[:, :, :10].sum(axis=2)/10) / np.sqrt(Pu/2000)
############################################################################################################################################
# 遠方界チャネル

    """
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
            
            h_far = np.sum(h_dash_far, (2,3))
            H_compact = h_far[:, valid] 
            if NS == 'Wo_NS':
                H_est_compact = H_compact + n_dash_w[:, :, 0] / np.sqrt(Pu/2000)
            elif NS == 'W_NS':
                H_est_compact = H_compact + (n_dash_w[:, :, :10].sum(axis=2)/10) / np.sqrt(Pu/2000)
            # 全てのマルチパスの寄与
            h_tru_obj[Channel_number, w] = H_compact
            h_est_obj[Channel_number, w] = H_est_compact
    """

    return H_w

# 以下E-SDM
Pt_mW = 1000/2000
P_noise_mW = 6.31e-12
U=8
def mutual_coherence_matrix_from_Gram(G):
    """
    G: Gram行列 = H^H H, shape (Nt, Nt)

    戻り値:
        C: shape (Nt, Nt)
    """
    diag = np.real(np.diag(G))
    diag = np.where(diag <= 0, 1e-15, diag)
    denom = np.sqrt(np.outer(diag, diag))
    MC = np.abs(G) / denom
    MC = np.clip(MC, 0.0, 1.0)
    return MC

# グラム行列の固有値と固有ベクトルを求める関数
def calc_eigval(H):
    print(np.shape(H))
    h_norm = H / np.linalg.norm(H, axis=0, keepdims=True)
    print(h_norm.conj().T @ h_norm)
    H_H_H = np.conjugate(H).T @ H
    MC = mutual_coherence_matrix_from_Gram(H_H_H)
    print("Mutual Coherence \n", MC)
    eigval, eigvec = np.linalg.eigh(H_H_H)
    # 固有値を降順にソートし、インデックスを取得
    sorted_indices = np.argsort(eigval)[::-1]
    # 固有値と固有ベクトルをソート
    eigval = np.real(eigval[sorted_indices])  # 固有値を降順でソート
    eigvec = eigvec[:, sorted_indices]        # 固有ベクトルを対応するインデックスで並べ替え
    # 固有値が負の値のものを0に置き換え
    eigval[eigval < 0] = 0
    eigval = eigval.astype(float)
    return eigval, eigvec

# 注水定理を実行する関数
def water_filling_ratio(eig_vals, Pt, P_noise, iters=200):
    """
    eig_vals: 固有値 λ_j
    Pt: 総送信電力
    P_noise: 雑音電力
    """
    tol = 1e-6
    g = np.asarray(eig_vals, float)
    g = np.maximum(g, 1e-15)  # ゼロ割防止

    lo, hi = 0.0, 1e12
    for _ in range(iters):
        alpha = (lo + hi) / 2.0
        P = np.maximum(1/alpha - P_noise / g, 0.0)
        S = P.sum()
        if abs(S - Pt) < tol:
            break
        if S > Pt:
            lo = alpha
        else:
            hi = alpha

    # === 修正ポイント ===
    Ly = int(np.count_nonzero(P > tol))  # 有効レイヤ数

    if Ly > 0:
        P_used = P[:Ly]                     # 有効レイヤだけ取り出す
        p_ratio = P_used / P_used.sum()     # 正規化も有効レイヤ内で
    else:
        p_ratio = np.zeros_like(P)

    # 二分探索の最後に:
    alpha = (lo + hi) / 2.0
    mu = 1.0 / alpha
    return p_ratio, Ly


def generate_s(Ly):
    # 複素ガウス乱数の生成（実部と虚部を分散0.5で生成）
    s = (np.random.randn(Ly) + 1j * np.random.randn(Ly)) / np.sqrt(2)
    return s

# チャネル容量を求める関数(注意!! nearとfarで用いるH_truが異なります！！！！)
def calc_channel_capacity(H, H_tru, Pt_mW, P_noise_mW):
    H_val , H_vec = calc_eigval(H)
    power_allo, Ly = water_filling_ratio(H_val, Pt_mW, P_noise_mW)
    p_sqrt = np.sqrt(power_allo)
    p_diag = np.diag(p_sqrt)

    A = np.sqrt(Pt_mW)*p_diag
    Te_H = H_vec[:,:Ly]

    n_dash_dash = channel.noise_dash_dash(U)
    s = generate_s(Ly)
    x = Te_H @ A @ s
    y = H_tru @ x + n_dash_dash
    
    N_dash_dash_ly = np.random.normal(0, 1.778e-6, (U,Ly)) + 1j * np.random.normal(0, 1.778e-6, (U,Ly))
    S = np.eye(Ly)
    H_eff = H_tru @ Te_H @ S + (N_dash_dash_ly / np.sqrt(Pt_mW))
    Wr = np.ones((U,Ly), dtype=np.complex128)
    gamma0 = Pt_mW / P_noise_mW
    I_Ly = np.identity(Ly)
    W_MMSE = ( np.linalg.inv(H_eff.conj().T @ H_eff + (Ly / gamma0) * I_Ly) @ H_eff.conj().T ).T
    
    Wr = W_MMSE
    r = Wr.T @ y
    B = Wr.T @ H_tru @ Te_H @ A
    S = np.abs(np.diag(B)) ** 2
    I = np.sum(np.abs(B) ** 2, axis=1) - np.abs(np.diag(B)) ** 2

    Rnn = P_noise_mW * Wr.T @ (Wr.T).conj().T
    N = np.diag(Rnn)
    C_ly = np.log2(S / (I + N) + 1)
    C = np.real(np.sum(C_ly))
    

    return C, Ly, H_val



d = 15
Ssub_lam = 0
channel_type = "InH"
NF_setting = "Near"
Method = "Mirror"

# [1,1]は#9
# 5,14
test_num = 9
th_dB = 0
base  = np.load("C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Data/Base_InH.npy", allow_pickle=True)
channel_base = base[test_num]

Channel_scatter1 = Add_scatter1(channel_base, d, Ssub_lam, channel_type)
P_Near_dBm, P_far_dBm = Power_calculation(Channel_scatter1, channel_type, NF_setting, Method, d, f, test_num, test_num+1)

threshold = -73
ba_m1, row_m1 = Beam_allocation_method_1(P_Near_dBm, threshold, test_num)
# ba_m2, row_m2 = Beam_allocation_method_2(P_Near_dBm, threshold, test_num)

# 上位4候補のデータフレーム（オプション表示）
df_top_per_v = pd.concat(row_m1, ignore_index=True)
print(f"Ssub = {Ssub_lam} lam")
print(f"d = {d}m")
print("Beam allocation: \n", ba_m1)
print("row_m1:\n", df_top_per_v)
H = Channel_Calculation(ba_m1, Channel_scatter1, "W_NS", Ssub_lam, NF_setting, th_dB)
C, Ly, Hval = calc_channel_capacity(H[:,:,1], H[:,:,1], Pt_mW, P_noise_mW)
print("Channel Capacity [bps/Hz]:", C)


# print("H shape:", np.shape(H))