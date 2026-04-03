# 260319_RCS石垣島_ゼロマスキング検討
import numpy as np
import math
import Channel_functions as channel
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (import for 3D)
from pathlib import Path

# グローバル変数定義
Q = 16 # サブアレーの素子数
V = 12 # サブアレー数
U = 8 # UEのアンテナ素子数
lam_cen = (3.0 * 1e8) / (142 * 1e9)
f = np.linspace(141.50025*1e9, 142.49975*1e9, 2000) #2000個の周波数
f_GHz = np.linspace(141.50025, 142.49975, 2000) #GHz単位
f_GHz_val = f_GHz[1000] #中央周波数
lam = 0.3 / f_GHz #2000個の波長
c_ns = 0.3 # 光速 m/ns

#################################################################################################################
# ビーム割当法２(各サブアレー最大受信電力)
def Beam_allocation_method_2(Power, threshold, testch_num=0):
    """
    ビーム割当法2：各サブアレーの最大受信電力を優先して割り当て
    """
    beam_allocation = np.full((3, 2, 4), -1, dtype=int)  # w削除済み
    rows_per_v = []

    for v in range(V):
        valid_mask = Power[v] >= threshold
        valid_values = Power[v][valid_mask]

        if valid_values.size == 0:
            df_v = pd.DataFrame([{
                "Channel": testch_num,
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
        df_v.insert(0, "Channel", testch_num)
        df_v[["v", "pa", "pe"]] = df_v[["v", "pa", "pe"]].astype("Int64")
        rows_per_v.append(df_v)
        

    # 割り当て処理
    for face in range(3):
        for k in range(4):
            v = face * 4 + k
            df_v = rows_per_v[v]
            if isinstance(df_v, pd.DataFrame) and df_v["rank"].notnull().any():
                row = df_v.iloc[0]
                if row is not None and pd.notnull(row["pa"]):
                    beam_allocation[face, 0, k] = row["pa"]
                    beam_allocation[face, 1, k] = row["pe"]
    
    return beam_allocation, rows_per_v

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

# グラム行列の固有値と固有ベクトルを求める関数
def calc_eigval(H):
    h_norm = H / np.linalg.norm(H, axis=0, keepdims=True)
    H_H_H = np.conjugate(H).T @ H
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

    lo, hi = 1e-15, 1e12
    for _ in range(iters):
        alpha = (lo + hi) / 2.0
        # 最適電力を計算
        P = np.maximum(1/alpha - P_noise / g, 0.0)
        S = P.sum()
        if abs(S - Pt) < tol:
            break
        if S > Pt:
            lo = alpha
        else:
            hi = alpha

    tol_layer = 1e-6*Pt
    Ly = int(np.count_nonzero(P > tol_layer))  # 有効レイヤ数判定

    if Ly > 0:
        P_used = P[:Ly]                     # 有効レイヤだけ取り出す
        p_ratio = P_used / P_used.sum()     # 正規化も有効レイヤ内で
    else:
        p_ratio = np.zeros_like(P)

    return p_ratio, Ly

# 信号を生成する関数
def generate_s(Ly):
    # 複素ガウス乱数の生成（実部と虚部を分散0.5で生成）
    s = (np.random.randn(Ly) + 1j * np.random.randn(Ly)) / np.sqrt(2)
    return s

# チャネル容量を求める関数(注意!! nearとfarで用いるH_truが異なります！！！！)
# チャネル容量を求める関数(全サブキャリア平均対応)
def calc_channel_capacity(H_all, H_tru_all, Pt_mW, P_noise_mW):
    # もし2次元配列が渡された場合はK=1として扱えるように次元を追加
    if H_all.ndim == 2:
        H_all = H_all[:, :, np.newaxis]
        H_tru_all = H_tru_all[:, :, np.newaxis]

    K = H_all.shape[2]
    C_sum = 0.0
    Ly_sum = 0.0

    # サブキャリアごとに容量を計算して足し合わせる
    for k in range(K):
        H = H_all[:, :, k]
        H_tru = H_tru_all[:, :, k]

        H_val, H_vec = calc_eigval(H)
        power_allo, Ly = water_filling_ratio(H_val, Pt_mW, P_noise_mW)
        
        if Ly == 0:
            continue

        p_sqrt = np.sqrt(power_allo)
        p_diag = np.diag(p_sqrt)

        A = np.sqrt(Pt_mW) * p_diag
        Te_H = H_vec[:, :Ly]

        n_dash_dash = channel.noise_dash_dash(U)
        s = generate_s(Ly)
        x = Te_H @ A @ s
        y = H_tru @ x + n_dash_dash
        
        N_dash_dash_ly = np.random.normal(0, 1.778e-6, (U, Ly)) + 1j * np.random.normal(0, 1.778e-6, (U, Ly))
        S = np.eye(Ly)
        H_eff = H_tru @ Te_H @ S + (N_dash_dash_ly / np.sqrt(Pt_mW))
        gamma0 = Pt_mW / P_noise_mW
        I_Ly = np.identity(Ly)
        W_MMSE = (np.linalg.inv(H_eff.conj().T @ H_eff + (Ly / gamma0) * I_Ly) @ H_eff.conj().T).T
        
        Wr = W_MMSE
        r = Wr.T @ y
        B = Wr.T @ H_tru @ Te_H @ A
        S_pow = np.abs(np.diag(B)) ** 2
        I_pow = np.sum(np.abs(B) ** 2, axis=1) - np.abs(np.diag(B)) ** 2

        Rnn = P_noise_mW * Wr.T @ (Wr.T).conj().T
        N_pow = np.diag(Rnn)
        C_ly = np.log2(S_pow / (I_pow + N_pow) + 1)
        C_sum += np.real(np.sum(C_ly))
        Ly_sum += Ly
        
    # 全サブキャリアの平均を返す
    return C_sum / K, Ly_sum / K, None

def calc_channel_capacity_SingleLayer(H_all, H_tru_all, Pt_mW, P_noise_mW):
    if H_all.ndim == 2:
        H_all = H_all[:, :, np.newaxis]
        H_tru_all = H_tru_all[:, :, np.newaxis]

    K = H_all.shape[2]
    C_sum = 0.0

    for k in range(K):
        H = H_all[:, :, k]
        H_tru = H_tru_all[:, :, k]

        # 固有値・固有ベクトル（右特異ベクトル）
        H_val, H_vec = calc_eigval(H)

        Ly = 1
        power_allo = np.array([1.0])           # 全電力を第1層へ
        A = np.array([[np.sqrt(Pt_mW)]], dtype=np.complex128)
        Te_H = H_vec[:, :Ly]                   # 第1モードのみ

        n_dash_dash = channel.noise_dash_dash(U)
        s = generate_s(Ly)
        x = Te_H @ A @ s
        y = H_tru @ x + n_dash_dash

        # 受信側（MMSE）も Ly=1 でそのまま
        N_dash_dash_ly = (np.random.normal(0, 1.778e-6, (U, Ly))+ 1j*np.random.normal(0, 1.778e-6, (U, Ly)))
        H_eff = H_tru @ Te_H + (N_dash_dash_ly / np.sqrt(Pt_mW))

        gamma0 = Pt_mW / P_noise_mW
        I_Ly = np.identity(Ly)
        W_MMSE = (np.linalg.inv(H_eff.conj().T @ H_eff + (Ly / gamma0) * I_Ly) @ H_eff.conj().T).T

        Wr = W_MMSE
        B = Wr.T @ H_tru @ Te_H @ A
        S_pow = np.abs(np.diag(B)) ** 2
        I_pow = np.sum(np.abs(B) ** 2, axis=1) - S_pow

        Rnn = P_noise_mW * Wr.T @ (Wr.T).conj().T
        N_pow = np.diag(Rnn)

        C_ly = np.log2(S_pow / (I_pow + N_pow) + 1)
        C_sum += float(np.real(np.sum(C_ly)))

    return C_sum / K, 1.0, None


##################################################################################################################
def setting_NYUSIM_synario(Base_data_num, d):
    chi = Base_data_num['chi']
    N = Base_data_num['N']
    M = Base_data_num['M']
    Z = Base_data_num['Z']
    U_nm = Base_data_num['U']
    rho = Base_data_num['rho']
    tau = Base_data_num['tau']
    beta = Base_data_num['beta']
    theta_ND_deg = Base_data_num['theta_ND_deg']
    eta_ND_deg = Base_data_num['eta_ND_deg']
    eta_dir = np.degrees(np.arcsin(1 / d))
    theta_dir = -eta_dir
    theta_0_0 = theta_ND_deg[0][0] if len(theta_ND_deg[0]) > 0 else theta_ND_deg[0]
    eta_0_0 = eta_ND_deg[0][0] if len(eta_ND_deg[0]) > 0 else eta_ND_deg[0]

    Delta_EOD = theta_dir - theta_0_0
    Delta_EOA = eta_dir - eta_0_0

    theta_deg = [theta_ND_deg[n] + Delta_EOD for n in range(N)]
    eta_deg = [eta_ND_deg[n] + Delta_EOA for n in range(N)]
    
    NYUSIM_Synario_Data = {
        'chi': chi,
        'N': N,
        'M': M,
        'Z': Z,
        'U_nm': U_nm,
        'rho': rho,
        'tau': tau,
        'beta': beta,
        'phi_deg': Base_data_num['phi_deg'],
        'theta_deg': theta_deg,
        'varphi_deg': Base_data_num['varphi_deg'],
        'eta_deg': eta_deg
    }
    
    return NYUSIM_Synario_Data

def simulation_core(channel_type, Q, lam, d, Pu_dBm, Ssub_lam, Synario_Data, use_H):
    chi = Synario_Data['chi']
    N = Synario_Data['N']
    M = Synario_Data['M']
    Z = Synario_Data['Z']
    U_nm = Synario_Data['U_nm']
    rho = Synario_Data['rho']
    tau = Synario_Data['tau']
    beta = Synario_Data['beta']
    phi_deg = Synario_Data['phi_deg']
    theta_deg = Synario_Data['theta_deg']
    varphi_deg = Synario_Data['varphi_deg']
    eta_deg = Synario_Data['eta_deg']
    
    ##############################################################################
    # ここからパイロット信号送受信
    Pu_mW = 10 ** (Pu_dBm / 10) # mW単位
    Pu_dBm_per_carrer = 10 * np.log10(Pu_mW / 2000) # UEの1キャリアあたりの送信電力
    Pu_mW_per_carrer = Pu_mW / 2000 # UEの1キャリアあたりの送信電力mW単位

    # 受信電力計算
    
    theta_rad = np.zeros((N,max(M)))
    phi_rad = np.zeros((N,max(M)))
    varphi_rad = np.zeros((N,max(M)))
    eta_rad = np.zeros((N,max(M)))
    
    phi_rad_v = np.zeros((V,N,max(M)))
    theta_rad_v = np.zeros((V,N,max(M)))
    varphi_rad_v = np.zeros((V,N,max(M)))
    eta_rad_v = np.zeros((V,N,max(M)))
    
    for n in range(N):
        for m in  range(M[n]):
            theta_rad[n][m] = np.radians(theta_deg[n][m])
            phi_rad[n][m] = np.radians(phi_deg[n][m])
            varphi_rad[n][m] = np.radians(varphi_deg[n][m])
            eta_rad[n][m] = np.radians(eta_deg[n][m])
    
    
    Pr_dBm = channel.calc_Pr(lam_cen, d, chi, Pu_dBm, channel_type, do=1)
    Pr_dBm_each_career = channel.calc_Pr_each_career(lam_cen, d, chi, Pu_dBm_per_carrer, channel_type, do=1)
    t_nm = channel.abs_timedelays(d, rho, tau, N, M)
    
    R, MUE_coordinate  = channel.Mirror_UE_positions(d, N, M, rho, tau, phi_rad, theta_rad)
    
    # サブアレー各素子の座標wideversion
    subarray_v_qy_qz = channel.calc_anntena_xyz_Ssub(lam_cen, V, Q, Ssub_lam)
    # サブアレーの各素子をプロットしたグラフ
    # plot_subarray_antennas(subarray_v_qy_qz)
    
    D_subarray, Rayleigh_distance = channel.calc_Rayleigh_distance(subarray_v_qy_qz)
    print(f"Rayleigh_distance: {Rayleigh_distance} m, D_subarray: {D_subarray} m")
    
    # n番目のTCのm番目のUE鏡像体#0と，v番目のサブアレーのqy, qz番目のアンテナ素子間の距離をrm,n,v, qy, qz 式14
    r_mnv0qyqz = channel.distance_to_eachanntena(MUE_coordinate, subarray_v_qy_qz, N, M, V, Q)
    # UEの#0から各アンテナ素子への伝搬時間 ㉓
    tau_mnv0qyqz = r_mnv0qyqz / c_ns 
    phi_deg_v, theta_deg_v = channel.calc_each_subarray_AOD_EOD(N, M, MUE_coordinate, subarray_v_qy_qz, V)
    
    for v in range(V):
        for n in range(N):
            for m in range(M[n]):
                phi_rad_v[v, n, m]    = np.radians(phi_deg_v[v][n][m])
                theta_rad_v[v, n, m]  = np.radians(theta_deg_v[v][n][m])
                varphi_rad_v[v, n, m] = np.radians(varphi_deg[n][m] - phi_deg[n][m] + phi_deg_v[v][n][m])
                eta_rad_v[v, n, m]    = np.radians(eta_deg[n][m]    + theta_deg[n][m] - theta_deg_v[v][n][m])

    t_nm = channel.abs_timedelays(d, rho, tau, N, M) #絶対遅延
    P_mW = channel.cluster_power(Pr_dBm, N, tau, Z, channel_type)
    Pi_mW = channel.SP_power(N, M, P_mW, rho, U_nm, channel_type)
    print("Pi_mW:", Pi_mW)
    P_each_career_mW = channel.cluster_Power_each_career(Pr_dBm_each_career, Z, N, tau, channel_type)
    Pi_each_career_mW = channel.SP_Power_each_career(N, M, P_each_career_mW, rho, U_nm, channel_type)
    
    DFT_weights = channel.DFT_weight_calc(Q)
    

    b_varphi_eta = channel.define_b_verphi_eta(N, M, eta_rad)
    Amp_per_career_mW = np.sqrt(Pi_each_career_mW)
    

    complex_Amp_at_MUE = []
    for n in range(N):
        mlen = int(M[n])

        amp = Amp_per_career_mW[n, :mlen]      # (mlen,)
        ph  = np.exp(1j * beta[n])          # (mlen,)
        bve = np.array(b_varphi_eta[n])[:mlen]  # ★ここ追加 (mlen,)

        complex_Amp_at_MUE.append(amp * ph * bve)

    a_phi_theta = channel.define_a(V, N, M, phi_rad_v, theta_rad_v,NF_setting)
    complex_Amp_at_each_antena = channel.complex_Amp_at_each_antena(f, V, Q, N, M, r_mnv0qyqz, complex_Amp_at_MUE, a_phi_theta)

    n_k_v = channel.noise_n_k_v(V)
    P_sub_dash_dBm = channel.near_Power_inc_noise_optimized(V, Q, f, DFT_weights, complex_Amp_at_each_antena, n_k_v)
    

    # ここまでパイロット信号受信電力計算##############################################################
    # ここからチャネル計算
    
    # チャネルモデルに合わせて振幅１の複素振幅
    # (本来はサブキャリアごとに異なるが、ここでは代表値として中央周波数で計算)
    Amp_desital = np.sqrt(Pi_mW) / np.sqrt(Pu_mW)
    K = len(f)
    
    b_varphi_eta = channel.define_b_verphi_eta(N, M, eta_rad)
    b_varphi_eta_v = channel.define_b_verphi_eta_v(V, N, M, eta_rad_v)
    
    u = np.arange(U)  # shape: (U,)
    # ベータのデータ形式をnumpy配列に変換
    beta_tmp = np.zeros((N,max(M)))
    for i, values in enumerate(beta):
        beta_tmp[i, :len(values)] = values  # 長さが足りない部分はそのままゼロ
    beta = np.array(beta_tmp)
    t_nm_tmp = np.zeros((N,max(M)))
    for i, values in enumerate(t_nm):
        t_nm_tmp[i, :len(values)] = values  # 長さが足りない部分はそのままゼロ
    t_nm = np.array(t_nm_tmp)
    
    b_varphi_eta = np.array(b_varphi_eta)
    
    a_MUE_vnm = Amp_desital * np.exp(1j * beta) * b_varphi_eta_v
    
    a_mn0vkqyqz = np.zeros((N, max(M), V, Q, Q, K), dtype=np.complex64)
    lam = 0.3 / f_GHz

    exp_term = np.exp(-2j * np.pi * f_GHz[None, None, None, None, None, :] * tau_mnv0qyqz[:, :, :, :, :, None])  # shape: (N, max(M), V, Q, Q, K)

    # 各アンテナ素子においての複素振幅　式47
    a_mn0vkqyqz = np.einsum('vnm,vnm,vnmyzk->nmvyzk', a_MUE_vnm, a_phi_theta, exp_term, optimize=True)

    c = np.cos(eta_rad_v) * np.sin(varphi_rad_v)
    # (V, N, M) -> (N, M, V) へ明示的に入れ替え
    c = np.transpose(c, (1, 2, 0))
    
    # 位相 (N, M, U, V, 1, 1, 1)
    phase = np.exp(
        -1j * np.pi 
        * u[None, None, :, None, None, None, None]     # (1,1,U,1,1,1,1)
        * c[:, :, None, :, None, None, None]           # (N,M,1,V,1,1,1)
    )

    # a_mn0vkqyqz を (N, M, 1, V, Q, Q, K) にして掛ける → (N, M, U, V, Q, Q, K)
    a_mnuvkqyqz = a_mn0vkqyqz[:, :, None, :, :, :, :] * phase
    # すべてのマルチパスについて和を取る　式48  
    a_uvkqyqz = np.sum(a_mnuvkqyqz, (0,1))

    # 以下ビーム割当###############################################################
    w_DD_pape = np.zeros((V, Q, Q), dtype=np.complex64)
    invalid_indices = []  # vの削除対象インデックスを格納するリスト

    # 各サブアレーの最大受信電力を優先してビーム割当
    threshold = -73 # ビーム割当の閾値
    ba, row = Beam_allocation_method_2(P_sub_dash_dBm, threshold, testch_num=0)
    print(ba)
    
    invalid_indices = []  # 各wごとに初期化
    v = 0  # v をループ内で一意に管理
    for face in range(3):
        for array in range(4):
            pa = ba[face, 0, array]
            pe = ba[face, 1, array]
            if pa == -100 and pe == -100:
                invalid_indices.append(v)  # 削除対象のvを記録
            elif pa == -1 and pe == -1:
                invalid_indices.append(v)
            else:
                w_DD_pape[v,:,:] = channel.DFT_weight_calc_pape(Q, pa, pe)
            v += 1  # ループごとに v を増加

    # vの次元を削除して V' にする
    valid = np.setdiff1d(np.arange(V), invalid_indices)  # 残すvのインデックス
    w_DD_pape_red = w_DD_pape[valid]      # (V′,Q,Q)
    a_red = a_uvkqyqz[:,valid,:,:,:]  # (U, V′, Q, Q, K)
    n_dash_uv_full = channel.noise_dash_K_10(U, V)
    n_dash_uv_vdash = n_dash_uv_full[:, valid, :]

    # チャネル行列を計算　式49
    h_uvk = np.einsum('vyz,uvyzk->uvk', w_DD_pape_red, a_red, optimize=True)  # (V′)
    if use_H == "E_wo":
        h_uvk = h_uvk + n_dash_uv_vdash[:,:,:,0]  # (U, V′, K)
    
###########################################################################################################################.
    h_uvk *= np.exp(1j * 2 * np.pi * f_GHz[None, None, :]*t_nm[0, 0])

    # 周波数軸
    f_Hz = f_GHz * 1e9
    K = f_Hz.size
    df = f_Hz[1] - f_Hz[0]

    # 遅延軸（IFFTの遅延サンプル位置）
    tau = np.arange(K) / (K * df)      # [s]
    tau_ns = tau * 1e9                 # [ns]

    H_mag_org = np.linalg.norm(h_uvk, axis=(0, 1))  # (K,)

    # plt.figure(figsize=(8, 5))
    # plt.plot(H_mag_org)
    # plt.xlabel("Frequency index k")
    # plt.ylabel("|H_bb,k|")
    # title = "Channel amplitude in frequency domain"
    # plt.title(title)
    # plt.grid(True)
    # plt.tight_layout()
    # # channel.save_current_fig_pdf(title)
    # plt.show()

    # ==================================================
    # ① 対象拡張と 2K IDFT
    # ==================================================
    # H(0)...H(K-1) の後ろに H(K-1)...H(0) を結合して 2K ポイントにする
    # [..., ::-1] は最後の次元（K）を逆順にする Python のスライス操作です
    H_w_rev = h_uvk[..., ::-1]
    H_w_2K = np.concatenate([h_uvk, H_w_rev], axis=-1)  # shape: (U, V', 2K)

    # 2K IDFT を実行
    h_tau_2K = np.fft.ifft(H_w_2K, axis=-1)

    # ==================================================
    # ② ゼロマスキング (100 ns ～ 1900 ns)
    # ==================================================
    # 拡張前の遅延軸における 100 ns のインデックス L を計算
    # 元の遅延分解能 dt = 1 / (K * df)
    dt = 1.0 / (K * df)
    L = int(np.round(100e-9 / dt))  # 100 ns に対応するインデックス

    # 指定通り、h^(2L) ～ h^(2K - 1 - 2L) を 0 に置き換える
    h_tau_2K_masked = h_tau_2K.copy()
    # Pythonのスライスは終端を含まないため、後ろのインデックスは 2*K - 2*L とします
    h_tau_2K_masked[..., 2*L : 2*K - 2*L] = 0

    # ==================================================
    # ③ 2K DFT と前半部分の抽出 (雑音抑圧チャネルの取得)
    # ==================================================
    # DFT を適用
    H_w_2K_masked = np.fft.fft(h_tau_2K_masked, axis=-1)

    # 前半 K ポイント分を取り出し、雑音抑圧されたチャネルとする
    H_w_denoised = H_w_2K_masked[..., :K]  # shape: (U, V', K)

    # ==================================================
    # ④ 結果の比較プロット（おまけ：マスキングの効果確認）
    # ==================================================
    # マスキング前後の周波数特性の振幅
    H_mag_denoised = np.linalg.norm(H_w_denoised, axis=(0, 1))

    """
    plt.figure(figsize=(8, 5))
    plt.plot(H_mag_org, label="Original", alpha=0.7)
    plt.plot(H_mag_denoised, label="Denoised (Zero-Masked)", linestyle='--')
    plt.xlabel("Frequency index k")
    plt.ylabel("|H_bb,k|")
    plt.title("Channel amplitude in frequency domain (Comparison)")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# ==================================================
    # （修正版）時間領域・遅延プロファイルのプロット（絶対値表記）
    # ==================================================
    # 2Kポイントの遅延軸（ns）を作成
    dt_2K = 1.0 / (2 * K * df)
    tau_2K_ns = np.arange(2 * K) * dt_2K * 1e9

    # 空間次元（U, V'）にわたってノルム（振幅・絶対値）をとる
    h_mag_2K = np.linalg.norm(h_tau_2K, axis=(0, 1))               # マスキング前
    h_mag_2K_masked = np.linalg.norm(h_tau_2K_masked, axis=(0, 1)) # マスキング後

    # グラフ描画
    plt.figure(figsize=(8, 5))
    # 絶対値をそのままプロット
    plt.plot(tau_2K_ns, h_mag_2K, label="Original (with Noise)", color='tab:red', alpha=0.7)
    plt.plot(tau_2K_ns, h_mag_2K_masked, label="Masked (Denoised)", color='tab:blue', linestyle='--')
    
    # ゼロマスキングした範囲（100ns ～ 1900ns）を背景色で強調（オプション）
    # plt.axvspan(100, 1900, color='gray', alpha=0.2, label="Masked Region")

    plt.xlabel("Delay [ns]")
    plt.ylabel("Amplitude |h(τ)|")  # 縦軸のラベルを絶対値表記に変更
    plt.title("Delay Profile in Time Domain (Absolute Value)")
    
    # 横軸の表示範囲（1000nsまで）
    plt.xlim(-50, 2050)
    plt.xticks([0, 100, 500, 1000, 1500, 1900, 2000], fontsize="small")
    
    # 縦軸は0からスタートさせ、上は自動調整（ピークより少し上まで）
    plt.ylim(bottom=0)
    
    plt.grid(True, linestyle=':')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    """
    
    print(h_uvk)
    
    return h_uvk, H_w_denoised, ba


def sweep_capacity_vs_d_mc(
    channel_indices,
    channel_type, Q, lam,
    d_values, Ssub_list,
    base_seed, MC,
    Base_data,
    Pu_dBm,
    Pt_mW, P_noise_mW,
    use_H="T",
    return_std=True,
    save_folder=None
):
    results = {}

    for Ssub_lam in Ssub_list:
        results[Ssub_lam] = {
            "d": [],
            "C_mean": [],
            "C_std": [],
            "C_single_mean": [],
            "C_single_std": [],
            "Ly": [],
        }

        for d in d_values:
            C_list = []
            C_single_list = []
            Ly_list = []

            for testch_num in channel_indices:
                np.random.seed(base_seed + testch_num)

                Synario_Data = setting_NYUSIM_synario(Base_data[testch_num], d)
                H, H_use, ba = simulation_core(
                    channel_type, Q, lam, d, Pu_dBm,Ssub_lam,
                    Synario_Data, use_H=use_H
                )
                
                C, Ly, _ = calc_channel_capacity(H_use, H, Pt_mW, P_noise_mW)
                C_single, _, _ = calc_channel_capacity_SingleLayer(H_use, H, Pt_mW, P_noise_mW)

                C_list.append(C)
                C_single_list.append(C_single)
                Ly_list.append(Ly)

            C_arr = np.asarray(C_list, float)
            C1_arr = np.asarray(C_single_list, float)
            Ly_arr = np.asarray(Ly_list, float)

            results[Ssub_lam]["d"].append(d)
            results[Ssub_lam]["C_mean"].append(C_arr.mean())
            results[Ssub_lam]["C_std"].append(C_arr.std(ddof=1) if len(C_arr) >= 2 else 0.0)
            results[Ssub_lam]["C_single_mean"].append(C1_arr.mean())
            results[Ssub_lam]["C_single_std"].append(C1_arr.std(ddof=1) if len(C1_arr) >= 2 else 0.0)
            results[Ssub_lam]["Ly"].append(Ly_arr.mean())

        print(f"Ssub={Ssub_lam}lam done.")

    # plot_capacity(results, Ssub_list, return_std=return_std, MC=MC, use_H=use_H, save_folder=save_folder)
    # plot_layers(results, Ssub_list, use_H=use_H, save_folder=save_folder)
    return results


def plot_capacity(results, Ssub_list, return_std=False, MC=1, use_H="T", save_folder=None):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",    # ← これが「本物の斜体」の鍵です！
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

    if use_H == "T":
        title_str = "Channel Capacity vs BS-UE Distance (True Channel)"
    elif use_H == "E_w":
        title_str = "Channel Capacity vs BS-UE Distance (Estimated Channel W/ NS)"
    elif use_H == "E_wo":
        title_str = "Channel Capacity vs BS-UE Distance (Estimated Channel W/o NS)"

    color_map = {0:"tab:green", 50:"tab:blue"}
    default_color = "tab:red"

    for Ssub_lam in Ssub_list:
        d = results[Ssub_lam]["d"]
        Cm = results[Ssub_lam]["C_mean"]
        Cs = results[Ssub_lam]["C_std"]
        C1 = results[Ssub_lam]["C_single_mean"]

        color = color_map.get(Ssub_lam, default_color)

        ax.plot(d, Cm, marker="o", markersize=4.5, lw=1.6, color=color, label=fr"{Ssub_lam}$\lambda$ Multi")
        ax.plot(d, C1, marker="s", markersize=4.5, ls="--", lw=1.6, color=color, markerfacecolor="none", label=fr"{Ssub_lam}$\lambda$ Single")

        if return_std and MC >= 2:
            ax.fill_between(d, Cm-Cs, Cm+Cs, color=color, alpha=0.15)

    ax.set_xlabel("BS-UE Distance (m)")
    ax.set_ylabel("Channel Capacity (bps/Hz)")
    ax.set_ylim(0, 60)
    ax.grid(True)
    ax.legend(title=r"$S_\mathrm{sub}$ Layer")

    ax.set_title("")

    if save_folder is not None:
        channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall原稿使用"), folder=save_folder, variants=("Paper",)) #←　カンマ必須！！！
        
        ax.set_title(title_str, size=15)
        channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall原稿使用"), folder=save_folder, variants=("Slide",)) #←　カンマ必須！！！

    plt.show()
    plt.close(fig)
    
def plot_layers(results, Ssub_list, use_H="T", save_folder=None):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",    # ← これが「本物の斜体」の鍵です！
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

    if use_H == "T":
        title_str = "Number of Layers vs BS-UE Distance (True Channel)"
    elif use_H == "E_w":
        title_str = "Number of Layers vs BS-UE Distance (Estimated Channel W/ NS)"
    elif use_H == "E_wo":
        title_str = "Number of Layers vs BS-UE Distance (Estimated Channel W/o NS)"

    color_map = {0:"tab:green", 50:"tab:blue"}
    default_color = "tab:red"
    offset = {0:-0.4, 50:0.0, 100:+0.4}
    for Ssub_lam in Ssub_list:
        d0 = np.array(results[Ssub_lam]["d"], float)
        d  = d0 + offset.get(Ssub_lam, 0.0)
        Ly = np.array(results[Ssub_lam]["Ly"], float)

        color = color_map.get(Ssub_lam, default_color)

        ax.plot(d, Ly, lw=1.6, color=color, zorder=1)
        ax.scatter(
            d, Ly,
            marker="o",
            s=35,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=2,
            label=fr"{Ssub_lam}$\lambda$"
        )


        ax.set_xlabel("BS-UE Distance (m)")
        ax.set_ylabel("Number of Layers")
        ax.set_ylim(0, 8.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linewidth=0.4, alpha=0.3)
        ax.legend(title=r"$S_\mathrm{sub}$")

    ax.set_title("")

    if save_folder is not None:
        channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall原稿使用"), folder=save_folder, variants=("Paper",)) #←　カンマ必須！！！
        
        ax.set_title(title_str, size=15)
        channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall原稿使用"), folder=save_folder, variants=("Slide",)) #←　カンマ必須！！！

    plt.show()
    plt.close(fig)

def plot_eigs(results, Ssub_list, k_list=(1,2), use_H="T", save_folder=None):
    fig, ax = plt.subplots(constrained_layout=True)

    if use_H == "T":
        title_str = "Eigenvalues vs BS-UE Distance (True Channel)"
    elif use_H == "E_w":
        title_str = "Eigenvalues vs BS-UE Distance (Estimated Channel W/ NS)"
    elif use_H == "E_wo":
        title_str = "Eigenvalues vs BS-UE Distance (Estimated Channel W/o NS)"
    else:
        title_str = "Eigenvalues vs BS-UE Distance"

    color_map = {0:"tab:green", 50:"tab:blue"}
    default_color = "tab:red"
    ls_map = {1:"-", 2:"--", 3:":", 4:"-."}  # λ1,λ2,...で線種変える

    for Ssub_lam in Ssub_list:
        d = np.array(results[Ssub_lam]["d"], float)
        eig_mean_list = results[Ssub_lam]["eig_mean"]  # list of (Nt,)
        eig_mat = np.vstack(eig_mean_list)             # (len(d), Nt)

        color = color_map.get(Ssub_lam, default_color)

        for k in k_list:
            if k-1 >= eig_mat.shape[1]:
                continue
            ax.plot(d, eig_mat[:, k-1],
                    lw=2.2, color=color, ls=ls_map.get(k, "-"),
                    marker=None,
                    label=f"Ssub={Ssub_lam}λ, λ{k}")

    ax.set_xlabel("BS-UE Distance [m]", size=12)
    ax.set_ylabel("Eigenvalue", size=12)
    ax.set_yscale("log")  # ★必須（桁が違いすぎる）
    ax.grid(True, which="both")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9)

    ax.set_title("")

    if save_folder is not None:
        channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall"), folder=save_folder, variants=("Paper",))
        ax.set_title(title_str, size=15)
        channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall"), folder=save_folder, variants=("Slide",))

    plt.show()
    plt.close(fig)

###############################################################
channel_type = "InH"
NF_setting = "Near"
Method = "Mirror"

MC = 1             # モンテカルロ回数

Synario = "NYUSIM"  # "Direct" or "NYUSIM"
# channel_indices = range(1, 2, 1)  # NYUSIMチャネルの場合のインデックスリスト(範囲取って平均)
channel_indices = [0]  # NYUSIMチャネルの場合のインデックスリスト(個別指定)
use_H = "T" # 'T' : 真のチャネル行列 , 'E_w' : 推定&同相加算　''E_wo' : 推定&非同相加算

# パイロット信号送信電力パラメータ
Pu_dBm = 30  # UEの送信電力(dBm)※全サブキャリア

save_folder = None #: グラフを保存しない, フォルダ名 : 保存するフォルダ名
# save_folder =  f"Channel_{channel_indices[0]}" 
# save_folder = "Resized"
################################################################pr

Base_data = np.load(f"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Data/Base_{channel_type}.npy", allow_pickle=True)
print("N =", Base_data[channel_indices[0]]["N"], "M =", Base_data[channel_indices[0]]["M"])
phi_disp = [arr % 360 for arr in Base_data[channel_indices[0]]["phi_deg"]]
print("phi_deg_disp =", phi_disp)
print("rho =", Base_data[channel_indices[0]]["rho"])
print("tau =", Base_data[channel_indices[0]]["tau"])
print("beta =", Base_data[channel_indices[0]]["beta"])

base_seed = 9  # 今までの固定seed

#E-SDMパラメータ
Pt_mW = 1000/2000  #通信時のサブキャリアごとの基地局送信電力
P_noise_mW = 6.31e-12 #500kHzあたりの雑音電力

# シミュレーションパラメータ
d_values = list(range(5, 51, 5))
Ssub_list = [0, 50, 100]


# picked_idx, C_med, idx_list, C_list = pick_typical_channel_by_capacity_totalpaths(
#     Base_data,
#     channel_type=channel_type, Q=Q, lam=lam,
#     d_pick=5, Ssub_lam=0,
#     Pu_dBm=Pu_dBm,
#     Pt_mW=Pt_mW, P_noise_mW=P_noise_mW,
#     use_H="T",
#     total_paths_target=2,
#     base_seed=base_seed
# )
# channel_indices = [picked_idx]
# print("Picked channel index for sweep:", channel_indices)

results = sweep_capacity_vs_d_mc(
    channel_indices=channel_indices,
    channel_type=channel_type, Q=Q, lam=lam,
    d_values=d_values, Ssub_list=Ssub_list,
    base_seed=base_seed, MC=MC,
    Base_data=Base_data,
    Pu_dBm=Pu_dBm,
    Pt_mW=Pt_mW, P_noise_mW=P_noise_mW,
    use_H=use_H,
    return_std=False,
    save_folder=save_folder
)

