# シミュレーションシナリオ 251117 Channel_number=3 でSsubごとのチャネル容量を確認する
import numpy as np
import math
import Channel_functions as channel
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (import for 3D)

# 必要なパラメータ設定
lam_cen = (3.0 * 1e8) / (142 * 1e9)
Pt = 10 # BSの送信電力
Pu_dBm_per_carrer = -23 # UEの1キャリアあたりの送信電力
g_dB = 0 # アンテナゲイン
Q = 16 # サブアレーの素子数
V = 12 # サブアレー数
U=8 # UEのアンテナ素子数
f = np.linspace(141.50025*1e9, 142.49975*1e9, 2000) #2000個の周波数
f_GHz = np.linspace(141.50025, 142.49975, 2000) #GHz単位
f_GHz_val = f_GHz[1000] #中央周波数
lam = 0.3 / f_GHz #2000個の波長
c_ns = 0.3 # 光速 m/ns
Pu = 10 
threshold = -73 # ビーム割当の閾値



# シミュレーションシナリオ 251120 正面から到来するシナリオ
#################################################################################################################
# サブアレーの位置があっているか確認するプロット用関数
def plot_subarray_antennas(subarray_v_qy_qz):
    """
    subarray_v_qy_qz : shape = (V, Q, Q, 3)
        [v, qy, qz, coord] でアンテナ座標 (x,y,z) が入っている配列
    """
    V, Qy, Qz, coord_dim = subarray_v_qy_qz.shape
    assert coord_dim == 3, "最後の次元は (x,y,z) の3次元になっている必要があります"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 各サブアレーごとにプロット
    for v in range(V):
        # (Qy, Qz, 3) → (Qy*Qz, 3) に reshape
        xyz = subarray_v_qy_qz[v].reshape(-1, 3)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        ax.scatter(x, y, z, s=10)  # 色は自動に任せる

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Subarray antenna positions (3D)")

    # 軸スケールを揃える（立方体で表示）
    all_xyz = subarray_v_qy_qz.reshape(-1, 3)
    x_min, y_min, z_min = all_xyz.min(axis=0)
    x_max, y_max, z_max = all_xyz.max(axis=0)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0
    z_mid = (z_max + z_min) / 2.0

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    plt.tight_layout()
    plt.show()

# ビーム割当法１多様性重視の割り当て
def Beam_allocation_method_1(Power, threshold, test_num):
    """
    ビーム割当法1：1,2,3,4位と順に割り当て、ダイバーシティを優先
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
                row = df_v.iloc[k] if k < len(df_v) else None
                if row is not None and pd.notnull(row["pa"]):
                    beam_allocation[face, 0, k] = row["pa"]
                    beam_allocation[face, 1, k] = row["pe"]
    
    return beam_allocation, rows_per_v

# ビーム割当法２(各サブアレー最大受信電力)
def Beam_allocation_method_2(Power, threshold, test_num):
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

# 以下E-SDM
Pt_mW = 1000/2000
P_noise_mW = 6.31e-12
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
    return MC


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

def mutual_coherence_hi_hj(H: np.ndarray, i: int, j: int) -> float:
    """
    i番目の列要素から成る8次元列ベクトルを hi とする。
    j番目の列要素から成る8次元列ベクトルを hj とする。

    ||hi|| = √(hi^H hi),  ||hj|| = √(hj^H hj)
    μ_{i,j} = |hi^H hj| / (||hi|| ||hj||)

    Parameters
    ----------
    H : np.ndarray
        下り回線のチャネル行列。想定は shape (8, 4)（複素数OK）。
    i, j : int
        列番号（1始まりで渡す想定：i=1..4, j=1..4）

    Returns
    -------
    float
        mutual coherence μ_{i,j}
    """
    # --- i番目・j番目の列要素から成る8次元列ベクトルを取り出す ---
    hi = H[:, i-1]  # shape (8,)
    hj = H[:, j-1]  # shape (8,)

    # --- ノルム ||hi|| = √(hi^H hi), ||hj|| = √(hj^H hj) ---
    # hi^H hi は np.vdot(hi, hi) で計算できる（vdotは第1引数を共役にする）
    norm_hi = np.sqrt(np.vdot(hi, hi))
    norm_hj = np.sqrt(np.vdot(hj, hj))

    # 0割り防止（ほぼ起きない想定だけど安全に）
    denom = np.abs(norm_hi) * np.abs(norm_hj)
    if denom == 0:
        return 0.0

    # --- μ_{i,j} = |hi^H hj| / (||hi|| ||hj||) ---
    mu_ij = np.abs(np.vdot(hi, hj)) / denom

    # 実数スカラーとして返す
    return float(np.real(mu_ij))
##################################################################################################################
# 直接波のみ、正面からの設定を作るための関数
def setting_Direct_synario(channel_type, Q, lam, d, Ssub_lam, NS, threshold, test_num):
    chi = channel.define_chi(channel_type)
    N = 1
    M = np.ones(1, dtype=int)
    Z = channel.define_Zn(N,channel_type)
    U_nm = channel.define_Unm(N,M,channel_type)
    rho = channel.intracluster_delays(M, N, setting='InH')
    tau = channel.cluster_excess_delays(rho, N, M, channel_type)
    beta = channel.SP_phases(M,N)
    phi_deg = [np.zeros(M[i]) for i in range(N)]
    theta_deg = [np.zeros(M[i]) for i in range(N)]
    eta_deg = [np.zeros(M[i]) for i in range(N)]
    varphi_deg = [np.zeros(M[i]) for i in range(N)]
    eta_dir = np.degrees(np.arcsin(1 / d))
    eta_deg[0][0] = eta_dir
    theta_dir = -eta_dir
    theta_deg[0][0] = theta_dir
    varphi_deg[0][0] = 0
        
    Pr = channel.calc_Pr(lam_cen, d, chi, Pt, g_dB, channel_type, do=1)
    Pr_each_career = channel.calc_Pr_each_career(lam, d, chi, Pu_dBm_per_carrer, g_dB, channel_type, do=1)
    t_nm = channel.abs_timedelays(d, rho, tau, N, M)
    phi_rad = [np.radians(phi_deg[n]) for n in range(N)]
    theta_rad = [np.radians(theta_deg[n]) for n in range(N)]
    
    R, MUE_coordinate  = channel.Mirror_UE_positions(d, N, M, rho, tau, phi_rad, theta_rad)
    # サブアレー各素子の座標Ssub導入版　式11, 12, 13
    subarray_v_qy_qz = channel.calc_anntena_xyz_Ssub(lam_cen, V, Q, Ssub_lam)
    # n番目のTCのm番目のSPの第1散乱体と，v番目のサブアレーのqy, qz番目のアンテナ素子間の距離をrm,n,v, qy, qz 式14
    dis_MUE_to_antenna = channel.distance_to_eachanntena(MUE_coordinate, subarray_v_qy_qz, N, M, V, Q)
    phi_deg_v, theta_deg_v = channel.calc_each_subarray_AOD_EOD(N, M, MUE_coordinate, subarray_v_qy_qz, V)

    t_nm = channel.abs_timedelays(d, rho, tau, N, M) #絶対遅延
    P = channel.cluster_power(Pr, N, tau, Z, channel_type)
    Pi = channel.SP_power(N, M, P, rho, U_nm, channel_type)
    P_each_career = channel.cluster_Power_each_career(Pr_each_career, Z, N, tau, channel_type)
    Pi_each_career = channel.SP_Power_each_career(N, M, P_each_career, rho, U_nm, channel_type)

    phi_rad = [np.radians(phi_deg[n]) for n in range(N)]
    theta_rad = [np.radians(theta_deg[n]) for n in range(N)]
    # 引数Rで鏡像法
    MUE_coordinate = channel.scatter1_xyz_cordinate(N, M, R, phi_rad, theta_rad)
    # print("MUE_coordinate:", np.array(MUE_coordinate))
    # サブアレー各素子の座標wideversion
    subarray_v_qy_qz = channel.calc_anntena_xyz_Ssub(lam_cen, V, Q, Ssub_lam)
    # サブアレーの各素子をプロットしたグラフ
    # plot_subarray_antennas(subarray_v_qy_qz)
    
    phi_deg_v, theta_deg_v = channel.calc_each_subarray_AOD_EOD(N, M, MUE_coordinate, subarray_v_qy_qz, V)
    DFT_weights = channel.DFT_weight_calc(Q)
    
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
            
    for v in range(V):
        for n in range(N):
            for m in range(M[n]):
                phi_rad_v[v, n, m]   = np.radians(phi_deg_v[v][n][m])
                theta_rad_v[v, n, m] = np.radians(theta_deg_v[v][n][m])
                
    # print("phi_rad_v :", np.degrees(phi_rad_v))
    # print("theta_rad_v :", np.degrees(theta_rad_v))
    print("eta_rad :", np.degrees(eta_rad))
    print("varphi_rad :", np.degrees(varphi_rad))
    

    b_varphi_eta = channel.define_b_verphi_eta(N, M, eta_rad)
    # 指向性を計算（近傍界の場合サブアレー毎）
    if NF_setting == "Near":
        a_phi_theta = channel.define_a(V, N, M, phi_rad_v, theta_rad_v, NF_setting)

    Amp_per_career = np.sqrt(Pi_each_career)
    complex_Amp_at_O = channel.calc_complex_Amp_at_O(f, V, N, M, Amp_per_career, beta, t_nm, b_varphi_eta)
    complex_Amp_at_MUE = channel.complex_Amp_at_there(N, M, f, R, complex_Amp_at_O)

    complex_Amp_at_each_antena = channel.complex_Amp_at_each_antena(f, V, Q, N, M, dis_MUE_to_antenna, complex_Amp_at_MUE, a_phi_theta)

    n_k_v = channel.noise_n_k_v(V)

    K = len(f)
    P_sub_dash_dBm = channel.near_Power_inc_noise_optimized(V, Q, f, DFT_weights, complex_Amp_at_each_antena, n_k_v)

            
    # 近傍界の場合、指向性をサブアレー毎に計算する
    if NF_setting == "Near":
        a_phi_theta = channel.define_a(V, N, M, phi_rad_v, theta_rad_v,NF_setting)
    elif NF_setting == "Far":
        a_phi_theta = channel.define_a(V, N, M, phi_rad, theta_rad, NF_setting)
    
    b_varphi_eta = channel.define_b_verphi_eta(N, M, eta_rad)
    Amp_per_career_desital = np.sqrt(Pi) / np.sqrt(Pu)
    
    # UEのアンテナ素子#0から送信されたSPm,nの，原点Oでの，周波数fkにおける複素振幅 ㊺
    a_O_m_n_u_k = np.zeros((N, max(M), U), dtype=np.complex64)
    u = np.arange(U)  # shape: (U,)
    print("u =", u)
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
    
    f_common = Amp_per_career_desital * np.exp(1j * beta) * np.exp(-1j * 2 * np.pi * f_GHz_val * t_nm) * b_varphi_eta
    # eta_rad と varphi_rad の組み合わせから位相計算
    # co = np.cos(eta_rad) * np.sin(varphi_rad)  # shape: (N, M)
    # 各 u に対してブロードキャストで位相成分を計算！
    phase = np.exp(-1j * np.pi * u[:, None, None])  # shape: (U, N, M)
    #  * co[None, :, :]
    


    # 共通因子と位相成分を掛け合わせる
    a_temp = phase * f_common[None, :, :]  # shape: (U, N, M)
    # 最終的な出力を (N, M, U) に転置
    a_O_m_n_u_k = np.transpose(a_temp, (1, 2, 0))  # shape: (N, M, U)

    # 第1散乱体と原点O間で電波が伝搬に要する時間 ㉒
    R_tmp = np.zeros((N,max(M)))
    for i, values in enumerate(R):
        R_tmp[i, :len(values)] = values  # 長さが足りない部分はそのままゼロ
    R = R_tmp
    tau_1_ns = R / c_ns

    ##########################################################################
    # --- 共通因子（指向性 b(eta) はそのまま残す） ---


    phase_ue = np.exp(1j * 2 * np.pi * f_GHz_val * tau_1_ns)   # (N,M,U)
    a_MUE_m_n_u_k = a_O_m_n_u_k * phase_ue  # shape: (N, M, U)

    # MUEとアンテナ間の伝搬時間 ㉓
    tau_nmvqyqz = dis_MUE_to_antenna / c_ns
    
    a = np.zeros((N, max(M), V, U, Q, Q), dtype=np.complex64)
    
    ############################################################################################
    # U: UE素子数
    # f_GHz_val: 周波数[GHz]
    lam = 0.3 / f_GHz_val
    d_u = lam / 2
    ue_xyz_u = np.zeros((U,3))
    ue_xyz_u[:,1] = (np.arange(U) - (U-1)/2) * d_u
    
    # MUE_coordinates: (N, maxM, 3)  ※いま使ってるやつ（基準点）
    MUE = np.asarray(MUE_coordinate, dtype=np.float64)      # (1,1,3)

    delta_u = (ue_xyz_u - ue_xyz_u[0])[None, None, :, :]     # (1,1,U,3)
    MUE_nmu = MUE[:, :, None, :] + delta_u                   # (1,1,U,3)

    diff = subarray_v_qy_qz[None,None,None,:,:,:] - MUE_nmu[:, :, :, None, None, None, :]  # (N,maxM,U,V,Q,Q,3)
    d_nmu_vqq = np.linalg.norm(diff, axis=-1)                                        # (N,maxM,U,V,Q,Q)

    tau_nmuvqyqz = d_nmu_vqq / c_ns                                                    # (N,maxM,U,V,Q,Q)
    exp_term = np.exp(-2j*np.pi * f_GHz_val * tau_nmuvqyqz)                            # 同shape

    # tau_nmvqyqz は (N, max(M), V, Q, Q) で与えられている
    exp_term = np.exp(-2j * np.pi * f_GHz_val * tau_nmuvqyqz)  # shape: (N, max(M), V, Q, Q)

    # 各アンテナ素子においての複素振幅　式47
    a = np.einsum('nmu,vnm,vnumyz->nmvuyz', a_MUE_m_n_u_k, a_phi_theta, exp_term, optimize=True)
    # すべてのマルチパスについて和を取る　式48
    a = np.sum(a, (0,1))
    ############################################################################################
    w_DD_pape = np.zeros((V, Q, Q), dtype=np.complex64)
    invalid_indices = []  # vの削除対象インデックスを格納するリスト

    n_dash_full = channel.noise_u_v_k(U, V)  # フルサイズを保持（後で毎回切る）
    
    ba, row = Beam_allocation_method_2(P_sub_dash_dBm, threshold, test_num)
    
    invalid_indices = []  # 各wごとに初期化
    v = 0  # v をループ内で一意に管理
    for face in range(3):
        for array in range(4):
            pa = ba[face, 0, array]
            pe = ba[face, 1, array]
            if pa == -100 and pe == -100:
                invalid_indices.append(v)  # 削除対象のvを記録
                # print(f"v={v} は削除されます。")
            elif pa == -1 and pe == -1:
                invalid_indices.append(v)
            else:
                w_DD_pape[v,:,:] = channel.DFT_weight_calc_pape(Q, pa, pe)
            v += 1  # ループごとに v を増加

    # vの次元を削除して V' にする
    valid = np.setdiff1d(np.arange(V), invalid_indices)  # 残すvのインデックス
    w_DD_pape_red = w_DD_pape[valid]      # (V′,Q,Q)
    a_red = a[valid]                      # (V′,U,Q,Q)
    n_dash_w = n_dash_full[:, valid, :]

    # 近傍界チャネル計算
    if NF_setting == 'Near':
        # チャネル行列を計算　式49
        H_w = np.einsum('vyz,vuyz->uv', w_DD_pape_red, a_red, optimize=True)  # (U, V′)
        if NS == 'Wo_NS':
            H_w_est = H_w + n_dash_w[:, :, 0] / np.sqrt(Pu/2000)
        elif NS == 'W_NS':
            H_w_est = H_w + (n_dash_w[:, :, :10].sum(axis=2)/10) / np.sqrt(Pu/2000)
    print("max |H[u]-H[0]| over u, for each column:")
    print(np.max(np.abs(H_w - H_w[0:1, :]), axis=0))
    
    return H_w, H_w_est, ba



channel_type = "InH"
NF_setting = "Near"
Method = "Mirror"
d = 5  # UEまでの距離
test_num = 9
np.random.seed(test_num)

for Ssub_lam in [0, 50, 100, 1000]:
    NS = 'Wo_NS'
    H, H_est, ba = setting_Direct_synario(channel_type, Q, lam, d, Ssub_lam, NS, threshold, test_num)
    print("========================================")
    print(f"Ssub = {Ssub_lam} lam")
    print(f"d = {d}m")
    C, Ly, Hval = calc_channel_capacity(H, H, Pt_mW, P_noise_mW)
    # C_est, Ly_est, Hval_est = calc_channel_capacity(H_est, H, Pt_mW, P_noise_mW)
    print("ba :", ba)
    print("C, Ly : ", round(C, 3), Ly)
    print("Hval :", "  ".join(f"{x:.3e}" for x in Hval))
    print("----------------------------------------")
    # print("Cest, Ly_est :", round(C_est, 3), Ly_est)
    # print("Hval_est:", "  ".join(f"{x:.3e}" for x in Hval_est))

    mc12 = mutual_coherence_hi_hj(H, 1, 2)
    mc13 = mutual_coherence_hi_hj(H, 1, 3)
    mc14 = mutual_coherence_hi_hj(H, 1, 4)
    print("Mutual Coherence mu12, mu13, mu14 :", mc12, mc13, mc14)

    # print("H", H)