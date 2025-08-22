import warnings, numpy as np
import pandas as pd
import Channel_functions as cf

warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(all='ignore')  # もしくは divide='ignore', invalid='ignore'

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

# パラメータ設定
Pu = 10
f_GHz = np.linspace(141.50025, 142.49975, 2000)
f_GHz_val = f_GHz[1000]
lam = 0.3 / f_GHz #2000個の波長
lam_val = 0.3 /f_GHz_val
c_ns = 0.3
threshold = -73

# アンテナ設定
V=12
U=8
Q=64

channel_type = "InH"
NF_setting = "Near"
d = 5
Channel_number = 0

channel_data_1000 = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Scatter1/d={d}/Scatter1.npy", allow_pickle=True)
channel_data = channel_data_1000[Channel_number]
Power_1000 = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Power/d={d}/Power_{NF_setting}.npy")
Power = Power_1000[Channel_number]

# ビーム割当法１
def Beam_allocation_method_1(Power, threshold):
    """
    ビーム割当法１：1,2,3,4位と順に割り当て、ダイバーシティを優先
    """
    beam_allocation = np.full((3, 2, 4), -1, dtype=int)  # w削除済み
    rows_per_v = []

    for v in range(V):
        valid_mask = Power[v] >= threshold
        valid_values = Power[v][valid_mask]

        if valid_values.size == 0:
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

        pa_indices, pe_indices = np.where(valid_mask)
        candidates = np.stack([valid_values, np.full_like(pa_indices, v), pa_indices, pe_indices], axis=1)
        top4 = candidates[np.argsort(-candidates[:, 0])][:4]

        df_v = pd.DataFrame(top4, columns=["Power_dBm", "v", "pa", "pe"])
        df_v.insert(0, "rank", np.arange(1, len(df_v)+1))
        df_v.insert(0, "Channel", Channel_number)
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

ba_m1, row_m1 = Beam_allocation_method_1(Power, threshold)

# 上位4候補のデータフレーム（オプション表示）
df_top_per_v = pd.concat(row_m1, ignore_index=True)
# print(df_top_per_v.to_string(index=False))
# print("Beam allocation: \n", ba_m1)

# ビーム割当法２
def Beam_allocation_method_2(Power, threshold):
    """
    ビーム割当法１：1,2,3,4位と順に割り当て、ダイバーシティを優先
    """
    beam_allocation = np.full((3, 2, 4), -1, dtype=int)  # w削除済み
    rows_per_v = []

    for v in range(V):
        valid_mask = Power[v] >= threshold
        valid_values = Power[v][valid_mask]

        if valid_values.size == 0:
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

        pa_indices, pe_indices = np.where(valid_mask)
        candidates = np.stack([valid_values, np.full_like(pa_indices, v), pa_indices, pe_indices], axis=1)
        top4 = candidates[np.argsort(-candidates[:, 0])][:4]

        df_v = pd.DataFrame(top4, columns=["Power_dBm", "v", "pa", "pe"])
        df_v.insert(0, "rank", np.arange(1, len(df_v)+1))
        df_v.insert(0, "Channel", Channel_number)
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

ba_m2, row_m2 = Beam_allocation_method_2(Power, threshold)

# チャネル容量の計算
def ChannelMatrix_Calculation(beam_allocation, channel_data, f_GHz_val, Pu, c_ns):
    N = channel_data["N"]
    M = channel_data["M"]
    phi_deg = channel_data["phi_deg"]
    theta_deg = channel_data["theta_deg"]
    eta_deg = channel_data["eta_deg"]
    varphi_deg = channel_data["varphi_deg"]
    Pi = channel_data["Pi"]
    beta = channel_data["beta"]
    t_nm = channel_data["t_nm"]
    R = channel_data["R"]
    dis_sca1_to_anntena = channel_data["dis_sca1_to_anntena"]

    phi_rad, theta_rad, varphi_rad, eta_rad = transform_angle_to_numpy(phi_deg, theta_deg, varphi_deg, eta_deg, N, M)

    # ビーム割当に必要な関数
    a_phi_theta = cf.define_a(V, N, M, phi_rad, theta_rad)
    b_varphi_eta = cf.define_b_verphi_eta(N, M, eta_rad)
    zeta = cf.define_zeta(V)
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

    h_u_v_k = np.zeros((U,V), dtype=complex)
    h_u_v_k_est = np.zeros((U,V), dtype=complex)
    n_dash = cf.noise_u_v_k(U,V)
    invalid_indices = []  # 各wごとに初期化
    v = 0  # v をループ内で一意に管理
    for face in range(3):
        for array in range(4):
            pa = beam_allocation[face, 0, array]
            pe = beam_allocation[face, 1, array]
            if pa == -1 and pe == -1:
                invalid_indices.append(v)  # 削除対象のvを記録
                # print(f"v={v} は削除されます。")
            else:
                w_DD_pape[v,:,:] = cf.DFT_weight_calc_pape(Q, pa, pe)
                g_DD_pape[v,:,:] = cf.g_dd_depend_on_pape(N, M, phi_rad, theta_rad, zeta, Q, v, pa, pe)
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
        h_u_v_k = np.einsum('vyz,vuyz->uv', w_DD_pape_reduced, a, optimize=True)
        h_u_v_k_est = h_u_v_k + n_dash[:,:,0] / np.sqrt(Pu/2000)
        
    return h_u_v_k, h_u_v_k_est

Htru_m1, Hest_m1 = ChannelMatrix_Calculation(ba_m1, channel_data, f_GHz_val, Pu, c_ns)
Htru_m2, Hest_m2 = ChannelMatrix_Calculation(ba_m2, channel_data, f_GHz_val, Pu, c_ns)

# チャネル容量計算
Pt_mW = 1000/2000
P_noise_mW = 6.31*1e-12

# グラム行列の固有値と固有ベクトルを求める関数
def calc_eigval(H):
    H_H_H = np.conjugate(H).T @ H
    eigval, eigvec = np.linalg.eig(H_H_H)
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
def water_filling_ratio(eig_vals, Pt, P_noise, tol=1e-12):
    """
    固有値 eig_vals に対して、注水定理に基づく電力分配比 p'_j を計算する関数。
    固有値は大きい順（λ1,λ2,...,λ_L）で与えられるものとする。
    
    Parameters:
        eig_vals (numpy array): 固有値の配列 (サイズ L_total)
        Pt (float): 総送信電力
        P_noise (float): 端末の雑音電力
        tol (float): 数値誤差の許容範囲
        
    Returns:
        p_alloc_subset (numpy array): 使用したレイヤ数 L_used に対する電力分配比 (要素数 L_used, 合計1)
        L_used (int): 使用したレイヤ数
    """
    norm_factor = Pt / P_noise
    L_total = len(eig_vals)
    
    # 上位 L_total から順に、使用レイヤ数 L を減らして試す
    for L in range(L_total, 0, -1):
        lambda_subset = eig_vals[:L]  # 上位 L 個の固有値
        inv_vals = 1 / (norm_factor * lambda_subset)  # 逆数を計算
        nu = (1 + np.sum(inv_vals)) / L  # 水準 ν を計算
        p_alloc_subset = nu - inv_vals  # 各層への分配比 p'_j
        
        # すべての p'_j がほぼ正であれば採用
        if np.all(p_alloc_subset > -tol):
            p_alloc_subset = np.maximum(p_alloc_subset, 0)  # 数値誤差対策
            p_alloc_subset /= np.sum(p_alloc_subset)  # 合計1に正規化
            return p_alloc_subset, L
    
    # すべての固有値が極小の場合は、全電力を最も大きい固有値に割り当てる
    return np.array([1.0]), 1

def generate_s(Ly):
    # 複素ガウス乱数の生成（実部と虚部を分散0.5で生成）
    # こうすると、各エントリの平均パワーは 1 
    s = (np.random.randn(Ly) + 1j * np.random.randn(Ly)) / np.sqrt(2)
    return s

# チャネル容量を求める関数(注意!! nearとfarで用いるH_truが異なります！！！！)
def calc_channel_capacity(H, H_tru):
    H_val , H_vec = calc_eigval(H)
    eigval = H_val  # 固有値を保存

    power_allo, Ly = water_filling_ratio(H_val, Pt_mW, P_noise_mW)
    p_sqrt = np.sqrt(power_allo)
    p_diag = np.diag(p_sqrt)

    A = np.sqrt(Pt_mW)*p_diag
    Te_H = H_vec[:,:Ly]

    n_dash_dash = cf.noise_dash_dash(U)
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
    Capacity = C
    
    # # 累積分布の計算
    # cdf = np.cumsum(counts) / len(L_used_array)

    # プロット
    # plt.figure(figsize=(8, 5))
    # plt.step(L_values, cdf, where='post', label='CDF of L_used')
    # plt.xlabel('L_used (Number of layers)')
    # plt.ylabel('Cumulative Probability')
    # plt.title('CDF of Number of Layers Used in Water-Filling')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    return Capacity, eigval, Ly


Capacity_Far_tru = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{channel_type}_1000/ChannelCapacity/Channel_{channel_type}_step5_ChannelCapacity_d={d}/Channel_{channel_type}_step5_Htru_Far_ChannelCapacity_d={5}.npy")
Capacity_Far_est = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{channel_type}_1000/ChannelCapacity/Channel_{channel_type}_step5_ChannelCapacity_d={d}/Channel_{channel_type}_step5_Hmea_Far_ChannelCapacity_d={5}.npy")


method1_tru_list = []
method1_est_list = []
method2_tru_list = []
method2_est_list = []
far_tru_list = []
far_est_list = []
channel_data_1000 = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Scatter1/d={d}/Scatter1.npy", allow_pickle=True)
Power_1000 = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Power/d={d}/Power_{NF_setting}.npy")



for Channel_number in range(1000):
    channel_data = channel_data_1000[Channel_number]
    Power = Power_1000[Channel_number]

    ba_m1, row_m1 = Beam_allocation_method_1(Power, threshold)
    ba_m2, row_m2 = Beam_allocation_method_2(Power, threshold)

    Htru_m1, Hest_m1 = ChannelMatrix_Calculation(ba_m1, channel_data, f_GHz_val, Pu, c_ns)
    Htru_m2, Hest_m2 = ChannelMatrix_Calculation(ba_m2, channel_data, f_GHz_val, Pu, c_ns)

    cap_tru_m1, eigval_tru_m1, Ly_tru_m1 = calc_channel_capacity(Htru_m1, Htru_m1)
    cap_est_m1, eigval_est_m1, Ly_est_m1 = calc_channel_capacity(Hest_m1, Htru_m1)
    cap_tru_m2, eigval_tru_m2, Ly_tru_m2 = calc_channel_capacity(Htru_m2, Htru_m2)
    cap_est_m2, eigval_est_m2, Ly_est_m2 = calc_channel_capacity(Hest_m2, Htru_m2)

    method1_tru_list.append(cap_tru_m1)
    method1_est_list.append(cap_est_m1)
    method2_tru_list.append(cap_tru_m2)
    method2_est_list.append(cap_est_m2)
    far_tru_list.append(np.max(Capacity_Far_tru[Channel_number]))
    far_est_list.append(np.max(Capacity_Far_est[Channel_number]))
    print(f"Channel #{Channel_number} processed.")

print('Average Channel Capacities over 100 Channels:')
print('Method 1 Tru', np.mean(method1_tru_list).round(3))
print('Method 1 Est', np.mean(method1_est_list).round(3))
print('')
print('Method 2 Tru', np.mean(method2_tru_list).round(3))
print('Method 2 Est', np.mean(method2_est_list).round(3))
print('')
print('Far Tru', np.mean(far_tru_list).round(3))
print('Far Est', np.mean(far_est_list).round(3))

