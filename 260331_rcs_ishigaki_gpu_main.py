# 260331_rcs_ishigaki_gpu_main.py
import numpy as np
import Channel_function_gpu as ch_func
import Channel_functions as channel
import pandas as pd
import torch
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# システムパラメータ
Q = torch.tensor(16, device=device) # 各サブアレーの一辺のアンテナ素子数
V = torch.tensor(12, device=device) # サブアレー数
U = torch.tensor(8, device=device) # UEのアンテナ素子数
# 周波数関連 (2000サブキャリアを一括処理)
f_GHz = torch.linspace(141.50025, 142.49975, 2000, device=device)
lam = 0.3 / f_GHz # 波長

# 1. データのロード (ここだけは一旦CPU)
base_all = np.load("Base_InH_rect.npy", allow_pickle=True).item()
B = 100 # バッチサイズ

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

#################################################################################################################
# この範囲未修正
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
def simulation_core_gpu(base_batch, d, Q, lam, Pu_dBm, Ssub_lam, scenario):
    """
    base_data_batch: (B, N, M_max) のテンソル（パディング済み）
    d: 通信距離 (m)
    """
    path_mask = base_batch['mask'].unsqueeze(1) # (B, 1, N, M) に拡張
    lam_cen = 0.3 / 142.0 # 中心周波数の波長 (m)
    
    eta_dir = np.degrees(np.arcsin(1 / d))
    theta_dir = -eta_dir
    
    delta_EOD = theta_dir - base_batch['theta_ND_00']
    delta_EOA = eta_dir - base_batch['eta_ND_00']
    theta_deg = base_batch['theta_ND_deg'] + delta_EOD[:, None, None]
    eta_deg = base_batch['eta_ND_deg'] + delta_EOA[:, None, None]
    
    # 角度をラジアンに変換 (これ以降の複素平面計算用)
    theta_rad = torch.deg2rad(theta_deg)
    eta_rad = torch.deg2rad(eta_deg)
    
    # phi と varphi は d に依存しないため、そのまま転送
    phi_rad = torch.deg2rad(base_batch['phi_deg'])
    varphi_rad = torch.deg2rad(base_batch['varphi_deg'])
    
##############################################################################
# ここからパイロット信号送受信
    Pu_mW = 10 ** (Pu_dBm / 10) # mW単位
    num_carriers = 2000 # サブキャリア数
    
    # 1キャリアあたりの送信電力算出
    Pu_mW_per_carrier = Pu_mW / num_carriers # (スカラまたはB,)
    Pu_dBm_per_carrier = 10 * torch.log10(torch.tensor(Pu_mW_per_carrier, device=device))

    # 受信電力計算    
    Pr_dBm = ch_func.calc_Pr_batched(lam, d, base_batch['chi'], scenario, Pu_dBm=10, do=1.0) # (B,)
    Pr_dBm_per_carrier = Pr_dBm - 10 * np.log10(num_carriers)
    t_nm = ch_func.abs_timedelays_batched(d, base_batch)
    
    # 端末の鏡像体の座標
    R, MUE_coordinate  = ch_func.Mirror_UE_positions_batched(d, base_batch, theta_rad, phi_rad)
    # サブアレー各素子のSsub含む座標
    subarray_v_qy_qz = ch_func.calc_anntena_xyz_Ssub_gpu(lam_cen, V, Q, Ssub_lam, device='cuda')

    # n番目のTCのm番目のUE鏡像体#0と，v番目のサブアレーのqy, qz番目のアンテナ素子間の距離をrm,n,v, qy, qz 式14
    r_mnv0qyqz = ch_func.distance_to_eachanntena_batched(MUE_coordinate, subarray_v_qy_qz) # (B, N, M, V, Q, Q)
    # UEの#0から各アンテナ素子への伝搬時間 ㉓
    tau_mnv0qyqz = r_mnv0qyqz / 0.3
    # サブアレー毎のAODとEODを計算
    phi_rad_v, theta_rad_v, varphi_rad_v, eta_rad_v = ch_func.calc_all_angles_batched(base_batch, MUE_coordinate, subarray_v_qy_qz) # (B, V, N, M)

    Pi_mW = ch_func.calc_Pi_mW_batched(Pr_dBm, base_batch['Z'], base_batch['N'], base_batch['tau'], scenario) # (B, N, M)
    Pi_mW_per_carrier = Pi_mW / num_carriers # (B, N, M)
    
    DFT_weights = ch_func.DFT_weight_calc_gpu(Q) # (Q, Q, Q, Q)
    
    b_varphi_eta_v = ch_func.define_b_phi_eta_batched(eta_rad_v) # (B, V, N, M)
    Amp_per_carrier = torch.sqrt(Pi_mW_per_carrier)
    

    a_phi_theta_v = ch_func.define_a_batched(V, phi_rad, theta_rad) # (B, V, N, M)
    
    # --- 5. 複素振幅の合成 (ここが「GPU高速化ブロック⓪」のバッチ化) ---
    # 初期位相 beta (B, N, M)
    beta_rad = base_batch['beta_rad'] 
    
    # パイロット信号成分 (B, V, N, M)
    # Amp(B,N,M) * exp(j*beta) * b_gain(B,V,N,M) * a_gain(B,V,N,M)
    # ※次元を合わせて一気に掛け算
    pilot_signal = Amp_per_carrier.unsqueeze(1) * torch.exp(1j * beta_rad).unsqueeze(1) * b_varphi_eta_v * a_phi_theta_v * path_mask

    # 位相回転項の生成 (B, N, M, V, Q, Q, K)
    # r_mnv0qyqz (B, N, M, V, Q, Q) / 0.3ns
    tau = r_mnv0qyqz / 0.3  # (B, N, M, V, Q, Q)
    
    # 巨大テンソルのメモリ節約のため、複素単精度(complex64)を使用
    # (B, N, M, V, Q, Q, 1) * (1, 1, 1, 1, 1, 1, K)
    phase_term = torch.exp(-2j * torch.pi * f_GHz.view(1, 1, 1, 1, 1, 1, -1) * tau.unsqueeze(-1))

    # --- 6. アンテナ素子ごとの信号合成 (einsum) ---
    # pilot_signal: bvnm
    # phase_term: bnmvyzk
    # n, m 次元（クラスタとサブパス）で和をとって消す
    # 結果: (B, V, K, Q, Q)
    complex_Amp_antena = torch.einsum('bvnm, bnmvyzk -> bvkyz', pilot_signal, phase_term)

    # --- 1. 雑音の生成 (B, V, K) ---
    # B: バッチサイズ, V: サブアレー数(12), num_carriers: 周波数波(2000)
    n_k_v = ch_func.noise_n_k_v_batched(B, V, num_carriers, device)

    # --- 2. BF後の受信電力算出 (B, V, Q, Q) ---
    # DFT_weights: (Q, Q, Q, Q) -> ビーム辞書
    # complex_amp_antena: (B, V, K, Q, Q) -> 各素子の複素振幅
    # 戻り値 P_sub_dash_dBm は (B, V, 16, 16) の形状
    P_sub_dash_dBm = ch_func.near_Power_inc_noise_batched(V, Q, DFT_weights, complex_Amp_antena, n_k_v)
    

# ここまでパイロット信号受信電力計算
#############################################################################
# ここからチャネル計算
    
    Amp_desital = torch.sqrt(Pi_mW) / torch.sqrt(Pu_mW).view(-1, 1, 1)

    # beta: (B, N, M) ※初期位相
    a_MUE_vnm = Amp_desital.unsqueeze(1) * torch.exp(1j * beta_rad).unsqueeze(1) * b_varphi_eta_v * path_mask

    lam = 0.3 / f_GHz


    # 入力: a_MUE_vnm(B,V,N,M), a_phi_theta(B,V,N,M), tau_mnv0qyqz(B,N,M,V,Q,Q), f_ghz(K), u(U)
    # 伝搬遅延による位相回転項 (B, N, M, V, Q, Q, K)
    # メモリ節約のため complex64（複素単精度）を強く推奨
    exp_term = torch.exp(-2j * torch.pi * f_GHz.view(1, 1, 1, 1, 1, 1, -1) * tau_mnv0qyqz.unsqueeze(-1))

    # UE側のアンテナ素子位置による位相項 (B, V, N, M, U)
    c = torch.cos(eta_rad_v) * torch.sin(varphi_rad_v) # (B, V, N, M)
    u = torch.arange(U, device=device).float()
    ue_phase = torch.exp(-1j * torch.pi * u.view(1, 1, 1, 1, -1) * c.unsqueeze(-1))

    # 全パスを合成してチャネル行列を算出 (B, U, V, K, Q, Q)
    # a_MUE_vnm: bvnm, a_phi_theta: bvnm, ue_phase: bvnmu, exp_term: bnmvyzk
    a_uvkqyqz = torch.einsum('bvnm, bvnm, bvnmu, bnmvyzk -> buvkyz', 
                            a_MUE_vnm, a_phi_theta_v, ue_phase, exp_term)
    
    """
    ここまで修正完了
    """

    ###############################################################]
    # 以下ビーム割当
    threshold_dBm = -73
    # --- 1. 各サブアレー内での最強ビームを抽出 (B, V) ---
    # (B, V, 256) に平坦化して最大電力を取得
    P_max_per_sub, flat_beam_idx = torch.max(P_sub_dash_dBm.view(B, V, -1), dim=2)

    # 各サブアレーの最強ビーム座標 (pa, pe) を保持
    best_pa = flat_beam_idx // Q  # (B, V)
    best_pe = flat_beam_idx % Q   # (B, V)

    # --- 2. スレショルド判定と V' の決定 ---
    # threshold_dBm (スカラー) 以上の電力を持つサブアレーを True にする
    # active_mask の形状は (B, V)
    active_mask = P_max_per_sub > threshold_dBm

    # 各バッチにおける有効なサブアレー数 V_prime をカウント (B,)
    V_prime = torch.sum(active_mask.long(), dim=1)

    # --- 3. スレショルド以下のビーム情報を「切り捨てる」 ---
    # 無効なサブアレーの pa, pe を -1 (または 0) に埋める、あるいは電力自体を消す
    # ※後の計算でこの mask を使って信号を 0 にするのが一番盤石です
    best_pa_filtered = torch.where(active_mask, best_pa, torch.tensor(-1, device=device))
    best_pe_filtered = torch.where(active_mask, best_pe, torch.tensor(-1, device=device))
    P_max_filtered = torch.where(active_mask, P_max_per_sub, torch.tensor(-float('inf'), device=device))
    
    # --- 1. 選ばれたビームに対応する DFTウェイトの抽出 (B, V, Q, Q) ---
    # DFT_weights: (Q, Q, Q, Q) -> (pa, pe, qy, qz)
    # best_pa, best_pe: (B, V) -> 各サブアレーの最強ビーム番号

    # 全体から必要な pa, pe の組み合わせだけを抜き出す
    # (B, V, Q, Q) の形状で、各サブアレーに適用すべき 16x16 のウェイトを保持
    # まず pa を選択
    w_DD_pa = DFT_weights[best_pa] # (B, V, Q, Q, Q)
    # 次に その中から pe を選択 (少し特殊なインデックス操作)
    w_DD_selected = DFT_weights[best_pa, best_pe] # (B, V, Q, Q)

    # --- 2. チャネル行列 h_uvk の計算 (einsum) ---
    # a_uvkqyqz: (B, U, V, K, Q, Q) -> 全アンテナ素子の複素振幅
    # w_DD_selected: (B, V, Q, Q) -> 選ばれたビームのウェイト
    # 結果形状: (B, U, V, K)
    h_uvk = torch.einsum('bvyz, buvkyz -> buvk', w_DD_selected, a_uvkqyqz)

    # --- 3. 雑音生成と有効サブアレー V' への適用 ---
    # 雑音 n_dash_uv_full: (B, U, V, K) 
    # ここでも標準偏差 sigma_dash を使って一括生成
    sigma_dash = 1.778 * 1e-6 # 仮の定数
    n_dash_real = torch.randn((B, U, V, K), device=device) * sigma_dash
    n_dash_imag = torch.randn((B, U, V, K), device=device) * sigma_dash
    n_dash_uv_full = torch.complex(n_dash_real, n_dash_imag)

    # --- 4. スレショルド判定 (active_mask) に基づく処理 ---
    # 前のステップで作った active_mask (B, V) を活用
    # (B, V) を (B, 1, V, 1) に拡張してブロードキャスト
    mask_expanded = active_mask.view(B, 1, V, 1)
    # 無効なサブアレーの成分を 0 にする
    h_uvk_filtered = h_uvk * mask_expanded

    if use_H == "E_wo":
        # 雑音を加えて、かつ無効なサブアレーを 0 に落とす
        h_uvk_final = (h_uvk_filtered + n_dash_uv_full) * mask_expanded
    else:
        h_uvk_final = h_uvk_filtered

# ここまでチャネル計算
###########################################################################################################################
# ここからゼロマスキング
    # --- 1. 位相回転の補正 (B, U, V, K) ---
    # t_nm_min: 各バッチの最小遅延 (B,) を想定。
    # t_nm は以前のステップで出した (B, N, M) の中から最短パス [:, 0, 0] を抽出
    t_nm_min = t_nm[:, 0, 0].view(B, 1, 1, 1) 
    h_uvk = h_uvk * torch.exp(1j * 2 * torch.pi * f_GHz.view(1, 1, 1, -1) * t_nm_min)

    # --- 2. 2K IDFT の準備 ---
    df = (f_GHz[1] - f_GHz[0]) # 周波数ステップ
    K = f_GHz.size(0)

    # H(K) の後ろに逆順にしたものを結合して 2K ポイントにする
    # torch.flip で最後の次元 (K) を反転
    h_w_rev = torch.flip(h_uvk, dims=[-1])
    h_w_2k = torch.cat([h_uvk, h_w_rev], dim=-1) # (B, U, V, 2K)

    # 2K IDFT を実行 (最後の次元に対して適用)
    h_tau_2k = torch.fft.ifft(h_w_2k, dim=-1)

    # --- 3. ゼロマスキング (100 ns ～ 1900 ns) ---
    dt = 1.0 / (K * df)
    L = int(round(100e-9 / dt)) # 100ns に対応するインデックス

    # マスク処理：中間の高遅延成分を 0 にする
    # バッチ次元を維持したままスライスで 0 を代入
    # h_tau_2k_masked はコピーを作ってから処理
    h_tau_2k_masked = h_tau_2k.clone()
    h_tau_2k_masked[..., 2*L : 2*K - 2*L] = 0

    # --- 4. 2K DFT とデノイズ結果の抽出 ---
    h_w_2k_masked = torch.fft.fft(h_tau_2k_masked, dim=-1)

    # 前半 K ポイントを取り出して、ノイズ抑制後のチャネルとする
    h_w_denoised = h_w_2k_masked[..., :K] # (B, U, V, K)

    # --- 5. 評価用：電力（ノルム）の計算 (B, K) ---
    # 空間次元 (U, V) を潰して、バッチごとの周波数特性を出す
    h_mag_org = torch.linalg.norm(h_uvk, dim=(1, 2))
    h_mag_denoised = torch.linalg.norm(h_w_denoised, dim=(1, 2))
    
    return h_uvk_final



###############################################################
scenario = "InH"
NF_setting = "Near"
Method = "Mirror"

channel_indices = [0]  # NYUSIMチャネルの場合のインデックスリスト(個別指定)
use_H = "T" # 'T' : 真のチャネル行列 , 'E_w' : 推定&同相加算　''E_wo' : 推定&非同相加算

# パイロット信号送信電力パラメータ
Pu_dBm = 30  # UEの送信電力(dBm)※全サブキャリア

save_folder = None #: グラフを保存しない, フォルダ名 : 保存するフォルダ名
# save_folder =  f"Channel_{channel_indices[0]}" 
# save_folder = "Resized"
################################################################

Base_data = np.load(f"Base_{scenario}_rect.npy", allow_pickle=True)

base_seed = 9  # 今までの固定seed

#E-SDMパラメータ
Pt_mW = 1000/2000  #通信時のサブキャリアごとの基地局送信電力
P_noise_mW = 6.31e-12 #500kHzあたりの雑音電力

# シミュレーションパラメータ
d_values = list(range(5, 6, 5))
Ssub_list = [0]

