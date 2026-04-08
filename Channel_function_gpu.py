import numpy as np
import Channel_functions as channel
import pandas as pd
import torch
import time
import math
from dataclasses import dataclass, field

@dataclass
class SystemConfig:
    # デバイス設定
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # アンテナパラメータ
    Q: int = 16  # 各サブアレーの一辺のアンテナ素子数
    V: int = 12  # サブアレー数
    U: int = 8   # UEのアンテナ素子数
    K: int = 2000 # サブキャリア数
    
    # 周波数パラメータ (2000サブキャリア)
    f_GHz: torch.Tensor = field(default_factory=lambda: torch.linspace(141.50025, 142.49975, 2000))
    
    # 電力・雑音パラメータ
    Pu_dBm: float = 30.0      # UEの送信電力(dBm)
    Pt_mW: float = 1000/2000  # 基地局送信電力(mW/サブキャリア)
    P_noise_mW: float = 6.31e-12 # 雑音電力(mW)

    @property
    def lam(self):
        # 計算時に常に最新のdeviceにいることを保証
        return 0.3 / self.f_GHz.to(self.device)

def calc_Pr_batched(lam, d, chi_batch, scenario, Pt_dBm=10, do=1.0):
    if scenario == 'InH':
        n = 1.8
    elif scenario == 'InF':
        n = 1.7
    
    fspl = 20 * torch.log10(4 * torch.pi * do / lam)
    PL = fspl + 10*n*torch.log10(d/do) + chi_batch
    Pr_dbm = Pt_dBm - PL
    
    return Pr_dbm

def abs_timedelays_batched(d, batch):
    """
    バッチ分の絶対遅延を一括計算する (B, N, M)
    d: 通信距離 (m)
    batch['tau']: クラスタ遅延 (B, N)
    batch['rho']: サブパス遅延 (B, N, M)
    """
    c = 0.3  # 光速 m/ns
    t0 = d / c  # 直接波の到達時間 (ns)

    # --- GPUでの一括計算 (B, N, M) ---
    t_nm = t0 + batch['tau'].unsqueeze(-1) + batch['rho'] #
    
    return t_nm

def Mirror_UE_positions_batched(d, batch, theta_rad, phi_rad):
    """
    バッチ分のミラーUE座標を一括計算する (B, N, M, 3)
    d: 通信距離 (m)
    batch: パディング済みデータ (B, N, M)
    theta_rad, phi_rad: ステップ3で算出したラジアン角度 (B, N, M)
    """
    # --- 1. 距離 R の一括計算 (B, N, M) ---
    R = d + 0.3 * (batch['tau'].unsqueeze(-1) + batch['rho']) #
    # 全バッチ (B) に対して一斉適用
    R[:, 0, 0] = d

    # --- 2. デカルト座標 (x, y, z) への変換 (B, N, M) ---
    cos_theta = torch.cos(theta_rad)
    
    x = R * cos_theta * torch.cos(phi_rad)
    y = R * cos_theta * torch.sin(phi_rad)
    z = R * torch.sin(theta_rad)

    # --- 3. 座標のスタック (B, N, M, 3) ---
    MUE_coordinates = torch.stack([x, y, z], dim=-1)
    
    return R, MUE_coordinates

def calc_anntena_xyz_Ssub_gpu(lam_cen, V, Q, Ssub_lam, device='cuda'):
    """
    基地局サブアレーの3D座標を一括計算する (V, Q, Q, 3)
    lam: 中心周波数の波長 (m)
    V: サブアレー総数 (12)
    Q: サブアレー一辺の素子数 (16)
    """
    S_sub = Ssub_lam * lam_cen # サブアレー間の追加間隔
    p = lam_cen / 2 # 素子間隔 (λ/2)
    Vf = V // 3 # 1面あたりのサブアレー数 (4)
    
    # サブアレーの配置に必要な全幅 L
    L = (Q + 1) * p * Vf + (Vf - 1) * S_sub 
    
    # 座標格納用テンソル (V, Q, Q, 3)
    subarray_v_qy_qz = torch.zeros((V, Q, Q, 3), device=device, dtype=torch.float32)
    
    # サブアレー内の素子インデックスグリッドを作成
    # indexing="ij" で元の numpy と挙動を合わせます
    qy_idx, qz_idx = torch.meshgrid(
        torch.arange(Q, device=device), 
        torch.arange(Q, device=device), 
        indexing="ij"
    )
    
    # 全面に共通する z 座標（高さ方向）
    z = lam_cen / 2 + qz_idx * p

    for v in range(V):
        if 0 <= v < Vf:
            # -------- 0°面（正面） --------
            x_v = L / (2 * math.sqrt(3))
            y_v = -L/2 + p * (1 + (Q + 1) * v) + S_sub * v
            
            x_vqyqz = torch.full_like(qy_idx, x_v)
            y_vqyqz = y_v + qy_idx * p
            coords = torch.stack([x_vqyqz, y_vqyqz, z], dim=-1)

        elif Vf <= v < 2 * Vf:
            # -------- 120°面 --------
            v_rel = v - Vf
            x_v = L/(2*math.sqrt(3)) - (math.sqrt(3)/2) * (p*(1 + (Q+1)*v_rel) + S_sub*v_rel)
            y_v = L/2 - 0.5 * p * (1 + (Q+1)*v_rel) - 0.5 * S_sub * v_rel
            
            x_vqyqz = x_v - (math.sqrt(3)/2) * p * qy_idx
            y_vqyqz = y_v - qy_idx * p / 2
            coords = torch.stack([x_vqyqz, y_vqyqz, z], dim=-1)

        else:
            # -------- 240°面 --------
            v_rel = v - 2 * Vf
            x_v = -L/math.sqrt(3) + (math.sqrt(3)/2) * (p*(1 + (Q+1)*v_rel) + S_sub*v_rel)
            y_v = -0.5 * p * (1 + (Q+1)*v_rel) - 0.5 * S_sub * v_rel
            
            x_vqyqz = x_v + (math.sqrt(3)/2) * p * qy_idx
            y_vqyqz = y_v - qy_idx * p / 2
            coords = torch.stack([x_vqyqz, y_vqyqz, z], dim=-1)

        subarray_v_qy_qz[v] = coords

    return subarray_v_qy_qz

def distance_to_eachanntena_batched(MUE_coordinates, subarray_v_qy_qz):
    """
    バッチ分の全サブパス・全アンテナ素子間の距離を一括計算
    MUE_coordinates: (B, N, M, 3) -> バッチ分のUE鏡像座標
    subarray_v_qy_qz: (V, Q, Q, 3) -> 基地局アンテナの固定座標
    """
    B, N, M, _ = MUE_coordinates.shape # バッチサイズ, クラスタ数, サブパス数
    V, Q, _, _ = subarray_v_qy_qz.shape # サブアレー数, 素子数

    # --- GPUブロードキャスト用の次元拡張 (View) ---
    # UE座標: (B, N, M, 1, 1, 1, 3) に拡張
    ue_exp = MUE_coordinates.view(B, N, M, 1, 1, 1, 3) #
    
    # BS座標: (1, 1, 1, V, Q, Q, 3) に拡張
    bs_exp = subarray_v_qy_qz.view(1, 1, 1, V, Q, Q, 3) #

    # --- バッチ分の全素子間距離を並列計算 ---
    r_batch = torch.norm(ue_exp - bs_exp, dim=-1) #

    # 結果の形状: (B, N, M, V, Q, Q)
    return r_batch

def calc_all_angles_batched(batch, MUE_coordinates, subarray_v_qy_qz):
    """
    バッチ分の全サブアレーにおける4つの物理角度を一括計算する
    出力: phi_v, theta_v, varphi_v, eta_v (すべてラジアン、形状: B, V, N, M)
    """
    # --- 1. サブアレー中心と相対座標の計算 ---
    # 基地局サブアレーの中心座標 (V, 3)
    sub_center = subarray_v_qy_qz.mean(dim=(1, 2)) #

    # バッチ分の差分ベクトル (B, N, M, V, 3)
    diff = MUE_coordinates.unsqueeze(3) - sub_center.view(1, 1, 1, -1, 3) #
    
    dx, dy, dz = diff[..., 0], diff[..., 1], diff[..., 2] #
    r = torch.norm(diff, dim=-1) # 各サブアレー中心からの距離

    # --- 2. サブアレーごとの個別角度 (phi_v, theta_v) ---
    # ラジアンで算出し、次元を (B, V, N, M) に整える
    phi_rad_v = torch.atan2(dy, dx).permute(0, 3, 1, 2) #
    theta_rad_v = torch.asin(dz / r).permute(0, 3, 1, 2) #
    
    # 計算用に度数法バージョンも作成
    phi_deg_v = torch.rad2deg(phi_rad_v) #
    theta_deg_v = torch.rad2deg(theta_rad_v) #

    # --- 3. UE側角度の補正計算 (B, V, N, M) ---
    # バッチの代表角度 (B, N, M) を (B, 1, N, M) に拡張してブロードキャスト
    phi_ue = batch['phi_deg'].unsqueeze(1) #
    theta_ue = batch['theta_deg'].unsqueeze(1) #
    varphi_ue = batch['varphi_deg'].unsqueeze(1) #
    eta_ue = batch['eta_deg'].unsqueeze(1) #

    # サブアレーごとのズレを UE 側の角度に反映
    # 式: 基準方位角 + (個別方位角 - 基準方位角)
    varphi_rad_v = torch.deg2rad(varphi_ue - phi_ue + phi_deg_v) #
    eta_rad_v = torch.deg2rad(eta_ue + theta_ue - theta_deg_v) #

    return phi_rad_v, theta_rad_v, varphi_rad_v, eta_rad_v

def calc_Pi_mW_batched(Pr_dBm_batch, batch, scenario='InH'):
    """
    バッチ分の各サブパス電力を一括決定し、Pi[0,0]を最大値に調整する
    Pr_dBm_batch: (B,) -> バッチ分の受信電力
    batch: rho(B,N,M), tau(B,N), Z(B,N), U_nm(B,N,M) を含むテンソル
    """
    # --- 1. 定数設定 (環境に応じた gamma) ---
    if scenario == 'InH':
        gamma_cluster = 18.2
        gamma_sp = 2.0
    else: # InF
        gamma_cluster = 16.2
        gamma_sp = 4.7

    # --- 2. クラスタ電力 P_mW の算出 (B, N) ---
    Pr_mW = 10 ** (Pr_dBm_batch / 10.0) # (B,)
    # P_dash = exp(-tau/gamma) * 10^(Z/10)
    P_dash = torch.exp(-batch['tau'] / gamma_cluster) * (10 ** (batch['Z'] / 10.0)) # (B, N)
    P_dash_sum = torch.sum(P_dash, dim=1, keepdim=True) # (B, 1)
    P_mW = (P_dash / (P_dash_sum + 1e-15)) * Pr_mW.view(-1, 1) # (B, N)

    # --- 3. サブパス電力 Pi の算出 (B, N, M) ---
    # Pi_dash = exp(-rho/gamma) * 10^(U/10)
    Pi_dash = torch.exp(-batch['rho'] / gamma_sp) * (10 ** (batch['U_nm'] / 10.0)) # (B, N, M)
    Pi_dash_sum = torch.sum(Pi_dash, dim=2, keepdim=True) # (B, N, 1)
    # 各クラスタ電力をサブパスへ分配 (B, N, M)
    Pi = (Pi_dash / (Pi_dash_sum + 1e-15)) * P_mW.unsqueeze(-1) # (B, N, M)

    # --- 4. Pi[0,0] と最大値の入れ替え処理 (バッチごとに実行) ---
    # バッチごとの (N, M) 平面における最大値のインデックスを取得
    # Pi.view(B, -1) で (B, N*M) に平坦化して最大値を探す
    Pi_flat = Pi.view(Pi.size(0), -1) # (B, N*M)
    max_indices = torch.argmax(Pi_flat, dim=1) # 各バッチの最大値インデックス (B,)

    for b in range(Pi.size(0)):
        if max_indices[b] != 0:
            # 2次元インデックス (n, m) に戻す
            max_n = max_indices[b] // Pi.size(2)
            max_m = max_indices[b] % Pi.size(2)
            # Pi[0,0] と最大値を入れ替え
            target_val = Pi[b, 0, 0].clone()
            Pi[b, 0, 0] = Pi[b, max_n, max_m]
            Pi[b, max_n, max_m] = target_val

    return Pi

def DFT_weight_calc_gpu(Q, device='cuda'):
    """
    QxQ素子用のDFTウェイト（辞書）をGPU上で一括計算する (Q, Q, Q, Q)
    Q: サブアレー一辺の素子数 (16)
    出力: (pa, pe, qy, qz) -> pa, peはビーム番号、qy, qzは素子番号
    """
    # 0 ～ Q-1 のインデックスを作成
    indices = torch.arange(Q, device=device, dtype=torch.float32)
    
    # 4次元のメッシュグリッドを一気に作成 (indexing="ij" で元の挙動を維持)
    # pa: ビーム水平角, pe: ビーム仰角, qy: 素子y方向, qz: 素子z方向
    pa, pe, qy, qz = torch.meshgrid(indices, indices, indices, indices, indexing="ij")
    
    # --- DFTウェイトの計算式 (1/Q * exp...) ---
    # 元の式: (1/Q) * exp(-1j*pi*qy*(-Q+2*pa+2)/Q) * exp(-1j*pi*qz*(-Q+2*pe+2)/Q)
    
    # 位相項の計算
    phase_y = -torch.pi * qy * (-Q + 2 * pa + 2) / Q
    phase_z = -torch.pi * qz * (-Q + 2 * pe + 2) / Q
    
    # 複素指数の合成 (exp(jA) * exp(jB) = exp(j(A+B)))
    # torch.complex を使って GPU 上で複素テンソルを生成
    w_DD = (1.0 / Q) * torch.exp(1j * (phase_y + phase_z))
    
    return w_DD # 形状: (Q, Q, Q, Q)

def define_b_phi_eta_batched(eta_rad_v):
    """
    バッチ分の全サブアレーにおけるUE側素子利得を計算する (B, V, N, M)
    eta_rad_v: (B, V, N, M) -> 前のステップで算出したラジアン角度
    """
    # --- 1. 定数の準備 ---
    # sqrt(3/2) をあらかじめ計算しておく
    const_factor = torch.sqrt(torch.tensor(1.5, device=eta_rad_v.device))

    # --- 2. 一括計算 (B, V, N, M) ---
    # ループを使わず、テンソル全体に対して cos を適用
    # パディング部分は cos(0)=1 になりますが、後の計算で Pi(電力)=0 が掛かるので問題ありません
    b_varphi_eta_v = const_factor * torch.cos(eta_rad_v)

    return b_varphi_eta_v

def define_a_batched(V, phi_rad_v, theta_rad_v):
    """
    バッチ対応版のアンテナ指向性計算 (B, V, N, M)
    phi_rad, theta_rad: (B, V, N, M) 既に次元が揃っている前提
    """
    device = phi_rad_v.device
    
    # --- 1. サブアレーの設置角 zeta の定義 (V,) ---
    # 0, 120, 240 度の各面 4 枚ずつ
    zeta_deg = torch.tensor([0,0,0,0, 120,120,120,120, 240,240,240,240], device=device)
    # ブロードキャスト用に (1, V, 1, 1) へ変換
    zeta = torch.deg2rad(zeta_deg).view(1, -1, 1, 1)

    # --- 2. 指向性計算 (B, V, N, M) ---
    # 方位角の差分を -pi ~ pi の範囲で算出
    diff = (phi_rad_v - zeta + torch.pi) % (2 * torch.pi) - torch.pi
    
    cos_theta = torch.cos(theta_rad_v)
    # 前面放射の計算式
    a_val = 2.282 * cos_theta * torch.sin((torch.pi / 2) * cos_theta * torch.cos(diff))
    
    # --- 3. 背面放射の処理 (Masking) ---
    # abs(diff) > pi/2 (背面) の要素を 0 に置き換える
    a = torch.where(torch.abs(diff) > (torch.pi / 2), torch.zeros_like(a_val), a_val)
    
    return a

def noise_n_k_v_batched(B, V, K, device):
    """
    (B, V, K) サイズの複素乱数を一括生成する
    """
    # 標準偏差 1.778e-6
    sigma = 1.778 * 1e-6
    
    # torch.randn で一気に生成 (B, V, K)
    # 複素数として生成するために torch.complex を使用
    real_part = torch.randn((B, V, K), device=device) * sigma
    imag_part = torch.randn((B, V, K), device=device) * sigma
    
    n_k_v = torch.complex(real_part, imag_part)
    return n_k_v

def near_Power_inc_noise_batched(V, Q, DFT_weight, complex_amp_antena, n_k_v):
    """
    ビームフォーミング、雑音加算、電力算出をバッチ一括で行う
    DFT_weight: (Q, Q, Q, Q) -> (pa, pe, qy, qz)
    complex_amp_antena: (B, V, K, Q, Q) -> (バッチ, サブアレー, 周波数, qy, qz)
    n_k_v: (B, V, K) -> 雑音
    """
    # --- 1. ビームフォーミング (DFTウェイト適用) ---
    # pqyz (ビームpa, pe, 素子qy, qz) と bvkyz (バッチ, サブアレー, 周波数, 素子qy, qz) を合成
    # qy(y) と qz(z) の次元を潰して、各ビーム (p, q) の複素振幅を出す
    # 結果形状: (B, V, Q, Q, K)
    a_near = torch.einsum('pqyz, bvkyz -> bvpqk', DFT_weight, complex_amp_antena)

    # --- 2. 雑音の加算 ---
    # n_k_v (B, V, K) を (B, V, 1, 1, K) に拡張してブロードキャスト
    a_near_dash = a_near + n_k_v.view(n_k_v.size(0), n_k_v.size(1), 1, 1, -1) #

    # --- 3. 受信電力の算出 (dBm) ---
    # 各ビーム・各周波数の電力 P = |振幅|^2
    P_sub_each_dash = torch.abs(a_near_dash) ** 2
    
    # 周波数次元 (K, dim=4) で合計して全帯域電力を出す
    # 結果形状: (B, V, Q, Q) -> 各バッチ、各サブアレーのビームごとの電力
    P_sub_dash = torch.sum(P_sub_each_dash, dim=4)
    
    # dBm変換
    P_sub_dash_dBm = 10 * torch.log10(P_sub_dash)

    return P_sub_dash_dBm

# 注水定理を実行する関数　ここだけCPUで処理する関数
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

def calc_channel_capacity_hybrid_all_data(h_use, h_true, config, water_filling_func):
    """
    h_use:  重み（SVD・注水定理）の決定に使用するチャネル [B, U, V, K]
    h_true: 実際の伝搬環境（真のチャネル） [B, U, V, K]
    config: SystemConfig インスタンス
    water_filling_func: CPU側の注水定理関数
    """
    B, U, V, K = h_use.shape
    device = config.device
    
    # 1. GPUで一括SVD [B*K, U, V]
    # バッチ次元とサブキャリア次元を統合して並列処理
    h_use_flat = h_use.permute(0, 3, 1, 2).reshape(-1, U, V)
    h_true_flat = h_true.permute(0, 3, 1, 2).reshape(-1, U, V)
    
    # 特異値 S と 右特異ベクトル Vh を一括取得
    _, S, Vh = torch.linalg.svd(h_use_flat)
    
    # 2. 特異値をCPUへ転送して注水定理を実行（ハイブリッド処理）
    S_cpu = S.detach().cpu().numpy()
    eig_vals = S_cpu ** 2 # グラム行列の固有値に対応
    
    p_allo_list = []
    ly_list = []
    
    # CPUで各サブキャリアの電力配分を計算
    for ev in eig_vals:
        p_ratio, ly = water_filling_func(ev, config.Pt_mW, config.P_noise_mW)
        p_allo_list.append(p_ratio)
        ly_list.append(ly)
        
    max_ly = max(ly_list) if ly_list else 0
    if max_ly == 0:
        return np.zeros(B), np.zeros(B)
    
    # 3. 結果をGPUに戻して並列計算
    p_allo_gpu = torch.zeros((B*K, max_ly), device=device)
    for i, p in enumerate(p_allo_list):
        if ly_list[i] > 0:
            p_allo_gpu[i, :ly_list[i]] = torch.tensor(p, device=device)
    
    # 送信ウェイト Te_H: 右特異ベクトルの複素共役転置から有効分を抽出
    # ※ PyTorchのバージョンにより、Vh.mch() または Vh.conj().transpose(-2, -1) を使用
    Te_H = Vh.conj().transpose(-2, -1)[:, :, :max_ly]
    
    # 有効チャネル H_eff = H_true * Te_H
    h_eff = torch.bmm(h_true_flat, Te_H)
    
    # 電力行列 A の適用
    A = torch.sqrt(torch.tensor(config.Pt_mW, device=device)) * torch.diag_embed(torch.sqrt(p_allo_gpu))
    
    # 4. 受信側 MMSE 重みの算出 (B*K個を一気に計算)
    gamma0 = config.Pt_mW / config.P_noise_mW
    I_ly = torch.eye(max_ly, device=device).unsqueeze(0)
    
    h_eff_h = h_eff.conj().transpose(-2, -1)
    inv_term = torch.linalg.inv(torch.bmm(h_eff_h, h_eff) + (max_ly / gamma0) * I_ly)
    W_MMSE = torch.bmm(inv_term, h_eff_h).transpose(-2, -1)
    
    # 5. SINR計算
    B_mat = torch.bmm(torch.bmm(W_MMSE.conj().transpose(-2, -1), h_eff), A)
    
    s_pow = torch.abs(torch.diagonal(B_mat, dim1=-2, dim2=-1))**2
    total_pow = torch.sum(torch.abs(B_mat)**2, dim=-1)
    i_pow = total_pow - s_pow
    n_pow = config.P_noise_mW * torch.sum(torch.abs(W_MMSE)**2, dim=-2)
    
    # --- 全データ集計処理 ---
    # 各サブキャリア・各レイヤの容量算出
    c_ly = torch.log2(s_pow / (i_pow + n_pow) + 1.0)
    c_subcarrier = torch.sum(torch.real(c_ly), dim=1)
    
    # 形状を [B, K] に戻す
    c_trials_k = c_subcarrier.view(B, K)
    ly_trials_k = torch.tensor(ly_list, device=device).view(B, K).float()
    
    # 各試行における全サブキャリアの平均値を算出
    cap_per_trial = torch.mean(c_trials_k, dim=1).cpu().numpy()
    ly_per_trial = torch.mean(ly_trials_k, dim=1).cpu().numpy()
    
    return cap_per_trial, ly_per_trial

# シミュレーションの大筋のcore部分
def simulation_core_channelcalculation_gpu(base_batch, d, Ssub_lam, scenario, B, config: SystemConfig):
    """
    config (SystemConfig): システムパラメータの塊
    d (float): 通信距離
    Ssub_lam (float): サブアレー間隔 (ラムダ単位)
    """
    # configからパラメータを抽出
    device = config.device
    Q, V, U, K = config.Q, config.V, config.U, config.K
    f_GHz = config.f_GHz.to(device)
    lam = config.lam # propertyにより自動計算
    
    path_mask = base_batch['mask'].unsqueeze(1) # (B, 1, N, M)
    lam_cen = 0.3 / 142.0 

    # --- 物理座標・角度計算 ---
    eta_dir = np.degrees(np.arcsin(1 / d))
    theta_dir = -eta_dir
    delta_EOD = theta_dir - base_batch['theta_nd_00']
    delta_EOA = eta_dir - base_batch['eta_nd_00']
    theta_deg = base_batch['theta_nd_deg'] + delta_EOD[:, None, None]
    eta_deg = base_batch['eta_nd_deg'] + delta_EOA[:, None, None]
    
    theta_rad = torch.deg2rad(theta_deg)
    eta_rad = torch.deg2rad(eta_deg)
    phi_rad = torch.deg2rad(base_batch['phi_deg'])
    varphi_rad = torch.deg2rad(base_batch['varphi_deg'])

    # --- パイロット信号送受信 ---
    Pu_mW = 10 ** (config.Pu_dBm / 10)
    num_carriers = f_GHz.size(0)
    
    # 各種バッチ計算関数の呼び出し (ch_func等は定義済みとする)
    Pr_dBm = calc_Pr_batched(lam, d, base_batch['chi'], scenario, Pt_dBm=10, do=1.0)
    t_nm = abs_timedelays_batched(d, base_batch)
    R, MUE_coordinate = Mirror_UE_positions_batched(d, base_batch, theta_rad, phi_rad)
    subarray_v_qy_qz = calc_anntena_xyz_Ssub_gpu(lam_cen, V, Q, Ssub_lam, device=device)
    r_mnv0qyqz = distance_to_eachanntena_batched(MUE_coordinate, subarray_v_qy_qz)
    tau_mnv0qyqz = r_mnv0qyqz / 0.3
    phi_rad_v, theta_rad_v, varphi_rad_v, eta_rad_v = calc_all_angles_batched(base_batch, MUE_coordinate, subarray_v_qy_qz)

    Pi_mW = calc_Pi_mW_batched(Pr_dBm, base_batch, scenario=scenario)
    Pi_mW_per_carrier = Pi_mW / num_carriers
    
    DFT_weights = DFT_weight_calc_gpu(Q, device=device)
    b_varphi_eta_v = define_b_phi_eta_batched(eta_rad_v)
    Amp_per_carrier = torch.sqrt(Pi_mW_per_carrier)
    a_phi_theta_v = define_a_batched(V, phi_rad_v, theta_rad_v)

    # 複素振幅合成
    beta_rad = base_batch['beta_rad']
    pilot_signal = Amp_per_carrier.unsqueeze(1) * torch.exp(1j * beta_rad).unsqueeze(1) * b_varphi_eta_v * a_phi_theta_v * path_mask
    phase_term = torch.exp(-2j * torch.pi * f_GHz.view(1, 1, 1, 1, 1, 1, -1) * tau_mnv0qyqz.unsqueeze(-1))
    complex_Amp_antena = torch.einsum('bvnm, bnmvyzk -> bvkyz', pilot_signal, phase_term)

    # 雑音とビーム割当
    n_k_v = noise_n_k_v_batched(B, V, num_carriers, device)
    P_sub_dash_dBm = near_Power_inc_noise_batched(V, Q, DFT_weights, complex_Amp_antena, n_k_v)

    # --- チャネル行列算出 ---
    Amp_digital = torch.sqrt(Pi_mW) / torch.sqrt(torch.tensor(Pu_mW, device=device))
    a_MUE_vnm = Amp_digital.unsqueeze(1) * torch.exp(1j * beta_rad).unsqueeze(1) * b_varphi_eta_v * path_mask
    exp_term = torch.exp(-2j * torch.pi * f_GHz.view(1, 1, 1, 1, 1, 1, -1) * tau_mnv0qyqz.unsqueeze(-1))
    
    u_idx = torch.arange(U, device=device).float()
    c_val = torch.cos(eta_rad_v) * torch.sin(varphi_rad_v)
    ue_phase = torch.exp(-1j * torch.pi * u_idx.view(1, 1, 1, 1, -1) * c_val.unsqueeze(-1))

    a_uvkqyqz = torch.einsum('bvnm, bvnm, bvnmu, bnmvyzk -> buvkyz', a_MUE_vnm, a_phi_theta_v, ue_phase, exp_term)

    # ビーム選択ロジック
    threshold_dBm = -73
    P_max_per_sub, flat_beam_idx = torch.max(P_sub_dash_dBm.view(B, V, -1), dim=2)
    best_pa, best_pe = flat_beam_idx // Q, flat_beam_idx % Q
    active_mask = (P_max_per_sub > threshold_dBm).view(B, 1, V, 1)

    w_DD_selected = DFT_weights[best_pa, best_pe]
    h_uvk = torch.einsum('bvyz, buvkyz -> buvk', w_DD_selected, a_uvkqyqz)

    # 真のチャネル
    h_uvk_tru = h_uvk * active_mask

    # --- デノイズ (ゼロマスキング) ---
    sigma_dash = 1.778 * 1e-6
    n_dash = torch.complex(torch.randn((B, U, V, K), device=device), torch.randn((B, U, V, K), device=device)) * sigma_dash
    h_uvk_est = (h_uvk_tru + n_dash) * active_mask

    # 位相補正
    t_nm_min = t_nm[:, 0, 0].view(B, 1, 1, 1)
    h_est_corrected = h_uvk_est * torch.exp(1j * 2 * torch.pi * f_GHz.view(1, 1, 1, -1) * t_nm_min)

    # 2K-IDFT/DFT デノイズ
    h_w_rev = torch.flip(h_est_corrected, dims=[-1])
    h_w_2k = torch.cat([h_est_corrected, h_w_rev], dim=-1)
    h_tau_2k = torch.fft.ifft(h_w_2k, dim=-1)

    df = f_GHz[1] - f_GHz[0]
    dt = 1.0 / (2 * K * df) # 2Kポイントなので分母は2K
    L_idx = int(round(100e-9 / dt))
    
    h_tau_2k_masked = h_tau_2k.clone()
    h_tau_2k_masked[..., L_idx : 2*K - L_idx] = 0 # 100ns以降をマスク
    
    h_w_denoised = torch.fft.fft(h_tau_2k_masked, dim=-1)[..., :K]

    return h_uvk_tru, h_w_denoised
