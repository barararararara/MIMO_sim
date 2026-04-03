import numpy as np
import Channel_functions as channel
import pandas as pd
import torch
import time
import math

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

def define_a_batched(V, phi_rad, theta_rad):
    """
    バッチ対応版のアンテナ指向性計算 (B, V, N, M)
    phi_rad, theta_rad: (B, V, N, M) 既に次元が揃っている前提
    """
    device = phi_rad.device
    
    # --- 1. サブアレーの設置角 zeta の定義 (V,) ---
    # 0, 120, 240 度の各面 4 枚ずつ
    zeta_deg = torch.tensor([0,0,0,0, 120,120,120,120, 240,240,240,240], device=device)
    # ブロードキャスト用に (1, V, 1, 1) へ変換
    zeta = torch.deg2rad(zeta_deg).view(1, -1, 1, 1)

    # --- 2. 指向性計算 (B, V, N, M) ---
    # 方位角の差分を -pi ~ pi の範囲で算出
    diff = (phi_rad - zeta + torch.pi) % (2 * torch.pi) - torch.pi
    
    cos_theta = torch.cos(theta_rad)
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