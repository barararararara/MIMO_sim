import numpy as np
import random as rd
import scipy.stats as st
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import opt_einsum as oe
import numba
from numba import njit, prange

# step1　距離を設定
def define_d(d_min=5,d_max=30):

    # 一様乱数で基地局と端末の距離を設定
    d = np.random.uniform(d_min,d_max)

    return(d)

def define_chi(setting):
    chi = 0
    if setting == 'InH':
        chi = np.random.normal(0, 2.9) 
    elif setting == 'InF':
        chi = np.random.normal(0, 4.0)
    return (chi)
# step2 環境に依存する、全体の電力を計算
def calc_Pr(lam, d, chi, Pt=10, g_dB=0, setting='InH', do=1):
    # 環境に依る各パラメータを設定
    if setting == 'InH':
        n = 1.8
    elif setting == 'InF':
        n = 1.7

    # 受信電力を計算
    PL_FS = 20 * np.log10(4*np.pi*do/lam)
    PL = PL_FS + 10*n*np.log10(d/do) + chi
    Pr = Pt - PL + g_dB

    return(Pr)
def calc_Pr_each_career(lam, d, chi, Pu=-23, g_dB=0, setting='InH', do=1):
    # 環境に依る各パラメータを設定
    if setting == 'InH':
        n = 1.8
    elif setting == 'InF':
        n = 1.7
    # 受信電力を計算
    PL_FS = 20 * np.log10(4*np.pi*do/lam)
    PL = PL_FS + 10*n*np.log10(d/do) + chi
    Pr_each_career = Pu - PL + g_dB
    return(Pr_each_career)

# Lを求める
# AOD→azumith angle of departure, AOA→azumith angle of arrival
def calc_L(L_AOD_max=2, L_AOA_max=2, setting='InH'):
    if setting == 'InH':
        L_AOD_max = 2
        L_AOA_max = 2
        L_AOD = rd.randint(1,L_AOD_max)
        L_AOA = rd.randint(1,L_AOA_max)
    elif setting == 'InF':
        L_AOD = rd.poisson(1.8) + 1
        L_AOD = rd.poisson(1.9) + 1

    return(L_AOD, L_AOA)

#step3タイムクラスタ数を求める
def Poisson(setting = 'InH'):
    if setting == 'InH':
        lam_c = 0.9
    elif setting == 'InF':
        lam_c = 2.4
    N_dash = np.random.poisson(lam_c)
    N = N_dash + 1 #タイムクラスタ数
    return(N)

#各タイムクラスタのサブパス数M[n]を求めるstep4
def DED(N, setting='InH'):
    M = np.zeros(N, dtype=int)
    for n in range(N):
        # 平均値
        if setting == 'InH':
            mean = 1.4
        elif setting == 'InF':
            mean = 2.6
        # 連続指数分布の乱数を生成
        X = np.random.exponential(scale=mean)
        # 離散化
        X_discrete = np.ceil(X).astype(int) - 1
        M[n] = X_discrete + 1
    return(M)

# タイムクラスタ内での各サブパスの相対的な遅延step5
def intracluster_delays(M, N, setting='InH'):
    rho = [np.zeros(M[i]) for i in range(N)]  # 初期化

    if setting == 'InH':
        mu = 1.1
        for n in range(N):
            delays = np.random.exponential(mu, size=M[n])  # 遅延をまとめて生成
            delays[0] = 0  # 最初の遅延を 0 にする
            rho[n] = np.sort(delays)  # ここで正しくソート

    elif setting == 'InF':
        alpha_rho = 1.2
        beta_rho = 16.3
        for n in range(N):
            delays = np.random.gamma(alpha_rho, beta_rho, size=M[n])  # まとめて生成
            delays[0] = 0  # 最初の遅延を 0 にする
            rho[n] = np.sort(delays)  # ここで正しくソート

    return rho

# step6 1番目のタイムクラスタの先頭のサブパスからn番目のタイムクラスタの先頭のサブパスまでの遅延τ_n_step6
def cluster_excess_delays(rho, N, M, setting):
    tau = np.zeros(N)
    tau_dash_dash = np.zeros(N)
    delta_tau = np.zeros(N)
    if setting == 'InH':
        meu_tau=14.6
        MTI = 6
        tau_dash_dash = np.random.exponential(meu_tau, N)
    elif setting == 'InF':
        alpha_tau = 0.7
        beta_tau = 26.9
        tau_dash_dash = np.random.gamma(alpha_tau, beta_tau, N)
        MTI = 8

    for n in range(1,N):
        delta_tau = np.sort(tau_dash_dash) - np.min(tau_dash_dash)
        tau[0] = 0
        tau[n] = tau[n-1] + rho[n-1][-1] + delta_tau[n] + MTI
        print(n, M[n-1])
        print('rho[n][M[n]-1]',rho[n-1][-1])
        print('delta_tau', delta_tau)

    return(tau)

# step7の乱数による部分
def define_Zn(N,setting):
    if setting == 'InH':
        sigma_Z = 9.0
    elif setting == 'InF':
        sigma_Z = 10
    Z = np.random.normal(0,sigma_Z, N)
    return(Z)

# step7　n番目のクラスタ電力Pを決定する
def cluster_power(Pr,N,tau,Z,setting='InH'):
    if setting == 'InH':
        gamma = 18.2
    elif setting == 'InF':
        gamma = 16.2
    P_dash_sum = 0
    P = np.zeros(N)
    P_dash = np.zeros(N)
    Pr_mW = 10 ** (Pr / 10)

    P_dash = np.exp(-tau/gamma)*10**(Z/10)
    P_dash_sum = np.sum(P_dash)
    P = (P_dash/P_dash_sum)*Pr_mW
    
    return(P)

def cluster_Power_each_career(Pr_each_career,Z,N,tau,setting='InH'):
    if setting == 'InH':
        gamma = 18.2
    elif setting == 'InF':
        gamma = 16.2
    P_dash_sum = 0
    P_each_career = np.zeros(N)
    P_dash = np.zeros(N)
    Pr_mW = 10 ** (Pr_each_career / 10)

    P_dash = np.exp(-tau/gamma)*10**(Z/10)
    P_dash_sum = np.sum(P_dash)
    P_each_career = (P_dash/P_dash_sum)*Pr_mW
    
    return(P_each_career)

# step8の乱数によるUn,mを計算するとこ
def define_Unm(N,M,setting):
    if setting == 'InH':
        sigma_U = 5.0
    elif setting == 'InF':
        sigma_U = 13
    U = [np.zeros(M[i]) for i in range(N)]
    for n in range(N):
        for m in range(M[n]):
            U[n][m] = np.random.normal(0, sigma_U)
    return U

# step8 n番目のクラスタ内のm番目のサブパスの電力Π_m,nを決定
def SP_power(N,M,P,rho,U, setting='InH'):
    if setting == 'InH':
        gamma = 2.0
    elif setting == 'InF':
        gamma = 4.7
    Pi_dash = [np.zeros(M[i]) for i in range(N)]
    Pi = np.zeros((N,max(M)))
    for n in range(N):
        for m in range(M[n]):
            Pi_dash[n][m] = np.exp(-rho[n][m]/gamma)*(10**(U[n][m]/10))
            Pi[n][m] = (Pi_dash[n][m] / np.sum(Pi_dash[n])) * P[n]
            
    # Pi の最大値とそのインデックスを取得
    max_n, max_m = np.unravel_index(np.argmax(Pi, axis=None), Pi.shape)

    # Pi[0][0] が最大じゃない場合、入れ替える
    if (max_n, max_m) != (0, 0):
        Pi[0][0], Pi[max_n][max_m] = Pi[max_n][max_m], Pi[0][0]

    return(Pi)

def SP_Power_each_career(N,M,P_each_career,rho,U,setting='InH'):
    if setting == 'InH':
        gamma = 2.0
        sigma_U = 5.0
    elif setting == 'InF':
        gamma = 4.7
        sigma_U = 13
    Pi_dash = [np.zeros(M[i]) for i in range(N)]
    Pi_each_career = np.zeros((N,max(M)))

    for n in range(N):
        for m in range(M[n]):
            Pi_dash[n][m] = np.exp(-rho[n][m]/gamma)*(10**(U[n][m]/10))
            Pi_each_career[n][m] = (Pi_dash[n][m] / np.sum(Pi_dash[n])) * P_each_career[n]
            
    # Pi の最大値とそのインデックスを取得
    max_n, max_m = np.unravel_index(np.argmax(Pi_each_career, axis=None), Pi_each_career.shape)

    # Pi[0][0] が最大じゃない場合、入れ替える
    if (max_n, max_m) != (0, 0):
        Pi_each_career[0][0], Pi_each_career[max_n][max_m] = Pi_each_career[max_n][max_m], Pi_each_career[0][0]

    return(Pi_each_career)

# step9 各サブパスの位相を与える
def SP_phases(M,N):
    beta = [np.zeros(M[i]) for i in range(N)]

    for n in range (N):
        for m in range (M[n]):
            beta[n][m] = np.random.uniform(0, 2 * np.pi)

    return(beta)

# step10　各サブパスの絶対遅延時間を求める
def abs_timedelays(d,rho,tau,N,M):
    t_nm = [np.zeros(M[i]) for i in range(N)]
    c=0.3
    to = d/c
    for n in range(N):
        for m in range(M[n]):
            t_nm[n][m] = to + tau[n] + rho[n][m]
    return(t_nm)

# step11a　i番目のSPの方位角の平均値を決定
def mean_a_angles(L_AOD, L_AOA):
    varphi_mean_AOA = np.zeros(L_AOA)
    phi_mean_AOD = np.zeros(L_AOD)

    for i in range(L_AOD):
        phi_min = 360*(i)/L_AOD
        phi_max = 360*(i+1)/L_AOD
        phi_mean_AOD[i] = np.random.uniform(phi_min, phi_max)
    for j in range(L_AOA):
        phi_min = 360*(j)/L_AOA
        phi_max = 360*(j+1)/L_AOA
        varphi_mean_AOA[j] = np.random.uniform(phi_min, phi_max)

    # print('phi_AOD\n', phi_mean_AOD)
    # print('varphi_AOA\n', varphi_mean_AOA)
    return(phi_mean_AOD, varphi_mean_AOA)

# step11b i番目のAOAとAODの仰角の平均値を決定
def mean_e_angles(L_AOD, L_AOA, setting='InH'):
    theta_mean_AOD = np.zeros(L_AOD)
    eta_mean_AOA = np.zeros(L_AOA)
    if setting == 'InH':
        meu_AOD = -6.8
        meu_AOA = 7.4
        sigma_AOD = 4.9
        sigma_AOA = 4.5
    elif setting == 'InF':
        meu_AOD = -4.0
        meu_AOA = 4.0
        sigma_AOD = 4.3
        sigma_AOA = 4.3

    for i in range(L_AOD):
        theta_mean_AOD[i] = np.random.normal(meu_AOD,sigma_AOD) 
    for i in range(L_AOA):
        eta_mean_AOA[i] = np.random.normal(meu_AOA,sigma_AOA)

    return(theta_mean_AOD, eta_mean_AOA)

# step12 n番目のタイムクラスタのm番目のサブパスについて4つの角度を求める
def generate_AOD_AOA_angle(M, N, L_AOD, L_AOA, phi_mean_AOD, varphi_mean_AOA, theta_mean_AOD, eta_mean_AOA, setting):
    if setting == 'InH':
        sigma_phi_AOD = 4.8
        sigma_theta_AOD = 4.3
        sigma_phi_AOA = 4.7
        sigma_theta_AOA = 4.4
    elif setting == 'InF':
        sigma_phi_AOD = 6.7
        sigma_theta_AOD = 3.0
        sigma_phi_AOA = 11.7
        sigma_theta_AOA = 2.3

    phi = [np.zeros((M[i],)) for i in range(N)]
    theta = [np.zeros((M[i],)) for i in range(N)]
    varphi = [np.zeros((M[i],)) for i in range(N)]
    eta = [np.zeros((M[i],)) for i in range(N)]

    for n in range(N):
        for m in range(M[n]):
            if L_AOD > 1:
                i = np.random.randint(0,L_AOD)
            else:
                i = 0
            if L_AOA > 1:
                j = np.random.randint(0,L_AOA)
            else:
                j = 0
            if N == 1 and M[0] == 1:
                i = 0
                j = 0
            # print('i:AOD, j:AOA', i,j)
            phi[n][m] = phi_mean_AOD[i] + np.random.normal(0, sigma_phi_AOD)
            theta[n][m] = theta_mean_AOD[i] + np.random.normal(0, sigma_theta_AOD)
            varphi[n][m] = varphi_mean_AOA[j] + np.random.normal(0,sigma_phi_AOA)
            eta[n][m] = eta_mean_AOA[j] + np.random.normal(0,sigma_theta_AOA)
    
    
    phi_dash = [np.zeros(M[i]) for i in range(N)]
    varphi_dash = [np.zeros(M[i]) for i in range(N)]

    
    # 直接波の方向を決定
    phi_dir = np.random.uniform(0, 60)
    varphi_dir = phi_dir + 180

    for i in range(len(phi)):
        phi_0_0 = phi[0][0] if len(phi[i]) > 0 else phi[0]
        varphi_0_0 = varphi[0][0] if len(varphi[i]) > 0 else varphi[0]

    # 各角度の変化量を計算
    Delta_AOD = phi_dir - phi_0_0
    Delta_AOA = varphi_dir - varphi_0_0

    # 回転後の角度を計算
    for n in range(N):
        for m in range(M[n]):
            phi_dash[n][m] = phi[n][m] + Delta_AOD
            varphi_dash[n][m] = varphi[n][m] + Delta_AOA

    return phi_dash, varphi_dash, theta, eta

######################################################################################################################################################################

# 散乱体１までの距離r
def scatter1_distance(d, Q, V, N, M, rho, tau):
    lam = 3.0*1e8 / 142 / 1e9
    r_dash = lam*(Q+1)*V / (6*math.sqrt(3))
    R = [np.zeros(M[i]) for i in range(N)]
    for n in range(N):
        for m in range(M[n]):
            R[n][m] = d + 0.3*(tau[n] + rho[n][m])

    r = [np.zeros(M[i]) for i in range(N)]
    for n in range(N):
        for m in range(M[n]):
            r[n][m] = np.random.uniform(r_dash, R[n][m])
    r[0][0] = d
    return r, R

# 新バージョン
def scatter1_xyz_cordinate(N, M, r, phi_rad, theta_rad):
    max_M = max(M)
    xyz_coordinates = np.zeros((N, max_M, 3))

    for n in range(N):
        m_len = M[n]
        r_n = np.asarray(r[n][:m_len])
        theta_n = np.asarray(theta_rad[n][:m_len])
        phi_n = np.asarray(phi_rad[n][:m_len])

        cos_theta = np.cos(theta_n)
        x = r_n * cos_theta * np.cos(phi_n)
        y = r_n * cos_theta * np.sin(phi_n)
        z = r_n * np.sin(theta_n)

        xyz_coordinates[n, :m_len, 0] = x
        xyz_coordinates[n, :m_len, 1] = y
        xyz_coordinates[n, :m_len, 2] = z

    return xyz_coordinates

# サブアレーの各素子の座標を計算
def calc_anntena_xyz(lam, V, Q):
    L = (lam * (Q + 1) / 2) * V / 3
    subarray_v_qy_qz = np.zeros((V, Q, Q, 3))

    qy_idx, qz_idx = np.meshgrid(np.arange(Q), np.arange(Q), indexing='ij')

    for v in range(V):
        if 0 <= v < V / 3:
            x = L / (2 * math.sqrt(3))
            y_base = -L / 2 + lam / 2 * (1 + (Q + 1)) * v
            y = y_base + qy_idx * lam / 2
            z = qz_idx * lam / 2
            coords = np.stack([np.full_like(y, x), y, z], axis=-1)

        elif V / 3 <= v < 2 * V / 3:
            offset = v - V / 3
            x = L / (2 * math.sqrt(3)) - math.sqrt(3) / 2 * lam / 2 * (1 + (Q + 1) * offset)
            y = L / 2 - 0.5 * lam / 2 * (1 + (Q + 1) * offset)
            x_shift = -math.sqrt(3) * qy_idx * lam / 4
            y_shift = -qy_idx * lam / 4
            z = qz_idx * lam / 2
            coords = np.stack([x + x_shift, y + y_shift, z], axis=-1)

        else:
            offset = v - 2 * V / 3
            x = -L / math.sqrt(3) + math.sqrt(3) / 2 * lam / 2 * (1 + (Q + 1) * offset)
            y = -0.5 * lam / 2 * (1 + (Q + 1) * offset)
            x_shift = math.sqrt(3) * qy_idx * lam / 4
            y_shift = -qy_idx * lam / 4
            z = qz_idx * lam / 2
            coords = np.stack([x + x_shift, y + y_shift, z], axis=-1)

        subarray_v_qy_qz[v] = coords

    return subarray_v_qy_qz

# サブアレー間隔を広げるようなver
def calc_anntena_xyz_wide(lam, V, Q, S_sub):
    L = (lam * (Q + 1) / 2) * V / 3 + (V/3 - 1) * S_sub
    subarray_v_qy_qz = np.zeros((V, Q, Q, 3))

    qy_idx, qz_idx = np.meshgrid(np.arange(Q), np.arange(Q), indexing='ij')

    for v in range(V):
        if 0 <= v < V / 3:
            x = L / (2 * math.sqrt(3))
            y_base = -L / 2 + lam / 2 * (1 + (Q + 1)) * v + S_sub * v
            y = y_base + qy_idx * lam / 2
            z = qz_idx * lam / 2
            coords = np.stack([np.full_like(y, x), y, z], axis=-1)

        elif V / 3 <= v < 2 * V / 3:
            offset = v - V / 3
            x = L / (2 * math.sqrt(3)) - math.sqrt(3) / 2 * (lam / 2 * (1 + (Q + 1) * offset) + S_sub * offset)
            y = L / 2 - 0.5 * lam / 2 * (1 + (Q + 1) * offset) - 1/2 * S_sub * offset
            x_shift = -math.sqrt(3) * qy_idx * lam / 4
            y_shift = -qy_idx * lam / 4
            z = qz_idx * lam / 2
            coords = np.stack([x + x_shift, y + y_shift, z], axis=-1)

        else:
            offset = v - 2 * V / 3
            x = -L / math.sqrt(3) + math.sqrt(3) / 2 * lam / 2 * (1 + (Q + 1) * offset) + math.sqrt(3)/2 * S_sub * offset
            y = -0.5 * lam / 2 * (1 + (Q + 1) * offset) - 1/2 * S_sub * offset
            x_shift = math.sqrt(3) * qy_idx * lam / 4
            y_shift = -qy_idx * lam / 4
            z = qz_idx * lam / 2
            coords = np.stack([x + x_shift, y + y_shift, z], axis=-1)

        subarray_v_qy_qz[v] = coords
        
    return subarray_v_qy_qz

# 散乱体１と各素子の距離を計算 もっときれいに書けそう
def distance_scatter1_to_eachanntena(xyz_coordinates, subarray_v_qy_qz, N, M, V, Q):
    max_M = np.max(M)
    r_qy_qz = np.zeros((V, N, max_M, Q, Q), dtype=np.float64)

    for n in range(N):
        for m in range(M[n]):
            # (V, Q, Q, 3) - (1, 1, 1, 3) → (V, Q, Q, 3)
            diffs = subarray_v_qy_qz - xyz_coordinates[n, m][None, None, None, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=-1))  # → (V, Q, Q)
            r_qy_qz[:, n, m, :, :] = dists

    return r_qy_qz

# ビーム割り当て用のゼータを設定
def define_zeta(V):
    zeta = np.zeros(V, dtype=float)
    
    for v in range(V):
        if 0 <= v <= V/3-1 :
            zeta[v] = 0
        elif V/3 <= v <= (2*V/3 - 1):
            zeta[v] = 2*math.pi/3
        elif 2*V/3 <= v <= (V-1):
            zeta[v] = 4*math.pi/3
    
    return(zeta)

# ビーム割り当て用のベータを設定 (varphiには依存しない)　18
def define_b_verphi_eta(N, M, eta_rad):
    b_varphi_eta = [np.zeros(M[i]) for i in range(N)]
    for n in range(N):
        b_varphi_eta[n] = math.sqrt(3/2)*np.cos(eta_rad[n])
    return(b_varphi_eta)

# ビーム割り当て用のaを設定
def define_a(V, N, M, phi_rad, theta_rad):
    a = np.zeros((V, N, max(M)))
    zeta = define_zeta(V)
    for v in range(V):
        for n in range(N):
            for m in range(M[n]):
                diff = (phi_rad[n][m] - zeta[v] + math.pi) % (2*math.pi) - math.pi
                if abs(diff) > math.pi/2:  # 背面
                    a[v][n][m] = 0
                else:                      # 前面
                    a[v][n][m] = 2.282*np.cos(theta_rad[n][m]) * \
                                np.sin((math.pi/2)*np.cos(theta_rad[n][m]) * np.cos(diff))
                
    return(a)

# 各アンテナの原点においての複素振幅を計算
def calc_complex_Amp_at_O(f, V, N, M, Amp_per_career, beta, tau_nm, b_varphi_eta):
    num_freq = len(f)  # 周波数の数
    complex_Amp_at_O = np.zeros((N, max(M), num_freq), dtype=complex)  # 初期化

    for v in range(V):
        for n in range(N):
            for m in range(M[n]):
                # tau_nm の形状に応じて値を取得
                if np.isscalar(tau_nm[n]):
                    tau_value = tau_nm[n]
                elif len(np.shape(tau_nm[n])) == 1:
                    tau_value = tau_nm[n][m]
                else:
                    tau_value = tau_nm[n, m]

                # 振幅計算（ベクトル化）
                complex_Amp_at_O[n, m, :] = (
                    Amp_per_career[n][m]  
                    * np.exp(1j * beta[n][m]) 
                    * np.exp(-1j * 2 * np.pi * f * tau_value)  
                    * b_varphi_eta[n][m] 
                )

    # 無効な値のチェック（オプション）
    if not np.all(np.isfinite(complex_Amp_at_O)):
        raise ValueError("Complex amplitude contains NaN or inf values.")
    
    return complex_Amp_at_O

# v番目のサブアレーのqy,qz番目のアンテナ素子での周波数fkでの複素振幅を求める
def complex_Amp_at_each_antena(f, V, Q, N, M, distance_sca_to_anntena, complex_Amp_at_scatter1, a_varphi_theta):
    # tau_nm_v_qyqz は、distance_sca_to_anntena / 0.3 としてベクトル化して一度に計算
    tau_nm_v_qyqz = np.zeros((V,N,max(M),Q,Q))
    tau_nm_v_qyqz = distance_sca_to_anntena / 0.3 / 1e9 #ここに限っては、fをHzで初期化しているからe9はok

    # a_nm_v_qyqz の次元を V, N, max(M), len(f), Q, Q に合わせて初期化
    a_nm_v_qyqz = np.zeros((V, N, max(M), len(f), Q, Q), dtype=complex)

    # 周波数 f の全てに対して複素振幅を計算
    for v in range(V):
        for n in range(N):
            for m in range(M[n]):
                for qy in range(Q):
                    for qz in range(Q):
                        # tau_nm_v_qyqz と f[k] の計算をベクトル化して一括処理
                        a_nm_v_qyqz[v, n, m, :, qy, qz] = complex_Amp_at_scatter1[n,m,:] * a_varphi_theta[v][n][m] * np.exp(-2j * np.pi * f[:] * tau_nm_v_qyqz[v, n, m, qy, qz])

    # n と m の次元について合計（この処理は axis=(1, 2) で一括）
    a_v_qyqz = np.sum(a_nm_v_qyqz, axis=(1, 2))  # (V, len(f), Q, Q)

    return a_v_qyqz

# 第１散乱体での複素振幅を計算
def complex_Amp_at_scatter1(N, M, f, r, complex_Amp_at_O):
    complex_Amp_at_scatter1 = np.zeros((N,max(M),len(f)), dtype=complex)
    tau1 = [np.zeros(M[i]) for i in range(N)]
    for n in range(N):
        tau1[n] = r[n] / 0.3
    for n in range(N):
        for m in range(M[n]):
            for k, freq in enumerate(f):  # f[k] をループ
                complex_Amp_at_scatter1[n, m, k] = complex_Amp_at_O[n, m, k] * np.exp(1j * 2 * np.pi * freq * tau1[n][m])
    return complex_Amp_at_scatter1

# DFTのウエイトを計算する 近傍界用 26
def DFT_weight_calc(Q):
    w_DD = np.array((Q,Q,Q,Q), dtype=complex)
    indices = np.arange(Q)
    pa, pe, qy, qz = np.meshgrid(indices, indices, indices, indices, indexing="ij")
    
    w_DD = (1 / Q) * np.exp(-1j * np.pi * qy * (-Q + 2 * pa + 2) / Q) \
                * np.exp(-1j * np.pi * qz * (-Q + 2 * pe + 2) / Q)
    return(w_DD)

def DFT_weight_calc_pape(Q, pa, pe):
    w_DD = np.array((Q, Q), dtype=complex)
    indices = np.arange(Q)
    qy, qz = np.meshgrid(indices, indices, indexing="ij")
    w_DD = (1 / Q) * np.exp(-1j * np.pi * qy * (-Q + 2 * pa + 2) / Q) \
                * np.exp(-1j * np.pi * qz * (-Q + 2 * pe + 2) / Q)
    return(w_DD)

def g_DD_pa_pe(N, M, phi_rad, theta_rad, zeta, Q, V):
    pa, pe = np.meshgrid(np.arange(Q), np.arange(Q), indexing='ij')  # Q x Q のグリッド作成
    gDD = np.zeros((V, N, max(M), Q, Q), dtype=np.complex128)  # 複素数型にする
    
    for v in range(V):
        for n in range(N):
            for m in range(M[n]):  # M[n] の範囲内でループ                
                # Dpa, Dpe の計算（Q x Q の配列をそのまま代入）
                Dpa = np.pi * (np.cos(theta_rad[n][m]) * np.sin(phi_rad[n][m] - zeta[v]) + 1 - 2 * pa / Q - 2 / Q)
                Dpe = np.pi * (np.sin(theta_rad[n][m]) + 1 - 2 * pe / Q - 2 / Q)
                
                # ゼロ除算を防ぐための処理
                denominator_A = np.sin(Dpa / 2)
                gDD_A = np.where(np.abs(denominator_A) != 0, np.sin(Q * Dpa / 2) / denominator_A, Q)
                
                denominator_E = np.sin(Dpe / 2)
                gDD_E = np.where(np.abs(denominator_E) != 0, np.sin(Q * Dpe / 2) / denominator_E, Q)
                
                # 位相計算
                Phase = np.exp(1j * (Q - 1) * (Dpa + Dpe) / 2)

                # 指向性計算（Q x Q の配列をそのまま代入）
                gDD[v, n, m] = (1 / Q) * gDD_A * gDD_E * Phase

    return gDD

def g_dd_depend_on_pape(N, M, phi_rad, theta_rad, zeta, Q, V, pa, pe):
    gDD = np.zeros((N,max(M)), dtype=complex)
    gDD_A = [np.zeros(M[i],dtype=complex) for i in range(N)]
    gDD_E = [np.zeros(M[i], dtype=complex) for i in range(N)]
    Phase = [np.zeros(M[i], dtype=complex) for i in range(N)]
    Dpa = [np.zeros(M[i], dtype=float) for i in range(N)]
    Dpe = [np.zeros(M[i], dtype=float) for i in range(N)]
    Phase = [np.zeros(M[i], dtype=complex) for i in range(N)]
    for n in range(N):
        for m in range(M[n]):  # M[n] の範囲内でループ                
            # Dpa, Dpe の計算（Q x Q の配列をそのまま代入）
            Dpa[n][m] = np.pi * (np.cos(theta_rad[n][m]) * np.sin(phi_rad[n][m] - zeta[V]) + 1 - 2 * pa / Q - 2 / Q)
            Dpe[n][m] = np.pi * (np.sin(theta_rad[n][m]) + 1 - 2 * pe / Q - 2 / Q)
            
            # ゼロ除算を防ぐための処理
            denominator_A = np.sin(Dpa[n][m] / 2)
            gDD_A[n][m] = np.where(np.abs(denominator_A) != 0, np.sin(Q * Dpa[n][m] / 2) / denominator_A, Q)
            
            denominator_E = np.sin(Dpe[n][m] / 2)
            gDD_E[n][m] = np.where(np.abs(denominator_E) != 0, np.sin(Q * Dpe[n][m] / 2) / denominator_E, Q)
            
            # 位相計算
            Phase[n][m] = np.exp(1j * (Q - 1) * (Dpa[n][m] + Dpe[n][m]) / 2)

            # 指向性計算（Q x Q の配列をそのまま代入）
            gDD[n][m] = (1 / Q) * gDD_A[n][m] * gDD_E[n][m] * Phase[n][m]
    return gDD


# 雑音を生成する関数n_k_v
def noise_n_k_v(V):
    # (V, 2000) サイズの複素乱数を生成
    real_part = np.random.normal(0, 1.778 * 1e-6, (V, 2000))
    imag_part = np.random.normal(0, 1.778 * 1e-6, (V, 2000))
    n_k_v = real_part + 1j * imag_part
    return n_k_v

@njit(parallel=True, fastmath=True)
def fast_dot_einsum(DFT_weight, amp):
    V, K, Y, Z = amp.shape
    A, E, _, _ = DFT_weight.shape
    out = np.zeros((V, A, E, K), dtype=np.complex128)
    for v in prange(V):          # 並列ループは外側に
        for a in range(A):
            for e in range(E):
                for k in range(K):
                    val = 0.0 + 0.0j
                    for y in range(Y):
                        for z in range(Z):
                            val += DFT_weight[a,e,y,z] * amp[v,k,y,z]
                    out[v,a,e,k] = val
    return out

@njit(parallel=True, fastmath=True)
def fast_dot_einsum_optimized(DFT_weight, amp):
    V, K, Y, Z = amp.shape
    A, E, _, _ = DFT_weight.shape
    out = np.zeros((V, A, E, K), dtype=np.complex128)

    for v in prange(V):
        for a in range(A):
            for e in range(E):
                dft_local = DFT_weight[a, e]  # shape (Y,Z)
                for k in range(K):
                    amp_local = amp[v, k]     # shape (Y,Z)
                    # Z×Y個のスカラー積→スカラー加算に
                    val = 0.0 + 0.0j
                    for y in range(Y):
                        for z in range(Z):
                            val += dft_local[y, z] * amp_local[y, z]
                    out[v, a, e, k] = val
    return out


# 近傍界条件におけるビーム電力
# def near_Power_inc_noise(V,Q,f,DFT_weight,complex_Amp_at_each_antena,n_k_v):
#     # 近傍界　励振後のv番目のサブアレーの出力(複素振幅) 27
#     a_near = np.zeros((V,Q,Q,len(f)), dtype=complex)
#     a_near = np.einsum('aeyz,vkyz->vaek', DFT_weight, complex_Amp_at_each_antena, optimize=True)
#     # 近傍領域の信号成分＋雑音
#     a_near_dash = np.zeros_like(a_near)
#     a_near_dash = a_near + n_k_v[:,np.newaxis, np.newaxis, :] #35
#     # 近傍領域として扱ったときのサブアレー受信電力（信号成分＋雑音）
#     P_sub_each_dash = np.abs(a_near_dash) ** 2
#     P_sub_dash = np.sum(P_sub_each_dash, axis=(3))
#     P_sub_dash_dBm = 10 * np.log10(abs(P_sub_dash))
#     return P_sub_dash_dBm

def near_Power_inc_noise(V,Q,f,DFT_weight,complex_Amp_at_each_antena,n_k_v):
    # 近傍界　励振後のv番目のサブアレーの出力(複素振幅) 27
    a_near = np.zeros((V,Q,Q,len(f)), dtype=complex)
    a_near = fast_dot_einsum(DFT_weight, complex_Amp_at_each_antena)
    # 近傍領域の信号成分＋雑音
    a_near_dash = np.zeros_like(a_near)
    a_near_dash = a_near + n_k_v[:,np.newaxis, np.newaxis, :] #35
    # 近傍領域として扱ったときのサブアレー受信電力（信号成分＋雑音）
    P_sub_each_dash = np.abs(a_near_dash) ** 2
    P_sub_dash = np.sum(P_sub_each_dash, axis=(3))
    P_sub_dash_dBm = 10 * np.log10(abs(P_sub_dash))
    return P_sub_dash_dBm

def near_Power_inc_noise_optimized(V,Q,f,DFT_weight,complex_Amp_at_each_antena,n_k_v):
    # 近傍界　励振後のv番目のサブアレーの出力(複素振幅) 27
    a_near = np.zeros((V,Q,Q,len(f)), dtype=complex)
    a_near = fast_dot_einsum_optimized(DFT_weight, complex_Amp_at_each_antena)
    # 近傍領域の信号成分＋雑音
    a_near_dash = np.zeros_like(a_near)
    a_near_dash = a_near + n_k_v[:,np.newaxis, np.newaxis, :] #35
    # 近傍領域として扱ったときのサブアレー受信電力（信号成分＋雑音）
    P_sub_each_dash = np.abs(a_near_dash) ** 2
    P_sub_dash = np.sum(P_sub_each_dash, axis=(3))
    P_sub_dash_dBm = 10 * np.log10(abs(P_sub_dash))
    return P_sub_dash_dBm

# 遠方界条件における電力
def far_Power_inc_noise(V,Q,f,complex_Amp_at_O,g_DD,n_k_v,a):
    # 遠方界
    a_far = np.zeros((V, Q, Q, len(f)))
    a_far = np.einsum('nmk,vnm,vnmae->vaek', complex_Amp_at_O, a, g_DD, optimize=True)
    # 遠方界想定の信号成分＋雑音
    a_far_dash = np.zeros_like(a_far)
    a_far_dash = a_far + n_k_v[:,np.newaxis, np.newaxis, :] #44
    # 遠方界を想定した時のサブアレー受信電力（信号成分＋雑音）
    P_far_sub_each_dash = np.abs(a_far_dash) ** 2
    P_far_sub_dash = np.sum(P_far_sub_each_dash, axis=(3))
    P_far_sub_dash_dBm = 10* np.log10(abs(P_far_sub_dash))
    return P_far_sub_dash_dBm

# 現時点で検討している、0,0,5m,pe=31についてのグラフを表示する関数 25/01/21
def plot_graph_0(d,Q,P_sub_dash_dBm,P_far_sub_dash_dBm,theta_rad,phi_rad,pe=31):
    # 0度の面に関して
    phi_dire = np.zeros(Q)
    thete_dire = np.asin((2*(pe+1)/Q - 1))
    for pa in range(Q):
        phi_dire[pa] = np.asin(2*(pa+1)/Q - 1)
    y_near_dash_v1 = np.zeros(Q)
    y_far_dash_v1 = np.zeros(Q)
    for pa in range(Q):
        if P_sub_dash_dBm[1][pa][pe]!=None:
            y_near_dash_v1[pa] = P_sub_dash_dBm[1][pa][pe]
        else:
            y_near_dash_v1[pa] = None
        if P_far_sub_dash_dBm[1][pa][pe]!=None:
            y_far_dash_v1[pa] = P_far_sub_dash_dBm[1][pa][pe]
        else:
            y_far_dash_v1[pa] = None
    
    x = np.degrees(phi_dire)

    # print(f'y_near_v1[31] = {y_near_dash_v1[31]}')

    # プロット
    plt.scatter(x, y_near_dash_v1, color='green', label='Near-Field')
    plt.scatter(x, y_far_dash_v1, color='black', label='Far-Field')
    
    plt.title(f"theta_1,1 = {np.degrees(theta_rad[0])}, phi_1,1 = {np.degrees(phi_rad[0])}°, d = {d}, v = 1, pe = 31", fontsize=0)

    # ラベル
    plt.xlabel('Φ',fontsize=25)
    plt.ylabel('Power [dBm]',fontsize=25)
    plt.tick_params(axis='both', labelsize=20)  # x軸・y軸の目盛のフォントサイズを14に設定


    # 凡例
    plt.legend(fontsize=25)

    # グラフの表示
    plt.show()

# 現時点で検討している、0,0,5m,pe=31についてのグラフを表示する関数 25/01/21
def plot_graph_120(d,Q,P_sub_dash_dBm,P_far_sub_dash_dBm,theta_rad,phi_rad, pe=31):
    # 0度の面に関して
    phi_dire = np.zeros(Q)
    thete_dire = np.asin((2*(pe+1)/Q - 1))
    for pa in range(Q):
        phi_dire[pa] = np.asin(2*(pa+1)/Q - 1)
    y_near_dash_v1 = np.zeros(Q)
    y_far_dash_v1 = np.zeros(Q)
    for pa in range(Q):
        if P_sub_dash_dBm[5][pa][pe]!=None:
            y_near_dash_v1[pa] = P_sub_dash_dBm[5][pa][pe]
        else:
            y_near_dash_v1[pa] = None
        if P_far_sub_dash_dBm[5][pa][pe]!=None:
            y_far_dash_v1[pa] = P_far_sub_dash_dBm[5][pa][pe]
        else:
            y_far_dash_v1[pa] = None
    
    x = np.degrees(phi_dire) + 120

    print(f'y_near_v1[31] = {y_near_dash_v1[31]}')

    # プロット
    plt.scatter(x, y_near_dash_v1, color='green', label='near_dash')
    plt.scatter(x, y_far_dash_v1, color='black', label='far_dash')
    
    plt.title(f"theta_1,1 = {np.degrees(theta_rad[0])}, phi_1,1 = {np.degrees(phi_rad[0])}°, d = {d}, v = 5, pe = 31")

    # ラベル
    plt.xlabel('phi')
    plt.ylabel('Power[dBm]')

    # 凡例
    plt.legend()

    # グラフの表示
    plt.show()

# 現時点で検討している、0,0,5m,pe=31についてのグラフを表示する関数 25/01/21
def plot_graph_240(d,Q,P_sub_dash_dBm,P_far_sub_dash_dBm,theta_rad,phi_rad, pe=31):
    # 0度の面に関して
    phi_dire = np.zeros(Q)
    thete_dire = np.asin((2*(pe+1)/Q - 1))
    for pa in range(Q):
        phi_dire[pa] = np.asin(2*(pa+1)/Q - 1)
    y_near_dash_v1 = np.zeros(Q)
    y_far_dash_v1 = np.zeros(Q)
    for pa in range(Q):
        if P_sub_dash_dBm[9][pa][pe]!=None:
            y_near_dash_v1[pa] = P_sub_dash_dBm[9][pa][pe]
        else:
            y_near_dash_v1[pa] = None
        if P_far_sub_dash_dBm[9][pa][pe]!=None:
            y_far_dash_v1[pa] = P_far_sub_dash_dBm[9][pa][pe]
        else:
            y_far_dash_v1[pa] = None
    
    x = np.degrees(phi_dire) + 240

    print(f'y_near_v1[31] = {y_near_dash_v1[31]}')

    # プロット
    plt.scatter(x, y_near_dash_v1, color='green', label='near_dash')
    plt.scatter(x, y_far_dash_v1, color='black', label='far_dash')
    
    plt.title(f"theta_1,1 = {np.degrees(theta_rad[0])}, phi_1,1 = {np.degrees(phi_rad[0])}°, d = {d}, v = 9, pe = 31")

    # ラベル
    plt.xlabel('phi')
    plt.ylabel('Power[dBm]')

    # 凡例
    plt.legend()

    # グラフの表示
    plt.show()

##########################################################################################################
def plot_3d_histogram_from_2d_array(data,channel_num,V):
    """
    負の値を含む64x64の2Dデータを3Dヒストグラム風にプロットする関数。
    すべての値を正の範囲にシフトし、z=0 から上に伸びるように調整。
    Z軸の値は元のデータと一致するように補正。
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # データの最小値を0にシフト
    data_min = -80
    shifted_data = data - data_min  # すべての値を正の範囲に変換

    # X, Y 座標の作成
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")  # X, Y のグリッド
    
    # ヒストグラムの棒の底の位置
    xpos = X.ravel()
    ypos = Y.ravel()
    zpos = np.zeros_like(xpos)  # すべての棒の底を z=0 に

    # 棒のサイズ
    dx = dy = 0.5  # 各棒の幅
    dz = shifted_data.ravel()  # 高さをデータの値とする

    # 3Dヒストグラムの描画
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

    # Z軸の補正
    tick_positions = np.linspace(0, np.max(shifted_data), num=5)  # 目盛りの位置
    tick_labels = np.linspace(data_min, np.max(data), num=5)  # 元のデータの値に戻す
    ax.set_zticks(tick_positions)
    ax.set_zticklabels([f"{t:.1f}" for t in tick_labels])  # ラベルを見やすく

    ax.set_xlabel("pa axis")
    ax.set_ylabel("pe axis")
    ax.set_zlabel("Value")
    ax.set_title(f"Beam Power Channel Number={channel_num}, v={V}")

    plt.show()

##########################################################################################################
# チャネルの計算
# 各アンテナの原点においての複素振幅を計算
def calc_complex_Amp_at_O_u(f, U, N, M, Amp_per_career_desital, beta, tau_nm, b_varphi_eta, eta_rad, varphi_rad):
    num_freq = len(f)  # 周波数の数
    complex_Amp_at_O = np.zeros((N, max(M), num_freq), dtype=complex)  # 初期化

    for u in range(U):
        for n in range(N):
            for m in range(M[n]):
                # tau_nm の形状に応じて値を取得
                if np.isscalar(tau_nm[n]):
                    tau_value = tau_nm[n]
                elif len(np.shape(tau_nm[n])) == 1:
                    tau_value = tau_nm[n][m]
                else:
                    tau_value = tau_nm[n, m]
                
                # tau_value の妥当性チェック
                if not np.isfinite(tau_value):
                    raise ValueError(f"Invalid tau_value: {tau_value} at n={n}, m={m}")

                # 振幅計算（ベクトル化）
                complex_Amp_at_O[n, m, :] = (
                    Amp_per_career_desital[n][m]  
                    * np.exp(1j * beta[n][m]) 
                    * np.exp(-1j * 2 * np.pi * f * tau_value)  
                    * b_varphi_eta[n][m] 
                    * np.exp(-1j * np.pi * u * np.cos(eta_rad) * np.sin(varphi_rad))
                )

    # 無効な値のチェック（オプション）
    if not np.all(np.isfinite(complex_Amp_at_O)):
        raise ValueError("Complex amplitude contains NaN or inf values.")
    
    return complex_Amp_at_O

def noise_u_v_k(U, V):
    # (V, Q, Q, 2000) サイズの複素乱数を生成
    real_part = np.random.normal(0, 1.778 * 1e-6, (U,V, 2000))
    imag_part = np.random.normal(0, 1.778 * 1e-6, (U,V, 2000))
    n_u_k_v = real_part + 1j * imag_part
    return n_u_k_v

def noise_dash_dash(U):
    real_part = np.random.normal(0, 1.778 * 1e-6, (U))
    imag_part = np.random.normal(0, 1.778 * 1e-6, (U))
    n_dash_dash = real_part + 1j * imag_part
    return n_dash_dash

#########################################################################################################
def save_to_npy(file_name, data):
    # npyファイルに保存
    np.save(file_name, data)
    print(f"data have been saved to {file_name}")
