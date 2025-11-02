import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import Channel_functions as channel
import os

Pt_mW = 1000/2000
P_noise_mW = 6.31e-12
U=8

#####################################################################################################################
# グラム行列の固有値と固有ベクトルを求める関数
def calc_eigval(H):
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

    # ここで Ly と p_ratio を出す
    Ly = int(np.count_nonzero(P > tol))  # 有効レイヤ数
    if P.sum() > 0:
        p_ratio = P / P.sum()  # 割当比（合計1）
    else:
        p_ratio = P
    
    # 二分探索の最後に:
    alpha = (lo + hi) / 2.0
    mu = 1.0 / alpha
    return p_ratio[:Ly], Ly

def generate_s(Ly):
    # 複素ガウス乱数の生成（実部と虚部を分散0.5で生成）
    s = (np.random.randn(Ly) + 1j * np.random.randn(Ly)) / np.sqrt(2)
    return s

# チャネル容量を求める関数(注意!! nearとfarで用いるH_truが異なります！！！！)
def calc_channel_capacity(H, H_tru, Pt_mW, P_noise_mW, TorF):
    Capacity = np.zeros((1000,12))
    eigvals_all = [[None for _ in range(12)] for _ in range(1000)]  # 固有値を保存するための2次元リスト
    L_used = np.zeros((1000,12), dtype=int)  # 使用されたレイヤ数を保存する配列
    Ly_star = np.zeros(1000, dtype=int)  # 各チャネルで選択された最適レイヤ数を保存する配列
    for i in range(1000):
        for w in range(12):
            H_val , H_vec = calc_eigval(H[i,w])
            eigvals_all[i][w] = H_val  # 固有値を保存

            power_allo, Ly = water_filling_ratio(H_val, Pt_mW, P_noise_mW)
            p_sqrt = np.sqrt(power_allo)
            p_diag = np.diag(p_sqrt)

            A = np.sqrt(Pt_mW)*p_diag
            Te_H = H_vec[:,:Ly]

            n_dash_dash = channel.noise_dash_dash(U)
            s = generate_s(Ly)
            x = Te_H @ A @ s
            y = H_tru[i,w] @ x + n_dash_dash
            
            N_dash_dash_ly = np.random.normal(0, 1.778e-6, (U,Ly)) + 1j * np.random.normal(0, 1.778e-6, (U,Ly))
            S = np.eye(Ly)
            H_eff = H_tru[i,w] @ Te_H @ S + (N_dash_dash_ly / np.sqrt(Pt_mW))
            Wr = np.ones((U,Ly), dtype=np.complex128)
            gamma0 = Pt_mW / P_noise_mW
            I_Ly = np.identity(Ly)
            W_MMSE = ( np.linalg.inv(H_eff.conj().T @ H_eff + (Ly / gamma0) * I_Ly) @ H_eff.conj().T ).T
            
            Wr = W_MMSE
            r = Wr.T @ y
            B = Wr.T @ H_tru[i,w] @ Te_H @ A
            S = np.abs(np.diag(B)) ** 2
            I = np.sum(np.abs(B) ** 2, axis=1) - np.abs(np.diag(B)) ** 2

            Rnn = P_noise_mW * Wr.T @ (Wr.T).conj().T
            N = np.diag(Rnn)
            C_ly = np.log2(S / (I + N) + 1)
            C = np.real(np.sum(C_ly))
            Capacity[i,w] = C
            
            # L_used を収集
            L_used[i,w] = Ly
        
            w_star = np.argmax(Capacity[i,:])
            Ly_star[i] = L_used[i,w_star]

    print(w_star)
    print("Ly_star", Ly_star)
    # プロット
    min_L = int(Ly_star.min())
    max_L = int(Ly_star.max())
    bins  = np.arange(min_L - 0.5, max_L + 1.5, 1)  # 0.5オフセット

    plt.hist(Ly_star, bins=bins, color='b', alpha=0.6, rwidth=0.8)
    plt.xticks(range(1,13))  # 目盛りは整数に
    plt.xlabel('Used Layers')
    plt.ylabel('Frequency')
    plt.title(f'No. of Layers d={d} {NF_setting} {TorF}')
    plt.grid(alpha=0.3)
    plt.show()
    
    return Capacity, eigvals_all

####################################################################################################################

# メインの処理
# 設定
Method = 'Mirror'
channel_type = 'InH'
Ssub_lam = 10  # サブアレー間隔(単位: 波長)
NF_setting = 'Near'


d=5
load_dir = f"C:/Users/tai20/Downloads/sim_data/Data/{Method}/{channel_type}/Ssub={Ssub_lam}lam/Channel_Matrix/{NF_setting}"
H_tru = np.load(f"{load_dir}/H_tru/d={d}.npy", allow_pickle=True)
H_est = np.load(f"{load_dir}/H_est/d={d}.npy", allow_pickle=True)
# load_dir = f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/InH_CoherentSum/InH_Simple_CoherentSum/ChannelMatrix"
# H_tru = np.load(f"{load_dir}/Channel_InH_1000_step4_Htru_Near_d=5.npy", allow_pickle=True)
# H_est = np.load(f"{load_dir}/Channel_InH_1000_step4_Hmea_Near_d=5.npy", allow_pickle=True)
cap_tru, eigval_tru = calc_channel_capacity(H_tru, H_tru, Pt_mW, P_noise_mW, "True")
cap_est, eigval_est = calc_channel_capacity(H_est, H_tru, Pt_mW, P_noise_mW, "Est")



"""
# 保存パス
save_dir = f"C:/Users/tai20/Downloads/sim_data/Data/Mirror/{channel_type}/Ssub={Ssub_lam}lam/Channel_Capacity/{NF_setting}/d={d}"
os.makedirs(save_dir, exist_ok=True)

# # Capacityの保存
np.save(f"{save_dir}/Capacity_Htru.npy", cap_tru)
np.save(f"{save_dir}/Capacity_Hest.npy", cap_est)

# 固有値の保存（リストのままだとnp.saveできないのでobject指定）
np.save(f"{save_dir}/eigvals_Htru.npy", eigval_tru, allow_pickle=True)
np.save(f"{save_dir}/eigvals_Hest.npy", eigval_est, allow_pickle=True)
print(f'{NF_setting} : {d} has done')
"""
