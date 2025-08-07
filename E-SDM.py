import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import Channel_functions as channel
import os


Pt_mW = 1000/2000
P_noise_mW = 6.31*1e-12
U=8
d=5
#####################################################################################################################
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
    Capacity = np.zeros((1000,12))
    eigvals_all = [[None for _ in range(12)] for _ in range(1000)]  # 固有値を保存するための2次元リスト
    L_used_list = []


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
        L_used_list.append(Ly)

    L_used_array = np.array(L_used_list)

    # ユニークなLとその出現数
    L_values, counts = np.unique(L_used_array, return_counts=True)

    # 累積分布の計算
    cdf = np.cumsum(counts) / len(L_used_array)

    # プロット
    plt.figure(figsize=(8, 5))
    plt.step(L_values, cdf, where='post', label='CDF of L_used')
    plt.xlabel('L_used (Number of layers)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Number of Layers Used in Water-Filling')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    return Capacity, eigvals_all

####################################################################################################################

# メインの処理
# 設定
channel_type = 'InH'
NF_setting = 'Far'
load_dir = f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Channel_Matrix/{NF_setting}/H_tru"

for d in range(5, 31, 5):  # d = 5, 10, ..., 30

    H_tru = np.load(f"{load_dir}/d={d}.npy", allow_pickle=True)
    H_est = np.load(f"{load_dir}/d={d}.npy", allow_pickle=True)
    cap_tru, eigval_tru = calc_channel_capacity(H_tru, H_tru)
    cap_est, eigval_est = calc_channel_capacity(H_est, H_tru)
    
    # 保存パス
    save_dir = f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Channel_Capacity/{NF_setting}"
    # ディレクトリがなければ作る
    os.makedirs(save_dir, exist_ok=True)

    # # Capacityの保存
    np.save(f"{save_dir}", cap_tru)
    np.save(f"{save_dir}", cap_est)

    # 固有値の保存（リストのままだとnp.saveできないのでobject指定）
    np.save(f"{save_dir}/eigvals_Htru_d={d}.npy", eigval_tru, allow_pickle=True)
    np.save(f"{save_dir}/eigvals_Hest_d={d}.npy", eigval_est, allow_pickle=True)
    print(f'{d} has done')