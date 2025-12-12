import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import Channel_functions as channel
import os

Pt_mW = 1000/2000
P_noise_mW = 6.31e-12
U=8
def pick_best_eigs(capacity, eigvals_all, max_modes=8):
    """
    capacity: (I, W)
    eigvals_all: [[eigvals_w0, eigvals_w1, ...] for i in I]
                 各要素は 1D ndarray（長さは可変）
    返り値: (I, max_modes)  上位 max_modes の固有値（足りない分は0でパディング）
    """
    I, W = capacity.shape
    out = np.zeros((I, max_modes), dtype=float)
    for i in range(I):
        w_star = int(np.argmax(capacity[i]))
        vals = np.asarray(eigvals_all[i][w_star], dtype=float)  # (L_i,)
        if vals.size == 0:
            continue
        k = min(max_modes, vals.size)
        out[i, :k] = vals[:k]  # calc_eigvalで降順になってる前提
    return out

# 濱田さんの修論図3.8を描画する関数
def plot_multi_eigenvalue_cdf(eigval_tru, eigval_est=None, max_modes=8, eps=1e-30,
                            title='Eigenvalue CDF (Top modes)',
                            xlabel='Eigenvalue [dB]'):
    """
    eigval_tru, eigval_est : shape = (num_cases, num_eigs)
        各ケースごとの固有値配列（降順でも昇順でもOK）
    max_modes : 図に描く固有値順位の上限（例: 6なら1st〜6th）
    """
    plt.figure(figsize=(7, 5))

    num_modes = min(max_modes, eigval_tru.shape[1])
    colors_true = plt.cm.Reds(np.linspace(0.5, 1, num_modes))
    colors_est  = plt.cm.Blues(np.linspace(0.4, 0.9, num_modes))

    for j in range(num_modes):
        # ---- True channel ----
        vals_true = np.clip(eigval_tru[:, j], eps, None)
        x_true = 10 * np.log10(np.sort(vals_true))
        y_true = np.linspace(0, 1, len(x_true))
        lw = 2.2 if j == 0 else 1.3
        plt.plot(x_true, y_true, color=colors_true[j],
                lw=lw, marker='o', markersize='10', markevery=max(1, len(x_true)//10), 
                label='Actual channels' if j==0 else None)

        # ---- Estimated channel ----
        if eigval_est is not None:
            vals_est = np.clip(eigval_est[:, j], eps, None)
            x_est = 10 * np.log10(np.sort(vals_est))
            y_est = np.linspace(0, 1, len(x_est))
            plt.plot(x_est, y_est, color=colors_est[j],
                    lw=lw, ls='--',
                    marker='^', markersize='10',
                    markevery=max(1, len(x_est)//10),
                    label='Estimated channels' if j==0 else None)

    # 軸ラベルなど整える
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Cumulative distribution', fontsize=12)
    plt.title(title, fontsize=13.5)
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend(loc='lower right', framealpha=0.95)
    plt.xlim(-160, 0)   # 固有値[dB]の範囲。必要なら調整可
    plt.ylim(0, 1.0)
    plt.tight_layout()
    channel.save_current_fig_pdf(title)
    plt.show()

# 濱田さんの修論図3.9を描画する関数
def to_matrix(p_alloc_used, max_layers=None, normalize=True, eps=1e-12):
    num_cases = len(p_alloc_used)
    if max_layers is None:
        max_layers = max((len(p) for p in p_alloc_used if p is not None), default=0)

    M = np.zeros((num_cases, max_layers), dtype=float)
    for i, p in enumerate(p_alloc_used):
        if p is None:
            continue
        p = np.asarray(p, dtype=float)
        if normalize and p.sum() > eps:
            p = p / p.sum()
        L = min(len(p), max_layers)
        M[i, :L] = p[:L]
    # 数値誤差での微小値を0に
    M[np.abs(M) < eps] = 0.0
    return M

def sort_by_active_layers(M, eps=1e-12, secondary="first"):
    """
    有効レイヤ数（>eps）昇順でソート。
    secondary: 同数内の並び
      - 'first' : 1st layer の比率が大きい順（単峰→均等の順に）
      - 'entropy': エントロピー昇順（尖った→均等の順に）
      - None : そのまま
    """
    active = (M > eps).sum(axis=1)

    if secondary == "first":
        key2 = -M[:, 0]                 # 1stが大きいほど先
    elif secondary == "entropy":
        P = M.copy()
        P[P <= eps] = eps
        key2 = (P * np.log(P)).sum(axis=1)  # 実質エントロピーの符号反転
    else:
        key2 = np.zeros(len(active))

    order = np.lexsort((key2, active))  # まずactive昇順、次にkey2
    return M[order], order, active[order]

def plot_layer_ratio_lines(M, title=None, ylim=(0,1)):
    num_cases, max_layers = M.shape
    x = np.arange(1, num_cases+1)

    plt.figure(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, min(max_layers, 10)))
    for j in range(max_layers):
        c = colors[j % len(colors)]
        label = f'{j+1}st Layer' if j==0 else f'{j+1}th Layer'
        plt.plot(x, M[:, j], lw=1.0, label=label, color=c)

    plt.xlim(1, num_cases)
    if ylim: plt.ylim(*ylim)
    plt.xlabel('Case numbers', fontsize=12)
    plt.ylabel('Power allocation ratio', fontsize=12)
    if title: plt.title(title, fontsize=15)
    plt.grid(True, ls='--', alpha=0.4)
    plt.legend(ncol=3, fontsize=10, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    channel.save_current_fig_pdf(title)
    plt.show()

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
def calc_channel_capacity(H, H_tru, Pt_mW, P_noise_mW, TorE):
    Capacity = np.zeros((1000,12))
    eigvals_all = [[None for _ in range(12)] for _ in range(1000)]  # 固有値を保存するための2次元リスト
    p_alloc_all = [[None for _ in range(12)] for _ in range(1000)] # 各チャネルでの注水結果を保存するための2次元リスト
    L_used = np.zeros((1000,12), dtype=int)  # 使用されたレイヤ数を保存する配列
    Ly_star = np.zeros(1000, dtype=int)  # 各チャネルで選択された最適レイヤ数を保存する配列
    p_alloc_used = np.zeros(1000, dtype=object)  # 各チャネルでの注水結果を保存する配列
    for i in range(1000):
        for w in range(12):
            H_val , H_vec = calc_eigval(H[i][w])
            eigvals_all[i][w] = H_val  # 固有値を保存

            power_allo, Ly = water_filling_ratio(H_val, Pt_mW, P_noise_mW)
            p_alloc_all[i][w] = power_allo
            L_used[i,w] = Ly
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
            
        
        w_star = np.argmax(Capacity[i,:])
        Ly_star[i] = L_used[i,w_star]
        p_alloc_used[i] = p_alloc_all[i][w_star]

    # print("Ly_star", Ly_star)
    # プロット
    min_L = int(Ly_star.min())
    max_L = int(Ly_star.max())
    bins  = np.arange(min_L - 0.5, max_L + 1.5, 1)  # 0.5オフセット

    plt.figure(figsize=(7,5))
    plt.hist(Ly_star, bins=bins, color='b', alpha=0.6, rwidth=0.8)
    plt.xticks(np.arange(1, 12, 1))  # 目盛りは整数に
    plt.xlabel('Used Layers')
    plt.ylabel('Frequency')
    plt.title(f'No. of Layers d={d}m {NF_setting} {TorE}')
    plt.grid(alpha=0.3)
    # channel.save_current_fig_pdf(f'No_of_Layers_d={d}m_{NF_setting}_{TorE}')
    # plt.show()
    
    # 図3.9の描画部分
    M = to_matrix(p_alloc_used, normalize=True) # (1000, Lmax)
    M_sorted, order, L_active_sorted = sort_by_active_layers(M, secondary='first')

    # 可視化（“レイヤ数が少ない順”に並び替えてプロット）
    # plot_layer_ratio_lines(M_sorted, title=f'Power allocation ratio d={d}m {NF_setting} {TorE} (sorted)', ylim=(0,0.4) if TorE =="Est" else (0,1.0))
    
    return Capacity, eigvals_all

####################################################################################################################

# メインの処理
# 設定
Method = 'Mirror'
channel_type = 'InH'
Ssub_lam = 0  # サブアレー間隔(単位: 波長)
NS = 'W_NS'
NF_setting = 'Far'

for d in range(5, 31, 5):
    load_dir = f"C:/Users/tai20/Downloads/sim_data/Data/{Method}/{channel_type}/Ssub={Ssub_lam}lam/{NS}/Channel_Matrix/{NF_setting}"
    Beam_allocation = np.load(f"C:/Users/tai20/Downloads/sim_data/Data/{Method}/{channel_type}/Ssub={Ssub_lam}lam/{NS}/Beamallocation/{NF_setting}/d={d}.npy", allow_pickle=True)
    H_tru = np.load(f"{load_dir}/H_tru/d={d}.npy", allow_pickle=True)
    H_est = np.load(f"{load_dir}/H_est/d={d}.npy", allow_pickle=True)
    cap_tru, eigval_tru = calc_channel_capacity(H_tru, H_tru, Pt_mW, P_noise_mW, "True")
    cap_est, eigval_est = calc_channel_capacity(H_est, H_tru, Pt_mW, P_noise_mW, "Est")
    


    MAX_MODES_TO_SHOW = 8

    eig_true = pick_best_eigs(cap_tru, eigval_tru, max_modes=MAX_MODES_TO_SHOW)  # (I, max_modes)
    eig_est  = pick_best_eigs(cap_est,  eigval_est,  max_modes=MAX_MODES_TO_SHOW)

    # plot_multi_eigenvalue_cdf(eig_true, eig_est, title=f'Eigenvalue CDF Ssub={Ssub_lam}λ d={d}m {NF_setting}', xlabel='Eigenvalue [dB]')

    # 保存パス
    save_dir = f"C:/Users/tai20/Downloads/sim_data/Data/Mirror/{channel_type}/Ssub={Ssub_lam}lam/{NS}/Channel_Capacity/{NF_setting}/d={d}"
    os.makedirs(save_dir, exist_ok=True)

    # # Capacityの保存
    np.save(f"{save_dir}/Capacity_Htru.npy", cap_tru)
    np.save(f"{save_dir}/Capacity_Hest.npy", cap_est)

    # 固有値の保存（リストのままだとnp.saveできないのでobject指定）
    np.save(f"{save_dir}/eigvals_Htru.npy", np.array(eigval_tru, dtype=object), allow_pickle=True)
    np.save(f"{save_dir}/eigvals_Hest.npy", np.array(eigval_est, dtype=object), allow_pickle=True)
    print(f'{NF_setting} : {d} has done')
