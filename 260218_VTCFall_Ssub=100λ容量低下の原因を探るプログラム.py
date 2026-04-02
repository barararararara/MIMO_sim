# 260218_Ssub=100λ容量低下の原因を探るプログラム
import numpy as np
import math
import Channel_functions as channel
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (import for 3D)
from pathlib import Path

def mu_12_13_14(H):  # H: (U,4)
    H = np.asarray(H)
    norms = np.linalg.norm(H, axis=0) + 1e-12

    def mu(i, j):
        return float(
            np.abs(np.vdot(H[:, i], H[:, j])) / (norms[i] * norms[j])
        )

    mu12 = mu(0, 1)  # (1,2)
    mu13 = mu(0, 2)  # (1,3)
    mu14 = mu(0, 3)  # (1,4)
    return [mu12, mu13, mu14]
def plot_mu3_vs_Ssub(results, Ssub_list):
    S = np.array(sorted(Ssub_list), float)

    mu_mean = np.zeros((len(S), 3), float)  # 3本
    for i, s in enumerate(S):
        mu3_list = np.asarray(results[int(s)]["mu3"], float)  # shape:(Nd,3) ここではNd=1でもOK
        mu_mean[i, :] = mu3_list.mean(axis=0)

    fig, ax = plt.subplots(constrained_layout=True)
    for k in range(3):
        ax.plot(S, mu_mean[:, k], "o-", lw=2, label=f"μ(best, col{k+1})")
    ax.set_xlabel("Ssub [λ]")
    ax.set_ylabel("Mutual coherence μ")
    # ax.set_ylim(0.0, 1.01)
    ax.grid(True)
    ax.legend(ncol=3, fontsize=9)
    plt.show()

def mismatch_metrics_allV(subarray_v_qy_qz, ba, Q, MUE_coordinate, M, lam):
    V = subarray_v_qy_qz.shape[0]
    pts = _as_points_3d(MUE_coordinate, M)  # (K,3)
    k0 = 2*np.pi/lam

    w_DD = channel.DFT_weight_calc(Q)  # (Q,Q,Q^2?) ←あなたの実装に合わせて
    # ↑もし w_DD[pa,pe] が (Q,Q) で返る前提ならOK

    eta_list = []
    Gsel_list = []
    Gmax_list = []

    for v in range(V):
        face = v // 4
        aidx = v % 4
        pa_sel = int(ba[face, 0, aidx])
        pe_sel = int(ba[face, 1, aidx])
        if pa_sel < 0 or pe_sel < 0:
            continue

        xyz = subarray_v_qy_qz[v].reshape(-1,3)  # (Q^2,3)
        c = subarray_v_qy_qz[v].mean(axis=(0,1))
        vecs = pts - c[None,:]
        u = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)  # (K,3)

        phase = k0 * (xyz @ u.T)          # (Q^2,K)
        a = np.exp(1j*phase)
        a = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-12)        # (Q^2,K)

        # --- 選択ビーム利得 ---
        w_sel = w_DD[pa_sel, pe_sel].reshape(-1)
        w_sel = w_sel / (np.linalg.norm(w_sel) + 1e-12)
        G_sel = np.max(np.abs(np.conj(w_sel) @ a)**2)

        # --- 全ビーム探索での最大利得（上限） ---
        G_best = 0.0
        for pa in range(Q):
            for pe in range(Q):
                w = w_DD[pa, pe].reshape(-1)
                w = w / (np.linalg.norm(w) + 1e-12)
                g = np.max(np.abs(np.conj(w) @ a)**2)
                if g > G_best:
                    G_best = g

        eta = G_sel / (G_best + 1e-18)

        Gsel_list.append(float(G_sel))
        Gmax_list.append(float(G_best))
        eta_list.append(float(eta))

    return np.array(Gsel_list), np.array(Gmax_list), np.array(eta_list)


def mc_columns(H):
    # H: (U, V')
    Vp = H.shape[1]
    R = np.zeros((Vp, Vp), float)
    norms = np.linalg.norm(H, axis=0) + 1e-12
    for i in range(Vp):
        for j in range(Vp):
            R[i,j] = np.abs(np.vdot(H[:,i], H[:,j])) / (norms[i]*norms[j])
    off = R[~np.eye(Vp, dtype=bool)]
    return R, float(off.mean()), float(off.max())

def _as_points_3d(ue_xyz, M=None):
    """
    ue_xyz:
      - (3,)         1点
      - (K,3)        点群
      - (N,max_M,3)  Mirror UE点群（0埋め想定） -> M[n]で有効部だけ抽出
    """
    ue = np.asarray(ue_xyz)
    if ue.ndim == 1:
        return ue.reshape(1, 3)

    if ue.ndim == 2:
        return ue  # (K,3)

    if ue.ndim == 3:
        assert M is not None, "ue_xyz が (N,max_M,3) のときは M を渡して"
        xs, ys, zs = [], [], []
        for n in range(ue.shape[0]):
            m_len = M[n]
            xs.append(ue[n, :m_len, 0])
            ys.append(ue[n, :m_len, 1])
            zs.append(ue[n, :m_len, 2])
        return np.stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)], axis=1)

    raise ValueError("ue_xyz shape not supported")

def _pape_to_dirvec(pa: int, pe: int, Q: int, face: int):
    # (pa,pe) を [-pi/2, pi/2] に写像（あなたの既存角度マップに合わせた版）
    phi_local   = math.asin(2.0 * ((pa + 1) / Q) - 1.0)
    theta_local = math.asin(2.0 * ((pe + 1) / Q) - 1.0)

    # faceごとの回転（0,120,240 deg）
    zeta_face = [0.0, 2.0*math.pi/3.0, 4.0*math.pi/3.0][face]
    phi = phi_local + zeta_face
    theta = theta_local

    # unit vector
    return np.array([
        math.cos(theta) * math.cos(phi),
        math.cos(theta) * math.sin(phi),
        math.sin(theta),
    ], dtype=float)
def _pape_to_dirvec(pa: int, pe: int, Q: int, face: int):
    # (pa,pe) を [-pi/2, pi/2] に写像（あなたの既存角度マップに合わせた版）
    phi_local   = math.asin(2.0 * ((pa + 1) / Q) - 1.0)
    theta_local = math.asin(2.0 * ((pe + 1) / Q) - 1.0)

    # faceごとの回転（0,120,240 deg）
    zeta_face = [0.0, 2.0*math.pi/3.0, 4.0*math.pi/3.0][face]
    phi = phi_local + zeta_face
    theta = theta_local

    # unit vector
    return np.array([
        math.cos(theta) * math.cos(phi),
        math.cos(theta) * math.sin(phi),
        math.sin(theta),
    ], dtype=float)



def plot_bs_and_mainlobes_with_mue(
    subarray_v_qy_qz, ba, Q,
    ue_xyz, M=None,
    ray_len=100.0,          # ここを効かせたいなら固定にする
    draw_elements=False,
    linewidth=4,
    end_marker=True,
    view=(20, 45),
    debug=True,
):
    centers = subarray_v_qy_qz.mean(axis=(1, 2))  # (V,3)
    ue_pts  = _as_points_3d(ue_xyz, M=M)          # (K,3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # MUE
    if ue_pts.size:
        ax.scatter(ue_pts[:,0], ue_pts[:,1], ue_pts[:,2], s=20, alpha=0.7, label="MUE(s)")

    # BS
    if draw_elements:
        for v in range(subarray_v_qy_qz.shape[0]):
            xyz = subarray_v_qy_qz[v].reshape(-1, 3)
            ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=6)
    ax.scatter(centers[:,0], centers[:,1], centers[:,2], s=40, marker="o", label="Subarray centers")

    # ★軸範囲を「BS + UE」だけで決めて固定（ここが重要）
    bounds = [centers]
    if ue_pts.size:
        bounds.append(ue_pts)
    all_pts = np.vstack(bounds)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    mid  = (mins + maxs) / 2
    half = (maxs - mins).max() / 2 * 1.05

    ax.set_xlim(mid[0]-half, mid[0]+half)
    ax.set_ylim(mid[1]-half, mid[1]+half)
    ax.set_zlim(mid[2]-half, mid[2]+half)

    # 以降、レイを描いても軸が変わらないようにする
    ax.set_autoscale_on(False)

    # rays
    cnt = 0
    for face in range(3):
        for k in range(4):
            pa = int(ba[face, 0, k]); pe = int(ba[face, 1, k])
            if pa < 0 or pe < 0:
                continue
            dirvec = _pape_to_dirvec(pa, pe, Q, face)
            if not np.all(np.isfinite(dirvec)):
                continue

            for a in range(4):
                v = face*4 + a
                p0 = centers[v]
                p1 = p0 + ray_len * dirvec

                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=linewidth)
                if end_marker:
                    ax.scatter([p1[0]], [p1[1]], [p1[2]], s=70, marker="^")
                cnt += 1

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("BS subarrays + MUE(s) + mainlobe directions")
    ax.legend()
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])

    if debug:
        span = np.ptp(all_pts, axis=0).max()
        print("span(BS+UE) =", span, "ray_len =", ray_len, "ray_len/span =", ray_len/span, "lines =", cnt)

    plt.show()
    return fig, ax



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
def Beam_allocation_method_1(Power, threshold, testch_num=0):
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
                row = df_v.iloc[k] if k < len(df_v) else None
                if row is not None and pd.notnull(row["pa"]):
                    beam_allocation[face, 0, k] = row["pa"]
                    beam_allocation[face, 1, k] = row["pe"]
    
    return beam_allocation, rows_per_v

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

# ミューチュアルコヒーレンスを計算する関数二つ
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

def calc_channel_capacity_SingleLayer(H, H_tru, Pt_mW, P_noise_mW):
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
    C = float(np.real(np.sum(C_ly)))

    return C, Ly, H_val

def beam_gain_Gv_all(
    subarray_v_qy_qz,     # (V,Q,Q,3)
    MUE_coordinates, M,   # (N,max_M,3), list M
    pa, pe,              # 割当ビーム index
    lam,
    v_ids=None,          # 例: [0,1,2,3]（その面の4サブアレー）
    use_center=True,     # True: サブアレー中心から方向 u を作る
    ):
    """
    return:
      G: (len(v_ids), K)  各サブアレーv×各MUE点のビーム利得
      pts: (K,3)
    """
    V, Q, _, _ = subarray_v_qy_qz.shape
    if v_ids is None:
        v_ids = list(range(V))

    pts = _as_points_3d(MUE_coordinates, M)   # (K,3)
    K = pts.shape[0]

    # DFT weight (Q,Q) -> (Q^2,)
    w_DD = channel.DFT_weight_calc(Q)
    w = w_DD[int(pa), int(pe)].reshape(-1)
    w = w / (np.linalg.norm(w) + 1e-12)

    k0 = 2*np.pi/lam
    G = np.zeros((len(v_ids), K), dtype=float)

    for iv, v in enumerate(v_ids):
        xyz = subarray_v_qy_qz[v].reshape(-1, 3)  # (Q^2,3)
        c = subarray_v_qy_qz[v].mean(axis=(0,1))  # center (3,)

        # 各MUEへの方向 u（単位ベクトル）
        ref = c if use_center else xyz.mean(axis=0)
        vecs = pts - ref[None,:]                  # (K,3)
        u = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

        # steering a(u) = exp(j k r·u)
        # phase: (Q^2,K) = xyz @ u^T
        phase = k0 * (xyz @ u.T)                  # (Q^2,K)
        a = np.exp(1j * phase)                    # (Q^2,K)
        a = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-12)

        # G = |w^H a|^2
        wha = np.conj(w) @ a                      # (K,)
        G[iv, :] = np.abs(wha)**2

    return G, pts
##################################################################################################################
# 直接波のみ、正面からの設定を作るための関数
def setting_Direct_synario(channel_type, d):
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
    varphi_deg[0][0] = 180
    
    Direct_Syanario_Data = {
        'chi': chi,
        'N': N,
        'M': M,
        'Z': Z,
        'U_nm': U_nm,
        'rho': rho,
        'tau': tau,
        'beta': beta,
        'phi_deg': phi_deg,
        'theta_deg': theta_deg,
        'varphi_deg': varphi_deg,
        'eta_deg': eta_deg
    }
    
    return Direct_Syanario_Data

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
    # print(f"Rayleigh_distance: {Rayleigh_distance} m, D_subarray: {D_subarray} m")
    
    # n番目のTCのm番目のUE鏡像体#0と，v番目のサブアレーのqy, qz番目のアンテナ素子間の距離をrm,n,v, qy, qz 式14
    r_mnv0qyqz = channel.distance_to_eachanntena(MUE_coordinate, subarray_v_qy_qz, N, M, V, Q)
    # UEの#0から各アンテナ素子への伝搬時間 ㉓
    tau_mnv0qyqz = r_mnv0qyqz / c_ns  # (N, max(M), V, Q, Q)
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
    
    a_MUE_mn0k = Amp_desital * np.exp(1j * beta) * b_varphi_eta
    
    a_mn0vkqyqz = np.zeros((N, max(M), V, Q, Q), dtype=np.complex64)
    # f_GHz_val: 周波数[GHz]
    lam = 0.3 / f_GHz_val

    # tau_mnv0qyqz は (N, max(M), V, Q, Q) で与えられている
    exp_term = np.exp(-2j * np.pi * f_GHz_val * tau_mnv0qyqz)  # shape: (N, max(M), V, Q, Q)

    # 各アンテナ素子においての複素振幅　式47
    a_mn0vkqyqz = np.einsum('nm,vnm,vnmyz->nmvyz', a_MUE_mn0k, a_phi_theta, exp_term, optimize=True)

    c = np.cos(eta_rad_v) * np.sin(varphi_rad_v)

    # 1) c.shape == (N, M, V) ならそのまま
    if c.shape[-1] != a_mn0vkqyqz.shape[2]:
        # 2) c.shape == (V, N, M) みたいな場合 → (N, M, V) に移す
        #    (先頭がVだと仮定して moveaxis → 軸順を整える)
        c = np.moveaxis(c, 0, -1)  # (N, M, V) になる想定
    
    # 位相 (N, M, U, V, 1, 1)
    phase = np.exp(
        -1j * np.pi 
        * u[None, None, :, None, None, None]     # (1,1,U,1,1,1)
        * c[:, :, None, :, None, None]           # (N,M,1,V,1,1)
    )

    # a_mn0vkqyqz を (N, M, 1, V, Q, Q) にして掛ける → (N, M, U, V, Q, Q)
    a_mnuvkqyqz = a_mn0vkqyqz[:, :, None, :, :, :] * phase
    # すべてのマルチパスについて和を取る　式48  
    a_uvkqyqz = np.sum(a_mnuvkqyqz, (0,1))
    

    # 以下ビーム割当###############################################################
    w_DD_pape = np.zeros((V, Q, Q), dtype=np.complex64)
    invalid_indices = []  # vの削除対象インデックスを格納するリスト

    # 各サブアレーの最大受信電力を優先してビーム割当
    threshold = -73 # ビーム割当の閾値
    ba, row = Beam_allocation_method_2(P_sub_dash_dBm, threshold, testch_num=0)
    
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
    a_red = a_uvkqyqz[:,valid,:,:]
    n_dash_uv_full = channel.noise_dash(U, V)
    n_dash_uv_vdash = n_dash_uv_full[:, valid, :]

    # 近傍界チャネル計算
    # チャネル行列を計算　式49
    h_uvk = np.einsum('vyz,uvyz->uv', w_DD_pape_red, a_red, optimize=True)  # (V′)
    
    Pu_mW_per_carrer = Pu_mW / 2000  # 1キャリアあたりの送信電力(mW)
    
    if use_H == 'T':
        h_w_est = h_uvk
    elif use_H == 'E_w':
        h_w_est = h_uvk + n_dash_uv_vdash[:, :, :10].mean(axis=2) / np.sqrt(Pu_mW_per_carrer)
    elif use_H == 'E_wo':
        h_w_est = h_uvk + n_dash_uv_vdash[:, :, 0] / np.sqrt(Pu_mW_per_carrer)
    
    eig,_ = calc_eigval(h_w_est)
    # print(f"[DEBUG] d={d}, Ssub={Ssub_lam}, validV={len(valid)}, eig[:4]={eig[:4]}")
    # print(f"[DEBUG] Beam Allocation:\n{ba}")
    # print("MUE coordinate", MUE_coordinate)
    # subarray_v_qy_qz を既に作ってる前提（あなたのコード内のそれ）
    # ba は Beam_allocation_method_2 の出力（shape=(3,2,4)）【:contentReference[oaicite:4]{index=4}】
    # --- ★相性メトリクス：baで選ばれたビームを各サブアレー(v=0..3)で評価する ---
    v_ids = [0, 1, 2, 3]
    pts = _as_points_3d(MUE_coordinate, M)  # (K,3) ※点群

    w_DD = channel.DFT_weight_calc(Q)

    eta_vmax = []
    for v in v_ids:
        face = v // 4
        aidx = v % 4
        pa_sel = int(ba[face, 0, aidx])
        pe_sel = int(ba[face, 1, aidx])
        if pa_sel < 0 or pe_sel < 0:
            eta_vmax.append(0.0)
            continue

        # vごとの選択ビームで利得を計算
        G_v, _ = beam_gain_Gv_all(
            subarray_v_qy_qz,
            MUE_coordinate, M,
            pa_sel, pe_sel,
            lam,
            v_ids=[v],
        )  # shape (1,K)
        eta_vmax.append(float(G_v.max()))  # そのサブアレーで点群中最大利得

    eta_vmax = np.array(eta_vmax, float)
    eta_min  = float(eta_vmax.min())
    eta_std  = float(eta_vmax.std())
    eta_mean = float(eta_vmax.mean())


    # print(f"[METRIC] eta_vmax(4sub)={eta_vmax}")
    # print(f"[METRIC] eta_mean={eta_mean:.3e}, eta_std={eta_std:.3e}, eta_min={eta_min:.3e}ba
    # ")
    # print(f"[METRIC] eig2={eig[1]:.3e}, eig2/eig1={eig[1]/(eig[0]+1e-18):.3e}")
    
    # MC_H, mc_mean, mc_max = mc_columns(h_uvk)  # or h_uvk
    # print(f"[METRIC] MC(H) offdiag mean={mc_mean:.3f}, max={mc_max:.3f}")
    # # print("[METRIC] MC(H)=\n", MC_H)
    
    # print("azimuth angle of MUE from each subarray",phi_deg_v)
    
    Gsel, Gmax, eta = mismatch_metrics_allV(subarray_v_qy_qz, ba, Q, MUE_coordinate, M, lam)
    print(f"[METRIC] mismatch eta: mean={eta.mean():.3f}, min={eta.min():.3f}, std={eta.std():.3f}")
    print(f"[METRIC] Gsel/Gmax example (first5): {eta[:5]}")



    eig, _ = calc_eigval(h_uvk)
    eig_ratio = float(eig[1] / (eig[0] + 1e-18))  # eig2/eig1

    Gsel, Gmax, eta = mismatch_metrics_allV(subarray_v_qy_qz, ba, Q, MUE_coordinate, M, lam)
    H_for_mc = h_uvk
    _, mc_mean, mc_max = mc_columns(H_for_mc)

    # 0.000表示回避
    print(f"[METRIC] mismatch eta: mean={eta.mean():.3e}, min={eta.min():.3e}, std={eta.std():.3e}")
    print(f"[METRIC] eig2/eig1={eig_ratio:.3e}")
    lam2 = float(eig[1])  # 第2固有値
    lam1 = float(eig[0])  # ついでに
    
    mu3 = mu_12_13_14(h_uvk)

    return h_uvk, h_w_est, ba, eta, eig_ratio, mc_max, lam1, lam2, mu3




def sweep_capacity_vs_d_mc(
    channel_indices,
    Synario,
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
            "eta_all": [],        # (4,)をappend
            "eta_mean": [],
            "eta_min": [],
            "eig2eig1": [],
            "mc_max": [],
            "lam2": [],
            "lam1": [],
            "mu3": [],
        }

        for d in d_values:
            C_list = []
            C_single_list = []
            Ly_list = []
            
            
            if Synario == "Direct":
                np.random.seed(base_seed)

                Synario_Data = setting_Direct_synario(channel_type, d)
                H, H_use, ba = simulation_core(
                    channel_type, Q, lam, d, Pu_dBm, Ssub_lam,
                    Synario_Data=Synario_Data, use_H=use_H
                )
                

                C, Ly, _ = calc_channel_capacity(H_use, H, Pt_mW, P_noise_mW)
                C_single, _, _ = calc_channel_capacity_SingleLayer(H_use, H, Pt_mW, P_noise_mW)

                C_list.append(C)
                C_single_list.append(C_single)
                Ly_list.append(Ly)

            else:  # NYUSIM
                for testch_num in channel_indices:
                    np.random.seed(base_seed + testch_num)

                    Synario_Data = setting_NYUSIM_synario(Base_data[testch_num], d)
                    H, H_use, ba, eta, eig_ratio, mc_max, lam1, lam2, mu3 = simulation_core(
                        channel_type, Q, lam, d, Pu_dBm,Ssub_lam,
                        Synario_Data, use_H=use_H
                    )
                    
                    G = H.conj().T @ H
                    MC = mutual_coherence_matrix_from_Gram(G)
                    # print("MC =",MC)
                    

                    C, Ly, _ = calc_channel_capacity(H_use, H, Pt_mW, P_noise_mW)
                    C_single, _, _ = calc_channel_capacity_SingleLayer(H_use, H, Pt_mW, P_noise_mW)
                    
                    print(f"Ssub={Ssub_lam}lam", C)

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
            results[Ssub_lam]["eta_all"].append(eta.copy())
            results[Ssub_lam]["eta_mean"].append(float(np.mean(eta)))
            results[Ssub_lam]["eta_min"].append(float(np.min(eta)))
            results[Ssub_lam]["eig2eig1"].append(float(eig_ratio))
            results[Ssub_lam]["mc_max"].append(float(mc_max))
            results[Ssub_lam]["lam2"].append(lam2)
            results[Ssub_lam]["lam1"].append(lam1)  # optional
            results[Ssub_lam]["mu3"].append(mu3)



        # print(f"Ssub={Ssub_lam}lam done.")

    plot_capacity(results, Ssub_list, return_std=return_std, MC=MC, use_H=use_H, save_folder=save_folder)
    plot_layers(results, Ssub_list, use_H=use_H, save_folder=save_folder)
    return results

def plot_mu_vs_Ssub(results, Ssub_list):
    S = np.array(sorted(Ssub_list), float)
    y = np.array([np.mean(results[int(s)]["mc_max"]) for s in S], float)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(S, y, "d-.", lw=2)
    ax.set_xlabel("Ssub [λ]")
    ax.set_ylabel("μ_max (column correlation)")
    ax.set_ylim(0.97, 1.001)  # 必要なら調整
    ax.grid(True)
    plt.show()


def plot_capacity(results, Ssub_list, return_std=False, MC=1, use_H="T", save_folder=None):
    fig, ax = plt.subplots(constrained_layout=True)

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

        ax.plot(d, Cm, marker="o", lw=2.4, color=color, label=f"{Ssub_lam}λ Multi")
        ax.plot(d, C1, marker="s", ls="--", lw=2.4, color=color, label=f"{Ssub_lam}λ Single")

        if return_std and MC >= 2:
            ax.fill_between(d, Cm-Cs, Cm+Cs, color=color, alpha=0.2)

    ax.set_xlabel("BS-UE Distance [m]", size=12)
    ax.set_ylabel("Channel Capacity [bps/Hz]", size=12)
    ax.set_ylim(0, 60)
    ax.tick_params(labelsize=11)
    ax.grid(True)
    ax.legend(title="Ssub Layer",fontsize=11)

    ax.set_title("")

    if save_folder is not None:
        channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall"), folder=save_folder, variants=("Paper",)) #←　カンマ必須！！！
        
        ax.set_title(title_str, size=15)
        channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall"), folder=save_folder, variants=("Slide",)) #←　カンマ必須！！！

    plt.show()
    plt.close(fig)
    
def plot_layers(results, Ssub_list, use_H="T", save_folder=None):
    fig, ax = plt.subplots(constrained_layout=True)

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

        ax.plot(d, Ly, lw=2.5, color=color, zorder=1)
        ax.scatter(
            d, Ly,
            marker="o",
            s=90,
            facecolors="white",
            edgecolors=color,
            linewidths=2.5,
            zorder=2,
            label=f"{Ssub_lam}λ"
        )


        ax.set_xlabel("BS-UE Distance [m]", size=12)
        ax.set_ylabel("Number of Layers", size=12)
        ax.set_ylim(0, 8.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)
        ax.tick_params(labelsize=11)
        ax.legend(title="Ssub", fontsize=11)

    ax.set_title("")

    if save_folder is not None:
        channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall"), folder=save_folder, variants=("Paper",)) #←　カンマ必須！！！
        
        ax.set_title(title_str, size=15)
        channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall"), folder=save_folder, variants=("Slide",)) #←　カンマ必須！！！

    plt.show()
    plt.close(fig)

def plot_eig_ratio_vs_Ssub(results, Ssub_list):
    S = np.array(sorted(Ssub_list), float)
    y = np.array([np.mean(results[int(s)]["eig2eig1"]) for s in S], float)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(S, y, "o-", lw=2)
    ax.set_xlabel("Ssub [λ]")
    ax.set_ylabel("eig2/eig1")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    plt.show()

def plot_eta_vs_Ssub(results, Ssub_list, which="min"):
    key = "eta_min" if which=="min" else "eta_mean"
    S = np.array(sorted(Ssub_list), float)
    y = np.array([np.mean(results[int(s)][key]) for s in S], float)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(S, y, "s--", lw=2)
    ax.set_xlabel("Ssub [λ]")
    ax.set_ylabel(f"η_{which}")
    ax.grid(True)
    plt.show()

def plot_lam2_vs_Ssub(results, Ssub_list):
    S = np.array(sorted(Ssub_list), float)
    y = np.array([np.mean(results[int(s)]["lam2"]) for s in S], float)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(S, y, "o-", lw=2)
    ax.set_xlabel("Ssub [λ]")
    ax.set_ylabel("2nd eigenvalue λ2")
    ax.set_yscale("log")  # 桁が広いはずなのでlog推奨
    ax.grid(True, which="both")
    plt.show()

def plot_mu3_vs_Ssub(results, Ssub_list):
    S = np.array(sorted(Ssub_list), float)

    mu_mean = np.zeros((len(S), 3), float)  # 3本
    for i, s in enumerate(S):
        mu3_list = np.asarray(results[int(s)]["mu3"], float)  # shape:(Nd,3) ここではNd=1でもOK
        mu_mean[i, :] = mu3_list.mean(axis=0)

    fig, ax = plt.subplots(constrained_layout=True)
    for k in range(3):
        ax.plot(S, mu_mean[:, k], "o-", lw=2, label=f"μ(best, col{k+1})")
    ax.set_xlabel("Ssub [λ]")
    ax.set_ylabel("Mutual coherence μ")
    ax.set_ylim(0.0, 1.01)
    ax.grid(True)
    ax.legend(ncol=3, fontsize=9)
    plt.show()

def plot_mu3_vs_Ssub(results, Ssub_list):
    S = np.array(sorted(Ssub_list), float)
    mu_mean = np.zeros((len(S), 3), float)

    for i, s in enumerate(S):
        arr = np.asarray(results[int(s)]["mu3"], float)  # (Nd,3)
        mu_mean[i, :] = arr.mean(axis=0)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(S, mu_mean[:,0], "o-", lw=2, label="μ(1,2)")
    ax.plot(S, mu_mean[:,1], "s--", lw=2, label="μ(1,3)")
    ax.plot(S, mu_mean[:,2], "d-.", lw=2, label="μ(1,4)")
    ax.set_xlabel("Ssub [λ]")
    ax.set_ylabel("Mutual coherence μ")
    ax.grid(True)
    ax.legend(ncol=3)
    plt.show()
    
def total_paths_from_M(M):
    # M は list / np.ndarray / object配列 どれでも来うる想定
    return int(np.sum(np.asarray(M, dtype=int)))

def pick_typical_channel_by_capacity_totalpaths(
    Base_data,
    channel_type, Q, lam,
    d_pick=5, Ssub_lam=0,
    Pu_dBm=30,
    Pt_mW=1000/2000, P_noise_mW=6.31e-12,
    use_H="T",
    total_paths_target=2,
    base_seed=9,
):
    idx_list, C_list = [], []

    for idx in range(len(Base_data)):
        M = Base_data[idx]["M"]
        total_paths = total_paths_from_M(M)

        if total_paths != int(total_paths_target):
            continue

        np.random.seed(base_seed + idx)
        Synario_Data = setting_NYUSIM_synario(Base_data[idx], d_pick)
        H, H_use, ba = simulation_core(
            channel_type, Q, lam, d_pick, Pu_dBm, Ssub_lam,
            Synario_Data, use_H=use_H
        )
        C, Ly, _ = calc_channel_capacity(H_use, H, Pt_mW, P_noise_mW)

        idx_list.append(idx)
        C_list.append(float(C))
        print(f"[check] idx={idx}, total_paths={total_paths}")

    if not idx_list:
        raise ValueError(f"総パス数 ΣM = {total_paths_target} のチャネルが見つからない")

    C_arr = np.asarray(C_list, float)
    C_med = float(np.median(C_arr))
    pick_pos = int(np.argmin(np.abs(C_arr - C_med)))
    picked_idx = int(idx_list[pick_pos])

    print(f"[pick] total_paths(ΣM)={total_paths_target} count={len(idx_list)}")
    print(f"[pick] d={d_pick}m, Ssub={Ssub_lam}λ, use_H={use_H}")
    print(f"[pick] median C={C_med:.3f}, picked idx={picked_idx}, C={C_arr[pick_pos]:.3f}")

    return picked_idx, C_med, idx_list, C_list

###############################################################
channel_type = "InH"
NF_setting = "Near"
Method = "Mirror"

MC = 1             # モンテカルロ回数

Synario = "NYUSIM"  # "Direct" or "NYUSIM"
# channel_indices = range(1, 2, 1)  # NYUSIMチャネルの場合のインデックスリスト(範囲取って平均)
channel_indices = [391]  # NYUSIMチャネルの場合のインデックスリスト(個別指定)
use_H = "T" # 'T' : 真のチャネル行列 , 'E_w' : 推定&同相加算　''E_wo' : 推定&非同相加算

# パイロット信号送信電力パラメータ
Pu_dBm = 30  # UEの送信電力(dBm)※全サブキャリア

# False : グラフを保存しない, フォルダ名 : 保存するフォルダ名
# save_folder =  f"Channel_{channel_indices[0]}/Pu={Pu_dBm}dBm" 
save_folder = None
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
d_values = list(range(5, 6, 5))
Ssub_list = list(range(1,120,1))#[0, 50, 100]


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
    Synario=Synario,
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

# plot_eig_ratio_vs_Ssub(results, Ssub_list)
plot_eta_vs_Ssub(results, Ssub_list)
plot_mu_vs_Ssub(results, Ssub_list)
plot_lam2_vs_Ssub(results, Ssub_list)
plot_mu3_vs_Ssub(results, Ssub_list)
