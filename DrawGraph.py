import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import Channel_functions as channel
# NYUSIMチャネルモデルのマルチパス数を描画する関数
import pickle
from pathlib import Path
import matplotlib.ticker as ticker

# with open("C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall/Paper/Channel_391/Pu=10dBm/Fig_pickle/260212_Channel_Capacity_vs_BS-UE_Distance_(True_Channel)_1641.fig.pickle", "rb") as f:
#     fig = pickle.load(f)

# ax = fig.axes[0]

# for line in ax.get_lines():

#     label = line.get_label()

#     # --- マーカー設定 ---
#     if "Multi" in label:
#         # Multi は中塗り
#         line.set_markerfacecolor(line.get_color())
#     else:
#         # Single は中抜き
#         line.set_markerfacecolor("none")

#     line.set_markersize(7)
#     line.set_markeredgewidth(1.0)

#     # --- 線幅 ---
#     if line.get_linestyle() == "--":
#         line.set_linewidth(2.5)
#     else:
#         line.set_linewidth(2.0)

# # --- 凡例も同様に修正 ---
# leg = ax.get_legend()

# for h in leg.legend_handles:
#     label = h.get_label()

#     if "Multi" in label:
#         h.set_markerfacecolor(h.get_color())
#     else:
#         h.set_markerfacecolor("none")

#     h.set_markeredgewidth(1.0)
#     h.set_markersize(8)

#     if h.get_linestyle() == "--":
#         h.set_linewidth(2.5)
#     else:
#         h.set_linewidth(2.0)


# title_str = "forVTC_Capacity_図7"
# channel.save_current_fig(title_str, root=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures"), folder="", variants=("Paper",))

# plt.show()

# exit()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _get_M_list(one_channel, key_M="M"):
    """Return 1D int array of M (per-cluster multipath counts)."""
    if isinstance(one_channel, dict):
        if key_M not in one_channel:
            raise KeyError(f"key '{key_M}' not found. keys={list(one_channel.keys())}")
        M = one_channel[key_M]
        if np.isscalar(M):
            return np.array([int(M)], dtype=int)
        return np.array(M, dtype=int)

    # one_channel itself is M
    if np.isscalar(one_channel):
        return np.array([int(one_channel)], dtype=int)
    return np.array(one_channel, dtype=int)


def plot_multipath_hist_cdf_InH(
    npy_path,
    key_M="M",
    title=None,                 # 原稿ではタイトルは基本ナシ推奨
    max_x=None,
    figsize=(3.5, 2.8),         # 1カラム用
    save_folder=None,           # channel.save_current_figに渡す用
    save_name="InH_multipath_hist_cdf",
    root_fig=Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Figures/26_VTCFall原稿使用"),
):
    """
    npy_path: Base_InH.npy など（object配列 or list of dict）を想定
    ヒストグラム：Frequency (%)、CDF：Cumulative distribution (%)
    """

    # IEEE-ish font (固定)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    npy_path = Path(npy_path)

    data = np.load(npy_path, allow_pickle=True)
    channels = data.tolist() if isinstance(data, np.ndarray) else data

    counts = []
    for ch in channels:
        M = _get_M_list(ch, key_M=key_M)
        counts.append(int(np.sum(M)))  # 総マルチパス数 = Σ M[n]
    counts = np.array(counts, dtype=int)

    if max_x is None:
        max_x = int(counts.max())

    # 整数ビン（1刻み）
    bins = np.arange(0.5, max_x + 1.5, 1.0)

    hist, _ = np.histogram(counts, bins=bins)
    freq = hist / hist.sum() * 100.0
    xs = np.arange(1, max_x + 1)

    cdf = np.cumsum(freq)

    fig, ax1 = plt.subplots(figsize=figsize, constrained_layout=True)

    # Bar: Frequency (%)
    ax1.bar(xs, freq, width=0.9)
    ax1.set_xlabel("No. of multipath components")
    ax1.set_ylabel("Frequency (%)")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_xlim(0, max_x + 1)
    ax1.set_ylim(0, max(5, np.ceil(freq.max()/5)*5))  # ちょい見やすく

    # Line: CDF (%)
    ax2 = ax1.twinx()
    ax2.plot(xs, cdf, marker="o", markersize=3.5, lw=1.2, color="blue")
    ax2.set_ylabel("Cumulative distribution (%)")
    ax2.set_ylim(0, 100)

    # グリッドは薄め（論文向き）
    ax1.grid(True, linewidth=0.4, alpha=0.3)

    # 原稿では基本タイトル無し（必要ならtitle引数で）
    if title:
        ax1.set_title(title, fontsize=9)
    else:
        ax1.set_title("")

    # 保存（Paper/Slide の運用に合わせる）
    if save_folder is not None:
        # タイトル無しでPaper保存
        ax1.set_title("")
        channel.save_current_fig(
            save_name,
            root=root_fig,
            folder=save_folder,
            variants=("Paper",),
        )
        # タイトル付きでSlide保存（任意）
        if title:
            ax1.set_title(title, fontsize=9)
            channel.save_current_fig(
                save_name,
                root=root_fig,
                folder=save_folder,
                variants=("Slide",),
            )

    plt.show()
    plt.close(fig)


# ===== 使い方（パスもまとめて）=====
BASE = Path(r"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data")
plot_multipath_hist_cdf_InH(
    npy_path=BASE / "Data" / "Base_InH.npy",
    title=None,  # 原稿用はNone推奨
    figsize=(3.5, 2.8),
    save_folder="Resized",      # 例： "Paper_Fig"
    save_name="Fig_multipath_InH",
)

exit()

def plot_cdf(data, label, color, linestyle, linewidth=5):  # ← 線太くした
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label, linestyle=linestyle, color=color, linewidth=linewidth)


# 距離別の折れ線グラフ
def oresen(load_dir):
    d_list = [5, 10, 15, 20, 25, 30]
    mean_tru = []
    mean_est = []
    
    for d in d_list:
        H_tru = np.load(f"{load_dir}/d={d}/Capacity_Htru.npy", allow_pickle=True)
        H_est = np.load(f"{load_dir}/d={d}/Capacity_Hest.npy", allow_pickle=True)
        
        # チャネル容量
        mean_tru.append(np.mean(np.max(H_tru, axis=1)))
        mean_est.append(np.mean(np.max(H_est, axis=1)))
    return mean_tru, mean_est

Method = 'Mirror'
channel_type = 'InH'
Ssub_lam = 0  # サブアレー間隔(単位: 波長)
NS = 'W_NS'
NF_setting = 'Near'
load_dir = f"C:/Users/tai20/Downloads/sim_data/Data/{Method}/{channel_type}/Ssub={Ssub_lam}lam/{NS}/Channel_Capacity/{NF_setting}"
# 複数比較（例）
series = [
    {"label": f"Mirror Near Ssub={Ssub_lam}λ", "load_dir": load_dir},
    {"label": f"Far Ssub={Ssub_lam}λ",  "load_dir": load_dir.replace("/Near", "/Far")},
    # {"label": "Mirror Near Ssub=10",  "load_dir": load_dir.replace("10", "0")},
    # {"label": "Far Ssub=10",  "load_dir": load_dir.replace("10", "0").replace("/Near", "/Far")}
]
# —— 複数シリーズを重ね描き（任意）：load_dirごとにラベル指定して重ねる——
def plot_oresen_multi(series, ylim=(0,60), xlabel="BS-UE Distance d [m]",
                    ylabel="Channel Capacity [bps/Hz]", title=None, save_pdf=None):
    """
    series: list[{"label": str, "load_dir": str}]
    各load_dirごとに True/Est の2本を重ね描き
    """
    d_list = [5, 10, 15, 20, 25, 30]
    fig, ax = plt.subplots(figsize=(10, 8))
    for s in series:
        tru, est = oresen(s["load_dir"])
        ax.plot(d_list, tru, linestyle='-',  marker='o', label=f'{s["label"]} True')
        ax.plot(d_list, est, linestyle='-.', marker='o', label=f'{s["label"]} Est')
    ax.set_xlabel(xlabel, fontsize=18); ax.set_ylabel(ylabel,fontsize=18)
    ax.tick_params(labelsize=14)
    if ylim: ax.set_ylim(*ylim)
    ax.set_xticks(d_list)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(ncol=2, fontsize=16)
    if title: ax.set_title(title, fontsize=22)
    plt.tight_layout()
    if save_pdf:
        plt.savefig(save_pdf, format='pdf')
    plt.show()

plot_oresen_multi(series, title=f"{channel_type} W_NS")

exit()

def _load_ordered_eigs(npy_path, K):
    """
    np.save(…, allow_pickle=True) された固有値入れ物から、
    サンプルごとに降順ソート＆上位K本だけを抽出して 2D配列 (n_samples, K) を返す。
    None や空要素はスキップ。
    """
    raw = np.load(npy_path, allow_pickle=True)
    rows = []
    # 配列の形がどうであれ .flat で総当たりして配列(固有値ベクトル)だけ拾う
    for item in raw.flat:
        if item is None:
            continue
        v = np.asarray(item).ravel()
        if v.size == 0:
            continue
        v = np.real(v)                 # 念のため実数化
        v = np.sort(v)[::-1]           # 降順
        rows.append(v[:K])             # 上位K本
    if not rows:
        return np.empty((0, K))
    # 長さ不足を0埋めしないように、短いものはスキップ（通常は等長のはず）
    L = min(len(r) for r in rows)
    rows = [r[:L] for r in rows]
    return np.stack(rows, axis=0)      # (n_samples, L) with L<=K

def _make_cdf(series_1d):
    """1次元データのCDF（x_sorted, y∈[0,1]）を返す"""
    x = np.sort(np.asarray(series_1d))
    n = len(x)
    y = np.linspace(0, 1, n, endpoint=True)
    return x, y

def plot_eig_order_cdf(load_dir, d=5, K=5, which=("True","Est"), use_log10=True,
                    title=None, save_pdf=None):
    """
    指定dに対して、λ1..λK の “各順位別” CDF を描画（True/Estを実線/破線で重ねる）
    which: ("True","Est") / ("True",) / ("Est",)
    """
    # パス
    path_tru = f"{load_dir}/d={d}/eigvals_Htru.npy"
    path_est = f"{load_dir}/d={d}/eigvals_Hest.npy"

    # データ読み込み
    mats = {}
    if "True" in which:
        eigT = _load_ordered_eigs(path_tru, K)
        mats["True"] = eigT
    if "Est" in which:
        eigE = _load_ordered_eigs(path_est, K)
        mats["Est"] = eigE

    # 何サンプル拾えたかチェック
    for k, v in mats.items():
        if v.shape[0] == 0:
            print(f"[warn] {k}: サンプルが見つかりませんでした。 pathを確認してください。")

    # プロット
    fig, ax = plt.subplots(figsize=(10, 8))
    orders = None
    styles = {"True":"-", "Est":"--"}
    for name, M in mats.items():
        if M.shape[0] == 0:
            continue
        # M: (n_samples, L) ここで L は ≤ K
        L = M.shape[1]
        orders = list(range(1, L+1))
        for j in range(L):
            series = M[:, j]
            if use_log10:
                # 非正の値が万一混ざってたら微小値でクリップ
                series = np.clip(series, 1e-300, None)
                series = np.log10(series)
                xlabel = "Eigenvalue λ_j (log10)"
            else:
                xlabel = "Eigenvalue λ_j"
            xs, ys = _make_cdf(series)
            ax.plot(xs, ys, linestyle=styles[name], label=f"{name} λ{j+1}")

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Cumulative Probability", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    if title:
        ax.set_title(title + f" (d={d} m, top {K})", fontsize=22)
    else:
        ax.set_title(f"Eigenvalue Order CDF (d={d} m, top {K})", fontsize=22)
    ax.legend(ncol=2, fontsize=13)
    plt.tight_layout()
    if save_pdf:
        plt.savefig(save_pdf, format="pdf")
    plt.show()

# 使い方例
# Method = 'Mirror'; channel_type = 'InH'; Ssub_lam = 10; NF_setting = 'Near'
# load_dir = f"C:/Users/tai20/Downloads/sim_data/Data/{Method}/{channel_type}/Ssub={Ssub_lam}lam/Channel_Capacity/{NF_setting}"
# plot_eig_order_cdf(load_dir, d=5, K=5, which=("True","Est"), use_log10=True, title=f"{channel_type}")

plot_eig_order_cdf(load_dir, title=f"{channel_type} CDF")
"""
r_5 = np.zeros((1000), dtype=object)
r_10 = np.zeros((1000), dtype=object)
# 第1散乱体までの距離の累積分布を画くグラフ
channel_d5 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=5.npy", allow_pickle=True)
channel_d10 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=10.npy", allow_pickle=True)
channel_d15 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=15.npy", allow_pickle=True)
channel_d20 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=20.npy", allow_pickle=True)
channel_d25 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=25.npy", allow_pickle=True)
channel_d30 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=30.npy", allow_pickle=True)

multipath = 0
for p in range(1000):
    N = channel_d5[p]['N']
    for n in range(N):
        multipath += channel_d5[p]['M'][n]
print(multipath)

def r_flatten(channel):
    r_list = []

    for i in range(1000):
        N = channel[i]['N']
        M = channel[i]['M']
        for n in range(N):
            for m in range(M[n]):
                r_array = channel[i]['r'][n][m]  # n番目の要素のm番目の要素を取り出す
                r_array = np.array(r_array).flatten()  # 完全に平坦化（1次元化）
                r_list.extend(r_array)  # 要素を追加（extendで中身だけ追加）
        # 最終的に NumPy 配列に変換
        r_all = np.array(r_list)
    return r_all

r_5 = r_flatten(channel_d5)
count_5 = np.sum(r_5 == 5)  # 5の個数
total_count = r_5.size      # r_5 の要素数
print(total_count)
# 5の割合を計算
ratio_5 = count_5 / total_count

print(f"5: {count_5}")
print(f"ratio_5 {ratio_5}")
r_10 = r_flatten(channel_d10)
r_15 = r_flatten(channel_d15)
r_20 = r_flatten(channel_d20)
r_25 = r_flatten(channel_d25)
r_30 = r_flatten(channel_d30)
r_5_filtered = r_5[r_5 != 5]  # 5の値を取り除く
r_10_filterd = r_10[r_10 != 10]
r_15_filterd = r_15[r_15 != 15]
r_20_filterd = r_20[r_20 != 20]
r_25_filterd = r_25[r_25 != 25]
r_30_filterd = r_30[r_30 != 30]



# plot_cdf(r_5, 'd=5', 'blue', 3)
# plot_cdf(r_10, 'd=10', 'red', 3)
# plot_cdf(r_15, 'd=15', 'yellow', 3)
# plot_cdf(r_20, 'd=20', 'green', 3)
# plot_cdf(r_25, 'd=25', 'cyan', 3)
# plot_cdf(r_30, 'd=30', 'purple', 3)
plot_cdf(r_5_filtered, 'd=5 wo5', 'blue', 3)
plot_cdf(r_10_filterd, 'd=10 wo10', 'red', 3)
plot_cdf(r_15_filterd, 'd=15 wo15', 'yellow', 3)
plot_cdf(r_20_filterd, 'd=20 wo20', 'green', 3)
plot_cdf(r_25_filterd, 'd=25 wo25', 'cyan', 3)
plot_cdf(r_30_filterd, 'd=30 wo30', 'purple', 3)

# 軸ラベルとかタイトルもフォントサイズ
plt.xlabel('r [m]', fontsize=20)
plt.ylabel('Cumulative distribution (CDF)', fontsize=20)
plt.title(f'Distance between BS and First scatterer [m]', fontsize=20)
plt.xlim(left=0, right=40)
plt.ylim(bottom=0, top=1.0)

# 軸目盛りもフォント大きめに
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 判例は右下＆フォントサイズUP
plt.legend(loc='lower right', fontsize=15)

plt.grid(True)
plt.tight_layout()
plt.show()
"""


##########################################################################################################################################################

# チャネル容量累積分布のグラフ
setting = 'InF'
d=10
load_dir_r3 = f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_r=3/ChannelCapacity/Channel_{setting}_step5_ChannelCapacity_d={d}"
load_dir = f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/ChannelCapacity/Channel_{setting}_step5_ChannelCapacity_d={d}"
load_dir_CA = f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/{setting}_CoherentSum"

H_near_tru = np.load(f"{load_dir_CA}/{setting}_Simple_CoherentSum/ChannelCapacity/Channel_{setting}_step5_Htru_Near_ChannelCapacity_d={d}.npy")
H_near_mea = np.load(f"{load_dir_CA}/{setting}_Simple_CoherentSum/ChannelCapacity/Channel_{setting}_step5_Hmea_Near_ChannelCapacity_d={d}.npy")
H_far_tru = np.load(f"{load_dir_CA}/{setting}_Simple_CoherentSum/ChannelCapacity/Channel_{setting}_step5_Htru_Far_ChannelCapacity_d={d}.npy")
H_far_mea = np.load(f"{load_dir_CA}/{setting}_Simple_CoherentSum/ChannelCapacity/Channel_{setting}_step5_Hmea_Far_ChannelCapacity_d={d}.npy")
H_r3_tru = np.load(f"{load_dir_CA}/{setting}_r=3/ChannelCapacity/Channel_{setting}_step5_Htru_Near_ChannelCapacity_d={d}.npy")
H_r3_mea = np.load(f"{load_dir_CA}/{setting}_r=3/ChannelCapacity/Channel_{setting}_step5_Hmea_Near_ChannelCapacity_d={d}.npy")


H_near_tru_ad = np.max(H_near_tru, axis=1)
H_near_mea_ad = np.max(H_near_mea, axis=1)
H_far_tru_ad = np.max(H_far_tru, axis=1)
H_far_mea_ad = np.max(H_far_mea, axis=1)
H_r3_tru_ad = np.max(H_r3_tru, axis=1)
H_r3_mea_ad = np.max(H_r3_mea, axis=1)
print(np.shape(H_near_tru))

plt.figure(figsize=(10, 7))  # ← サイズも少し大きめに

# プロット
plot_cdf(H_far_tru_ad, 'Far True', 'darkblue', '-',3)
plot_cdf(H_far_mea_ad, 'Far Est', 'cyan','-.', 3)
plot_cdf(H_near_tru_ad, 'Near True (Unif)', 'orangered','-', 3)
plot_cdf(H_near_mea_ad, 'Near Est (Unif)', 'orange','-.', 3)
plot_cdf(H_r3_tru_ad, 'Near True (3m)', 'forestgreen','-', 3)
plot_cdf(H_r3_mea_ad, 'Near Est (3m)', 'limegreen','-.', 3)



# 軸ラベルとかタイトルもフォントサイズUP
plt.xlabel('Channel capacity [bps/Hz]', fontsize=20)
plt.ylabel('Cumulative distribution (CDF)', fontsize=20)
# plt.title(f'{setting} d = {d} [m]', fontsize=20)
plt.xlim(left=0, right=90)
plt.ylim(bottom=0, top=1.0)

# 軸目盛りもフォント大きめに
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 判例は右下＆フォントサイズUP
plt.legend(loc='lower right', fontsize=20)

plt.grid(True)
plt.tight_layout()
# PDFで保存
plt.savefig(f"{setting}_d=10_cum.pdf", format='pdf')
plt.show()

exit()


# --- 設定 ---
setting = 'InH'
d = 5
# base_dir = (
#     f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter"
#     f"/channel_{setting}_1000/Channel_{setting}_1000_step5_ChannelCapacity"
#     f"/Channel_{setting}_step5_ChannelCapacity_d={d}"
# )

base_dir = f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/InH_CoherentSum/InH_Simple_CoherentSum/ChannelCapacity"

# --- データロード関数 ---
def load_selected_db(NorF, mode):
    # NorF: 'Near' or 'Far'; mode: 'Htru' or 'Hmea'
    # チャネル容量ロード
    H = np.load(
        f"{base_dir}/Channel_{setting}_step5_{mode}_{NorF}_ChannelCapacity_d={d}.npy"
    )  # shape: (100, 12)
    beam_idx = np.argmax(H, axis=1)
    # 固有値ロード
    eig = np.load(
        f"{base_dir}/eigvals_{NorF}_{mode}_{NorF}_d={d}.npy"
    )  # shape: (100, 12, M)
    # 選択固有値抽出 & dB変換
    sel = eig[np.arange(beam_idx.size), beam_idx, :]  # (100, M)
    return 10 * np.log10(sel)  # (100, M) in dB

# --- 各データ取得 ---
selected_tru_near_db = load_selected_db('Near', 'Htru')
selected_est_near_db = load_selected_db('Near', 'Hmea')
selected_tru_far_db  = load_selected_db('Far',  'Htru')
selected_est_far_db  = load_selected_db('Far',  'Hmea')

# --- CDFプロット関数 ---
def plot_cdf_db(data_db, label=None, linewidth=2, linestyle='-'):
    sorted_db = np.sort(data_db)
    cdf = np.arange(1, len(sorted_db) + 1) / len(sorted_db)
    plt.plot(sorted_db, cdf, label=label, linewidth=linewidth, linestyle=linestyle)

# --- プロット: Near ---
plt.figure(figsize=(10, 7))
M = 8
for mode_idx in range(M):
    # True
    plot_cdf_db(selected_tru_near_db[:, mode_idx],
                label=f'Near True {mode_idx+1}',
                linewidth=2, linestyle='-')
    # Est
    plot_cdf_db(selected_est_near_db[:, mode_idx],
                label=f'Near Est {mode_idx+1}',
                linewidth=2, linestyle='--')
plt.xlabel('Eigenvalue [dB]', fontsize=16)
plt.ylabel('CDF', fontsize=16)
# plt.title(f'{setting} Near (True Est) CDF Eigenval, d={d} [m]', fontsize=18)
plt.xlim(-160, 0)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='', fontsize=10, loc='lower right')
plt.grid(True)
plt.tight_layout()


# --- プロット: Far ---
plt.figure(figsize=(10, 7))
for mode_idx in range(M):
    # True
    plot_cdf_db(selected_tru_far_db[:, mode_idx],
                label=f'Far True  {mode_idx+1}',
                linewidth=2, linestyle='-')
    # Est
    plot_cdf_db(selected_est_far_db[:, mode_idx],
                label=f'Far Est {mode_idx+1}',
                linewidth=2, linestyle='--')
plt.xlabel('Eigenvalue [dB]', fontsize=16)
plt.ylabel('CDF', fontsize=16)
plt.title(f'{setting} Far (True Est) CDF Eigenval, d={d} [m]', fontsize=18)
plt.xlim(-160, 0)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='', fontsize=10, loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
##########################################################################################################################################

