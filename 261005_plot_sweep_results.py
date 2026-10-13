# 261005_plot_sweep_results.py
# 260927_capacity_d_ssub_sweep_gpu.py が出力した Results_*.npz から図を作る。
# 図のスタイル・保存方法は 260205_VTCFall.py の plot_capacity / plot_layers / plot_eigs と
# Channel_functions.save_current_fig に合わせている (Paper: タイトルなし / Slide: タイトルあり)。
# 使い方: python 261005_plot_sweep_results.py <Results_xxx.npz> [出力ルート]

import re
import sys
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import Channel_functions as channel

LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 9
MARKER_SIZE = 6
LINE_WIDTH = 2

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
})

COLOR_MAP = {0: "tab:green", 50: "tab:blue"}
DEFAULT_COLOR = "tab:red"
MARKER_MAP = {0: "o", 50: "s", 100: "^"}
FONT_SETTINGS = {"fontname": "Times New Roman", "fontsize": LABEL_SIZE}

# capacity の Type 軸: 0=true, 1=est_avg_denoised, 2=est_single_denoised, 3=est_single_raw
TRUE, EST_AVG, EST_SINGLE, EST_RAW = 0, 1, 2, 3
TYPE_STYLE = {
    TRUE:       ("True",                     "k",       "-",  "o"),
    EST_AVG:    ("Est. (10x avg + denoise)", "tab:blue", "--", "s"),
    EST_SINGLE: ("Est. (single + denoise)",  "tab:orange", "-.", "^"),
    EST_RAW:    ("Est. (single, raw)",       "tab:red", ":",  "v"),
}
MULTIPLEX, SINGLE_LAYER = 0, 1


def color_of(s):
    return COLOR_MAP.get(int(s), DEFAULT_COLOR)


def style_axes(ax):
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")
    ax.grid(True, which="major", linestyle="-", color="#777777", alpha=0.7, linewidth=0.5)


def distance_axis(ax, dv):
    ax.set_xticks(list(range(0, 55, 5)))
    ax.set_xlim(2, 53)
    ax.set_xlabel("BS-UE Distance (m)", **FONT_SETTINGS)


def style_legend(ax, title=None, legend_size=LEGEND_SIZE, loc="upper right", **kw):
    """VTCFallと同じ規則: 凡例ラベルの "0λ" を "0" に簡略化し、フォントを統一する。"""
    handles, labels = ax.get_legend_handles_labels()
    labels = [re.sub(r"(?<!\d)0\$\\lambda\$", "0", re.sub(r"(?<!\d)0λ", "0", l)) for l in labels]
    leg = ax.legend(handles, labels, handlelength=2.5, loc=loc, title=title,
                    prop={"family": "Times New Roman", "size": legend_size}, **kw)
    if title:
        plt.setp(leg.get_title(), family="Times New Roman", size=legend_size)
    return leg


def outside_legend(fig, ax, ncol, legend_size=LEGEND_SIZE):
    """図全体の下に凡例を1行で出す (曲線・棒に重ならないようにする)。"""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False,
               prop={"family": "Times New Roman", "size": legend_size})


def save(fig, title, root, folder, slide_title=True):
    """Paper版(タイトルなし)とSlide版(タイトルあり)を save_current_fig で保存する。"""
    plt.figure(fig.number)
    channel.save_current_fig(title, root=root, folder=folder, variants=("Paper",))
    if slide_title:
        if len(fig.axes) > 1:
            fig.suptitle(title, size=15)
        else:
            fig.axes[0].set_title(textwrap.fill(title, 36), size=12)
    channel.save_current_fig(title, root=root, folder=folder, variants=("Slide",))
    plt.close(fig)


# ---------------------------------------------------------------- 図1: 容量
def plot_capacity(cap_mean, dv, sv, root, folder):
    """距離 vs 容量(真のチャネル)。凡例は各項目に Single/Multi を付けたまま、左列=Single、右列=Multi の2コラム。"""
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    multi_lines, single_lines = [], []
    for j, s in enumerate(sv):
        c = color_of(s)
        lm, = ax.plot(dv, cap_mean[:, j, TRUE, MULTIPLEX], marker="o", markersize=MARKER_SIZE,
                      lw=LINE_WIDTH, color=c)
        ls_, = ax.plot(dv, cap_mean[:, j, TRUE, SINGLE_LAYER], marker="s", markersize=MARKER_SIZE,
                       ls="--", lw=LINE_WIDTH, color=c, markerfacecolor="none")
        multi_lines.append(lm)
        single_lines.append(ls_)
    distance_axis(ax, dv)
    ax.set_ylabel("Channel Capacity (bps/Hz)", **FONT_SETTINGS)
    ax.set_ylim(0, 70)
    style_axes(ax)

    # 凡例: 各項目に "<Ssub> Single" / "<Ssub> Multi" のラベルを付けたまま、左列=Single、右列=Multi の
    # 2コラムにする。ncol=2 は列方向に詰めるので、[Single x3, Multi x3] の順で渡す。
    s_lab = [fr"{x}$\lambda$" if x != 0 else "0" for x in sv]
    handles = single_lines + multi_lines
    labels = [f"{t} Single" for t in s_lab] + [f"{t} Multi" for t in s_lab]
    leg = ax.legend(handles, labels, ncol=2, handlelength=2.5, columnspacing=1.2, loc="upper right",
                    title=r"$S_\mathrm{sub}$ Layer", prop={"family": "Times New Roman", "size": LEGEND_SIZE})
    plt.setp(leg.get_title(), family="Times New Roman", size=LEGEND_SIZE)
    save(fig, "Channel Capacity vs BS-UE Distance (True Channel)", root, folder)


# ---------------------------------------------------------------- 図2: レイヤ数
def plot_layers(ly_mean, dv, sv, root, folder, t=TRUE, name="True Channel"):
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    order = {50: 0, 0: 1, 100: 2}
    for j, s in enumerate(sv):
        c = color_of(s)
        ax.plot(dv, ly_mean[:, j, t], lw=1.8, color=c, zorder=-1)
        ax.scatter(dv, ly_mean[:, j, t], marker=MARKER_MAP.get(int(s), "D"), s=7 ** 2,
                   facecolors="white", edgecolors=c, linewidths=1.8,
                   zorder=order.get(int(s), 3), label=fr"{s}$\lambda$")
    distance_axis(ax, dv)
    ax.set_ylabel("Number of Layers", **FONT_SETTINGS)
    ax.set_ylim(0, 8.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    style_axes(ax)
    style_legend(ax, title=r"$S_\mathrm{sub}$", legend_size=12)
    save(fig, f"Number of Layers vs BS-UE Distance ({name})", root, folder)


# ---------------------------------------------------------------- 図3: 推定方式の比較 (Ssubごと)
def plot_capacity_by_type(cap_mean, dv, sv, root, folder):
    for j, s in enumerate(sv):
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        for t, (lab, c, ls, mk) in TYPE_STYLE.items():
            ax.plot(dv, cap_mean[:, j, t, MULTIPLEX], marker=mk, markersize=MARKER_SIZE - 1,
                    ls=ls, lw=LINE_WIDTH, color=c, markerfacecolor="none" if t else c, label=lab)
        distance_axis(ax, dv)
        ax.set_ylabel("Channel Capacity (bps/Hz)", **FONT_SETTINGS)
        ax.set_ylim(0, 70)
        style_axes(ax)
        style_legend(ax, legend_size=8, loc="upper right")
        save(fig, f"Capacity by Channel Type (Ssub={s}lam)", root, folder)


def plot_capacity_loss(cap_mean, dv, sv, root, folder):
    for j, s in enumerate(sv):
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        for t in (EST_AVG, EST_SINGLE, EST_RAW):
            lab, c, ls, mk = TYPE_STYLE[t]
            loss = cap_mean[:, j, TRUE, MULTIPLEX] - cap_mean[:, j, t, MULTIPLEX]
            ax.semilogy(dv, np.clip(loss, 1e-4, None), marker=mk, markersize=MARKER_SIZE - 1,
                        ls=ls, lw=LINE_WIDTH, color=c, markerfacecolor="none", label=lab)
        distance_axis(ax, dv)
        ax.set_ylabel("Capacity Loss (bps/Hz)", **FONT_SETTINGS)
        ax.set_ylim(1e-2, 10)
        style_axes(ax)
        style_legend(ax, legend_size=8, loc="center right")
        save(fig, f"Capacity Loss vs True Channel (Ssub={s}lam)", root, folder)


# ---------------------------------------------------------------- 図4: 固有値の累積分布 (CDF)
NUMERICAL_ZERO_REL = 1e-14   # 固有値CDFで、λ1に対してこの比以下の固有値は数値的にゼロとして扱う


def plot_eig_cdf(eig, dv, sv, root, folder, d_pick=(5, 25, 50), k_list=(1, 2, 3, 4), active_v=None, min_v=8):
    """試行(チャネル)ごとの第k固有値(サブキャリア平均)のCDF(既定はλ1〜λ4。k_list=tuple(range(1, 9)) でλ1〜λ8)。色=固有値の番号k、種類は線種+マーカーで区別する:
    True = 太い実線(背面、マーカーなし)、W/ NS(10回平均+遅延デノイズ) = 細い実線+白抜き丸、
    W/o NS(単発・生の推定) = 破線+塗りつぶし三角。重なっても見えるよう、マーカーは一定のCDF間隔で、
    種類ごとに位置をずらして打つ。数値的にゼロ(λ1の1e-14倍以下)の固有値は対数軸に出せないため描かないが、
    CDFの分母は全チャネル数のままなので、曲線はその分だけ下側を空けて始まる。
    active_v を渡すと、ビームが割り当てられたサブアレー数 V'>=min_v(2面以上で受信)のチャネルだけで描く。
    (d, Ssub) ごとに1枚ずつ出力する。"""
    cmap = plt.get_cmap("tab10")
    k_colors = {k: cmap(k - 1) for k in k_list}
    # (type, 線種, 線幅, マーカー, 白抜きか, マーカー位置のずらし, zorder, 名前)
    methods = ((TRUE,    "-",  3.2, None, False,   0, 1, "True"),
               (EST_AVG, "-",  1.3, "o",  True,   50, 2, "W/ NS"),
               (EST_RAW, "--", 1.3, "^",  False, 100, 3, "W/o NS"))
    MARK_EVERY = 150  # 約1000試行のうち150点ごと(CDFで約0.15間隔)にマーカーを打つ
    for d_val in d_pick:
        di = int(np.where(np.asarray(dv) == d_val)[0][0])
        for j, s in enumerate(sv):
            sel = np.ones(eig.shape[2], bool) if active_v is None else active_v[di, j] >= min_v
            n_ch = int(sel.sum())
            fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
            for t, ls, lw, mk, hollow, off, z, name in methods:
                for k in k_list:
                    x = eig[di, j, sel, t, k - 1]
                    # 最大固有値の1e-14倍以下は数値的にゼロ(GPUとCPUの固有値はこの比まで一致する)とみなし、描かない
                    x = np.sort(10 * np.log10(x[x > NUMERICAL_ZERO_REL * eig[di, j, sel, t, 0]]))
                    if len(x) == 0:
                        continue
                    c = k_colors[k]
                    # 描く点は一部だけなので、CDFは(対象)チャネル数を分母にして、数値的にゼロの分だけ下側を空ける
                    y = (np.arange(1, len(x) + 1) + (n_ch - len(x))) / n_ch
                    ax.plot(x, y, ls=ls, lw=lw, color=c, zorder=z,
                            marker=mk, ms=4.2, markevery=(off, MARK_EVERY),
                            markerfacecolor="white" if hollow else c, markeredgecolor=c, markeredgewidth=1.1)
            ax.set_xlabel("Eigenvalue (dB)", **FONT_SETTINGS)
            ax.set_ylabel("CDF", **FONT_SETTINGS)
            ax.set_ylim(0, 1)
            style_axes(ax)
            # 凡例: 列方向に詰めて、固有値の色(1列に最大4行)の列の右に、線種+マーカー(True / W/ NS / W/o NS)の列を置く
            n_rows = min(len(k_list), 4)
            blank = Line2D([], [], color="none")
            k_handles = [Line2D([], [], color=k_colors[k], lw=2.2) for k in k_list]
            k_labels = [fr"$\lambda_{k}$" for k in k_list]
            m_handles = [Line2D([], [], color="k", ls=ls, lw=2.6 if mk is None else 1.3, marker=mk, ms=4.2,
                                markerfacecolor="white" if hollow else "k")
                         for _, ls, _, mk, hollow, _, _, _ in methods]
            m_labels = [name for *_, name in methods]
            for hs, ls_ in ((k_handles, k_labels), (m_handles, m_labels)):
                while len(hs) % n_rows:
                    hs.append(blank)
                    ls_.append("")
            fig.legend(k_handles + m_handles, k_labels + m_labels, loc="outside lower center",
                       ncol=(len(k_handles) + len(m_handles)) // n_rows, frameon=False, handlelength=2.6,
                       columnspacing=1.4, prop={"family": "Times New Roman", "size": 8})
            if active_v is None:
                save(fig, f"CDF of Eigenvalues (d={d_val}m, Ssub={s}lam)", root, folder)
            else:
                save(fig, f"CDF of Eigenvalues, 2+ Faces (d={d_val}m, Ssub={s}lam, n={n_ch})", root, folder)


# ---------------------------------------------------------------- 図5: 水注水の電力配分
def plot_wf_stacked(pr_mean, dv, sv, root, folder, n_show=6):
    """真のチャネルの水注水による層ごとの電力配分比率(試行平均)を距離ごとの積み上げ棒で示す。Ssubごとに1枚。"""
    cmap = plt.get_cmap("viridis")
    for j, s in enumerate(sv):
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        bottom = np.zeros(len(dv))
        for l in range(n_show):
            v = pr_mean[:, j, TRUE, l]
            ax.bar(dv, v, bottom=bottom, width=3.5, color=cmap(l / max(n_show - 1, 1)),
                   edgecolor="white", linewidth=0.4, label=f"Layer {l + 1}")
            bottom += v
        distance_axis(ax, dv)
        ax.set_ylabel("Power Allocation Ratio", **FONT_SETTINGS)
        ax.set_ylim(0, 1)
        style_axes(ax)
        outside_legend(fig, ax, ncol=3, legend_size=8)
        save(fig, f"Water-filling Power Allocation vs Distance (True Channel, Ssub={s}lam)", root, folder)


def plot_wf_stacked_row(pr_mean, dv, sv, root, folder, n_show=6):
    """距離ごとの層別電力配分比率(真のチャネル、試行平均)の積み上げ棒を、Ssub=0/50/100λ で3つ横に並べた1枚にする。
    縦軸・横軸を共通にして、Ssubによる違いを比較しやすくする。"""
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(sv), figsize=(4 * len(sv), 3), sharex=True, sharey=True, constrained_layout=True)
    for j, s in enumerate(sv):
        ax = axes[j]
        bottom = np.zeros(len(dv))
        for l in range(n_show):
            v = pr_mean[:, j, TRUE, l]
            ax.bar(dv, v, bottom=bottom, width=3.5, color=cmap(l / max(n_show - 1, 1)),
                   edgecolor="white", linewidth=0.4, label=f"Layer {l + 1}")
            bottom += v
        ax.set_title(fr"$S_\mathrm{{sub}}={s}\lambda$", fontsize=13)
        distance_axis(ax, dv)
        ax.set_ylim(0, 1)
        style_axes(ax)
    axes[0].set_ylabel("Power Allocation Ratio", **FONT_SETTINGS)
    outside_legend(fig, axes[-1], ncol=n_show, legend_size=9)
    # 3つ並べた1枚なので、各パネルのタイトルはそのまま残し、全体のタイトルはSlide版だけに付ける
    save(fig, "Water-filling Power Allocation vs Distance (True Channel, Ssub Comparison)", root, folder)


def plot_wf_by_type(pr_mean, dv, sv, root, folder, d_pick=(5, 25, 50), n_show=8):
    """層ごとの電力配分比率の比較: 真 / 推定(10回平均+デノイズ) / 推定(単発・生)。(d, Ssub)ごとに1枚。"""
    w = 0.27
    x = np.arange(1, n_show + 1)
    for d_val in d_pick:
        di = int(np.where(np.asarray(dv) == d_val)[0][0])
        for j, s in enumerate(sv):
            fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
            for i, t in enumerate((TRUE, EST_AVG, EST_RAW)):
                lab, c, _, _ = TYPE_STYLE[t]
                ax.bar(x + (i - 1) * w, pr_mean[di, j, t, :n_show], width=w,
                       color="gray" if t == TRUE else c, label=lab)
            ax.set_xlabel("Layer Index", **FONT_SETTINGS)
            ax.set_ylabel("Power Allocation Ratio", **FONT_SETTINGS)
            ax.set_xticks(x)
            ax.set_ylim(0, 0.5)
            style_axes(ax)
            style_legend(ax, legend_size=8, loc="upper right")
            save(fig, f"Water-filling Power Allocation by Channel Type (d={d_val}m, Ssub={s}lam)", root, folder)


# ---------------------------------------------------------------- 2面以上で受信するチャネル(V'>=8)に絞った図
# active_v: ビームが割り当てられたサブアレー数 V' (d, Ssub, Trial)。V'=4が1面、8が2面、12が3面。
LAMBDA_REF_DB = 10 * np.log10(6.31e-12 / 0.5)   # P_noise/Pt: この固有値で、全電力を1層に集中したときSNR=0 dB


def plot_eig_cdf_multiface(eig, active_v, dv, sv, root, folder, d_pick=(5, 25, 50), min_v=8):
    """V'>=min_v のチャネルだけの、全8固有値(真のチャネル、サブキャリア平均)のCDF。(d, Ssub)ごとに1枚。"""
    cmap = plt.get_cmap("viridis")
    for d_val in d_pick:
        di = int(np.where(np.asarray(dv) == d_val)[0][0])
        for j, s in enumerate(sv):
            m = active_v[di, j] >= min_v
            fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
            for k in range(1, eig.shape[-1] + 1):
                x = eig[di, j, m, TRUE, k - 1]
                x = np.sort(10 * np.log10(x[x > 0]))
                if len(x) == 0:
                    continue
                # 固有値が0(構造的にゼロ)のチャネルはCDFの下側に含めない: 分母は全チャネル数
                ax.plot(x, np.arange(1, len(x) + 1) / m.sum(), lw=1.6, color=cmap((k - 1) / 7), label=fr"$\lambda_{k}$")
            ax.axvline(LAMBDA_REF_DB, color="gray", ls="--", lw=1.0, label=r"$P_\mathrm{noise}/P_t$")
            ax.set_xlabel("Eigenvalue (dB)", **FONT_SETTINGS)
            ax.set_ylabel("CDF", **FONT_SETTINGS)
            ax.set_ylim(0, 1)
            style_axes(ax)
            outside_legend(fig, ax, ncol=5, legend_size=8)
            save(fig, f"CDF of Eigenvalues, 2+ Faces (d={d_val}m, Ssub={s}lam, n={int(m.sum())})", root, folder)


def plot_wf_by_layer_multiface(pr, active_v, dv, sv, root, folder, d_pick=(5, 25, 50), min_v=8, n_show=8):
    """V'>=min_v のチャネルの、層ごとの電力配分比率(チャネル平均)。Ssubを色分けして1枚にまとめる(dごとに1枚)。"""
    w = 0.27
    x = np.arange(1, n_show + 1)
    for d_val in d_pick:
        di = int(np.where(np.asarray(dv) == d_val)[0][0])
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        for j, s in enumerate(sv):
            m = active_v[di, j] >= min_v
            ax.bar(x + (j - 1) * w, pr[di, j, m, TRUE, :n_show].mean(axis=0), width=w,
                   color=color_of(s), label=fr"{s}$\lambda$")
        ax.set_xlabel("Layer Index", **FONT_SETTINGS)
        ax.set_ylabel("Power Allocation Ratio", **FONT_SETTINGS)
        ax.set_xticks(x)
        ax.set_ylim(0, 0.5)
        style_axes(ax)
        style_legend(ax, title=r"$S_\mathrm{sub}$", legend_size=9, loc="upper right")
        save(fig, f"Water-filling Power Allocation by Layer, 2+ Faces (d={d_val}m)", root, folder)


def plot_layer_hist_multiface(ly, active_v, dv, sv, root, folder, d_pick=(5, 25, 50), min_v=8):
    """V'>=min_v のチャネルの、レイヤ数(試行ごと、四捨五入)の分布。Ssubを色分けして1枚にまとめる(dごとに1枚)。"""
    w = 0.27
    x = np.arange(1, 9)
    for d_val in d_pick:
        di = int(np.where(np.asarray(dv) == d_val)[0][0])
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        for j, s in enumerate(sv):
            m = active_v[di, j] >= min_v
            cnt = np.bincount(np.round(ly[di, j, m]).astype(int), minlength=9)[1:9] / m.sum()
            ax.bar(x + (j - 1) * w, cnt, width=w, color=color_of(s), label=fr"{s}$\lambda$")
        ax.set_xlabel("Number of Layers", **FONT_SETTINGS)
        ax.set_ylabel("Fraction of Channels", **FONT_SETTINGS)
        ax.set_xticks(x)
        style_axes(ax)
        style_legend(ax, title=r"$S_\mathrm{sub}$", legend_size=9, loc="upper right")
        save(fig, f"Distribution of Number of Layers, 2+ Faces (d={d_val}m)", root, folder)


def main_multiface(npz_path, root):
    d = np.load(npz_path, allow_pickle=True)
    ly = d["layers"][:, :, :, TRUE]
    eig, pr, act = d["eigenvalues"], d["power_ratio"], d["active_v"]
    dv, sv = np.asarray(d["d"]), np.asarray(d["Ssub"])
    folder = Path(npz_path).stem + "_2faces"
    plot_eig_cdf(eig, dv, sv, root, folder, active_v=act)   # 全チャネル版と同じ形式(V'>=8に絞る)
    plot_wf_by_layer_multiface(pr, act, dv, sv, root, folder)
    plot_layer_hist_multiface(ly, act, dv, sv, root, folder)


def main(npz_path, root):
    d = np.load(npz_path, allow_pickle=True)
    cap, ly, eig, pr = d["capacity"], d["layers"], d["eigenvalues"], d["power_ratio"]
    dv, sv = np.asarray(d["d"]), np.asarray(d["Ssub"])
    folder = Path(npz_path).stem

    # 試行平均 (d, Ssub, Type, Mode) / (d, Ssub, Type) / (d, Ssub, Type, U)
    cap_mean = np.nanmean(cap, axis=2)
    ly_mean = np.nanmean(ly, axis=2)
    pr_mean = np.nanmean(pr, axis=2)

    plot_capacity(cap_mean, dv, sv, root, folder)
    plot_layers(ly_mean, dv, sv, root, folder, TRUE, "True Channel")
    plot_layers(ly_mean, dv, sv, root, folder, EST_RAW, "Estimated Channel W/o NS")
    plot_capacity_by_type(cap_mean, dv, sv, root, folder)
    plot_capacity_loss(cap_mean, dv, sv, root, folder)
    plot_eig_cdf(eig, dv, sv, root, folder)
    plot_wf_stacked(pr_mean, dv, sv, root, folder)
    plot_wf_by_type(pr_mean, dv, sv, root, folder)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python 261005_plot_sweep_results.py <Results_xxx.npz> [出力ルート] [2faces]")
    p = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else p.parent / "Figures"
    if len(sys.argv) > 3 and sys.argv[3] == "2faces":
        main_multiface(p, out)   # 2面以上(V'>=8)で受信するチャネルに絞った図だけ作る
    else:
        main(p, out)
