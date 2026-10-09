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
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    for j, s in enumerate(sv):
        c = color_of(s)
        ax.plot(dv, cap_mean[:, j, TRUE, MULTIPLEX], marker="o", markersize=MARKER_SIZE,
                lw=LINE_WIDTH, color=c, label=fr"{s}$\lambda$ Multi")
        ax.plot(dv, cap_mean[:, j, TRUE, SINGLE_LAYER], marker="s", markersize=MARKER_SIZE,
                ls="--", lw=LINE_WIDTH, color=c, markerfacecolor="none", label=fr"{s}$\lambda$ Single")
    distance_axis(ax, dv)
    ax.set_ylabel("Channel Capacity (bps/Hz)", **FONT_SETTINGS)
    ax.set_ylim(0, 70)
    style_axes(ax)
    style_legend(ax, title=r"$S_\mathrm{sub}$ Layer", ncol=2, columnspacing=1.0)
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
def plot_eig_cdf(eig, dv, sv, root, folder, d_pick=(5, 25, 50), k_list=(1, 2, 3, 4)):
    """試行(チャネル)ごとの第k固有値(サブキャリア平均)のCDF。実線=真のチャネル、破線=推定(単発・生)。
    (d, Ssub) ごとに1枚ずつ出力する。"""
    k_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    for d_val in d_pick:
        di = int(np.where(np.asarray(dv) == d_val)[0][0])
        for j, s in enumerate(sv):
            fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
            for k in k_list:
                for t, ls in ((TRUE, "-"), (EST_RAW, "--")):
                    x = eig[di, j, :, t, k - 1]
                    x = np.sort(10 * np.log10(x[x > 0]))
                    if len(x) == 0:
                        continue
                    ax.plot(x, np.arange(1, len(x) + 1) / eig.shape[2], ls=ls, lw=1.8,
                            color=k_colors[k], label=fr"$\lambda_{k}$ " + ("True" if t == TRUE else "Raw"))
            ax.set_xlabel("Eigenvalue (dB)", **FONT_SETTINGS)
            ax.set_ylabel("CDF", **FONT_SETTINGS)
            ax.set_ylim(0, 1)
            style_axes(ax)
            outside_legend(fig, ax, ncol=4, legend_size=8)
            save(fig, f"CDF of Eigenvalues (d={d_val}m, Ssub={s}lam)", root, folder)


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
        sys.exit("usage: python 261005_plot_sweep_results.py <Results_xxx.npz> [出力ルート]")
    p = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else p.parent / "Figures"
    main(p, out)
