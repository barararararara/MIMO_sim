# 261006_compare_gpu_cpu.py
# GPU版の結果(Results_*.npz)を、CPU版(# 260319_RCS石垣島.py)で同じ試行・d・Ssubを個別に再計算した結果と比べる。
# 使い方: python 261006_compare_gpu_cpu.py <Results_xxx.npz> [ケース数] [乱数seed]
#
# 比較する量: 容量(空間多重/単一レイヤ)、レイヤ数、固有値(サブキャリア平均)、電力配分比率、
#             有効サブアレー数V'、ビーム割当、推定チャネルの容量損失。
# CPU版とGPU版では乱数列が異なる(CPUはnp.random、GPUは試行ごとのtorch.Generator)ため、
# 雑音が絡む量(推定チャネル、ビーム選択の閾値付近)は完全一致せず、統計的な大きさで比べる。
#
# 切り分けのため、次の3点をCPU側で変えて比べる:
#  (c) Channel_functions.SP_power / SP_Power_each_career(サブパス電力の配分)は、正規化に「途中までの和」を
#      使う実装ミス(VTCFall時代のミス)を含む。GPU版は全サブパスの和で正規化している(電力保存)ので、
#      CPU側でこの2関数だけ正しい実装に差し替えた場合と、元のままの場合の両方で比べる
#  (a) 受信側MMSEの重みに入れている雑音(CPU版のH_effへの1.778e-6)の有無 (GPU版には無い)
#  (b) 推定チャネルの雑音の大きさ: CPU版(1.778e-6) と GPU版相当(2.512e-6/sqrt(Pu/K)=3.553e-6)

import sys
import io
import contextlib
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
BASE_FILE = Path("C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Data/Base_InH.npy")
CPU_SCRIPT = REPO / "# 260319_RCS石垣島.py"
Pt_mW, P_noise_mW, Pu_dBm, base_seed = 0.5, 6.31e-12, 30, 9
U = 8
SIGMA_CPU = 1.778e-6                  # CPU版 noise_dash_K_10 の実部・虚部それぞれの標準偏差
SIGMA_GPU = 2.512e-6 / (0.5 ** 0.5)   # GPU版 sigma_dash (= 2.512e-6 / sqrt(Pu_mW/K))
RX_ON, RX_OFF = "rx雑音あり", "rx雑音なし"


def _sp_power_fixed(N, M, P, rho, U_, setting="InH", per_carrier=False):
    """Channel_functions.SP_power と同じ入出力で、正規化に全サブパスの和を使う正しい実装。"""
    gamma = {"InH": 2.0, "InF": 4.7}[setting]
    Pi = np.zeros((N, max(M)))
    for n in range(N):
        dash = np.array([np.exp(-rho[n][m] / gamma) * (10 ** (U_[n][m] / 10)) for m in range(M[n])])
        Pi[n, :M[n]] = dash / dash.sum() * P[n]
    max_n, max_m = np.unravel_index(np.argmax(Pi, axis=None), Pi.shape)
    if (max_n, max_m) != (0, 0):
        Pi[0][0], Pi[max_n][max_m] = Pi[max_n][max_m], Pi[0][0]
    return Pi


@contextlib.contextmanager
def sp_power_mode(cpu, fixed):
    """fixed=True の間だけ、CPU版が呼ぶ channel.SP_power / SP_Power_each_career を正しい実装に差し替える。"""
    ch = cpu["channel"]
    orig = (ch.SP_power, ch.SP_Power_each_career)
    if fixed:
        ch.SP_power = _sp_power_fixed
        ch.SP_Power_each_career = _sp_power_fixed
    try:
        yield
    finally:
        ch.SP_power, ch.SP_Power_each_career = orig


def load_cpu_module(rx_noise=True):
    """CPU版を、末尾の「スイープ実行」より手前までだけ読み込む(import時の自動実行を避ける)。
    rx_noise=False なら、受信側MMSE重みの計算に入れている雑音(H_effへの1.778e-6)を0にする。"""
    sys.path.insert(0, str(REPO))
    src = CPU_SCRIPT.read_text(encoding="utf-8")
    if not rx_noise:
        assert "np.random.normal(0, 1.778e-6, (U, Ly))" in src
        src = src.replace("np.random.normal(0, 1.778e-6, (U, Ly))", "np.random.normal(0, 0.0, (U, Ly))")
    cut = src.index("results = sweep_capacity_vs_d_mc(")
    ns = {"__name__": "cpu_ref", "__file__": str(CPU_SCRIPT)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src[:cut], "cpu_ref", "exec"), ns)
    return ns


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def eig_and_ratio(cpu, H):
    """サブキャリアごとの固有値・水注水の電力比率をサブキャリア平均 (U,) で返す。"""
    K = H.shape[2]
    eig_sum, p_sum = np.zeros(U), np.zeros(U)
    for k in range(K):
        ev, _ = cpu["calc_eigval"](H[:, :, k])
        ev = ev[:U]
        eig_sum[:len(ev)] += ev
        p, _ = cpu["water_filling_ratio"](ev, Pt_mW, P_noise_mW)
        p_sum[:len(p)] += p
    return eig_sum / K, p_sum / K


def with_est_sigma(cpu, sigma, fn):
    """CPU版の推定雑音(noise_dash_K_10)の標準偏差(実部・虚部それぞれ)を一時的に sigma に差し替えて fn() を実行。"""
    ch = cpu["channel"]
    orig = ch.noise_dash_K_10
    ch.noise_dash_K_10 = lambda U_, V_: (np.random.normal(0, sigma, (U_, V_, 2000, 10))
                                         + 1j * np.random.normal(0, sigma, (U_, V_, 2000, 10)))
    try:
        return fn()
    finally:
        ch.noise_dash_K_10 = orig


def run_case(cpu, cpu0, Base, trial, d, Ssub):
    """CPU版の再計算。SP_power を元のまま('orig')/修正版('fix')で、真のチャネルの容量・固有値を出す。
    推定チャネルの容量損失は修正版のチャネルで、受信側雑音なし(cpu0)の容量どうしの差として出す。"""
    out = {}
    for mode, fixed in (("orig", False), ("fix", True)):
        with sp_power_mode(cpu, fixed):
            np.random.seed(base_seed + trial)
            sd = cpu["setting_NYUSIM_synario"](Base[trial], d)
            H, _, ba = quiet(cpu["simulation_core"], "InH", 16, cpu["lam"], d, Pu_dBm, Ssub, sd, "T")
        out[mode, "H"], out[mode, "ba"] = H, ba
        for tag, m_ in (("rxon", cpu), ("rxoff", cpu0)):
            np.random.seed(base_seed + trial)
            C, Ly, _ = quiet(m_["calc_channel_capacity"], H, H, Pt_mW, P_noise_mW)
            np.random.seed(base_seed + trial)
            C1, _, _ = quiet(m_["calc_channel_capacity_SingleLayer"], H, H, Pt_mW, P_noise_mW)
            out[mode, "C", tag], out[mode, "C1", tag], out[mode, "Ly", tag] = C, C1, Ly
        out[mode, "eig"], out[mode, "pr"] = eig_and_ratio(cpu, H)
        out[mode, "Vact"] = H.shape[1]
    # 推定チャネル(修正版のチャネルで、CPU雑音/GPU雑音の両方)
    with sp_power_mode(cpu, True):
        for tag, sigma in (("cpuσ", SIGMA_CPU), ("gpuσ", SIGMA_GPU)):
            for use_H in ("E_w", "E_wo"):
                def _go():
                    np.random.seed(base_seed + trial)
                    sd = cpu["setting_NYUSIM_synario"](Base[trial], d)
                    return quiet(cpu["simulation_core"], "InH", 16, cpu["lam"], d, Pu_dBm, Ssub, sd, use_H)
                H, Hest, _ = with_est_sigma(cpu, sigma, _go)
                np.random.seed(base_seed + trial)
                out[tag, use_H] = quiet(cpu0["calc_channel_capacity"], Hest, H, Pt_mW, P_noise_mW)[0]
    return out


def main(npz_path, n_cases=6, seed=0):
    g = np.load(npz_path, allow_pickle=True)
    cap, ly, eig, pr = g["capacity"], g["layers"], g["eigenvalues"], g["power_ratio"]
    act, bpa, bpe = g["active_v"], g["beam_pa"], g["beam_pe"]
    dv, sv = list(g["d"]), list(g["Ssub"])
    n_trials = cap.shape[2]

    rng = np.random.default_rng(seed)
    cases = [(int(rng.integers(n_trials)), int(rng.choice(dv)), int(rng.choice(sv))) for _ in range(n_cases)]

    cpu = load_cpu_module(rx_noise=True)
    cpu0 = load_cpu_module(rx_noise=False)
    Base = np.load(BASE_FILE, allow_pickle=True)
    np.set_printoptions(precision=3, linewidth=200, suppress=False)

    rows = []
    for trial, d, S in cases:
        di, si = dv.index(d), sv.index(S)
        t0 = time.time()
        r = run_case(cpu, cpu0, Base, trial, d, S)
        gc, gc1 = cap[di, si, trial, 0, 0], cap[di, si, trial, 0, 1]
        g_loss_avg = gc - cap[di, si, trial, 1, 0]
        g_loss_single = gc - cap[di, si, trial, 2, 0]
        print(f"=== trial={trial} d={d}m Ssub={S}λ  (CPU {time.time() - t0:.0f}s) ===")
        print(f"  容量(多重) GPU {gc:8.3f} | CPU元のまま {r['orig','C','rxoff']:8.3f} (差 {gc - r['orig','C','rxoff']:+.3f})"
              f" | CPU SP_power修正 {r['fix','C','rxoff']:8.3f} (差 {gc - r['fix','C','rxoff']:+.3f})   ※受信側雑音なし")
        print(f"  容量(多重) 〃       | CPU元のまま[rx雑音あり] {r['orig','C','rxon']:8.3f} | CPU修正[rx雑音あり] {r['fix','C','rxon']:8.3f}")
        print(f"  容量(単一) GPU {gc1:8.3f} | CPU元のまま {r['orig','C1','rxoff']:8.3f} (差 {gc1 - r['orig','C1','rxoff']:+.3f})"
              f" | CPU SP_power修正 {r['fix','C1','rxoff']:8.3f} (差 {gc1 - r['fix','C1','rxoff']:+.3f})")
        print(f"  レイヤ数 GPU {ly[di, si, trial, 0]:.3f} | CPU元 {r['orig','Ly','rxoff']:.3f} | CPU修正 {r['fix','Ly','rxoff']:.3f}"
              f"   有効V' GPU {int(act[di, si, trial])} CPU {r['fix','Vact']}")
        for tag in ("cpuσ", "gpuσ"):
            la = r["fix", "C", "rxoff"] - r[tag, "E_w"]
            ls = r["fix", "C", "rxoff"] - r[tag, "E_wo"]
            print(f"  推定の容量損失 CPU修正(推定雑音={tag}): 10回平均+デノイズ {la:+.4f}   単発+デノイズ {ls:+.4f}")
        print(f"  推定の容量損失 GPU                  : 10回平均+デノイズ {g_loss_avg:+.4f}   単発+デノイズ {g_loss_single:+.4f}")
        print(f"  固有値 GPU     {eig[di, si, trial, 0]}")
        print(f"  固有値 CPU修正 {r['fix','eig']}")
        print(f"  固有値 CPU元   {r['orig','eig']}")
        print(f"  電力比 GPU     {pr[di, si, trial, 0]}")
        print(f"  電力比 CPU修正 {r['fix','pr']}")
        ba = r["fix", "ba"]
        cpu_pa = [int(ba[f, 0, a]) for f in range(3) for a in range(4)]
        cpu_pe = [int(ba[f, 1, a]) for f in range(3) for a in range(4)]
        print(f"  ビームpa GPU {bpa[di, si, trial].tolist()}   CPU {cpu_pa}")
        print(f"  ビームpe GPU {bpe[di, si, trial].tolist()}   CPU {cpu_pe}", flush=True)
        rows.append((trial, d, S, gc, r["orig", "C", "rxoff"], r["fix", "C", "rxoff"],
                     gc1, r["orig", "C1", "rxoff"], r["fix", "C1", "rxoff"]))

    print("\n--- まとめ (GPU - CPU, 受信側雑音なし) ---")
    for trial, d, S, gc, co, cf, gc1, c1o, c1f in rows:
        print(f"trial={trial:4d} d={d:2d} S={S:3d}  多重: 元のまま {gc - co:+.4f} / SP_power修正 {gc - cf:+.4f}"
              f"   単一: 元のまま {gc1 - c1o:+.4f} / SP_power修正 {gc1 - c1f:+.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python 261006_compare_gpu_cpu.py <Results_xxx.npz> [ケース数] [seed]")
    main(sys.argv[1],
         int(sys.argv[2]) if len(sys.argv) > 2 else 6,
         int(sys.argv[3]) if len(sys.argv) > 3 else 0)
