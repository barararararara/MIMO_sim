# 261019_test_pu_cpu.py
# CPU版(# 260319_RCS石垣島.py)で、UE送信電力(パイロット)Puを変えたときの特性を調べる。
#   A: ビーム割当も推定雑音もPu=30dBm(現在の設定)
#   B: ビーム割当も推定雑音もPu=10dBm(濱田さんの修論の表3.3)
#   C: ビーム割当はPu=30dBm、推定雑音だけPu=10dBm相当(ビーム割当の精度と推定雑音の影響を切り分ける)
# 各ケースで、有効サブアレー数V'、真のチャネルの容量、W/ NS(10回平均+遅延ゼロマスキング)・Mask only(単発+ゼロマスキング)・
# W/o NS(単発、雑音低減なし)の容量損失とレイヤ数を出す。
# CPU版の推定雑音(noise_dash_K_10)にはPuでの正規化がないため、GPU版と同じ 1.778e-6/sqrt(Pu_mW/K) に差し替える。
# CPU版 SP_power の実装ミス(サブパス電力の配分)は、修正版に差し替えて計算する。
# 使い方: python 261019_test_pu_cpu.py [ケース数] [seed]

import sys
import io
import time
import importlib.util
import contextlib
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("cmp", REPO / "261006_compare_gpu_cpu.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

K = 2000
Pt_mW, P_noise_mW, base_seed = 0.5, 6.31e-12, 9


def sigma_est(pu_dbm):
    """推定雑音の標準偏差(実部・虚部それぞれ) = 1.778e-6 / sqrt(Pu_mW/K)。"""
    return 1.778e-6 / np.sqrt(10 ** (pu_dbm / 10) / K)


def load_cpu():
    """CPU版を読み込み、雑音低減前(位相補正後)の推定チャネルも返すよう、simulation_coreの戻り値を1つ増やす。"""
    src = cmp.CPU_SCRIPT.read_text(encoding="utf-8")
    key = "    h_uvk *= np.exp(1j * 2 * np.pi * f_GHz[None, None, :]*t_nm[0, 0])\n"
    assert key in src and "return h_uvk_true, H_w_denoised, ba" in src
    src = src.replace(key, key + "    h_uvk_raw = h_uvk.copy()\n")
    src = src.replace("return h_uvk_true, H_w_denoised, ba", "return h_uvk_true, H_w_denoised, ba, h_uvk_raw")
    src = src.replace("np.random.normal(0, 1.778e-6, (U, Ly))", "np.random.normal(0, 0.0, (U, Ly))")  # 受信側雑音なし(比較を簡単にする)
    cut = src.index("results = sweep_capacity_vs_d_mc(")
    ns = {"__name__": "cpu_ref", "__file__": str(cmp.CPU_SCRIPT)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src[:cut], "cpu_ref", "exec"), ns)
    return ns


def run_variant(cpu, Base, trial, d, S, pu_beam, sig_dbm):
    """ビーム割当にpu_beam、推定雑音にsig_dbm(Pu相当)を使って、真/W/ NS/Mask only/W/o NS の容量とレイヤ数を返す。"""
    out = {}
    sd = None
    res = {}
    for use_H in ("E_w", "E_wo"):
        def _go():
            np.random.seed(base_seed + trial)
            sd_ = cpu["setting_NYUSIM_synario"](Base[trial], d)
            return cmp.quiet(cpu["simulation_core"], "InH", 16, cpu["lam"], d, pu_beam, S, sd_, use_H)
        with cmp.sp_power_mode(cpu, True):
            H, Hest, ba, Hraw = cmp.with_est_sigma(cpu, sigma_est(sig_dbm), _go)
        res[use_H] = (Hest, Hraw)
    cap = lambda Hu: cmp.quiet(cpu["calc_channel_capacity"], Hu, H, Pt_mW, P_noise_mW)
    Ct, Lt, _ = cap(H)
    Cw, Lw, _ = cap(res["E_w"][0])          # W/ NS: 10回平均 + ゼロマスキング
    Cm, Lm, _ = cap(res["E_wo"][0])         # Mask only: 単発 + ゼロマスキング
    Cr, Lr, _ = cap(res["E_wo"][1])         # W/o NS: 単発、雑音低減なし
    return dict(V=H.shape[1], ba=ba, Ct=Ct, Lt=Lt, loss_w=Ct - Cw, loss_m=Ct - Cm, loss_r=Ct - Cr, Lw=Lw, Lr=Lr)


def main(n_cases=12, seed=0):
    cpu = load_cpu()
    Base = np.load(cmp.BASE_FILE, allow_pickle=True)
    rng = np.random.default_rng(seed)
    cases = [(int(rng.integers(1000)), int(rng.choice([5, 25])), 50) for _ in range(n_cases)]
    variants = [("A: ビーム30/推定30", 30, 30), ("B: ビーム10/推定10", 10, 10), ("C: ビーム30/推定10", 30, 10)]
    rows = {v[0]: [] for v in variants}
    print(f"推定雑音(実部・虚部の標準偏差): Pu=30dBm {sigma_est(30):.3e}, Pu=10dBm {sigma_est(10):.3e}", flush=True)
    for trial, d, S in cases:
        t0 = time.time()
        print(f"\n=== trial={trial} d={d}m Ssub={S}λ ===", flush=True)
        ba_ref = None
        for name, pu_b, pu_s in variants:
            # AとCはビーム割当が同じ(同じ真のチャネル)なので、雑音だけが違う
            r = run_variant(cpu, Base, trial, d, S, pu_b, pu_s)
            rows[name].append(r)
            ba = r["ba"]
            pa = [int(ba[f, 0, a]) for f in range(3) for a in range(4)]
            if ba_ref is None:
                ba_ref = pa
            same = sum(1 for x, y in zip(pa, ba_ref) if x == y and x not in (-1, -100))
            print(f"  {name}: V'={r['V']:2d}(ビームがAと同じ{same:2d}/12) | 真の容量 {r['Ct']:6.2f} Ly {r['Lt']:.2f} | 損失 W/ NS {r['loss_w']:+.3f} "
                  f"Mask only {r['loss_m']:+.3f} W/o NS {r['loss_r']:+.3f} | Ly: W/ NS {r['Lw']:.2f} W/o NS {r['Lr']:.2f}", flush=True)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
    print("\n--- まとめ(全ケース平均) ---")
    for name, _, _ in variants:
        R = rows[name]
        m = lambda k: np.mean([x[k] for x in R])
        print(f"{name}: V' {m('V'):.1f} | 真の容量 {m('Ct'):.2f} | 損失 W/ NS {m('loss_w'):.3f} Mask only {m('loss_m'):.3f} W/o NS {m('loss_r'):.3f} | "
              f"Ly 真 {m('Lt'):.2f} W/ NS {m('Lw'):.2f} W/o NS {m('Lr'):.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
