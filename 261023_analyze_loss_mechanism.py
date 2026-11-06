# 261023_analyze_loss_mechanism.py
# Pu=30 dBm の W/o NS で、「雑音の層に電力が配分される(レイヤ数が増える)のに、容量があまり減らない」理由を、
# CPU版と同じ手順(サブキャリアごとのSVD・注水・MMSE)で、要因ごとに分解して調べる。
#   C_true : 真のチャネルの固有ベクトル・注水結果(上限)
#   C_est  : 推定チャネル(W/o NS)の固有ベクトル・注水結果(= W/o NS の容量、損失の基準)
#   C_trunc: 推定の固有ベクトルのうち、真のレイヤ数Ltまでだけ使い、全電力をそこに配分(雑音の層を使わない)
#   C_dir  : 推定の固有ベクトル(上位Lt本) + 真の注水結果(固有ベクトルのずれだけの影響)
#   C_pow  : 真の固有ベクトル + 推定の注水結果(電力配分のずれだけの影響)
# 損失の分解: (C_true-C_trunc)=強い層の方向誤差による損失, (C_trunc-C_est)=雑音の層に電力を取られた損失
# 使い方: python 261023_analyze_loss_mechanism.py [ケース数] [seed] [Pu_dBm]

import sys
import time
import importlib.util
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, REPO / fn)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t = _load("tpu", "261019_test_pu_cpu.py")
cmp = t.cmp
Pt_mW, P_noise_mW, base_seed, K = t.Pt_mW, t.P_noise_mW, t.base_seed, t.K


def cap(Htru, Te, p, Pt=Pt_mW, Pn=P_noise_mW):
    """CPU版 calc_channel_capacity と同じMMSE・容量計算(受信側雑音は入れない)。Te:(V,L) p:(L,) 電力比率"""
    Ly = Te.shape[1]
    A = np.sqrt(Pt) * np.diag(np.sqrt(p))
    H_eff = Htru @ Te
    W = (np.linalg.inv(H_eff.conj().T @ H_eff + (Ly / (Pt / Pn)) * np.eye(Ly)) @ H_eff.conj().T).T
    B = W.T @ Htru @ Te @ A
    S = np.abs(np.diag(B)) ** 2
    I = np.sum(np.abs(B) ** 2, axis=1) - S
    N = np.diag(Pn * W.T @ W.T.conj().T).real
    return float(np.sum(np.log2(S / (I + N) + 1)))


def analyze(cpu, H, Hest, ks):
    """サブキャリア ks について、各量を平均して返す。"""
    eig, wf = cpu["calc_eigval"], cpu["water_filling_ratio"]
    acc = {k: 0.0 for k in ("Ct", "Ce", "Ctr", "Cdir", "Cpow", "Lt", "Le", "f", "pred", "Cred", "xs", "xn", "pn")}
    n = 0
    pe_acc = np.zeros(8)
    pt_acc = np.zeros(8)
    for k in ks:
        Ht, He = H[:, :, k], Hest[:, :, k]
        lt, Vt = eig(Ht)
        le, Ve = eig(He)
        pt, Lt = wf(lt, Pt_mW, P_noise_mW)
        pe, Le = wf(le, Pt_mW, P_noise_mW)
        if Lt == 0 or Le == 0:
            continue
        pt, pe = pt[:Lt], pe[:Le]
        Ct = cap(Ht, Vt[:, :Lt], pt)
        Ce = cap(Ht, Ve[:, :Le], pe)
        Lm = min(Lt, Le)
        # 雑音の層を使わない: 上位Lt本だけ、電力を比率のまま(推定の注水の比率を正規化)配分
        pp = pe[:Lt] / pe[:Lt].sum()
        Ctr = cap(Ht, Ve[:, :Lt], pp)
        Cdir = cap(Ht, Ve[:, :Lt], pt)
        Cpow = cap(Ht, Vt[:, :Le], pe)
        Cred = cap(Ht, Ve[:, :Lt], pe[:Lt])   # 推定の向き + 強い層の電力だけ(雑音の層の電力は捨てる=電力が1-fに減るだけ)
        # 注水レベル mu: 有効な層は p*Pt = mu - Pn/λ。しきい値の固有値は Pn/mu。
        mu = pe[Le - 1] * Pt_mW + P_noise_mW / le[Le - 1]
        xs = np.mean(10 * np.log10(le[:Lt] * mu / P_noise_mW))                          # 強い層の(しきい値に対する)固有値[dB]
        xn = np.mean(10 * np.log10(le[Lt:Le] * mu / P_noise_mW)) if Le > Lt else 0.0    # 雑音の層の固有値[dB]
        pn = pe[Lt:].sum() / max(Le - Lt, 1)                                            # 雑音の層1本あたりの電力比率
        pe_acc[:len(pe)] += pe
        pt_acc[:len(pt)] += pt
        f = pe[Lt:].sum()   # 真のレイヤ数より後ろの層に配分された電力の比率
        pred = Lt * np.log2(1 / max(1 - f, 1e-12))   # 強い層が全て同じ割合で電力を失うとしたときの損失の見積り(高SNR近似)
        for key, v in zip(acc, (Ct, Ce, Ctr, Cdir, Cpow, Lt, Le, f, pred, Cred, xs, xn, pn)):
            acc[key] += v
        n += 1
    out = {k: v / n for k, v in acc.items()}
    out["pe"] = pe_acc / n
    out["pt"] = pt_acc / n
    return out


def main(n_cases=12, seed=0, pu=30):
    cpu = t.load_cpu()
    Base = np.load(cmp.BASE_FILE, allow_pickle=True)
    rng = np.random.default_rng(seed)
    cases = [(int(rng.integers(1000)), int(rng.choice([5, 25])), 50) for _ in range(n_cases)]
    ks = range(0, K, 10)     # 200サブキャリアを抽出(全体の傾向を見るには十分)
    rows = []
    print(f"Pu={pu} dBm の W/o NS(推定雑音 {t.sigma_est(pu):.3e})、サブキャリア{len(ks)}本の平均")
    print(f"{'trial':>5} {'d':>3} | {'V':>2} Lt  Le | {'C_true':>6} {'C_est':>6} {'損失':>5} | {'方向':>5} {'雑音層':>5} | {'f':>5} {'予測':>5} | {'C_dir損':>6} {'C_pow損':>6}")
    for trial, d, S in cases:
        def _go():
            np.random.seed(base_seed + trial)
            sd = cpu["setting_NYUSIM_synario"](Base[trial], d)
            return cmp.quiet(cpu["simulation_core"], "InH", 16, cpu["lam"], d, 30, S, sd, "E_wo")
        with cmp.sp_power_mode(cpu, True):
            H, Hest, ba, Hraw = cmp.with_est_sigma(cpu, t.sigma_est(pu), _go)
        r = analyze(cpu, H, Hraw, ks)
        r["V"] = H.shape[1]
        r["d"] = d
        rows.append(r)
        print(f"{trial:5d} {d:3d} | {r['V']:2d} {r['Lt']:.1f} {r['Le']:.1f} | {r['Ct']:6.2f} {r['Ce']:6.2f} {r['Ct']-r['Ce']:5.2f} | "
              f"{r['Ct']-r['Ctr']:5.2f} {r['Ctr']-r['Ce']:5.2f} | {r['f']:5.2f} {r['pred']:5.2f} | {r['Ct']-r['Cdir']:6.2f} {r['Ct']-r['Cpow']:6.2f}", flush=True)
    m = lambda k: np.mean([x[k] for x in rows])
    print("\n--- 全ケース平均 ---")
    print(f"真の容量 {m('Ct'):.2f}, W/o NS の容量 {m('Ce'):.2f}  → 損失 {m('Ct') - m('Ce'):.2f} bps/Hz")
    print(f"  内訳  強い層の方向誤差による損失(C_true-C_trunc): {m('Ct') - m('Ctr'):.2f}")
    print(f"        雑音の層に電力を取られた損失(C_trunc-C_est):  {m('Ctr') - m('Ce'):.2f}")
    print(f"  方向誤差だけ(推定の方向+真の電力配分)の損失: {m('Ct') - m('Cdir'):.2f}")
    print(f"  電力配分だけ(真の方向+推定の電力配分)の損失:   {m('Ct') - m('Cpow'):.2f}")
    print(f"真のレイヤ数 {m('Lt'):.2f}, 推定のレイヤ数 {m('Le'):.2f}, 真のレイヤ数より後ろへの電力配分 f={m('f'):.2f}")
    print(f"  f から見積もった損失 Lt*log2(1/(1-f)) = {m('pred'):.2f}")
    print(f"  推定の向き + 強い層の電力だけ(雑音の層の電力は捨てる)の損失: {m('Ct') - m('Cred'):.2f}  → 雑音の層が実際に加える損失: {m('Cred') - m('Ce'):.2f}")
    print(f"  しきい値(注水の水位)に対する固有値: 強い層 {m('xs'):.1f} dB, 雑音の層 {m('xn'):.1f} dB / 雑音の層1本あたりの電力 {m('pn'):.2f}")
    np.set_printoptions(precision=3, suppress=True)
    print("  レイヤごとの電力配分(全ケース平均) True:", np.mean([x["pt"] for x in rows], axis=0), " W/o NS:", np.mean([x["pe"] for x in rows], axis=0))
    print(f"  1レイヤあたりの容量(真) = {m('Ct') / m('Lt'):.1f} bps/Hz")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12,
         int(sys.argv[2]) if len(sys.argv) > 2 else 0,
         float(sys.argv[3]) if len(sys.argv) > 3 else 30)
