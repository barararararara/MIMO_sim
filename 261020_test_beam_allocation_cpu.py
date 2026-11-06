# 261020_test_beam_allocation_cpu.py
# ビーム割当法の違いが、固有値分布・レイヤ数・チャネル容量に与える影響をCPU版で調べる。
#   current: 各サブアレーが、自分の最大受信電力のビーム1本を使う(現在のプログラム)
#   H(M)   : 濱田さんの修論の方式(3.1.2節)。しきい値を超えた主ビーム(局所最大)を受信電力の降順に並べ、上位M本を採用し、
#            各面で、その面の採用ビームを降順に、面内のサブアレーへ循環的に割り当てる(表3.1)。
#            Mは最大ビーム数。ビームが1本も採用されなかった面のサブアレーには、ビームを割り当てない。
# 比べる量(真のチャネル): 有効サブアレー数V'、容量(空間多重/単一レイヤ)、レイヤ数、固有値(サブキャリア平均)。
# 使い方: python 261020_test_beam_allocation_cpu.py [ケース数] [seed]

import sys
import time
import importlib.util
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("cmp", REPO / "261006_compare_gpu_cpu.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

Pt_mW, P_noise_mW, base_seed = 0.5, 6.31e-12, 9
THRESHOLD_DBM = -73   # CPU版と同じビーム割当のしきい値


def make_alloc_H(M):
    """濱田さん流のビーム割当関数(CPU版の Beam_allocation_method_2 と同じ入出力)を返す。"""
    def alloc(Power, threshold, testch_num=0):
        cand = []
        for face in range(3):
            Pf = Power[face * 4:(face + 1) * 4].max(axis=0)      # その面の4サブアレーのうち最大の受信電力 (Q,Q)
            # ビーム候補は、(pa,pe)平面で受信電力が局所最大(周囲8近傍以上)のビームだけにする。
            # DFTビームのサイドローブ・隣接ビームは、同じ波の別ビームなので、候補に入れない(個々の波の主ビームを候補とする)。
            pad = np.pad(Pf, 1, constant_values=-np.inf)
            nbr = np.max([pad[i:i + Pf.shape[0], j:j + Pf.shape[1]] for i in range(3) for j in range(3) if (i, j) != (1, 1)], axis=0)
            pa, pe = np.where((Pf >= threshold) & (Pf >= nbr))
            cand += [(Pf[a, e], face, int(a), int(e)) for a, e in zip(pa, pe)]
        cand.sort(key=lambda c: -c[0])
        adopted = cand[:M]
        ba = np.full((3, 2, 4), -1, dtype=int)
        for face in range(3):
            beams = [(a, e) for _, f, a, e in adopted if f == face]   # 受信電力の降順
            if not beams:
                continue
            for k in range(4):
                a, e = beams[k % len(beams)]
                ba[face, 0, k], ba[face, 1, k] = a, e
        return ba, None
    return alloc


def evaluate(cpu0, cpu, Base, trial, d, S, alloc):
    orig = cpu["Beam_allocation_method_2"]
    if alloc is not None:
        cpu["Beam_allocation_method_2"] = alloc
    try:
        with cmp.sp_power_mode(cpu, True):
            np.random.seed(base_seed + trial)
            sd = cpu["setting_NYUSIM_synario"](Base[trial], d)
            H, _, ba = cmp.quiet(cpu["simulation_core"], "InH", 16, cpu["lam"], d, 30, S, sd, "T")
    finally:
        cpu["Beam_allocation_method_2"] = orig
    np.random.seed(base_seed + trial)
    C, Ly, _ = cmp.quiet(cpu0["calc_channel_capacity"], H, H, Pt_mW, P_noise_mW)
    C1, _, _ = cmp.quiet(cpu0["calc_channel_capacity_SingleLayer"], H, H, Pt_mW, P_noise_mW)
    eig, pr = cmp.eig_and_ratio(cpu, H)
    nb = len({(int(ba[f, 0, a]), int(ba[f, 1, a])) for f in range(3) for a in range(4) if ba[f, 0, a] >= 0})
    return dict(V=H.shape[1], nbeam=nb, C=C, C1=C1, Ly=Ly, eig=eig)


def main(n_cases=8, seed=0):
    cpu = cmp.load_cpu_module(rx_noise=True)
    cpu0 = cmp.load_cpu_module(rx_noise=False)
    Base = np.load(cmp.BASE_FILE, allow_pickle=True)
    rng = np.random.default_rng(seed)
    cases = [(int(rng.integers(1000)), int(rng.choice([5, 25])), 50) for _ in range(n_cases)]
    variants = [("current", None)] + [(f"H(M={M})", make_alloc_H(M)) for M in (2, 3, 4, 6, 8, 12)]
    store = {v[0]: [] for v in variants}
    np.set_printoptions(precision=1, linewidth=200)
    for trial, d, S in cases:
        t0 = time.time()
        print(f"\n=== trial={trial} d={d}m Ssub={S}λ ===", flush=True)
        for name, alloc in variants:
            r = evaluate(cpu0, cpu, Base, trial, d, S, alloc)
            store[name].append(r)
            e_db = 10 * np.log10(np.maximum(r["eig"], 1e-300))
            print(f"  {name:9s}: V'={r['V']:2d} 異なるビーム{r['nbeam']:2d}本 | 多重 {r['C']:6.2f} 単一 {r['C1']:6.2f} Ly {r['Ly']:.2f} | λ(dB) {e_db[:6]}", flush=True)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
    print("\n--- まとめ(全ケース平均、真のチャネル) ---")
    for name, _ in variants:
        R = store[name]
        m = lambda k: np.mean([x[k] for x in R])
        eig_db = np.mean([10 * np.log10(np.maximum(x["eig"], 1e-300)) for x in R], axis=0)
        print(f"{name:9s}: V' {m('V'):4.1f} 異なるビーム {m('nbeam'):4.1f}本 | 多重 {m('C'):6.2f} 単一 {m('C1'):6.2f} Ly {m('Ly'):.2f} | λ1-6(dB) {np.round(eig_db[:6], 1)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
