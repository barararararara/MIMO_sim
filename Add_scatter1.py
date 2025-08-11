import numpy as np
import math
from multiprocessing import Pool
import Channel_functions as channel
import os

# 設定
channel_type = "InF"
NF_setting = "Near"
d= 5

# 必要なパラメータ設定
lam = (3.0 * 1e8) / (142 * 1e9)
Pt = 10
Pu_dBm_per_carrer = -23
g_dB = 0
Q = 64
V = 12


# ベースのチャネル情報を取得
Base = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/{channel_type}/Base.npy", allow_pickle=True)


def process_single_link(channel_base,channel_type,d):
    L_AOD = channel_base['L_AOD']
    L_AOA = channel_base['L_AOA']
    chi = channel_base['chi']
    N = channel_base['N']
    M = channel_base['M']
    Z = channel_base['Z']
    U = channel_base['U']
    rho = channel_base['rho']
    tau = channel_base['tau']
    beta = channel_base['beta']
    phi_mean_AOD = channel_base['phi_mean_AOD']
    varphi_mean_AOA = channel_base['varphi_mean_AOA']
    theta_mean_AOD = channel_base['theta_mean_AOD']
    eta_mean_AOA = channel_base['eta_mean_AOA']
    phi_deg = channel_base['phi_deg']
    varphi_deg = channel_base['varphi_deg']
    theta_ND_deg = channel_base['theta_ND_deg']
    eta_ND_deg = channel_base['eta_ND_deg']

    eta_dir = np.degrees(np.arcsin(1 / d))
    theta_dir = -eta_dir
    theta_0_0 = theta_ND_deg[0][0] if len(theta_ND_deg[0]) > 0 else theta_ND_deg[0]
    eta_0_0 = eta_ND_deg[0][0] if len(eta_ND_deg[0]) > 0 else eta_ND_deg[0]

    Delta_EOD = theta_dir - theta_0_0
    Delta_EOA = eta_dir - eta_0_0

    theta_deg = [theta_ND_deg[n] + Delta_EOD for n in range(N)]
    eta_deg = [eta_ND_deg[n] + Delta_EOA for n in range(N)]

    Pr = channel.calc_Pr(lam, d, chi, Pt, g_dB, channel_type, do=1)
    Pr_each_career = channel.calc_Pr_each_career(lam, d, chi, Pu_dBm_per_carrer, g_dB, channel_type, do=1)
    t_nm = channel.abs_timedelays(d, rho, tau, N, M)
    P = channel.cluster_power(Pr, N, tau, Z, channel_type)
    Pi = channel.SP_power(N, M, P, rho, U, channel_type)
    P_each_career = channel.cluster_Power_each_career(Pr_each_career, Z, N, tau, channel_type)
    Pi_each_career = channel.SP_Power_each_career(N, M, P_each_career, rho, U, channel_type)
    r_dash = lam * (Q + 1) * V / (6 * math.sqrt(3))
    r, R = channel.scatter1_distance(d, Q, V, N, M, rho, tau)
    phi_rad = [np.radians(phi_deg[n]) for n in range(N)]
    theta_rad = [np.radians(theta_deg[n]) for n in range(N)]
    sca1_xyz_co = channel.scatter1_xyz_cordinate(N, M, r, phi_rad, theta_rad)
    subarray_v_qy_qz = channel.calc_anntena_xyz(lam, V, Q)
    dis_sca1_to_anntena = channel.distance_scatter1_to_eachanntena(sca1_xyz_co, subarray_v_qy_qz, N, M, V, Q)

    return {
        "L_AOD": L_AOD, "L_AOA": L_AOA, "chi": chi, "N": N, "M": M, "Z": Z, "U": U,
        "rho": rho, "tau": tau, "beta": beta, "phi_mean_AOD": phi_mean_AOD, "varphi_mean_AOA": varphi_mean_AOA,
        "theta_mean_AOD": theta_mean_AOD, "eta_mean_AOA": eta_mean_AOA,
        "phi_deg": phi_deg, "varphi_deg": varphi_deg, "theta_deg": theta_deg, "eta_deg": eta_deg,
        "Pr": Pr, "Pr_each_career": Pr_each_career, "t_nm": t_nm, "P": P, "Pi": Pi,
        "P_each_career": P_each_career, "Pi_each_career": Pi_each_career,
        "r_dash": r_dash, "r": r, "R": R, "sca1_xyz_co": sca1_xyz_co,
        "subarray_v_qy_qz": subarray_v_qy_qz, "dis_sca1_to_anntena": dis_sca1_to_anntena
    }

save_dir = f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Scatter1/d={d}"
os.makedirs(save_dir, exist_ok=True)
if __name__ == '__main__':
    with Pool() as pool:
        args = [(channel_base, channel_type, d) for channel_base in Base.tolist()]
        result = pool.starmap(process_single_link, args)

    np.save(f"{save_dir}/Scatter1.npy", result)