import numpy as np
import Channel_functions as channel
import multiprocessing as mp
import time

# 設定
channel_type = "InH"
NF_setting = "Near_Unif"
V = 12
Q = 64
lam = (3*1e8) / (142*1e9)
Pt = 10
Pu = -23
Pu_dBm_per_carrer = -23.0
g_dB = 0
f = np.linspace(141.50025*1e9, 142.49975*1e9, 2000) #2000個の周波数


# 毎回求める必要のないものゾーン
DFT_weights = channel.DFT_weight_calc(Q)
zeta = channel.define_zeta(V)


def Power_calculation(Channel_scatter1, channel_type, d, start_idx, end_idx):
    P_sub_dash_dBm_list = []
    P_sub_far_dash_dBm_list = []

    for number in range(start_idx, end_idx):
        data = Channel_scatter1[number]
        N, M = data['N'], data['M']
        r, r_dash = data['r'], data['r_dash']
        t_nm = data['t_nm']
        Pi_each_career = data['Pi_each_career']
        beta = data['beta']
        phi_deg, varphi_deg = data['phi_deg'], data['varphi_deg']
        theta_deg, eta_deg = data['theta_deg'], data['eta_deg']
        dis_sca1_to_anntena = data['dis_sca1_to_anntena']
        
        theta_rad = np.zeros((N,max(M)))
        phi_rad = np.zeros((N,max(M)))
        varphi_rad = np.zeros((N,max(M)))
        eta_rad = np.zeros((N,max(M)))

        for n in range(N):
            for m in  range(M[n]):
                theta_rad[n][m] = np.radians(theta_deg[n][m])
                phi_rad[n][m] = np.radians(phi_deg[n][m])
                varphi_rad[n][m] = np.radians(varphi_deg[n][m])
                eta_rad[n][m] = np.radians(eta_deg[n][m])
                
        b_varphi_eta = channel.define_b_verphi_eta(N, M, eta_rad)
        a_phi_theta = channel.define_a(V, N, M, phi_rad, theta_rad)

        Amp_per_career = np.sqrt(Pi_each_career)
        complex_Amp_at_O = channel.calc_complex_Amp_at_O(f, V, N, M, Amp_per_career, beta, t_nm, b_varphi_eta)
        complex_Amp_at_scatter1 = channel.complex_Amp_at_scatter1(N, M, f, r, complex_Amp_at_O)
        complex_Amp_at_each_antena = channel.complex_Amp_at_each_antena(f, V, Q, N, M, dis_sca1_to_anntena, complex_Amp_at_scatter1, a_phi_theta)
        g_DD = channel.g_DD_pa_pe(N, M, phi_rad, theta_rad, zeta, Q, V)
        n_k_v = channel.noise_n_k_v(V)

        K = len(f)
        P_sub_dash_dBm = channel.near_Power_inc_noise(V, Q, K, DFT_weights, complex_Amp_at_each_antena, n_k_v)
        P_far_sub_dash_dBm = channel.far_Power_inc_noise(V,Q,f,complex_Amp_at_O,g_DD,n_k_v,a_phi_theta)
        
        P_sub_dash_dBm_list.append(P_sub_dash_dBm)
        P_sub_far_dash_dBm_list.append(P_far_sub_dash_dBm)

    log_msg = f'#{number} done (d={d})'
    print(log_msg)
    np.save(f"C:/Users/tai20/Downloads/研究データ/Data/{channel_type}/{NF_setting}/d={d}", P_sub_dash_dBm_list)
    print(f'{start_idx} to {end_idx} has done')

d=30
Channel_Scatter1 = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/{channel_type}/{NF_setting}/d={d}/Scatter1.npy", allow_pickle=True)
for i in range(7,10): 
    Power_calculation(Channel_Scatter1, 'InH', d, i*100, (i+1)*100)


