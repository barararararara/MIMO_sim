import numpy as np

eigvals = np.load("C:/Users/tai20/Downloads/sim_data/Data/Mirror/InH/Ssub=10lam/Channel_Capacity/Near/d=5/eigvals_Htru.npy")
print(np.shape(eigvals))
beamallocation = np.load("C:/Users/tai20/Downloads/sim_data/Data/Mirror/InH/Ssub=0lam/Beamallocation/Near/d=5.npy", allow_pickle=True)
print(np.shape(beamallocation))
print(beamallocation[:20,:,11,:,:])
# channel_type = "InH"
# NF_setting = "Near"
# d=5

# Scatter1 = np.load(f'C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Scatter1/d={d}/Scatter1.npy', allow_pickle=True)
# P_sub_dash_dBm = np.load(f'C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Power/d={d}/Power_Near.npy', allow_pickle=True)
# P_far_sub_dash_dBm = np.load(f'C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Power/d={d}/Power_Far.npy', allow_pickle=True)
# Beam_allocation = np.load(f'C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Beamallocation/{NF_setting}/d={d}.npy', allow_pickle=True)
# Channel_Capacity = np.load(f'C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Channel_Capacity/{NF_setting}/d={d}/Capacity_Htru.npy', allow_pickle=True)
# print(np.shape(Beam_allocation))

# pe=24
# for channel_number in range(1):
#     print(f'Channel {channel_number}:')
#     Allocated_idx = np.argmax(Channel_Capacity[channel_number], axis=0)
#     print(Beam_allocation[channel_number, :, Allocated_idx, :, :])
#     print('\n')
#     theta_rad = np.radians(Scatter1[channel_number]['theta_deg'])
#     phi_rad = np.radians(Scatter1[channel_number]['phi_deg'])
#     channel.plot_graph_0(d, 64, P_sub_dash_dBm[channel_number], P_far_sub_dash_dBm[channel_number], theta_rad, phi_rad, pe)  
#     channel.plot_graph_120(d, 64, P_sub_dash_dBm[channel_number], P_far_sub_dash_dBm[channel_number], theta_rad, phi_rad, pe)
#     channel.plot_graph_240(d, 64, P_sub_dash_dBm[channel_number], P_far_sub_dash_dBm[channel_number], theta_rad, phi_rad, pe)
    
# import math 

# pa= 58
# pe= 24

# import Channel_functions as cf
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# アンテナの配置確認コード
# V=12
# Q=64
# lam = (3.0 * 1e8) / (142 * 1e9)
# S_sub=100*lam

# coords = cf.calc_anntena_xyz_wide(lam, V, Q, S_sub)

# # プロット
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # V, Q, Q の点を展開して散布図にする
# for v in range(V):
#     x = coords[v, :, :, 0].flatten()
#     y = coords[v, :, :, 1].flatten()
#     z = coords[v, :, :, 2].flatten()
#     ax.scatter(x, y, z, label=f'Subarray {v}', s=1)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# ax.set_title("Antenna Array Coordinates")

# plt.show()

# d=30の値エラーの原因特定
# channel_type = "InF"
# NF_setting = "Near"
# d=30

# load_dir_Mirror = f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Channel_Capacity/{NF_setting}/d={d}"
# H_near_est_Mirror = np.load(f"{load_dir_Mirror}/Capacity_Hest.npy", allow_pickle=True)

# print(np.min(H_near_est_Mirror))

# Pi_each_career = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Scatter1/d={d}/Flattened/Pi_each_career_flat.npy", allow_pickle=True)

# print(Pi_each_career)

# d=25
# Pi_each_career = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Scatter1/d={d}/Flattened/Pi_each_career_flat.npy", allow_pickle=True)
# print(Pi_each_career)

# d=30
# NF_setting = "Far"
# channel_type = "InF"
# beam_allocation = np.load(f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Beamallocation/{NF_setting}/d={d}.npy", allow_pickle=True)

# print(beam_allocation[0])