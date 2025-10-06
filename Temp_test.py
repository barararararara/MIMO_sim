# import numpy as np
# import Channel_functions as channel

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

import Channel_functions as cf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

V=12
Q=64
lam = (3.0 * 1e8) / (142 * 1e9)
S_sub=10*lam

coords = cf.calc_anntena_xyz_wide(lam, V, Q, S_sub)

# プロット
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# V, Q, Q の点を展開して散布図にする
for v in range(V):
    x = coords[v, :, :, 0].flatten()
    y = coords[v, :, :, 1].flatten()
    z = coords[v, :, :, 2].flatten()
    ax.scatter(x, y, z, label=f'Subarray {v}', s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title("Antenna Array Coordinates")

plt.show()

