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
    
import math 

pa= 58
pe= 24

