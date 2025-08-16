import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def plot_cdf(data, label, color, linestyle, linewidth=5):  # ← 線太くした
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label, linestyle=linestyle, color=color, linewidth=linewidth)


# 距離別の折れ線グラフ
def oresen(channel_type):
    d_list = [5, 10, 15, 20, 25, 30]
    mean_near_tru_Mirror = []
    mean_near_est_Mirror = []
    mean_near_tru_Scatter1 = []
    mean_near_est_Scatter1 = []
    mean_far_tru = []
    mean_far_est = []
    
    for d in d_list:
        
        # 近傍界条件
        NF_setting = 'Near'
        # 鏡面反射条件
        load_dir_Mirror = f"C:/Users/tai20/Downloads/研究データ/Data/Mirror/{channel_type}/Channel_Capacity/{NF_setting}/d={d}"
        H_near_tru_Mirror = np.load(f"{load_dir_Mirror}/Capacity_Htru.npy", allow_pickle=True)
        H_near_est_Mirror = np.load(f"{load_dir_Mirror}/Capacity_Hest.npy", allow_pickle=True)
        # 第1散乱体条件
        load_dir_Scatter1 = f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{channel_type}_1000/ChannelCapacity"
        H_near_tru_Scatter1 = np.load(f"{load_dir_Scatter1}/Channel_{channel_type}_step5_ChannelCapacity_d={d}/Channel_{channel_type}_step5_Htru_{NF_setting}_ChannelCapacity_d={d}.npy")
        H_near_est_Scatter1 = np.load(f"{load_dir_Scatter1}/Channel_{channel_type}_step5_ChannelCapacity_d={d}/Channel_{channel_type}_step5_Hmea_{NF_setting}_ChannelCapacity_d={d}.npy")
        
        # 遠方界条件
        NF_setting = 'Far'
        H_far_tru = np.load(f"{load_dir_Scatter1}/Channel_{channel_type}_step5_ChannelCapacity_d={d}/Channel_{channel_type}_step5_Htru_{NF_setting}_ChannelCapacity_d={d}.npy")
        H_far_est = np.load(f"{load_dir_Scatter1}/Channel_{channel_type}_step5_ChannelCapacity_d={d}/Channel_{channel_type}_step5_Hmea_{NF_setting}_ChannelCapacity_d={d}.npy")
        
        # 近傍界におけるチャネル容量
        mean_near_tru_Mirror.append(np.mean(np.max(H_near_tru_Mirror, axis=1)))
        mean_near_est_Mirror.append(np.mean(np.max(H_near_est_Mirror, axis=1)))
        mean_near_tru_Scatter1.append(np.mean(np.max(H_near_tru_Scatter1, axis=1)))
        mean_near_est_Scatter1.append(np.mean(np.max(H_near_est_Scatter1, axis=1)))
        # 遠方界仮定のチャネル容量
        mean_far_tru.append(np.mean(np.max(H_far_tru, axis=1)))
        mean_far_est.append(np.mean(np.max(H_far_est, axis=1)))
        
        
    fig, ax = plt.subplots(figsize=(10, 8))  # 幅6インチ、高さ4インチ（縦横比3:2）
    
    plt.plot(d_list, mean_far_tru, color='darkblue', linestyle='-', marker='o',markersize=8, label='Far True')
    plt.plot(d_list, mean_far_est, marker='o', linestyle='-.', markersize=8, label='Far Est')
    plt.plot(d_list, mean_near_tru_Mirror, color='orangered', linestyle='-', marker='^', markersize=8, label='Near True (Mirror)')
    plt.plot(d_list, mean_near_est_Mirror, linestyle='-.', marker='^', markersize=8, label='Near Est (Mirror)')
    plt.plot(d_list, mean_near_tru_Scatter1, color='forestgreen', linestyle='-', marker='*', markersize=12, label='Near True (Scatter1 Unif)')
    plt.plot(d_list, mean_near_est_Scatter1, color='limegreen', linestyle='-.', marker='*', markersize=12, label='Near Est (Scatter1 Unif)')

    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'{channel_type}',fontsize=20)
    plt.xlabel('BS-UE Distance d[m]',fontsize=20)
    plt.ylabel('Channel Capacity [bps/Hz]',fontsize=20)
    plt.ylim(bottom=0, top=60)
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.legend(fontsize=18)
    
    # PDFで保存
    # plt.savefig(f"{channel_type}_oresen.pdf", format='pdf')
    plt.show()

    
oresen('InH')
exit()

"""
r_5 = np.zeros((1000), dtype=object)
r_10 = np.zeros((1000), dtype=object)
# 第1散乱体までの距離の累積分布を画くグラフ
channel_d5 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=5.npy", allow_pickle=True)
channel_d10 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=10.npy", allow_pickle=True)
channel_d15 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=15.npy", allow_pickle=True)
channel_d20 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=20.npy", allow_pickle=True)
channel_d25 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=25.npy", allow_pickle=True)
channel_d30 = np.load(f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/Channel_{setting}_1000_step1_IncludeScatter1/Channel_{setting}_1000_step1_d=30.npy", allow_pickle=True)

multipath = 0
for p in range(1000):
    N = channel_d5[p]['N']
    for n in range(N):
        multipath += channel_d5[p]['M'][n]
print(multipath)

def r_flatten(channel):
    r_list = []

    for i in range(1000):
        N = channel[i]['N']
        M = channel[i]['M']
        for n in range(N):
            for m in range(M[n]):
                r_array = channel[i]['r'][n][m]  # n番目の要素のm番目の要素を取り出す
                r_array = np.array(r_array).flatten()  # 完全に平坦化（1次元化）
                r_list.extend(r_array)  # 要素を追加（extendで中身だけ追加）
        # 最終的に NumPy 配列に変換
        r_all = np.array(r_list)
    return r_all

r_5 = r_flatten(channel_d5)
count_5 = np.sum(r_5 == 5)  # 5の個数
total_count = r_5.size      # r_5 の要素数
print(total_count)
# 5の割合を計算
ratio_5 = count_5 / total_count

print(f"5: {count_5}")
print(f"ratio_5 {ratio_5}")
r_10 = r_flatten(channel_d10)
r_15 = r_flatten(channel_d15)
r_20 = r_flatten(channel_d20)
r_25 = r_flatten(channel_d25)
r_30 = r_flatten(channel_d30)
r_5_filtered = r_5[r_5 != 5]  # 5の値を取り除く
r_10_filterd = r_10[r_10 != 10]
r_15_filterd = r_15[r_15 != 15]
r_20_filterd = r_20[r_20 != 20]
r_25_filterd = r_25[r_25 != 25]
r_30_filterd = r_30[r_30 != 30]



# plot_cdf(r_5, 'd=5', 'blue', 3)
# plot_cdf(r_10, 'd=10', 'red', 3)
# plot_cdf(r_15, 'd=15', 'yellow', 3)
# plot_cdf(r_20, 'd=20', 'green', 3)
# plot_cdf(r_25, 'd=25', 'cyan', 3)
# plot_cdf(r_30, 'd=30', 'purple', 3)
plot_cdf(r_5_filtered, 'd=5 wo5', 'blue', 3)
plot_cdf(r_10_filterd, 'd=10 wo10', 'red', 3)
plot_cdf(r_15_filterd, 'd=15 wo15', 'yellow', 3)
plot_cdf(r_20_filterd, 'd=20 wo20', 'green', 3)
plot_cdf(r_25_filterd, 'd=25 wo25', 'cyan', 3)
plot_cdf(r_30_filterd, 'd=30 wo30', 'purple', 3)

# 軸ラベルとかタイトルもフォントサイズ
plt.xlabel('r [m]', fontsize=20)
plt.ylabel('Cumulative distribution (CDF)', fontsize=20)
plt.title(f'Distance between BS and First scatterer [m]', fontsize=20)
plt.xlim(left=0, right=40)
plt.ylim(bottom=0, top=1.0)

# 軸目盛りもフォント大きめに
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 判例は右下＆フォントサイズUP
plt.legend(loc='lower right', fontsize=15)

plt.grid(True)
plt.tight_layout()
plt.show()
"""


##########################################################################################################################################################

# チャネル容量累積分布のグラフ
setting = 'InF'
d=10
load_dir_r3 = f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_r=3/ChannelCapacity/Channel_{setting}_step5_ChannelCapacity_d={d}"
load_dir = f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/channel_{setting}_1000/ChannelCapacity/Channel_{setting}_step5_ChannelCapacity_d={d}"
load_dir_CA = f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/{setting}_CoherentSum"

H_near_tru = np.load(f"{load_dir_CA}/{setting}_Simple_CoherentSum/ChannelCapacity/Channel_{setting}_step5_Htru_Near_ChannelCapacity_d={d}.npy")
H_near_mea = np.load(f"{load_dir_CA}/{setting}_Simple_CoherentSum/ChannelCapacity/Channel_{setting}_step5_Hmea_Near_ChannelCapacity_d={d}.npy")
H_far_tru = np.load(f"{load_dir_CA}/{setting}_Simple_CoherentSum/ChannelCapacity/Channel_{setting}_step5_Htru_Far_ChannelCapacity_d={d}.npy")
H_far_mea = np.load(f"{load_dir_CA}/{setting}_Simple_CoherentSum/ChannelCapacity/Channel_{setting}_step5_Hmea_Far_ChannelCapacity_d={d}.npy")
H_r3_tru = np.load(f"{load_dir_CA}/{setting}_r=3/ChannelCapacity/Channel_{setting}_step5_Htru_Near_ChannelCapacity_d={d}.npy")
H_r3_mea = np.load(f"{load_dir_CA}/{setting}_r=3/ChannelCapacity/Channel_{setting}_step5_Hmea_Near_ChannelCapacity_d={d}.npy")


H_near_tru_ad = np.max(H_near_tru, axis=1)
H_near_mea_ad = np.max(H_near_mea, axis=1)
H_far_tru_ad = np.max(H_far_tru, axis=1)
H_far_mea_ad = np.max(H_far_mea, axis=1)
H_r3_tru_ad = np.max(H_r3_tru, axis=1)
H_r3_mea_ad = np.max(H_r3_mea, axis=1)
print(np.shape(H_near_tru))

plt.figure(figsize=(10, 7))  # ← サイズも少し大きめに

# プロット
plot_cdf(H_far_tru_ad, 'Far True', 'darkblue', '-',3)
plot_cdf(H_far_mea_ad, 'Far Est', 'cyan','-.', 3)
plot_cdf(H_near_tru_ad, 'Near True (Unif)', 'orangered','-', 3)
plot_cdf(H_near_mea_ad, 'Near Est (Unif)', 'orange','-.', 3)
plot_cdf(H_r3_tru_ad, 'Near True (3m)', 'forestgreen','-', 3)
plot_cdf(H_r3_mea_ad, 'Near Est (3m)', 'limegreen','-.', 3)



# 軸ラベルとかタイトルもフォントサイズUP
plt.xlabel('Channel capacity [bps/Hz]', fontsize=20)
plt.ylabel('Cumulative distribution (CDF)', fontsize=20)
# plt.title(f'{setting} d = {d} [m]', fontsize=20)
plt.xlim(left=0, right=90)
plt.ylim(bottom=0, top=1.0)

# 軸目盛りもフォント大きめに
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 判例は右下＆フォントサイズUP
plt.legend(loc='lower right', fontsize=20)

plt.grid(True)
plt.tight_layout()
# PDFで保存
plt.savefig(f"{setting}_d=10_cum.pdf", format='pdf')
plt.show()

exit()


# --- 設定 ---
setting = 'InH'
d = 5
# base_dir = (
#     f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter"
#     f"/channel_{setting}_1000/Channel_{setting}_1000_step5_ChannelCapacity"
#     f"/Channel_{setting}_step5_ChannelCapacity_d={d}"
# )

base_dir = f"C:/Users/tai20/Downloads/NYUSIMchannel_shelter/InH_CoherentSum/InH_Simple_CoherentSum/ChannelCapacity"

# --- データロード関数 ---
def load_selected_db(NorF, mode):
    # NorF: 'Near' or 'Far'; mode: 'Htru' or 'Hmea'
    # チャネル容量ロード
    H = np.load(
        f"{base_dir}/Channel_{setting}_step5_{mode}_{NorF}_ChannelCapacity_d={d}.npy"
    )  # shape: (100, 12)
    beam_idx = np.argmax(H, axis=1)
    # 固有値ロード
    eig = np.load(
        f"{base_dir}/eigvals_{NorF}_{mode}_{NorF}_d={d}.npy"
    )  # shape: (100, 12, M)
    # 選択固有値抽出 & dB変換
    sel = eig[np.arange(beam_idx.size), beam_idx, :]  # (100, M)
    return 10 * np.log10(sel)  # (100, M) in dB

# --- 各データ取得 ---
selected_tru_near_db = load_selected_db('Near', 'Htru')
selected_est_near_db = load_selected_db('Near', 'Hmea')
selected_tru_far_db  = load_selected_db('Far',  'Htru')
selected_est_far_db  = load_selected_db('Far',  'Hmea')

# --- CDFプロット関数 ---
def plot_cdf_db(data_db, label=None, linewidth=2, linestyle='-'):
    sorted_db = np.sort(data_db)
    cdf = np.arange(1, len(sorted_db) + 1) / len(sorted_db)
    plt.plot(sorted_db, cdf, label=label, linewidth=linewidth, linestyle=linestyle)

# --- プロット: Near ---
plt.figure(figsize=(10, 7))
M = 8
for mode_idx in range(M):
    # True
    plot_cdf_db(selected_tru_near_db[:, mode_idx],
                label=f'Near True {mode_idx+1}',
                linewidth=2, linestyle='-')
    # Est
    plot_cdf_db(selected_est_near_db[:, mode_idx],
                label=f'Near Est {mode_idx+1}',
                linewidth=2, linestyle='--')
plt.xlabel('Eigenvalue [dB]', fontsize=16)
plt.ylabel('CDF', fontsize=16)
# plt.title(f'{setting} Near (True Est) CDF Eigenval, d={d} [m]', fontsize=18)
plt.xlim(-160, 0)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='', fontsize=10, loc='lower right')
plt.grid(True)
plt.tight_layout()


# --- プロット: Far ---
plt.figure(figsize=(10, 7))
for mode_idx in range(M):
    # True
    plot_cdf_db(selected_tru_far_db[:, mode_idx],
                label=f'Far True  {mode_idx+1}',
                linewidth=2, linestyle='-')
    # Est
    plot_cdf_db(selected_est_far_db[:, mode_idx],
                label=f'Far Est {mode_idx+1}',
                linewidth=2, linestyle='--')
plt.xlabel('Eigenvalue [dB]', fontsize=16)
plt.ylabel('CDF', fontsize=16)
plt.title(f'{setting} Far (True Est) CDF Eigenval, d={d} [m]', fontsize=18)
plt.xlim(-160, 0)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='', fontsize=10, loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
##########################################################################################################################################
