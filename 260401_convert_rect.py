# 260401_convert_rect.py
# Baseデータを長方形型に整えるためのコード

import numpy as np

hf ="InH" # "InH" もしくは "InF" を指定
source_file = f"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Data/Base_{hf}.npy"
output_file = f"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Data/Base_{hf}_rect.npy"

def reshape_to_rect(source_file, output_file):
    # 1. データのロード (1000個の辞書が入ったリスト)
    data = np.load(source_file, allow_pickle=True)
    num_samples = len(data) # 1000

    # 2. 最大サイズ (N:クラスター数, M:サブパス数) を自動取得
    max_n = max([d['N'] for d in data])
    max_m = max([np.max(d['M']) for d in data])
    print(f"File: {source_file}")
    print(f"Max Clusters (N): {max_n}, Max Subpaths (M): {max_m}")

    # 3. 格納用の長方形行列を準備 (全て0で初期化)
    # 形状: (シナリオ数, クラスター数, サブパス数)
    shape_3d = (num_samples, max_n, max_m)
    shape_2d = (num_samples, max_n)

    rect_dict = {
        # スカラー系 (1000,)
        'chi': np.zeros(num_samples),
        'N_actual': np.zeros(num_samples, dtype=int),
        
        # クラスター単位 (1000, N_max)
        'M_actual': np.zeros(shape_2d, dtype=int),
        'tau':      np.zeros(shape_2d),
        'Z':        np.zeros(shape_2d),
        
        # サブパス単位 (1000, N_max, M_max)
        'phi_deg':      np.zeros(shape_3d),
        'beta':         np.zeros(shape_3d),
        'rho':          np.zeros(shape_3d),
        'theta_ND_deg': np.zeros(shape_3d),
        'eta_ND_deg':   np.zeros(shape_3d),
        'varphi_deg':   np.zeros(shape_3d),
        'U_nm':         np.zeros(shape_3d),
        
        # 有効なパスを識別するマスク (重要！)
        'mask':         np.zeros(shape_3d)
    }

    # 4. データの流し込み
    for s in range(num_samples):
        d = data[s]
        n_count = d['N']
        rect_dict['chi'][s] = d['chi']
        rect_dict['N_actual'][s] = n_count

        for n in range(n_count):
            m_count = int(d['M'][n])
            rect_dict['M_actual'][s, n] = m_count
            rect_dict['tau'][s, n]      = d['tau'][n]
            rect_dict['Z'][s, n]        = d['Z'][n]

            # サブパスデータをコピー (パディング部分は0のまま残る)
            rect_dict['phi_deg'][s, n, :m_count]      = d['phi_deg'][n]
            rect_dict['beta'][s, n, :m_count]         = d['beta'][n]
            rect_dict['rho'][s, n, :m_count]          = d['rho'][n]
            rect_dict['theta_ND_deg'][s, n, :m_count] = d['theta_ND_deg'][n]
            rect_dict['eta_ND_deg'][s, n, :m_count]   = d['eta_ND_deg'][n]
            rect_dict['varphi_deg'][s, n, :m_count]   = d['varphi_deg'][n]
            rect_dict['U_nm'][s, n, :m_count]         = d['U'][n]
            
            # 有効なデータが入っている場所だけ1にする
            rect_dict['mask'][s, n, :m_count] = 1.0

    # 5. 保存
    np.save(output_file, rect_dict)
    print(f"Successfully saved to {output_file}\n")

# reshape_to_rect(source_file, output_file)

def full_verification(original_path, rect_path):
    # 1. データのロード
    orig = np.load(original_path, allow_pickle=True)
    rect = np.load(rect_path, allow_pickle=True).item()
    num_samples = len(orig) # 1000
    
    print(f"--- Full Verification: {original_path} vs {rect_path} ---")
    
    mismatches = []
    
    for s in range(num_samples):
        d_orig = orig[s]
        n_count = d_orig['N']
        
        # チェック項目リスト
        checks = {
            'chi': np.isclose(d_orig['chi'], rect['chi'][s]),
            'N': d_orig['N'] == rect['N_actual'][s]
        }
        
        # クラスター/サブパスデータの詳細比較
        for n in range(n_count):
            m_count = int(d_orig['M'][n])
            
            # 各パラメータが一致するか (有効領域のみスライスして比較)
            # phi_deg, beta, rho, theta_ND_deg, eta_ND_deg, varphi_deg, U_nm
            params = ['phi_deg', 'beta', 'rho', 'theta_ND_deg', 'eta_ND_deg', 'varphi_deg']
            for p in params:
                match = np.allclose(d_orig[p][n], rect[p][s, n, :m_count])
                if not match:
                    checks[f"{p}_s{s}_n{n}"] = False
            
            # マスクが正しく1になっているか
            if not np.all(rect['mask'][s, n, :m_count] == 1.0):
                checks[f"mask_active_s{s}_n{n}"] = False
            
            # パディング領域が正しく0になっているか (クラスター内の余白)
            if m_count < rect['phi_deg'].shape[2]:
                if not np.all(rect['mask'][s, n, m_count:] == 0.0):
                    checks[f"mask_padding_s{s}_n{n}"] = False

        # 一つでもFalseがあれば記録
        if not all(checks.values()):
            mismatches.append(s)

    # 2. 結果表示
    if len(mismatches) == 0:
        print(f"全 {num_samples} インデックスの照合が完了しました。")
        print("ALL MATCHED!")
    else:
        print(f"以下のインデックスで不一致が検出されました: {mismatches}")

# 実行
original_path = f"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Data/Base_{hf}.npy"
rect_path = f"C:/Users/tai20/OneDrive - 国立大学法人 北海道大学/sim_data/Data/Base_{hf}_rect.npy"
full_verification(original_path, rect_path)