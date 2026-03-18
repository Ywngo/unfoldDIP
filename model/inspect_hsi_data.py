import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def inspect_hsi_data(mat_path):
    print(f"������������: {mat_path} ...")
    try:
        mat_data = sio.loadmat(mat_path)
    except Exception as e:
        print(f"��������: {e}")
        return

    # 1. �������������� HSI ������������
    # �������������� key (�� __header__, __version__, __globals__)
    valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]

    target_key = None
    max_size = 0

    print(f"��������������: {valid_keys}")

    # ������������������������������������������������������
    for key in valid_keys:
        val = mat_data[key]
        if isinstance(val, np.ndarray):
            print(f"  -> ���� '{key}': Shape={val.shape}, Type={val.dtype}")
            if val.size > max_size:
                max_size = val.size
                target_key = key

    if target_key is None:
        print("��������������������������������")
        return

    print(f"\n����������������: ['{target_key}']")
    data = mat_data[target_key]

    # ������������������������������
    data = data.astype(np.float32)

    # 2. ������������
    d_min = np.min(data)
    d_max = np.max(data)
    d_mean = np.mean(data)
    d_std = np.std(data)

    # 3. ������������
    print("-" * 40)
    print("��������������������")
    print("-" * 40)
    print(f"������ (Min) : {d_min:.6f}")
    print(f"������ (Max) : {d_max:.6f}")
    print(f"����   (Mean): {d_mean:.6f}")
    print(f"������ (Std) : {d_std:.6f}")
    print("-" * 40)

    # 4. ��������������
    status = []

    # �������� (��������������������)
    if d_min < 0:
        status.append("? ����������������(���������� Zero-mean ��������������)")

    # ���������� [0, 1] ����
    if d_min >= 0 and d_max <= 1.0:
        status.append("? �������������� [0, 1] ������")
    elif d_min >= 0 and d_max > 1.0:
        if d_max <= 255:
            status.append("?? ������������ [0, 255] (8-bit)������������")
        elif d_max <= 65535:
            status.append("?? ������������ [0, 65535] (16-bit) �� ����������������������")
        else:
            status.append(f"?? ������������ (Max={d_max:.2f})������������")

    # ���� NaN �� Inf
    if np.isnan(data).any():
        status.append("? ������������������ NaN (����)��")
    if np.isinf(data).any():
        status.append("? ������������������ Inf (������)��")

    for s in status:
        print(s)

    print("-" * 40)

    # 5. �������� (����������)
    # ������������������������������������������
    flatten_data = data.flatten()
    if len(flatten_data) > 100000:
        sample_data = np.random.choice(flatten_data, 100000, replace=False)
    else:
        sample_data = flatten_data

    plt.figure(figsize=(10, 5))
    plt.hist(sample_data, bins=100, color='blue', alpha=0.7, log=True)
    plt.title(f"Data Distribution Histogram ({target_key})")
    plt.xlabel("Value")
    plt.ylabel("Count (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.show()


# --- �������� ---
# ����������������
file_path = "/home/ubuntu/YuanWenJie_workspace/MODEL/EDIP/EDIP-Net_TGRS-main_v17/checkpoints/WaDCSF8_band240_S3_0.0005_3000_3000_factorized_INR_band_k_14/lr_hsi.mat"

# ������������������������������ `ls` ��������������
try:
    inspect_hsi_data(file_path)
except FileNotFoundError:
    print("������������������ file_path ��������������")