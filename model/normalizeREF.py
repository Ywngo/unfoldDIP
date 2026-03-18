# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.io as io

# ================= �������� =================
# ������������ (������������������)
ROOT_PATH = "data/EDIP-Net"

# ���������������� (��������������)
EXCLUDE_DIRS = ['spectral_response']


# ===========================================

def normalize_data(data):
    """
    ����������: (x - min) / (max - min + eps)
    """
    img = data.astype(np.float64)
    min_val = img.min()
    max_val = img.max()
    normalized = (img - min_val) / (max_val - min_val + 1e-8)
    return normalized


def process_folder(folder_name):
    """
    ��������������������
    """
    # ������������
    folder_path = os.path.join(ROOT_PATH, folder_name)
    file_path = os.path.join(folder_path, "REF.mat")

    # 1. ����������������
    if not os.path.exists(file_path):
        print(f"[����] {folder_name}: ������ REF.mat")
        return

    print(f"[{folder_name}] ��������...", end="")

    try:
        # 2. ��������
        mat_data = io.loadmat(file_path)

        # 3. �������������� (������ 'REF', ������������������������)
        key = 'REF'
        if key not in mat_data:
            # ������ __header__ ����������
            valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if valid_keys:
                key = valid_keys[-1]
            else:
                key = list(mat_data.keys())[-1]

        raw_img = mat_data[key]

        # 4. ������
        norm_img = normalize_data(raw_img)

        # 5. �������� (��������������)
        # ������������������������������������������ "REF_norm.mat"
        io.savemat(file_path, {key: norm_img})

        print(f" -> ����! (Range: {norm_img.min():.1f} - {norm_img.max():.1f})")

    except Exception as e:
        print(f" -> [����] {e}")


if __name__ == "__main__":
    if not os.path.exists(ROOT_PATH):
        print(f"����: ���������� {ROOT_PATH}")
        exit()

    # ����������������������
    sub_dirs = sorted(os.listdir(ROOT_PATH))

    print(f"������������ {ROOT_PATH} ����������...\n")

    count = 0
    for folder_name in sub_dirs:
        # ����������������������
        if not os.path.isdir(os.path.join(ROOT_PATH, folder_name)):
            continue
        if folder_name in EXCLUDE_DIRS:
            continue

        process_folder(folder_name)
        count += 1

    print(f"\n�������������������� {count} ����������")