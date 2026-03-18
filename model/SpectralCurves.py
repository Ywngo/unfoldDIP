import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def save_spectral_curves_png(mat_path, save_name='spectral_analysis.png', data_key=None):
    """
    ����mat����������������������������������������������PNG��
    """
    # -------------------------------------------------
    # 1. �������� (����������������������������)
    # -------------------------------------------------
    if not os.path.exists(mat_path):
        print(f"���������������� {mat_path}")
        return

    try:
        mat_data = sio.loadmat(mat_path)
    except Exception as e:
        print(f"��������: {e}")
        return

    hsi_image = None
    if data_key:
        hsi_image = mat_data[data_key]
    else:
        # ��������3D����
        for key in mat_data:
            if not key.startswith('__'):
                val = mat_data[key]
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    hsi_image = val
                    print(f"-> ������������: '{key}' {hsi_image.shape}")
                    break

    if hsi_image is None:
        print("����������������������������")
        return

    # �������������� (H, W, C)���������� (C, H, W) ��������
    if hsi_image.shape[0] < hsi_image.shape[2]:  # ����������������������������C������H��W������������������������������
        print("������������������ (C, H, W) ���������������� (H, W, C)...")
        hsi_image = hsi_image.transpose(1, 2, 0)

    H, W, C = hsi_image.shape

    # -------------------------------------------------
    # 2. ��������
    # -------------------------------------------------
    # ���� 'seaborn-v0_8-whitegrid' �� 'ggplot' ����������������
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')  # ������������

    plt.figure(figsize=(10, 6), dpi=300)  # DPI=300 ����PNG������

    # -------------------------------------------------
    # 3. �������������� (Random Pixels)
    # -------------------------------------------------
    num_pixels = 3
    # ����������������������
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(num_pixels):
        rx = random.randint(0, H - 1)
        ry = random.randint(0, W - 1)

        spectrum = hsi_image[rx, ry, :]

        # ����������
        plt.plot(spectrum, color=colors[i], linewidth=1.5, alpha=0.8,
                 label=f'Pixel ({rx}, {ry})')

    # -------------------------------------------------
    # 4. ���������������� (Region Average)
    # -------------------------------------------------
    region_size = 20
    sx = random.randint(0, H - region_size)
    sy = random.randint(0, W - region_size)

    region_cube = hsi_image[sx:sx + region_size, sy:sy + region_size, :]
    mean_spectrum = np.mean(region_cube, axis=(0, 1))

    # ��������������������������
    plt.plot(mean_spectrum, color='#d62728', linewidth=3, linestyle='--',
             label=f'Avg Region [{sx}:{sx + region_size}, {sy}:{sy + region_size}]')

    # -------------------------------------------------
    # 5. ���� PNG
    # -------------------------------------------------
    plt.title(f'Spectral Signature Analysis (Bands: {C})', fontsize=14, fontweight='bold')
    plt.xlabel('Band Index', fontsize=12)
    plt.ylabel('Intensity / Reflectance', fontsize=12)
    plt.legend(frameon=True, fancybox=True, shadow=True)  # ��������
    plt.tight_layout()  # ��������������

    # ��������
    plt.savefig(save_name, format='png', bbox_inches='tight')
    print(f"? ������PNG������������: {os.path.abspath(save_name)}")

    # ������������������������IDE����
    # plt.show()
    plt.close()  # ����������������


# ==========================================
# ��������
# ==========================================
if __name__ == "__main__":
    # ������������������
    fake_data = np.random.rand(128, 128, 150)
    sio.savemat('test_data.mat', {'hsi': fake_data})

    # ��������
    save_spectral_curves_png('test_data.mat')