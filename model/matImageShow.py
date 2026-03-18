import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, img_as_ubyte
import os

# 1. ���� .mat ����
mat_path = r"/home/ubuntu/YuanWenJie_workspace/MODEL/EDIP/EDIP-Net_TGRS-main_v17/compareModel/CS2DIP/data/PaviaU/REF.mat"
data = sio.loadmat(mat_path)

# 2. ������������
print("MAT ��������������:")
print(data.keys())
data = sio.loadmat(mat_path)

# ================= ������������ =================
# 2. ���������������������� 'Out'��
# �������� '__' ����������������������������������
real_keys = [k for k in data.keys() if not k.startswith('__')]

if len(real_keys) == 0:
    raise ValueError("������MAT ������������������������")

# ����������������������
var_name = real_keys[0]
hsi = data[var_name]

# 4. ������������
print("��������:", hsi.shape)  # ���� (rows, cols, bands)

# 5. �������� (R=30, G=20, B=7)
bandR, bandG, bandB = 30, 15, 10

# ������MATLAB ������ 1 ������Python �� 0 �������������� 1
rgb = np.stack([
    hsi[:, :, bandR - 1],
    hsi[:, :, bandG - 1],
    hsi[:, :, bandB - 1]
], axis=2)

# 6. �������� [0,1]
rgb_min, rgb_max = rgb.min(), rgb.max()
rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)

# 7. ��������
plt.imshow(rgb_norm)
plt.title(f'RGB �������� (R:{bandR}, G:{bandG}, B:{bandB})')
plt.axis('off')
plt.show()

# 8. ��������
save_path = r"/home/ubuntu/YuanWenJie_workspace/MODEL/EDIP/EDIP-Net_TGRS-main_v17/compareModel/CS2DIP/data/TG/TG.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ������ 8-bit ����������
rgb_8bit = img_as_ubyte(rgb_norm)
plt.imsave(save_path, rgb_8bit)

print(f"? ������������: {save_path}")
