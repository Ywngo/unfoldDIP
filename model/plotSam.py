import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
# ==========================
# Step 1: ���� .mat ����
# ==========================
# ���� output.mat ���������� 'output'
# ���� gt.mat ���������� 'gt'

output_mat = loadmat('/home/ubuntu/MODEL/EDIP/EDIP-Net_TGRS-main_v12_11base_LE_fusion/checkpoints/PaviaCSF8_band240_S1_0.001_2000_2000_S2_0.004_2000_2000_S3_0.002_7000_7000_gauss_band=20/Out_fhsi_S3.mat')
gt_mat = loadmat('/home/ubuntu/MODEL/EDIP/EDIP-Net_TGRS-main_v12_11base_LE_fusion/checkpoints/PaviaCSF8_band240_S1_0.001_2000_2000_S2_0.004_2000_2000_S3_0.002_7000_7000_gauss_band=20/REF.mat')
# output_mat = loadmat('/home/ubuntu/MODEL/EDIP/EDIP-Net_TGRS-main_v12_11base_LE_fusion/checkpoints/TGSF12_band240_S1_0.001_2000_2000_S2_0.004_2000_2000_S3_0.002_7000_7000_gauss_band=10/Out_fhsi_S3.mat')
# gt_mat = loadmat('/home/ubuntu/MODEL/EDIP/EDIP-Net_TGRS-main_v12_11base_LE_fusion/data/EDIP-Net/TG/REF.mat')
# ������������������
x_pred = output_mat['Out']  # ��������
x_true = gt_mat['data']          # Ground Truth

# ������������
print("GT shape:", x_true.shape)
print("Pred shape:", x_pred.shape)

assert x_true.shape == x_pred.shape, "GT �� Pred ������������"

# ==========================
# Step 2: SAM ��������
# ==========================
def compute_sam_map(im1, im2, eps=1e-8):
    """
    �������������� SAM (����: ��)
    """
    w, h, c = im1.shape
    im1_flat = im1.reshape(-1, c)
    im2_flat = im2.reshape(-1, c)

    dot = np.sum(im1_flat * im2_flat, axis=1)
    denom = np.linalg.norm(im1_flat, axis=1) * np.linalg.norm(im2_flat, axis=1)

    sam = np.arccos(np.clip(dot / (denom + eps), -1, 1))
    sam_deg = np.rad2deg(sam)
    return sam_deg.reshape(w, h)


# ==========================
# Step 3: ����������
# ==========================
# ����SAM����������
scale = np.max(x_true)
x_true_norm = x_true / (scale + 1e-8)
x_pred_norm = x_pred / (scale + 1e-8)

# ==========================
# Step 4: ���� SAM
# ==========================
sam_map = compute_sam_map(x_true_norm, x_pred_norm)
mean_sam = np.mean(sam_map)
print(f"Mean SAM: {mean_sam:.3f}��")

# ==========================
# Step 5: ��������������
# ==========================
save_dir = '/home/ubuntu/MODEL/EDIP/EDIP-Net_TGRS-main_v12_11base_LE_fusion/checkpoints/PaviaCSF8_band240_S1_0.001_2000_2000_S2_0.004_2000_2000_S3_0.002_7000_7000_gauss_band=20/'
os.makedirs(save_dir, exist_ok=True)

# ==========================
# Step 6: ����������������
# ==========================
plt.figure(figsize=(7,5))
plt.hist(sam_map.ravel(), bins=100, edgecolor='black', alpha=0.7)
plt.xlabel("Spectral Angle (degrees)")
plt.ylabel("Pixel Count")
plt.title("SAM Distribution Histogram")
plt.grid(alpha=0.3)
hist_path = os.path.join(save_dir, 'sam_histogram.png')
plt.savefig(hist_path, dpi=300)
plt.close()
print(f"SAM histogram saved to {hist_path}")

# ==========================
# Step 7: ����������������
# ==========================
plt.figure(figsize=(6,5))
plt.imshow(sam_map, cmap='hot')
plt.colorbar(label='SAM (degrees)')
plt.title("SAM Spatial Heatmap")
plt.axis('off')
heatmap_path = os.path.join(save_dir, 'sam_heatmap.png')
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"SAM heatmap saved to {heatmap_path}")

# ==========================
# Step 8: ����������������������
# ==========================
high_angle_thresh = 5
high_mask = sam_map > high_angle_thresh

overlay = np.zeros((*sam_map.shape, 3))
overlay[..., 0] = high_mask  # ��������

plt.figure(figsize=(6,5))
plt.imshow(sam_map, cmap='gray')
plt.imshow(overlay, alpha=0.5)
plt.title(f'High-Angle Regions (>{high_angle_thresh}��)')
plt.axis('off')
high_mask_path = os.path.join(save_dir, 'sam_high_angle_regions.png')
plt.savefig(high_mask_path, dpi=300)
plt.close()
print(f"SAM high-angle regions saved to {high_mask_path}")

print(f"���� {high_angle_thresh}�� ����������: {np.mean(high_mask)*100:.2f}%")