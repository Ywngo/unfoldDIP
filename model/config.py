import argparse
import torch
import os
from datetime import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# --- 1. Basic Experiment Settings ---
parser.add_argument('--exp_name', type=str, default='Baseline', help='Experiment name')
parser.add_argument('--data_name', type=str, default="PaviaU", help='Dataset name')
parser.add_argument('--scale_factor', type=int, default=8, help='Downsampling scale factor (HSI vs MSI)')
parser.add_argument("--gpu_ids", type=str, default='0', help='GPU ID')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Main directory for checkpoints')
parser.add_argument('--seed', type=int, default=30, help='Random seed')
parser.add_argument('--target_scale', type=float, default=1.0, help=' traget scale of pred')

# --- 2. Core Hyperparameters (Deep Unfolding & Tucker) ---

parser.add_argument('--gamma', type=float, default=0.8, help='Decay coefficient for unfolding stage loss weights')
parser.add_argument('--sam_weight', type=float, default=0.1, help='Weight for SAM (Spectral Angle Mapper) loss')
parser.add_argument('--num_endmembers', type=int, default=32, help='Number of Endmembers (Rank of decomposition)')
parser.add_argument('--K_iters', type=int, default=5, help='Number of deep unfolding stages')

# --- 3. Learning Rates & Iterations ---
parser.add_argument("--lr_stage1", type=float, default=0.001)
parser.add_argument('--niter1', type=int, default=2000)
parser.add_argument('--niter_decay1', type=int, default=2000)
parser.add_argument("--lr_stage3_dip", type=float, default=0.003)
parser.add_argument('--niter3_dip', type=int, default=500)
parser.add_argument('--niter_decay3_dip', type=int, default=2000)

# --- 4. Data Paths & Noise Settings ---
parser.add_argument('--sp_root_path', type=str, default='data/EDIP-Net/spectral_response/')
parser.add_argument('--default_datapath', type=str, default="data/EDIP-Net/")
parser.add_argument("--band", type=int, default=240)
parser.add_argument('--noise', type=str, default="No")
parser.add_argument('--nSNR', type=int, default=35)

# --- 5. Network Architecture (V3/V4 ResDCT + UNet Version) ---

# 5a. Spatial U-Net Encoder
parser.add_argument('--msi_depth', type=int, default=3, help='Depth of Spatial U-Net (number of Down-Up stages)')
parser.add_argument('--msi_base_dim', type=int, default=32, help='Base channel dimension of Spatial U-Net (Crucial for high frequency)')

# 5b. Spectral ResDCT Encoder
parser.add_argument('--hsi_depth', type=int, default=2, help='Depth of HSI spectral ResDCT blocks')
parser.add_argument('--hsi_base_dim', type=int, default=128, help='Base channel dimension of HSI spectral extractor')

# 5c. Spatial MLP (INR base without continuous coordinates)
parser.add_argument('--inr_spat_depth', type=int, default=4, help='Number of hidden layers in spatial MLP')
parser.add_argument('--inr_spat_dim', type=int, default=192, help='Hidden dimension of spatial MLP')

# 5d. Spectral MLP
parser.add_argument('--inr_spec_depth', type=int, default=4, help='Number of hidden layers in spectral MLP')
parser.add_argument('--inr_spec_dim', type=int, default=128, help='Hidden dimension of spectral MLP')


args = parser.parse_args()

# --- Device & Environment Configuration ---
device = torch.device('cuda:{}'.format(args.gpu_ids)) if torch.cuda.is_available() else torch.device('cpu')
args.device = device
args.sigma = args.scale_factor / 2.35482

# --- Standardized Experiment Directory ---
time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
exp_folder_name = f"{args.exp_name}_SF{args.scale_factor}_{time_stamp}"
args.expr_dir = os.path.join(args.checkpoints_dir, args.data_name, exp_folder_name)