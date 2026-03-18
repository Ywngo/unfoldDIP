import scipy.io

# �������� mat �������� 'data.mat'
mat = scipy.io.loadmat('/home/ubuntu/YuanWenJie_workspace/MODEL/EDIP/EDIP-Net_TGRS-main_v17/data/EDIP-Net/PaviaU/REF.mat')

# mat ������������������ mat ������������������
# ������������������ 'X' ����������:
print(mat['REF'].shape)

# ��������������������������������������
print(mat.keys())

# ����������������������������:
for key in mat:
    if not key.startswith('__'):
        print(f"{key}: {mat[key].shape}")