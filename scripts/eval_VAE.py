import os, sys
sys.path.append(os.getcwd())

import time
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import torch
from torch import nn
from torchvision import datasets, transforms

from model.AutoEncoder import AutoEncoder, Encoder, Decoder
from utils import train_VAE, evaluate_VAE, test_VAE, MSCOCOImageDataset

from datasets import load_dataset

# 清理CUDA缓存，避免内存不足
torch.cuda.empty_cache()

# 加载配置文件
cfg = OmegaConf.load('./config/autoencoder_kl_32x32x4.yaml')

# 设置训练参数
device = "cuda" if torch.cuda.is_available() and cfg.training.use_cuda else "cpu"
image_size = cfg.training.image_size
batch_size = 1  # 保持批量大小为1
vae_model_path = cfg.training.model_path

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# 加载数据集
cfg_data = OmegaConf.load('./config/dataset_mscoco.yaml')
download_dir = 'D:/datasets'
os.makedirs(download_dir, exist_ok=True)  # 确保目录存在
ds = load_dataset(cfg_data.dataset.name, cache_dir=download_dir)

dataset_val = MSCOCOImageDataset(ds, splits=['val'], transform=transform, filter_channels=False, target_size=(image_size, image_size))
val_loader = torch.utils.data.DataLoader(
    dataset_val, 
    batch_size=batch_size, 
    shuffle=False,
)

# 确保结果目录存在
os.makedirs("./results", exist_ok=True)

encoder = Encoder(**cfg.model.encoder).to(device)
decoder = Decoder(**cfg.model.decoder).to(device)

auto_encoder = AutoEncoder(
    encoder=encoder,
    decoder=decoder,
    **cfg.model.auto_encoder
).to(device)

auto_encoder.load_state_dict(torch.load(vae_model_path))
auto_encoder.eval()  # 设置为评估模式

# 创建一个2x4的网格，上面一行为原始图像，下面一行为重建图像
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 收集4张图片及其重建结果
original_images = []
reconstructed_images = []

# 获取4张不同的图片
for i, (pics, _) in enumerate(val_loader):
    if i >= 4:  # 只处理4张图片
        break
        
    pics = pics.to(device)
    
    with torch.no_grad():  # 不计算梯度，节省内存
        z = auto_encoder.encode(pics)
        recon_image = auto_encoder.decode(z.sample())
    
    # 将图像移回CPU并转换为numpy数组
    original = pics[0].permute(1, 2, 0).cpu().detach().numpy()
    reconstructed = recon_image[0].permute(1, 2, 0).cpu().detach().numpy()
    
    original_images.append(original)
    reconstructed_images.append(reconstructed)

# 显示图像
for i in range(4):
    # 显示原始图像
    axes[0, i].imshow(original_images[i])
    axes[0, i].set_title(f"original {i+1}")
    axes[0, i].axis("off")
    
    # 显示重建图像
    axes[1, i].imshow(reconstructed_images[i])
    axes[1, i].set_title(f"reconstructed {i+1}")
    axes[1, i].axis("off")

# 添加总标题
plt.suptitle("VAE", fontsize=16)
plt.tight_layout()

# 保存图像
timestamp = int(time.time())
save_path = f"./results/MS_COCO_VAE_{timestamp}.png"
plt.savefig(save_path, dpi=300)
print(f"结果已保存到 {save_path}")

# 显示图像（如果在交互环境中）
plt.show()
plt.close()