import os, sys
sys.path.append(os.getcwd())

import time
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import torch
from torch import nn
from torchvision import datasets, transforms

from model.AutoEncoder import AutoEncoder, Encoder, Decoder, loss
from utils import train_VAE, MSCOCOImageDataset

from datasets import load_dataset

# 加载配置文件
cfg = OmegaConf.load('./config/autoencoder_kl_32x32x4.yaml')

# 设置训练参数
device = "cuda:0" if torch.cuda.is_available() and cfg.training.use_cuda else "cpu"
image_size = cfg.training.image_size
batch_size = cfg.training.batch_size
epochs = cfg.training.epochs
learning_rate = cfg.training.learning_rate
vae_model_path = cfg.training.model_path
weight_decay = cfg.training.weight_decay
load_model = cfg.training.load_model

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# 加载数据集
cfg_data = OmegaConf.load('./config/dataset_mscoco.yaml')
download_dir = cfg_data.dataset.cache_dir
os.makedirs(download_dir, exist_ok=True)  # 确保目录存在
ds = load_dataset(cfg_data.dataset.name, cache_dir=download_dir)
dataset = ds

dataset = MSCOCOImageDataset(ds, splits=['train', 'val', 'test', 'restval'], transform=transform)
ds_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
encoder = Encoder(**cfg.model.encoder).to(device)
decoder = Decoder(**cfg.model.decoder).to(device)

auto_encoder = AutoEncoder(
    encoder=encoder,
    decoder=decoder,
    **cfg.model.auto_encoder
).to(device)

if load_model:
    auto_encoder.load_state_dict(torch.load(vae_model_path))

optimizer = torch.optim.AdamW(auto_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

total_loss_history = []
recon_loss_history = []
kl_loss_history = []

start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    avg_epoch_loss, avg_epoch_recon_loss, avg_epoch_kl_loss = train_VAE(ds_loader, optimizer, auto_encoder, loss, device)
    total_loss_history.append(avg_epoch_loss)
    recon_loss_history.append(avg_epoch_recon_loss)
    kl_loss_history.append(avg_epoch_kl_loss)
    torch.save(auto_encoder.state_dict(), vae_model_path)
    print(f"Epoch {epoch + 1} Completed. Average Loss: {avg_epoch_loss:.4f}, Average Recon Loss: {avg_epoch_recon_loss:.4f}, Average KL Loss: {avg_epoch_kl_loss:.4f}")
    print("-" * 50)

print("Training completed!")
print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes.")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(total_loss_history) + 1), total_loss_history, marker='o', label="Training Total Loss")
plt.title("Training Total Loss Over Epochs")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()
plt.savefig(f"./losses/MS_COCO_VAE_Total_Loss_{time.time()}.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(recon_loss_history) + 1), recon_loss_history, marker='o', color='orange', label="Training Recon Loss")
plt.title("Training Recon Loss Over Epochs")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig(f"./losses/MS_COCO_VAE_Recon_Loss_{time.time()}.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(kl_loss_history) + 1), kl_loss_history, marker='o', color='green', label="KL Loss")
plt.title("Training Recon Loss Over Epochs")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig(f"./losses/MS_COCO_VAE_KL_Loss_{time.time()}.png")
plt.close()
