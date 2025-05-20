import os, sys
sys.path.append(os.getcwd())

import time
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import torch
from torch import nn
from torchvision import datasets, transforms

from model.AutoEncoder import AutoEncoder, Encoder, Decoder, loss
from utils import train_VAE, evaluate_VAE, test_VAE, MSCOCOImageDataset

from datasets import load_dataset

# 加载配置文件
cfg = OmegaConf.load('./config/autoencoder_kl_32x32x4.yaml')

# 设置训练参数
# 修改设备选择，使用CUDA设备0和1
if torch.cuda.is_available() and cfg.training.use_cuda:
    # 指定要使用的GPU设备
    device_ids = [0, 1]  # 使用GPU 0和1
    device = f"cuda:{device_ids[0]}"  # 主设备设为cuda:0
    print(f"使用GPU设备: {device_ids}")
else:
    device = "cpu"
    print("使用CPU进行训练")

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

# 创建训练集、验证集和测试集
dataset_train = MSCOCOImageDataset(ds, splits=['train'], transform=transform, filter_channels=False, target_size=(image_size, image_size))
dataset_val = MSCOCOImageDataset(ds, splits=['val'], transform=transform, filter_channels=False, target_size=(image_size, image_size))
dataset_test = MSCOCOImageDataset(ds, splits=['test'], transform=transform, filter_channels=False, target_size=(image_size, image_size))

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    dataset_train, 
    batch_size=batch_size, 
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    dataset_val, 
    batch_size=batch_size, 
    shuffle=False,
)
test_loader = torch.utils.data.DataLoader(
    dataset_test, 
    batch_size=batch_size, 
    shuffle=False,
)

# 初始化模型
encoder = Encoder(**cfg.model.encoder).to(device)
decoder = Decoder(**cfg.model.decoder).to(device)

auto_encoder = AutoEncoder(
    encoder=encoder,
    decoder=decoder,
    **cfg.model.auto_encoder
).to(device)

# 先加载模型，再使用DataParallel包装
if load_model:
    auto_encoder.load_state_dict(torch.load(vae_model_path))

# 如果使用多GPU，则使用DataParallel包装模型
if torch.cuda.is_available() and cfg.training.use_cuda and len(device_ids) > 1:
    auto_encoder = nn.DataParallel(auto_encoder, device_ids=device_ids)
    print(f"模型已使用DataParallel在{len(device_ids)}个GPU上并行")

optimizer = torch.optim.AdamW(auto_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 记录训练和验证损失
total_loss_history = []
recon_loss_history = []
kl_loss_history = []
val_total_loss_history = []
val_recon_loss_history = []
val_kl_loss_history = []

start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # 训练阶段
    avg_epoch_loss, avg_epoch_recon_loss, avg_epoch_kl_loss = train_VAE(train_loader, optimizer, auto_encoder, loss, device)
    total_loss_history.append(avg_epoch_loss)
    recon_loss_history.append(avg_epoch_recon_loss)
    kl_loss_history.append(avg_epoch_kl_loss)
    
    # 验证阶段
    avg_val_loss, avg_val_recon_loss, avg_val_kl_loss = evaluate_VAE(val_loader, auto_encoder, loss, device)
    val_total_loss_history.append(avg_val_loss)
    val_recon_loss_history.append(avg_val_recon_loss)
    val_kl_loss_history.append(avg_val_kl_loss)
    
    # 保存模型
    if isinstance(auto_encoder, nn.DataParallel):
        torch.save(auto_encoder.module.state_dict(), vae_model_path)
    else:
        torch.save(auto_encoder.state_dict(), vae_model_path)
    
    # 记录训练进度
    with open('./checkpoint/vae_progress.txt', 'w') as f:
        f.write(f"训练进度: {epoch+1}/{epochs} epochs ({(epoch+1)/epochs*100:.2f}%)\n")
        f.write(f"已用时间: {(time.time() - start_time) / 60:.2f} 分钟\n")
        f.write(f"当前训练总损失: {avg_epoch_loss:.6f}\n")
        f.write(f"当前训练重构损失: {avg_epoch_recon_loss:.6f}\n")
        f.write(f"当前训练KL损失: {avg_epoch_kl_loss:.6f}\n")
        f.write(f"当前验证总损失: {avg_val_loss:.6f}\n")
        f.write(f"当前验证重构损失: {avg_val_recon_loss:.6f}\n")
        f.write(f"当前验证KL损失: {avg_val_kl_loss:.6f}\n")
    
    print(f"Epoch {epoch + 1} Completed.")
    print(f"训练 - 总损失: {avg_epoch_loss:.6f}, 重构损失: {avg_epoch_recon_loss:.6f}, KL损失: {avg_epoch_kl_loss:.6f}")
    print(f"验证 - 总损失: {avg_val_loss:.6f}, 重构损失: {avg_val_recon_loss:.6f}, KL损失: {avg_val_kl_loss:.6f}")
    print("-" * 50)

# 测试阶段
avg_test_loss, avg_test_recon_loss, avg_test_kl_loss = test_VAE(test_loader, auto_encoder, loss, device)
print("训练完成!")
print(f"总训练时间: {(time.time() - start_time) / 60:.2f} 分钟.")
print(f"最终测试总损失: {avg_test_loss:.6f}, 重构损失: {avg_test_recon_loss:.6f}, KL损失: {avg_test_kl_loss:.6f}")

# 绘制训练和验证的总损失
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(total_loss_history) + 1), total_loss_history, marker='o', label="训练总损失")
plt.plot(range(1, len(val_total_loss_history) + 1), val_total_loss_history, marker='s', label="验证总损失")
plt.title("训练和验证总损失随时间变化")
plt.xlabel("Epoch")
plt.ylabel("损失")
plt.grid()
plt.legend()
plt.savefig(f"./losses/MS_COCO_VAE_Total_Loss_{time.time()}.png")
plt.close()

# 绘制训练和验证的重构损失
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(recon_loss_history) + 1), recon_loss_history, marker='o', color='orange', label="训练重构损失")
plt.plot(range(1, len(val_recon_loss_history) + 1), val_recon_loss_history, marker='s', color='red', label="验证重构损失")
plt.title("训练和验证重构损失随时间变化")
plt.xlabel("Epoch")
plt.ylabel("损失")
plt.grid()
plt.legend()
plt.savefig(f"./losses/MS_COCO_VAE_Recon_Loss_{time.time()}.png")
plt.close()

# 绘制训练和验证的KL损失
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(kl_loss_history) + 1), kl_loss_history, marker='o', color='green', label="训练KL损失")
plt.plot(range(1, len(val_kl_loss_history) + 1), val_kl_loss_history, marker='s', color='purple', label="验证KL损失")
plt.title("训练和验证KL损失随时间变化")
plt.xlabel("Epoch")
plt.ylabel("损失")
plt.grid()
plt.legend()
plt.savefig(f"./losses/MS_COCO_VAE_KL_Loss_{time.time()}.png")
plt.close()
