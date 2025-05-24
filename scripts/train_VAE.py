import os, sys
sys.path.append(os.getcwd())

import time
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import torch
from torch import nn
from torchvision import datasets, transforms

from model.AutoEncoder import AutoEncoder, Encoder, Decoder
from loss.VAE_loss import LossWithLPIPS
from utils import train_VAE, evaluate_VAE, test_VAE, MSCOCOImageDataset, init_wandb

from datasets import load_dataset

from argparse import Namespace

# 加载配置文件
cfg = OmegaConf.load('./config/autoencoder_kl_32x32x4.yaml')

# 设备选择
if torch.cuda.is_available() and len(cfg.training.use_cuda):
    # 指定要使用的GPU设备
    device_ids = cfg.training.use_cuda
    device = f"cuda:{device_ids[0]}"  # 主设备设为cuda:0
    print(f"使用GPU设备: {device_ids}")
else:
    device = "cpu"
    print("使用CPU进行训练")

# 设置训练参数
image_size = cfg.training.image_size
batch_size = cfg.training.batch_size
epochs = cfg.training.epochs
learning_rate = cfg.training.learning_rate
vae_model_path = cfg.training.model_path
weight_decay = cfg.training.weight_decay
load_model = cfg.training.load_model

# 初始化wandb
config = Namespace(
    project_name="VAE",
    batch_size = batch_size,
    lr = learning_rate,
    weight_decay = weight_decay,
    optim_type = "AdamW",
    epochs = epochs,
    f = 2 ** (len(cfg.model.encoder.channel_multipliers) - 1),
    c = cfg.model.auto_encoder.z_channels,
    kl_weight = cfg.model.loss.kl_weight,
    ckpt_path = vae_model_path,
)
my_wandb =init_wandb(config=config, name=f"VAE_{time.time()}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# 加载数据集
cfg_data = OmegaConf.load('./config/dataset_mscoco.yaml')
# 本地
# download_dir = 'D:/datasets'
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
loss = LossWithLPIPS(**cfg.model.loss).to(device)

start_time = time.time()
auto_encoder.best_metric = -1.0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # 训练阶段
    avg_epoch_loss, avg_epoch_recon_loss, avg_epoch_kl_loss, avg_epoch_nll_loss = train_VAE(train_loader, optimizer, auto_encoder, loss, device)
    
    # 验证阶段
    avg_val_loss, avg_val_recon_loss, avg_val_kl_loss, avg_val_nll_loss = evaluate_VAE(val_loader, auto_encoder, loss, device)
    
    # 保存模型
    if avg_val_loss < auto_encoder.best_metric or auto_encoder.best_metric == -1.0:
        auto_encoder.best_metric = avg_val_loss
        if isinstance(auto_encoder, nn.DataParallel):
            torch.save(auto_encoder.module.state_dict(), vae_model_path)
        else:
            torch.save(auto_encoder.state_dict(), vae_model_path)
        
        # 使用wandb保存模型参数
        arti_model = my_wandb.Artifact(f'vae_model_epoch_{epoch}', type='model')
        arti_model.add_file(config.ckpt_path)
        my_wandb.log_artifact(arti_model)

    my_wandb.log({
        'epoch': epoch,
        'train/total_loss': avg_epoch_loss,
        'train/recon_loss': avg_epoch_recon_loss,
        'train/kl_loss': avg_epoch_kl_loss,
        'train/nll_loss': avg_epoch_nll_loss,
        'val/total_loss': avg_val_loss,
        'val/recon_loss': avg_val_recon_loss,
        'val/kl_loss': avg_val_kl_loss,
        'val/nll_loss': avg_val_nll_loss,
        'best_val_total_loss': auto_encoder.best_metric,
    })
    print(f"第{epoch + 1}批次,训练完毕.")
    print(f"训练 - 总损失: {avg_epoch_loss:.6f}, 重构损失: {avg_epoch_recon_loss:.6f}, KL损失: {avg_epoch_kl_loss:.6f}, NLL损失: {avg_epoch_nll_loss:.6f}")
    print(f"验证 - 总损失: {avg_val_loss:.6f}, 重构损失: {avg_val_recon_loss:.6f}, KL损失: {avg_val_kl_loss:.6f}, NLL损失: {avg_val_nll_loss:.6f}")
    print("-" * 50)

my_wandb.finish()

# 测试阶段 - 加载最佳模型参数
print("加载最佳模型参数进行测试...")
# 加载保存的最佳模型
if isinstance(auto_encoder, nn.DataParallel):
    auto_encoder.module.load_state_dict(torch.load(vae_model_path))
else:
    auto_encoder.load_state_dict(torch.load(vae_model_path))

avg_test_loss, avg_test_recon_loss, avg_test_kl_loss, avg_test_nll_loss = test_VAE(test_loader, auto_encoder, loss, device)
print("训练完成!")
print(f"总训练时间: {(time.time() - start_time) / 60:.2f} 分钟.")
print(f"最终测试总损失: {avg_test_loss:.6f}, 重构损失: {avg_test_recon_loss:.6f}, KL损失: {avg_test_kl_loss:.6f}, NLL损失: {avg_test_nll_loss:.6f}")
