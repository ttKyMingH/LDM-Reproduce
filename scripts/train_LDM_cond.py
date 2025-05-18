import os, sys
sys.path.append(os.getcwd())

import time
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import torch
from torch import nn
from torchvision import datasets, transforms

from model.UNet import UNetModel
from LatentDiffusion import LatentDiffusion
from sampler.DDPMSampler import DDPMSampler
from model.AutoEncoder import AutoEncoder, Encoder, Decoder
from model.Context_embedder import ContextEmbedder
from utils import train_LDM

# 加载配置文件
cfg = OmegaConf.load('./config/ldm_cond.yaml')

# 设置训练参数
device = "cuda" if torch.cuda.is_available() and cfg.training.use_cuda else "cpu"
image_size = cfg.training.image_size
batch_size = cfg.training.batch_size
epochs = cfg.training.epochs
learning_rate = cfg.training.learning_rate
model_path = cfg.training.model_path
vae_model_path = cfg.training.vae_model_path
weight_decay = cfg.training.weight_decay
load_model = cfg.training.load_model
d_cond = cfg.model.unet.d_cond

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

dataset_train = MSCOCOImageDataset(ds, splits=['train'], transform=transform)
dataset_val = MSCOCOImageDataset(ds, splits=['val'], transform=transform)
dataset_test = MSCOCOImageDataset(ds, splits=['test'], transform=transform)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

unet = UNetModel(
    **cfg.model.unet
)

encoder = Encoder(
    **cfg.model.encoder
)

decoder = Decoder(
    **cfg.model.decoder
)

auto_encoder = AutoEncoder(
    encoder=encoder,
    decoder=decoder,
    **cfg.model.auto_encoder
)

cond_encoder = ContextEmbedder(
    **cfg.model.context_embedder
)

auto_encoder.load_state_dict(torch.load(vae_model_path))

ldm = LatentDiffusion(
    unet_model=unet,
    auto_encoder=auto_encoder,
    context_embedder=cond_encoder,
    **cfg.model.ldm
).to(device)

if load_model:
    unet.load_state_dict(torch.load(model_path))

ddpm = DDPMSampler(ldm)

optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate, weight_decay=weight_decay)

loss_history = []
val_loss_history = []
start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    ldm.train()
    avg_epoch_loss = train_LDM(train_loader, optimizer, ldm, ddpm, d_cond, device)
    loss_history.append(avg_epoch_loss)
    
    avg_val_loss = evaluate_LDM(val_loader, ldm, ddpm, d_cond, device)
    val_loss_history.append(avg_val_loss)

    torch.save(unet.state_dict(), model_path)
    print(f"Epoch {epoch + 1} Completed. Average Loss: {avg_epoch_loss:.4f}")
    print("-" * 50)

avg_test_loss = test_LDM(test_loader, ldm, ddpm, d_cond, device)
print("Training completed!")
print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes.")
print(f"Final test loss: {avg_test_loss:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label="Training Loss")
plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, marker='s', label="Validation Loss")
plt.title("Training Loss and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()
plt.savefig(f"./losses/MS_COCO_ldm_cond_train_val_{time.time()}.png")
plt.close()
