from utils import train_VAE, MSCOCOImageDataset
import os, sys
sys.path.append(os.getcwd())
from omegaconf import OmegaConf
from datasets import load_dataset
import torch
from torchvision import transforms

# 定义一些基本参数
batch_size = 32
image_size = 256
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# 加载数据集
cfg_data = OmegaConf.load('./config/dataset_mscoco.yaml')
download_dir = cfg_data.dataset.cache_dir
os.makedirs(download_dir, exist_ok=True)  # 确保目录存在
ds = load_dataset(cfg_data.dataset.name, cache_dir=download_dir)

# 使用 MSCOCOImageDataset 处理数据集
dataset = MSCOCOImageDataset(ds, splits=['train', 'val', 'test', 'restval'], transform=transform)
ds_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"数据加载器: {ds_loader}")
print(f"数据集大小: {len(dataset)}")

# 查看第一个批次的数据 (DataLoader不支持索引访问)
for batch in ds_loader:
    images, captions = batch
    print(f"图像批次形状: {images.shape}")
    print(f"第一个样本的描述: {captions[0]}")
    break