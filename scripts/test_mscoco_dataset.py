import os, sys
sys.path.append(os.getcwd())

import torch
from torchvision import transforms
from datasets import load_dataset
from scripts.utils import MSCOCOImageDataset
from omegaconf import OmegaConf

# 加载配置文件
cfg_data = OmegaConf.load('./config/dataset_mscoco.yaml')

# 设置参数
image_size = 256
batch_size = 4

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# 加载数据集
download_dir = 'D:/datasets'
os.makedirs(download_dir, exist_ok=True)  # 确保目录存在
ds = load_dataset(cfg_data.dataset.name, cache_dir=download_dir)

# 创建MSCOCOImageDataset实例
dataset_test = MSCOCOImageDataset(
    dataset=ds, 
    splits=['test'],  # 可以是'train', 'val', 'test'
    transform=transform, 
    filter_channels=False,  # 是否只保留3通道图片
    target_size=(image_size, image_size)
)

# 创建DataLoader
test_loader = torch.utils.data.DataLoader(
    dataset_test, 
    batch_size=batch_size, 
    shuffle=False
)

# 测试数据集
print(f"数据集大小: {len(dataset_test)}")

# 获取一个批次并显示信息
for images, captions in test_loader:
    print(f"图像批次形状: {images.shape}")
    print(f"图像数据类型: {images.dtype}")
    print(f"标题数据: {captions}")
    break  # 只显示第一个批次