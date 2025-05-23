import torch
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
# 加载配置文件
cfg = OmegaConf.load('./config/ldm_cond.yaml')

def train_VAE(train_loader, optimizer, auto_encoder, loss, device):
    epoch_total_loss = []
    epoch_recon_loss = []
    epoch_kl_loss = []
    epoch_nll_loss = []
    for step, batch in enumerate(train_loader):  
        pics, _ = batch
        pics = pics.to(device)
        optimizer.zero_grad()

        # 检查是否为DataParallel模型，如果是则使用module属性访问原始模型
        if isinstance(auto_encoder, torch.nn.DataParallel):
            z = auto_encoder.module.encode(pics)
            pics_hat = auto_encoder.module.decode(z.sample())
        else:
            z = auto_encoder.encode(pics)
            pics_hat = auto_encoder.decode(z.sample())

        ls, log = loss(pics, pics_hat, z, split="train")

        ls.backward()
        optimizer.step()

        epoch_total_loss.append(ls.item())
        epoch_recon_loss.append(log['train/rec_loss'].item())
        epoch_kl_loss.append(log['train/kl_loss'].item())
        epoch_nll_loss.append(log['train/nll_loss'].item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Recon Loss: {log['train/rec_loss'].item():.6f} - KL Loss: {log['train/kl_loss'].item():.6f} - NLL Loss: {log['train/nll_loss'].item():.6f} - Total Loss: {ls.item():.6f}")

    avg_epoch_loss = torch.tensor(epoch_total_loss).mean().item()
    avg_epoch_recon_loss = torch.tensor(epoch_recon_loss).mean().item()
    avg_epoch_kl_loss = torch.tensor(epoch_kl_loss).mean().item()
    avg_epoch_nll_loss = torch.tensor(epoch_nll_loss).mean().item()

    return avg_epoch_loss, avg_epoch_recon_loss, avg_epoch_kl_loss, avg_epoch_nll_loss

def evaluate_VAE(val_loader, auto_encoder, loss, device):
    auto_encoder.eval()
    with torch.no_grad():
        epoch_total_loss = []
        epoch_recon_loss = []
        epoch_kl_loss = []
        epoch_nll_loss = []
        for step, batch in enumerate(val_loader):
            pics, _ = batch
            pics = pics.to(device)
            
            # 检查是否为DataParallel模型
            if isinstance(auto_encoder, torch.nn.DataParallel):
                z = auto_encoder.module.encode(pics)
                pics_hat = auto_encoder.module.decode(z.sample())
            else:
                z = auto_encoder.encode(pics)
                pics_hat = auto_encoder.decode(z.sample())
            
            ls, log = loss(pics, pics_hat, z, split="val")
            
            epoch_total_loss.append(ls.item())
            epoch_recon_loss.append(log['val/rec_loss'].item())
            epoch_kl_loss.append(log['val/kl_loss'].item())
            epoch_nll_loss.append(log['val/nll_loss'].item())
            
        avg_epoch_loss = torch.tensor(epoch_total_loss).mean().item()
        avg_epoch_recon_loss = torch.tensor(epoch_recon_loss).mean().item()
        avg_epoch_kl_loss = torch.tensor(epoch_kl_loss).mean().item()
        avg_epoch_nll_loss = torch.tensor(epoch_nll_loss).mean().item()
        
        print(f"验证集 - 总损失: {avg_epoch_loss:.6f} - 重构损失: {avg_epoch_recon_loss:.6f} - KL损失: {avg_epoch_kl_loss:.6f} - NLL损失: {avg_epoch_nll_loss:.6f}")
        
    return avg_epoch_loss, avg_epoch_recon_loss, avg_epoch_kl_loss, avg_epoch_nll_loss

def test_VAE(test_loader, auto_encoder, loss, device):
    auto_encoder.eval()
    with torch.no_grad():
        epoch_total_loss = []
        epoch_recon_loss = []
        epoch_kl_loss = []
        epoch_nll_loss = []
        for step, batch in enumerate(test_loader):
            pics, _ = batch
            pics = pics.to(device)
            
            # 检查是否为DataParallel模型
            if isinstance(auto_encoder, torch.nn.DataParallel):
                z = auto_encoder.module.encode(pics)
                pics_hat = auto_encoder.module.decode(z.sample())
            else:
                z = auto_encoder.encode(pics)
                pics_hat = auto_encoder.decode(z.sample())
            
            ls, log = loss(pics, pics_hat, z, split="test")
            
            epoch_total_loss.append(ls.item())
            epoch_recon_loss.append(log['test/rec_loss'].item())
            epoch_kl_loss.append(log['test/kl_loss'].item())
            epoch_nll_loss.append(log['test/nll_loss'].item())
            
        avg_epoch_loss = torch.tensor(epoch_total_loss).mean().item()
        avg_epoch_recon_loss = torch.tensor(epoch_recon_loss).mean().item()
        avg_epoch_kl_loss = torch.tensor(epoch_kl_loss).mean().item()
        avg_epoch_nll_loss = torch.tensor(epoch_nll_loss).mean().item()
        
        print(f"测试集 - 总损失: {avg_epoch_loss:.6f} - 重构损失: {avg_epoch_recon_loss:.6f} - KL损失: {avg_epoch_kl_loss:.6f} - NLL损失: {avg_epoch_nll_loss:.6f}")
        
    return avg_epoch_loss, avg_epoch_recon_loss, avg_epoch_kl_loss, avg_epoch_nll_loss

def train_LDM(train_loader, optimizer, ldm, sampler, d_cond, device):
    epoch_loss = []
    for step, (pics, labels) in enumerate(train_loader):
        pics = pics.to(device)
        optimizer.zero_grad()
        
        # 检查是否为DataParallel模型，如果是则使用module属性访问原始模型
        if isinstance(ldm, torch.nn.DataParallel):
            z = ldm.module.autoencoder_encode(pics)
        else:
            z = ldm.autoencoder_encode(pics)

        if d_cond != 0:
            if isinstance(ldm, torch.nn.DataParallel):
                cond = ldm.module.get_conditioning(labels).reshape(-1, 1, d_cond).to(device).to(torch.float32)
            else:
                cond = ldm.get_conditioning(labels).reshape(-1, 1, d_cond).to(device).to(torch.float32)
        else:
            cond = None

        loss = sampler.loss(z, cond)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Loss: {loss.item():.6f}")

    avg_epoch_loss = torch.tensor(epoch_loss).mean().item()
    return avg_epoch_loss

def evaluate_LDM(val_loader, ldm, sampler, d_cond, device):
    ldm.eval()
    with torch.no_grad():
        val_losses = []
        for pics, labels in val_loader:
            pics = pics.to(device)
            
            # 检查是否为DataParallel模型，如果是则使用module属性访问原始模型
            if isinstance(ldm, torch.nn.DataParallel):
                z = ldm.module.autoencoder_encode(pics)
            else:
                z = ldm.autoencoder_encode(pics)

            if d_cond != 0:
                if isinstance(ldm, torch.nn.DataParallel):
                    cond = ldm.module.get_conditioning(labels).reshape(-1, 1, d_cond).to(device).to(torch.float32)
                else:
                    cond = ldm.get_conditioning(labels).reshape(-1, 1, d_cond).to(device).to(torch.float32)
            else:
                cond = None

            loss = sampler.loss(z, cond)
            val_losses.append(loss.item())
        avg_val_loss = torch.tensor(val_losses).mean().item()
        print(f"Validation Loss: {avg_val_loss:.6f}")
    return avg_val_loss

def test_LDM(test_loader, ldm, sampler, d_cond, device):
    ldm.eval()
    with torch.no_grad():
        test_losses = []
        for pics, labels in test_loader:
            pics = pics.to(device)

            # 检查是否为DataParallel模型，如果是则使用module属性访问原始模型
            if isinstance(ldm, torch.nn.DataParallel):
                z = ldm.module.autoencoder_encode(pics)
            else:
                z = ldm.autoencoder_encode(pics)

            if d_cond != 0:
                if isinstance(ldm, torch.nn.DataParallel):
                    cond = ldm.module.get_conditioning(labels).reshape(-1, 1, d_cond).to(device).to(torch.float32)
                else:
                    cond = ldm.get_conditioning(labels).reshape(-1, 1, d_cond).to(device).to(torch.float32)
            else:
                cond = None
                
            loss = sampler.loss(z, cond)
            test_losses.append(loss.item())
        avg_test_loss = torch.tensor(test_losses).mean().item()
        print(f"Test Loss: {avg_test_loss:.6f}")
    return avg_test_loss

from datasets import concatenate_datasets
from torch.utils.data import Dataset

class MSCOCOImageDataset(Dataset):
    def __init__(self, dataset, splits=['train'], transform=None, filter_channels=False, target_size=(256, 256)):
        """
        初始化 MSCOCO 图像数据集
        
        参数:
            dataset: HuggingFace 数据集对象
            splits: 数据集分割名称列表，默认为['train']
            transform: 图像转换函数
            filter_channels: 是否只保留3通道图片，默认为True
            target_size: 目标图像尺寸，默认为(256, 256)
        """
        if isinstance(splits, str):
            splits = [splits]
            
        # 收集所有指定的分割
        datasets_to_concat = [dataset[split] for split in splits if split in dataset]
        
        if not datasets_to_concat:
            raise ValueError(f"没有找到有效的分割: {splits}")
            
        if len(datasets_to_concat) == 1:
            self.dataset = datasets_to_concat[0]
        else:
            self.dataset = concatenate_datasets(datasets_to_concat)
        
        self.transform = transform
        self.filter_channels = filter_channels
        self.target_size = target_size
        
        # 如果需要过滤通道，预处理数据集只保留3通道图片
        if self.filter_channels:
            self._filter_three_channels()
    
    def _filter_three_channels(self):
        """过滤数据集，只保留3通道且transform后形状正确的图片"""
        import numpy as np
        from tqdm import tqdm
        from PIL import Image
        
        print("正在过滤非标准图片...")
        valid_indices = []
        
        for i in tqdm(range(len(self.dataset))):
            try:
                img = self.dataset[i]['image']
                # 检查图像是否为3通道
                if hasattr(img, 'shape') and len(img.shape) == 3 and img.shape[2] == 3:
                    # 应用transform检查形状
                    if self.transform:
                        transformed_img = self.transform(img)
                        if transformed_img.shape == (3, self.target_size[0], self.target_size[1]):
                            valid_indices.append(i)
                    else:
                        valid_indices.append(i)
                # 对于PIL图像
                elif hasattr(img, 'mode') and img.mode == 'RGB':
                    # 应用transform检查形状
                    if self.transform:
                        transformed_img = self.transform(img)
                        if transformed_img.shape == (3, self.target_size[0], self.target_size[1]):
                            valid_indices.append(i)
                    else:
                        valid_indices.append(i)
            except Exception as e:
                print(f"处理图像 {i} 时出错: {e}")
                continue
        
        # 使用过滤后的索引创建新的数据集
        self.dataset = self.dataset.select(valid_indices)
        print(f"过滤完成，保留了 {len(valid_indices)} 张标准图片，占原数据集的 {len(valid_indices)/len(self.dataset)*100:.2f}%")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        en = self.dataset[idx]['en']
        
        # 确保图像是RGB格式
        if hasattr(image, 'mode') and image.mode != 'RGB':
            from PIL import Image
            image = image.convert('RGB')
        
        # 应用变换（包括调整大小到目标尺寸）
        if self.transform:
            image = self.transform(image)
        else:
            # 如果没有提供transform，手动确保图像尺寸和通道
            from torchvision import transforms
            default_transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
            ])
            image = default_transform(image)

        return image, en[0]

# Suggested by Arshad, 只取了一条描述，暂未使用，而且需要修改
def collate_fn(batch):
    # 解压批次数据
    image, en = zip(*batch)

    # 将图像堆叠成张量
    images = torch.stack(image, dim=0)

    # 将不同长度的文本描述填充到相同的长度,en需是tensor
    en = pad_sequence(en, batch_first=True, padding_value=0)

    return images, en

import wandb
from argparse import Namespace

def init_wandb(config, name):
    wandb.login(key="632cb2519658eb44f9bf49759ea46ffcf4f20a0c")
    wandb.init(
        project=config.project_name,
        name=name,
        config=config.__dict__,
    )
    
    return wandb