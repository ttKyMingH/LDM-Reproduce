import torch

def train_DDPM(train_loader, optimizer, sampler, d_cond, device):
    epoch_loss = []
    for step, (pics, labels) in enumerate(train_loader):
        pics = pics.to(device)
        optimizer.zero_grad()
        if d_cond != 0:
            cond = torch.repeat_interleave(labels, d_cond, dim=0).reshape(-1, 1, d_cond).to(device).to(torch.float32)
        else:
            cond = None

        loss = sampler.loss(pics, cond)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_epoch_loss = torch.tensor(epoch_loss).mean().item()
    return avg_epoch_loss

def train_VAE(train_loader, optimizer, auto_encoder, loss, device):
    epoch_total_loss = []
    epoch_recon_loss = []
    epoch_kl_loss = []
    for step, (pics, _) in enumerate(train_loader):
        pics = pics.to(device)
        optimizer.zero_grad()

        z = auto_encoder.encode(pics)
        pics_hat = auto_encoder.decode(z.sample())

        recon_loss, kl_loss = loss(pics, pics_hat, z.mean, z.log_var, 0.5)
        ls = recon_loss + kl_loss

        ls.backward()
        optimizer.step()

        epoch_loss.append(ls.item())
        epoch_recon_loss(recon_loss.item())
        epoch_kl_loss(kl_loss.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Recon Loss: {recon_loss.item():.4f} - KL Loss: {kl_loss.item():.4f}")

    avg_epoch_loss = torch.tensor(epoch_loss).mean().item()
    avg_epoch_recon_loss = torch.tensor(epoch_recon_loss).mean().item()
    avg_epoch_kl_loss = torch.tensor(epoch_kl_loss).mean().item()

    return avg_epoch_loss, avg_epoch_recon_loss, avg_epoch_kl_loss

def train_LDM(train_loader, optimizer, ldm, sampler, d_cond, device):
    epoch_loss = []
    for step, (pics, labels) in enumerate(train_loader):
        pics = pics.to(device)
        optimizer.zero_grad()

        z = ldm.autoencoder_encode(pics)
        if d_cond != 0:
            cond = torch.repeat_interleave(labels, d_cond, dim=0).reshape(-1, 1, d_cond).to(device).to(torch.float32)
        else:
            cond = None

        loss = sampler.loss(z, cond)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_epoch_loss = torch.tensor(epoch_loss).mean().item()
    return avg_epoch_loss

from datasets import concatenate_datasets
from torch.utils.data import Dataset

class MSCOCOImageDataset(Dataset):
    def __init__(self, dataset, splits=['train'], transform=None, filter_channels=False):
        """
        初始化 MSCOCO 图像数据集
        
        参数:
            dataset: HuggingFace 数据集对象
            splits: 数据集分割名称列表，默认为['train']
            transform: 图像转换函数
            filter_channels: 是否只保留3通道图片，默认为True
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
        
        # 如果需要过滤通道，预处理数据集只保留3通道图片
        if self.filter_channels:
            self._filter_three_channels()
    
    def _filter_three_channels(self):
        """过滤数据集，只保留3通道的图片"""
        import numpy as np
        from tqdm import tqdm
        
        print("正在过滤非3通道图片...")
        valid_indices = []
        
        for i in tqdm(range(len(self.dataset))):
            try:
                img = self.dataset[i]['image']
                # 检查图像是否为3通道
                if hasattr(img, 'shape') and len(img.shape) == 3 and img.shape[2] == 3:
                    valid_indices.append(i)
                # 对于PIL图像
                elif hasattr(img, 'mode') and img.mode == 'RGB':
                    valid_indices.append(i)
            except Exception as e:
                print(f"处理图像 {i} 时出错: {e}")
                continue
        
        # 使用过滤后的索引创建新的数据集
        self.dataset = self.dataset.select(valid_indices)
        print(f"过滤完成，保留了 {len(valid_indices)} 张3通道图片，占原数据集的 {len(valid_indices)/len(self.dataset)*100:.2f}%")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        en = self.dataset[idx]['en']
        if self.transform:
            image = self.transform(image)
        return image, en