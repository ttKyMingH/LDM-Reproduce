import math
import numpy as np
from typing import List
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, encoder: 'Encoder', decoder: 'Decoder', emb_channels: int, z_channels: int):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, img: torch.Tensor) -> 'GaussianDistribution':
        z = self.encoder(img)
        moments = self.quant_conv(z)
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        z = self.post_quant_conv(z)
        return self.decoder(z)

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=1):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=1):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # Swish激活函数
    return x*torch.sigmoid(x)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class Encoder(nn.Module):
    def __init__(self, *, 
            channels: int,
            in_channels: int,
            out_channels: int,
            channel_multipliers: List[int],
            num_resnet_blocks: int,
            dropout: float = 0.0,
            resolution: int,
            z_channels: int,
            double_z: bool = True,
            **ignore_kwargs
            ):
        """
        :param channels: 基础通道数, 第一个卷积层中的通道数
        :param in_channels: 输入图像的通道数
        :param out_channels: 输出通道数
        :param channel_multipliers: 通道数乘数列表, 下采样次数等于len(channel_multipliers)-1
        :param num_resnet_blocks: 每个分辨率的残差块数量
        :param dropout: 丢弃率
        :param resolution: 输入图像的分辨率
        :param z_channels: 潜在空间的通道数
        :param double_z: 是否将潜在空间的通道数加倍
        """
        super().__init__()
        self.ch = channels
        self.num_resolutions = len(channel_multipliers)
        self.num_resnet_blocks = num_resnet_blocks
        self.resolution = resolution 
        self.in_channels = in_channels

        # 下采样
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        current_resolution = resolution  
        self.in_channel_multipliers = [1,] + channel_multipliers    
        self.downsample = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in_channel  = self.ch*self.in_channel_multipliers[i_level]
            block_out_channel = self.ch*channel_multipliers[i_level]
            for i_block in range(self.num_resnet_blocks):
                block.append(ResnetBlock(in_channels=block_in_channel,
                                         out_channels=block_out_channel,
                                         dropout=dropout))
                block_in_channel = block_out_channel
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in_channel)
                current_resolution //= 2
            self.downsample.append(down)

        # 中间层
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in_channel,
                                       out_channels=block_in_channel,
                                       dropout=dropout)         
        self.mid.attn_1 = AttnBlock(block_in_channel)       
        self.mid.block_2 = ResnetBlock(in_channels=block_in_channel,
                                       out_channels=block_in_channel,
                                       dropout=dropout)

        # 结尾
        self.norm_out = Normalize(block_in_channel)
        self.conv_out = torch.nn.Conv2d(block_in_channel,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_resnet_blocks):
                h = self.downsample[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.downsample[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, 
            channels: int,
            in_channels: int,
            out_channels: int,
            channel_multipliers: List[int],
            num_resnet_blocks: int,
            dropout: float = 0.0,
            resolution: int,
            z_channels: int,
            **ignore_kwargs
            ):
        """
        :param channels: 基础通道数, 第一个卷积层中的通道数
        :param in_channels: 输入图像的通道数
        :param out_channels: 输出通道数
        :param channel_multipliers: 通道数乘数列表, 上采样次数等于len(channel_multipliers)-1
        :param num_resnet_blocks: 每个分辨率的残差块数量
        :param dropout: 丢弃率
        :param resolution: 输入图像的分辨率
        :param z_channels: 潜在空间的通道数
        """
        super().__init__()
        self.ch = channels
        self.num_resolutions = len(channel_multipliers)
        self.num_resnet_blocks = num_resnet_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        in_channel_multipliers = [1,] + channel_multipliers  
        block_in_channels = self.ch*channel_multipliers[self.num_resolutions-1]
        current_resolution = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1, z_channels, current_resolution, current_resolution)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in_channels,
                                       out_channels=block_in_channels,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in_channels)
        self.mid.block_2 = ResnetBlock(in_channels=block_in_channels,
                                       out_channels=block_in_channels,
                                       dropout=dropout)

        # upsampling
        self.upsample = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out_channels = self.ch*channel_multipliers[i_level]
            for i_block in range(self.num_resnet_blocks+1):
                block.append(ResnetBlock(in_channels=block_in_channels,
                                         out_channels=block_out_channels,
                                         dropout=dropout))
                block_in_channels = block_out_channels
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in_channels)
                current_resolution = current_resolution * 2
            self.upsample.insert(0, up) # prepend to get consistent order

         # end
        self.norm_out = Normalize(block_in_channels)
        self.conv_out = torch.nn.Conv2d(block_in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z: torch.Tensor):
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_resnet_blocks+1):
                h = self.upsample[i_level].block[i_block](h)
            if i_level != 0:
                h = self.upsample[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class GaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        """
        :param parameters: are the means and log of variances of the embedding of shape
            `[batch_size, z_channels * 2, z_height, z_height]`
        """
        # Split mean and log of variance
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        # Clamp the log of variances
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self):
        # Sample from the distribution
        return self.mean + self.std * torch.randn_like(self.std)

def loss(y, y_hat, mean, log_var, kl_scale):
    reconstruction_loss = torch.nn.functional.mse_loss(y, y_hat)
    kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var)))

    return reconstruction_loss, kl_loss * kl_scale
