from typing import List

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet_attention import SpatialTransformer

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    return GroupNorm32(32, channels)
    # return nn.InstanceNorm2d(channels)

class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, channels: int, d_t_emb: int, out_channels: int = None):
        super().__init__()

        if out_channels is None:
            out_channels = channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )

        self.out_layer = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.in_layers(x)
        t_emb = self.time_emb(t_emb)
        h = h + t_emb[:, :, None, None]
        h = self.out_layer(h)

        return self.skip_connection(x) + h

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x

class UNetModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channels: int,
                 n_res_blocks: int,
                 attention_levels: List[int],
                 channel_multipliers: List[int],
                 n_heads: int,
                 tf_layers: int = 1,
                 d_cond: int = 768,
                 ):
        super().__init__()

        self.channels = channels
        levels = len(channel_multipliers)

        d_time_emb = channels * 4
        self.time_emb = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        self.input_block = nn.ModuleList()
        self.input_block.append(TimestepEmbedSequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
        ))

        input_block_channels = [channels]
        channel_list = [channels * m for m in channel_multipliers]

        for i in range(levels):
            for _ in range(n_res_blocks):
                layers = [ResBlock(channels, d_time_emb, channel_list[i])]
                channels = channel_list[i]

                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))

                self.input_block.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)

            if i != levels - 1:
                self.input_block.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, n_heads, tf_layers, d_cond),
            ResBlock(channels, d_time_emb),
        )

        self.output_blocks = nn.ModuleList([])

        for i in reversed(range(levels)):
            for j in range(n_res_blocks + 1):
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_emb, channel_list[i])]
                channels = channel_list[i]

                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))

                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 1000):
        half = self.channels // 2

        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)

        args = time_steps[:, None].float() * frequencies[None]

        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor):
        x_input_blocks = []

        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_emb(t_emb)

        for module in self.input_block:
            x = module(x, t_emb, cond)
            x_input_blocks.append(x)

        # print(x.shape)

        x = self.middle_block(x, t_emb, cond)

        # print(x.shape)

        for module in self.output_blocks:
            x = torch.cat([x, x_input_blocks.pop()], dim=1)
            x = module(x, t_emb, cond)

        return self.out(x)


