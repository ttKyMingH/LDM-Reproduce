import os, sys
sys.path.append(os.getcwd())

import requests
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm

from sampler.DiffusionSampler import DiffusionSampler
import torch
import torch.nn.functional as F
import numpy as np

from LatentDiffusion import LatentDiffusion


class DDPMSampler(DiffusionSampler):
    def __init__(self, model: LatentDiffusion):
        super().__init__(model)

        self.time_steps = np.asarray(list(range(self.n_steps)))
        with torch.no_grad():
            # 检查模型是否为DataParallel类型
            if isinstance(self.model, torch.nn.DataParallel):
                alpha_bar = self.model.module.alpha_bar
                beta = self.model.module.beta
            else:
                alpha_bar = self.model.alpha_bar
                beta = self.model.beta

            alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.]), alpha_bar[:-1]])

            self.sqrt_alpha_bar = alpha_bar ** .5
            self.sqrt_1m_alpha_bar = (1. - alpha_bar) ** .5
            self.sqrt_recipe_alpha_bar = alpha_bar ** -.5
            self.sqrt_recipe_m1_alpha_bar = (1 / alpha_bar - 1) ** .5

            var = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
            self.log_var = torch.log(torch.clamp(var, min=1e-20))

            self.mean_x0_coef = beta * (alpha_bar_prev ** .5) / (1. - alpha_bar)
            self.mean_xt_coef = ((1 - beta) ** .5) * (1. - alpha_bar_prev) / (1. - alpha_bar)

    @torch.no_grad()
    def sample(self,
               shape,
               cond,
               repeat_noise=False,
               temperature=1,
               x_last=None,
               uncond_scale=1,
               uncond_cond=None,
               skip_steps=0):
        # 检查模型是否为DataParallel类型
        if isinstance(self.model, torch.nn.DataParallel):
            device = self.model.module.device()
        else:
            device = self.model.device()
            
        bs = shape[0]

        x = x_last if x_last is not None else torch.randn(shape, device=device)

        time_steps = np.flip(self.time_steps)[skip_steps:]

        for step in tqdm(time_steps):
            ts = x.new_full((bs,), step, dtype=torch.long)
            x, pred_x0, e_t = self.p_sample(x, cond, ts, step,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature,
                                            uncond_scale=uncond_scale, uncond_cond=uncond_cond)

        return x

    @torch.no_grad()
    def p_sample(self, x, c, t, step,
                 repeat_noise=False,
                 temperature=1,
                 uncond_scale=1, uncond_cond=None
                 ):
        eps_t = self.get_eps(x, t, c, uncond_scale, uncond_cond)
        bs = x.shape[0]

        sqrt_recip_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recipe_alpha_bar[step])
        sqrt_recip_m1_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recipe_m1_alpha_bar[step])

        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * eps_t

        mean_x0_coef = x.new_full((bs, 1, 1, 1), self.mean_x0_coef[step])
        mean_xt_coef = x.new_full((bs, 1, 1, 1), self.mean_xt_coef[step])

        mean = mean_x0_coef * x0 + mean_xt_coef * x

        log_var = x.new_full((bs, 1, 1, 1), self.log_var[step])

        if step == 0:
            noise = torch.Tensor([0])

        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]))

        else:
            noise = torch.randn(x.shape)

        noise = noise * temperature

        # 检查模型是否为DataParallel类型
        if isinstance(self.model, torch.nn.DataParallel):
            device = self.model.module.device()
        else:
            device = self.model.device()
            
        noise = noise.to(device)

        x_prev = mean + (log_var * 0.5).exp() * noise

        return x_prev, x0, eps_t

    # @torch.no_grad()
    def q_sample(self, x0, index, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        mean = self.sqrt_alpha_bar[index].view(-1, 1, 1, 1)
        var = self.sqrt_1m_alpha_bar[index].view(-1, 1, 1, 1)

        # return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise
        return mean * x0 + var * noise

    def loss(self, x0, c, noise=None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, noise).to(x0.device)
        eps_theta = self.model(xt, t, c)

        return F.mse_loss(eps_theta, noise)



def _test_q_sample():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    # image = Image.open("/data/000000039769.jpg")

    image_size = 128
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


    x_start = transform(image).unsqueeze(0)
    print(x_start.shape)

    ldm = LatentDiffusion(None, None, None, 1, 500, 0.0001, 0.02)
    ddpm = DDPMSampler(ldm)

    plt.figure(figsize=(16, 8))
    for idx, t in enumerate([0, 50, 100, 200, 300, 499]):
        x_noisy = ddpm.q_sample(x_start, t)
        noisy_image = (x_noisy.squeeze().permute(1, 2, 0) + 1) * 127.5
        noisy_image = noisy_image.numpy().astype(np.uint8)
        plt.subplot(1, 6, 1 + idx)
        plt.imshow(noisy_image)
        plt.axis("off")
        plt.title(f"t={t}")

    plt.show()


if __name__ == "__main__":
    _test_q_sample()


