import torch
from torch import nn

class DiffusionWrapper(nn.Module):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x, time_steps, context):
        return self.diffusion_model(x, time_steps, context)


class LatentDiffusion(nn.Module):
    def __init__(self,
                 unet_model,
                 auto_encoder,
                 context_embedder,
                 latent_scaling_factor,
                 n_steps,
                 linear_start,
                 linear_end
                 ):
        super().__init__()

        self.model = DiffusionWrapper(unet_model)

        self.first_stage_model = auto_encoder
        self.latent_scaling_factor = latent_scaling_factor

        self.cond_stage_model = context_embedder

        self.n_steps = n_steps

        # beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        beta = torch.linspace(linear_start, linear_end, n_steps, dtype=torch.float64)
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)

        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    @torch.no_grad()
    def device(self):
        return next(iter(self.model.parameters())).device

    def get_conditioning(self, prompts):
        return self.cond_stage_model(prompts)

    def autoencoder_encode(self, image):
        return self.first_stage_model.encode(image).sample() * self.latent_scaling_factor

    def autoencoder_decode(self, z):
        return self.first_stage_model.decode(z / self.latent_scaling_factor)

    def forward(self, x, t, context):
        return self.model(x, t, context)









