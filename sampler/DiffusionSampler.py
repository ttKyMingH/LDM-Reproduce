import torch

class DiffusionSampler:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_steps = model.n_steps

    def get_eps(self, x, t, c, uncond_scale, uncond_cond):
        if uncond_cond is None and uncond_scale == 1:
            return self.model(x, t, c)

        x_in = [x] * 2
        t_in = [t] * 2
        c_in = torch.cat([uncond_cond, c])

        eps_t_uncond, eps_t_cond = self.model(x_in, t_in, c_in)

        eps_t = eps_t_uncond + uncond_scale * (eps_t_cond - eps_t_uncond)

        return eps_t

    def sample(self,
               shape,
               cond,
               repeat_noise=False,
               temperatur=1,
               x_last=None,
               uncond_scale=1,
               uncond_cond=None,
               skip_steps=0):
        raise NotImplementedError()

    def paint(self, x, t, c,
              orig=None,
              mask=None,
              uncond_scale=1,
              uncond_cond=None,
              ):
        raise NotImplementedError()

    def q_sample(self, x0, index, noise=None):
        raise NotImplementedError()
