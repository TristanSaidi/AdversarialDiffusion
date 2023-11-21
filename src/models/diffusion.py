import torch
import torch.nn as nn
from utils.diff_utils import extract, linear_beta_schedule

class Diffusion(nn.Module):
    def __init__(self, data_shape, model, T=300, device='cuda'):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.data_shape = data_shape
        self.T = T
        self.beta = linear_beta_schedule(T).to(self.device)
        # constants for sampling
        self._init_scalars()

    def _init_scalars(self):
        self.alpha = 1 - self.beta
        # scalars related to alpha
        self.cumprod_alpha = torch.cumprod(self.alpha, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.cumprod_alpha[:-1], (1, 0), value=1)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alpha)
        # backward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.cumprod_alpha)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.cumprod_alpha)
        # forward process
        self.posterior_variance = self.beta * (1 - self.alphas_cumprod_prev) / (1 - self.cumprod_alpha)

    def forward(self, x, t, y):
        return self.model(x, t.squeeze(-1), y)

    def sample_q_t(self, x_0, t, noise=None):
        " forward noising step "
        if noise is None:
            noise = torch.randn_like(x_0).to(self.device)
        # scalars for sampling
        sqrt_alpha_cumprod = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        # sample
        x_t = sqrt_alpha_cumprod * x_0 +  sqrt_one_minus_alpha_cumprod * noise
        return x_t
    
    @torch.no_grad()
    def sample_p_t(self, x_t_prev, t, y):
        " backward denoising step "
        # scalars for sampling
        betas_t = extract(self.beta, t, x_t_prev.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t_prev.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t_prev.shape)

        mean = sqrt_recip_alphas_t * (x_t_prev - betas_t * self.model(x_t_prev, t, y) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variances_t = extract(self.posterior_variance, t, x_t_prev.shape)
        std = torch.sqrt(posterior_variances_t)
        
        noise = torch.randn_like(x_t_prev) if t > 0 else torch.zeros_like(x_t_prev)
        x_t = mean + std * noise.to(self.device)
        return x_t
    
    @torch.no_grad()
    def sample(self, y):
        " sample from the model "
        x_0 = torch.randn(size=self.data_shape).to(self.device)
        x_0 = x_0[None, None, :, :] # (B, C, H, W)
        x_t = x_0
        for t in reversed(range(self.T)):
            t = torch.tensor([t]).to(self.device)
            x_t = self.sample_p_t(x_t, t, y)
        return x_t
            
