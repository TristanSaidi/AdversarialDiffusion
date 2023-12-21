import torch
import torch.nn as nn
from utils.diff_utils import extract, linear_beta_schedule, cosine_beta_schedule

from models.diffusion import CondDiffusion


class Guidance(CondDiffusion):
    def __init__(self, data_shape, model, noise_schedule, T=300, p_uncond=0.1, device='cuda'):
        super().__init__(data_shape, model, noise_schedule, T, device)
        self.p_uncond = p_uncond
        self.null_class = 10

    @torch.no_grad()
    def sample_p_t(self, x_t_prev, t, y, w):
        """ Sample a single reverse process step

        Args:
            x_t_prev (tensor): previous denoising step sample
            t (int): time
            y (int): class label
            w (float): guidance strength

        Returns:
            tensor: sample from p(x_t | x_{t+1})
        """
        # scalars for sampling
        betas_t = extract(self.beta, t, x_t_prev.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t_prev.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t_prev.shape)

        # conditional score estimate
        pred_epsilon = self.model(x_t_prev, t, y)
        # unconditional score estimate
        pred_epsilon_null = self.model(x_t_prev, t, torch.tensor([self.null_class]).to(self.device))
        # weighted sum of conditional and unconditional score estimates
        pred_epsilon = (1 + w) * pred_epsilon - w * pred_epsilon_null

        mean = sqrt_recip_alphas_t * (x_t_prev - betas_t * pred_epsilon / sqrt_one_minus_alphas_cumprod_t)
        posterior_variances_t = extract(self.posterior_variance, t, x_t_prev.shape)
        std = torch.sqrt(posterior_variances_t)
        
        noise = torch.randn_like(x_t_prev) if t > 0 else torch.zeros_like(x_t_prev)
        x_t = mean + std * noise.to(self.device)
        return x_t
    
    def sample_p_t_grad(self, x_t_prev, t, y, w, deterministic=False):
        """ Sample a single reverse process step

        Args:
            x_t_prev (tensor): previous denoising step sample
            t (int): time
            y (int): class label
            w (float): guidance strength

        Returns:
            tensor: sample from p(x_t | x_{t+1})
        """
        # scalars for sampling
        betas_t = extract(self.beta, t, x_t_prev.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t_prev.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t_prev.shape)

        # conditional score estimate
        pred_epsilon = self.model(x_t_prev, t, y)
        # unconditional score estimate
        pred_epsilon_null = self.model(x_t_prev, t, torch.tensor([self.null_class]).to(self.device))
        # weighted sum of conditional and unconditional score estimates
        pred_epsilon = (1 + w) * pred_epsilon - w * pred_epsilon_null

        mean = sqrt_recip_alphas_t * (x_t_prev - betas_t * pred_epsilon / sqrt_one_minus_alphas_cumprod_t)
        posterior_variances_t = extract(self.posterior_variance, t, x_t_prev.shape)
        std = torch.sqrt(posterior_variances_t)
        
        noise = torch.randn_like(x_t_prev) if t > 0 else torch.zeros_like(x_t_prev)
        x_t = mean + std * noise.to(self.device)
        return x_t
    
    @torch.no_grad()
    def sample(self, y, w):
        """ Sample from the model

        Args:
            y (int): class label
            w (float): guidance strengh

        Returns:
            tensor: sample from model
        """
        x_0 = torch.randn(size=self.data_shape).to(self.device)
        x_0 = x_0[None, None, :, :] # (B, C, H, W)
        x_t = x_0
        for t in reversed(range(self.T)):
            t = torch.tensor([t]).to(self.device)
            x_t = self.sample_p_t(x_t, t, y, w)
        return x_t
    
    def explicit_sample(self, y, w, deterministic=False):
        " sample from the model and return latent noise"
        y = torch.tensor([y]).to(self.device)
        x_0 = torch.randn(size=self.data_shape).to(self.device)
        x_0 = x_0[None, None, :, :] # (B, C, H, W)
        x_t = x_0
        latents = [x_0]
        for t in reversed(range(self.T)):
            t = torch.tensor([t]).to(self.device)
            x_t = self.sample_p_t_grad(x_t, t, y, w, deterministic)
            latents.append(x_t)
        return latents