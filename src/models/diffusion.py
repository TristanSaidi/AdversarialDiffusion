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
    
    @torch.no_grad()
    def estimate_likelihood(self, x_0):
        x_1 = self.sample_q_t(x_0, torch.tensor([1]).to(self.device))
        t_1 = torch.tensor([0]*x_0.shape[0]).to(self.device)
        # from x_1, predict mu(x_0)
        eps = self.model(x_1, torch.tensor([1]).to(self.device))
        betas_t = extract(self.beta, t_1, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_1, x_0.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t_1, x_0.shape)
        mu = sqrt_recip_alphas_t * (x_1 - betas_t * eps / sqrt_one_minus_alphas_cumprod_t)
        sigma_1 = torch.ones((x_0.shape[0])).to(self.device) * self.posterior_variance[1]
        # reshape tensors
        x_0 = x_0.view(x_0.shape[0], -1)
        x_1 = x_1.view(x_1.shape[0], -1)
        mu = mu.view(mu.shape[0], -1)
        # iterate through each pixel
        num_dims = x_1.shape[-1]

        log_likelihood_tensor = torch.zeros((x_0.shape[0])).to(self.device)
        for b in range(x_0.shape[0]):
            log_likelihood = 0
            for i in range(num_dims):
                x_0_b_i = x_0[b, i]
                delta_minus = -torch.inf if x_0_b_i == -1 else x_0_b_i - (1/255)
                delta_plus = torch.inf if x_0_b_i == 1 else x_0_b_i + (1/255)
                # convert to tensors
                delta_minus = torch.tensor(delta_minus).to(self.device)
                delta_plus = torch.tensor(delta_plus).to(self.device)
                # compute dimension-wise likelihood
                dist = torch.distributions.normal.Normal(mu[b, i], sigma_1[b])
                cdf_plus = dist.cdf(delta_plus)
                cdf_minus = dist.cdf(delta_minus)
                likelihood = cdf_plus - cdf_minus
                likelihood = max(likelihood, 1e-5)
                log_likelihood += torch.log(torch.tensor(likelihood).to(self.device))
            log_likelihood_tensor[b] = log_likelihood
        prob = torch.mean(log_likelihood_tensor)
        return prob
    
    @torch.no_grad()
    def interpolate(self, x_0_a, x_0_b, n_interp=10):
        latent_a = self.sample_q_t(x_0_a, torch.tensor([self.T//5]).to(self.device))
        latent_b = self.sample_q_t(x_0_b, torch.tensor([self.T//5]).to(self.device))
        # interpolate
        images = []
        for l in torch.linspace(0, 1, n_interp):
            t = torch.tensor([self.T//5]).to(self.device)
            x_t = l * latent_a + (1-l) * latent_b
            x_t = self.sample_p_t(x_t, t)
            images.append(x_t)
        return images
            
