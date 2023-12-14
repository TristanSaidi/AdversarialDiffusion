import torch
import torch.nn as nn

class Adversary(nn.Module):
    def __init__(self, data_shape, diffusion_model, device='cuda'):
        super().__init__()
        self.diffusion = diffusion_model
        self.data_shape = data_shape
        self.device = device
        self.diffusion.to(device)

    def sample(self):
        """Return unperturbed sample from the diffusion model"""
        return self.diffusion.sample()
    
    def gaussian_perturb(self, t, scale=1e-1):
        """Return perturbed sample from the diffusion model"""
        assert t < self.diffusion.T, 't must be less than T'
        noises = self.diffusion.explicit_sample()
        latent_noise = noises[self.diffusion.T - t]
        perturbation = torch.randn_like(latent_noise).to(self.device)
        
        # perturb the intermediate representation
        perturbed_latent = latent_noise + scale * perturbation
        x_t = perturbed_latent
        # continue denoising
        for i in reversed(range(0, t)):
            t_cur = torch.tensor([i]).to(self.device)
            x_t = self.diffusion.sample_p_t(x_t, t_cur)
        perturbed_sample = x_t
        original_sample = noises[-1]
        return perturbed_sample, original_sample
    
    def gradient_perturb(self, t, target, scale=1e-1):
        """Return perturbed sample from the diffusion model"""
        assert t < self.diffusion.T, 't must be less than T'
        noises = self.diffusion.explicit_sample()
        latent = noises[self.diffusion.T - t]
        # compute loss between sampled image and adversarial target
        sample = noises[-1]
        loss = torch.norm(sample - target)
        # compute gradient of loss with respect to latent rep
        grad = torch.autograd.grad(loss, latent)[0]
        normed_grad = grad / torch.norm(grad)
        # perturb the intermediate representation
        perturbed_latent = latent - scale * normed_grad
        x_t = perturbed_latent
        # continue denoising
        for i in reversed(range(0, t)):
            t_cur = torch.tensor([i]).to(self.device)
            x_t = self.diffusion.sample_p_t(x_t, t_cur)
        perturbed_sample = x_t
        original_sample = noises[-1]
        return perturbed_sample, original_sample
    
    def gradient_descent_perturb(self, t, target, scale=1e-1, steps=15):
        """Return perturbed sample from the diffusion model"""
        assert t < self.diffusion.T, 't must be less than T'
        noises = self.diffusion.explicit_sample()
        latent = noises[self.diffusion.T - t]
        # compute loss between sampled image and adversarial target
        current_sample = noises[-1]
        # compute gradient of loss with respect to latent noise
        for _ in range(steps):
            self.diffusion.zero_grad()
            loss = torch.norm(current_sample - target)
            grad = torch.autograd.grad(loss, latent)[0]
            normed_grad = grad / torch.norm(grad)
            # perturb the intermediate representation
            latent = latent - scale * normed_grad
            perturbed_x_t = latent
            # continue denoising
            for i in reversed(range(0, t)):
                t_cur = torch.tensor([i]).to(self.device)
                perturbed_x_t = self.diffusion.sample_p_t_grad(perturbed_x_t, t_cur)
            current_sample = perturbed_x_t
        perturbed_sample = perturbed_x_t
        original_sample = noises[-1]
        return perturbed_sample, original_sample
    

class ConditionalAdversary(nn.Module):
    def __init__(self, data_shape, diffusion_model, device='cuda'):
        super().__init__()
        self.diffusion = diffusion_model
        self.data_shape = data_shape
        self.device = device
        self.diffusion.to(device)

    def sample(self, y):
        """Return unperturbed sample from the diffusion model"""
        return self.diffusion.sample(y)
    
    def gaussian_perturb(self, y, t, scale=1e-1):
        """Return perturbed sample from the diffusion model"""
        assert t < self.diffusion.T, 't must be less than T'
        noises = self.diffusion.explicit_sample(y)
        latent_noise = noises[self.diffusion.T - t]
        perturbation = torch.randn_like(latent_noise).to(self.device)
        
        # perturb the intermediate representation
        perturbed_latent = latent_noise + scale * perturbation
        x_t = perturbed_latent
        # continue denoising
        for i in reversed(range(0, t)):
            t_cur = torch.tensor([i]).to(self.device)
            x_t = self.diffusion.sample_p_t(x_t, t_cur, y)
        perturbed_sample = x_t
        original_sample = noises[-1]
        return perturbed_sample, original_sample
    
    def gradient_perturb(self, y, t, target, scale=1e-1):
        """Return perturbed sample from the diffusion model"""
        assert t < self.diffusion.T, 't must be less than T'
        noises = self.diffusion.explicit_sample(y)
        latent = noises[self.diffusion.T - t]
        # compute loss between sampled image and adversarial target
        sample = noises[-1]
        loss = torch.norm(sample - target)
        # compute gradient of loss with respect to latent rep
        grad = torch.autograd.grad(loss, latent)[0]
        normed_grad = grad / torch.norm(grad)
        # perturb the intermediate representation
        perturbed_latent = latent - scale * normed_grad
        x_t = perturbed_latent
        # continue denoising
        for i in reversed(range(0, t)):
            t_cur = torch.tensor([i]).to(self.device)
            x_t = self.diffusion.sample_p_t(x_t, t_cur, y)
        perturbed_sample = x_t
        original_sample = noises[-1]
        return perturbed_sample, original_sample
    
    def gradient_descent_perturb(self, y, t, target, scale=1e-1, steps=15):
        """Return perturbed sample from the diffusion model"""
        assert t < self.diffusion.T, 't must be less than T'
        noises = self.diffusion.explicit_sample(y)
        latent = noises[self.diffusion.T - t]
        # compute loss between sampled image and adversarial target
        current_sample = noises[-1]
        # compute gradient of loss with respect to latent noise
        for _ in range(steps):
            self.diffusion.zero_grad()
            loss = torch.norm(current_sample - target)
            grad = torch.autograd.grad(loss, latent)[0]
            normed_grad = grad / torch.norm(grad)
            # perturb the intermediate representation
            latent = latent - scale * normed_grad
            perturbed_x_t = latent
            # continue denoising
            for i in reversed(range(0, t)):
                t_cur = torch.tensor([i]).to(self.device)
                perturbed_x_t = self.diffusion.sample_p_t_grad(perturbed_x_t, t_cur, y)
            current_sample = perturbed_x_t
        perturbed_sample = perturbed_x_t
        original_sample = noises[-1]
        return perturbed_sample, original_sample