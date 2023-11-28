import torch

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# helper function from Ho et al.
def extract(scalars, t, shape):
    batch_size = t.shape[0]
    out = scalars.gather(-1, t)
    out = out.reshape(batch_size, *((1,)*(len(shape)-1))).to(t.device)
    return out