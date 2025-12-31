# License : CC BY-NC-SA 4.0

import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start = 0.0001, beta_end = 0.02):
    """linar schedule from the original DDPM paper https://arxiv.org/abs/2006.11239"""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008, min_beta=0.001, max_beta=0.999):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, min_beta, max_beta)


# Function to get alphas and betas
def get_alph_bet(timesteps, schedule=cosine_beta_schedule):
    """Function to get alphas and betas"""
    
    # define beta
    betas = schedule(timesteps)

    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0) # cumulative product of alpha
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # corresponding to the prev const
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    const_dict = {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,           # <-- add this
        'alphas_cumprod_prev': alphas_cumprod_prev, # useful for posterior mean
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance
    }
    
    return const_dict



def extract(constants, batch_t, x_shape):
    """Extract the values needed for time t"""
    
    diffusion_batch_size = batch_t.shape[0]
    
    # get a list of the appropriate constants of each timesteps
    out = constants.gather(-1, batch_t) 
    
    return out.reshape(diffusion_batch_size, *((1,) * (len(x_shape) - 1)))


def q_sample(constants_dict, batch_x0, batch_t, noise=None, clipping=1):
    """Forward diffusion (using the nice property)"""
    """ Implement clipping technique from https://arxiv.org/pdf/2301.10972 to allow """
    device = batch_x0.device
    if noise is None:
        noise = torch.randn_like(batch_x0)

    noise = noise.to(device)
    batch_t = batch_t.to(device)

    sqrt_alphas_cumprod_t = extract(constants_dict['sqrt_alphas_cumprod'].to(device), batch_t, batch_x0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(constants_dict['sqrt_one_minus_alphas_cumprod'].to(device), batch_t, batch_x0.shape)

    return sqrt_alphas_cumprod_t * batch_x0 * clipping + sqrt_one_minus_alphas_cumprod_t * noise

def q_sample_stochastic_interpolant(constants_dict, z0, z1, batch_t, s_func=None):
    """Forward diffusion for stochastic interpolant models"""
    device = z0.device
    batch_t = batch_t.to(device)

    sqrt_alphas_cumprod_t = extract(constants_dict['sqrt_alphas_cumprod'].to(device), batch_t, z0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(constants_dict['sqrt_one_minus_alphas_cumprod'].to(device), batch_t, z0.shape)

    if s_func is not None:
        s_t = s_func(batch_t).to(device)
        z_t = (1 - s_t) * (sqrt_alphas_cumprod_t * z0) + s_t * (sqrt_one_minus_alphas_cumprod_t * z1)
    else:
        z_t = sqrt_alphas_cumprod_t * z0 + sqrt_one_minus_alphas_cumprod_t * z1

    return z_t