# License : CC BY-NC-SA 4.0

import torch
from tqdm.auto import tqdm
from torch import nn
from torch.optim import Adam

from models.forward_diffusion import q_sample, extract


def train(epochs, train_dataloader, model, T, constants_dict):
    
    DEVICE = next(model.parameters()).device
    criterion = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            optimizer.zero_grad()

            batch_size_iter = batch["pixel_values"].shape[0]
            batch_image = batch["pixel_values"].to(DEVICE)

            # GENERATE NOISE LEVEL FOR EACH IMAGE
            batch_t = torch.randint(0, T, (batch_size_iter,), device=DEVICE).long()
            # GENERATE GAUSSIAN NOISE
            noise = torch.randn_like(batch_image)
            # CREATE NOISY IMAGE
            x_noisy = q_sample(constants_dict, batch_image, batch_t, noise=noise)
            # PREDICT NOISE
            predicted_noise = model(x_noisy, batch_t)
            # COMPUTE LOSS
            loss = criterion(noise, predicted_noise)

            loop.set_postfix(loss=loss.item())

            loss.backward()
            optimizer.step()
    
    return model


@torch.no_grad()
def p_sample(constants_dict, batch_xt, predicted_noise, batch_t):
    # We first get every constants needed
    # `extract` will get the output in the same device as batch_t
    betas_t = extract(constants_dict['betas'], batch_t, batch_xt.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        constants_dict['sqrt_one_minus_alphas_cumprod'], batch_t, batch_xt.shape
    )
    sqrt_recip_alphas_t = extract(
        constants_dict['sqrt_recip_alphas'], batch_t, batch_xt.shape
    )
    
    # Equation 11 in the ddpm paper
    # Use predicted noise to predict the mean (mu theta)
    model_mean = sqrt_recip_alphas_t * (
        batch_xt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    
    # We have to be careful to not add noise if we want to predict the final image
    predicted_image = torch.zeros(batch_xt.shape).to(batch_xt.device)
    t_zero_index = (batch_t == torch.zeros(batch_t.shape).to(batch_xt.device))
    
    # Algorithm 2 line 4, we add noise when timestep is not 1:
    posterior_variance_t = extract(constants_dict['posterior_variance'], batch_t, batch_xt.shape)
    noise = torch.randn_like(batch_xt)  # create noise, same shape as batch_x
    
    # If t>0, we add noise to mu 
    predicted_image[~t_zero_index] = model_mean[~t_zero_index] + (
        torch.sqrt(posterior_variance_t[~t_zero_index]) * noise[~t_zero_index]
    ) 
    # If t=0, we don't add noise to mu
    predicted_image[t_zero_index] = model_mean[t_zero_index]
    
    return predicted_image


@torch.no_grad()
def sampling(model, shape, T, constants_dict):
    # get the device used by the model
    DEVICE = next(model.parameters()).device

    # start from pure noise (for each example in the batch)
    # instantiate it on the same device as `model` 
    batch_xt = torch.randn(shape, device=DEVICE)
    
    # timestep of the noisy data (we begin at T for each data)
    batch_t = torch.ones(shape[0]) * T
    batch_t = batch_t.type(torch.int64).to(DEVICE)
    
    imgs = []

    for t in tqdm(reversed(range(0, T)), desc='Sampling Loop time step', total=T):
        batch_t -= 1
        
        predicted_noise = model(batch_xt, batch_t)
        
        batch_xt = p_sample(constants_dict, batch_xt, predicted_noise, batch_t)
        
        imgs.append(batch_xt.cpu())
        
    return imgs