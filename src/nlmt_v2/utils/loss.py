import torch
import torch.nn as nn

def vae_loss(recon_x, x, mu, logvar):
    bce = nn.BCEWithLogitsLoss()(recon_x, x)
    # KL Divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld
