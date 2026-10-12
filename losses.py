"""Objective functions from the paper: MSE + adversarial + L1."""
import torch
import torch.nn.functional as F


def discriminator_lsgan_loss(real_logits, fake_logits):
    real_loss = 0.5 * torch.mean((real_logits - 1.0) ** 2)
    fake_loss = 0.5 * torch.mean(fake_logits ** 2)
    return real_loss + fake_loss


def generator_losses(predicted, target, fake_logits,
                     alpha=1.0, beta=10.0, gamma=0.01):
    mse = F.mse_loss(predicted, target)
    adversarial = 0.5 * torch.mean((fake_logits - 1.0) ** 2)
    l1 = F.l1_loss(predicted, target)
    total = alpha * mse + beta * adversarial + gamma * l1
    return total, {"mse": mse.detach(), "adversarial": adversarial.detach(),
                   "l1": l1.detach(), "total": total.detach()}
