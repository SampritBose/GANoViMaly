"""Train GANoViMaly on normal video clips."""
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import Config
from dataset import VideoClipDataset
from models import GANoViMaly
from losses import discriminator_lsgan_loss, generator_losses


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train", help="Folder containing normal training videos")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Optional checkpoint path")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    dataset = VideoClipDataset(args.data, cfg.clip_length, cfg.image_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                        drop_last=True)
    if len(loader) == 0:
        raise RuntimeError("No full batches. Reduce --batch-size or add more training clips.")

    model = GANoViMaly(cfg.channels, cfg.feature_dim, cfg.embed_dim,
                       cfg.mamba_layers, cfg.patch_size, clip_length=cfg.clip_length).to(device)
    optimizer_g = Adam(model.generator.parameters(), lr=cfg.learning_rate,
                       betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    optimizer_d = Adam(model.discriminator.parameters(), lr=cfg.learning_rate,
                       betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=cfg.epochs)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=cfg.epochs)
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint.get("epoch", 0)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running = {"g": 0.0, "d": 0.0}
        for history, target in loader:
            history, target = history.to(device), target.to(device)

            # Discriminator update
            with torch.no_grad():
                predicted = model.generator(history)
            real_logits = model.discriminator(target)
            fake_logits = model.discriminator(predicted.detach())
            loss_d = discriminator_lsgan_loss(real_logits, fake_logits)
            optimizer_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optimizer_d.step()

            # Generator update
            predicted = model.generator(history)
            fake_logits = model.discriminator(predicted)
            loss_g, components = generator_losses(
                predicted, target, fake_logits, cfg.alpha_mse,
                cfg.beta_adv, cfg.gamma_l1
            )
            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 5.0)
            optimizer_g.step()

            running["g"] += loss_g.item()
            running["d"] += loss_d.item()

        scheduler_g.step()
        scheduler_d.step()
        denom = len(loader)
        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | "
              f"G={running['g']/denom:.5f} | D={running['d']/denom:.5f}")

        checkpoint_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
        torch.save({
            "epoch": epoch + 1, "model": model.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(), "config": cfg.__dict__
        }, checkpoint_path)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.checkpoint_dir, f"model_epoch_{epoch+1}.pt"))


if __name__ == "__main__":
    main()
