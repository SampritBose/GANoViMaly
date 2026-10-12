from dataclasses import dataclass

@dataclass
class Config:
    image_size: int = 256
    clip_length: int = 8
    channels: int = 1
    feature_dim: int = 64
    embed_dim: int = 128
    mamba_layers: int = 4
    patch_size: int = 16
    batch_size: int = 4
    epochs: int = 150
    learning_rate: float = 1e-5
    weight_decay: float = 0.05
    alpha_mse: float = 1.0
    beta_adv: float = 10.0
    gamma_l1: float = 0.01
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    seed: int = 42
