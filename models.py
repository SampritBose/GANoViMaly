import torch
from torch import nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class SpatiotemporalAttention(nn.Module):
    """3D convolutional feature extraction and local-window spatiotemporal attention."""
    def __init__(self, in_channels=1, feature_dim=64, window_size=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_dim),
            nn.SiLU(),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_dim),
            nn.SiLU(),
        )
        # Local attention map; each location is informed by its local 3D neighborhood.
        self.local_attention = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=window_size,
                      padding=window_size // 2, groups=1),
            nn.SiLU(),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, video):
        # video [B,T,C,H,W] -> Conv3D layout [B,C,T,H,W]
        x = video.permute(0, 2, 1, 3, 4).contiguous()
        features = self.features(x)
        weights = self.local_attention(features)
        return features * weights, weights


class BidirectionalVisionMambaBlock(nn.Module):
    """Bidirectional selective state-space block with Conv1D, gate and residual.

    Uses the selective SSM implementation from mamba-ssm rather than a
    convolutional stand-in. Install mamba-ssm in a supported Linux/CUDA setup.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "This paper-aligned implementation requires mamba-ssm. "
                "Install it in a supported PyTorch/CUDA environment; see README.md."
            )
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        inner = dim * expand
        self.forward_in = nn.Linear(dim, inner)
        self.backward_in = nn.Linear(dim, inner)
        self.forward_conv = nn.Conv1d(inner, inner, kernel_size=d_conv,
                                      padding=d_conv - 1, groups=inner)
        self.backward_conv = nn.Conv1d(inner, inner, kernel_size=d_conv,
                                       padding=d_conv - 1, groups=inner)
        self.forward_ssm = Mamba(d_model=inner, d_state=d_state,
                                 d_conv=d_conv, expand=1)
        self.backward_ssm = Mamba(d_model=inner, d_state=d_state,
                                  d_conv=d_conv, expand=1)
        self.gate_proj = nn.Linear(dim, inner)
        self.out_proj = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def _path(self, x, conv, ssm):
        # x [B,L,D]. Depthwise Conv1D is causalized by trimming right padding.
        x = conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        return ssm(x)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        content, gate = self.in_proj(x).chunk(2, dim=-1)
        f_in = self.forward_in(content)
        b_in = torch.flip(self.backward_in(torch.flip(content, dims=[1])), dims=[1])
        forward = self._path(f_in, self.forward_conv, self.forward_ssm)
        backward_reversed = self._path(torch.flip(b_in, dims=[1]),
                                       self.backward_conv, self.backward_ssm)
        backward = torch.flip(backward_reversed, dims=[1])
        # Gate combines both directional selective-SSM outputs.
        gate = torch.sigmoid(self.gate_proj(gate))
        fused = (forward + backward) * gate
        return residual + self.dropout(self.out_proj(F.silu(fused)))


class VisionMambaGenerator(nn.Module):
    """Predict I_(t+1) from preceding frames using 3D tokens and Vim blocks."""
    def __init__(self, image_channels=1, feature_dim=64, embed_dim=128,
                 layers=4, patch_size=16, clip_length=8, d_state=16):
        super().__init__()
        self.patch_size = patch_size
        self.clip_length = clip_length
        self.attention = SpatiotemporalAttention(image_channels, feature_dim)
        # Non-overlapping spatial patches represented as 3D video tokens.
        self.patch_embedding = nn.Conv3d(
            feature_dim, embed_dim, kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(
            1, clip_length * 256 + 1, embed_dim
        ))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList([
            BidirectionalVisionMambaBlock(embed_dim, d_state=d_state)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.context_projection = nn.Linear(embed_dim, embed_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, feature_dim * 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim * 2), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feature_dim // 2, image_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, video):
        # video [B,T,C,H,W], normalized to [-1,1]
        b, t, c, h, w = video.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("Height and width must be divisible by patch_size.")
        features, _ = self.attention(video)
        tokens = self.patch_embedding(features)  # [B,D,T,H/P,W/P]
        _, d, t2, hp, wp = tokens.shape
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(b, t2 * hp * wp, d)
        cls = self.class_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        # Interpolate learned positional embeddings if input clip/grid differs from config.
        if self.position_embedding.shape[1] != tokens.shape[1]:
            pos = self.position_embedding.transpose(1, 2)
            pos = F.interpolate(pos, size=tokens.shape[1], mode="linear",
                                align_corners=False).transpose(1, 2)
        else:
            pos = self.position_embedding
        tokens = tokens + pos
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        global_context = self.context_projection(tokens[:, :1])
        patch_tokens = tokens[:, 1:].reshape(b, t2, hp, wp, d)
        # Last observed timestep's spatial tokens form the future-frame representation.
        frame_tokens = patch_tokens[:, -1] + global_context[:, 0, None, None, :]
        feature_map = frame_tokens.permute(0, 3, 1, 2).contiguous()
        prediction = self.decoder(feature_map)
        if prediction.shape[-2:] != (h, w):
            prediction = F.interpolate(prediction, size=(h, w), mode="bilinear",
                                        align_corners=False)
        return prediction


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator ending in a 1x1 per-location classifier."""
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, kernel_size=1),
        )

    def forward(self, frame):
        return self.net(frame)


class GANoViMaly(nn.Module):
    def __init__(self, image_channels=1, feature_dim=64, embed_dim=128,
                 layers=4, patch_size=16, discriminator_channels=64,
                 clip_length=8, d_state=16):
        super().__init__()
        self.generator = VisionMambaGenerator(
            image_channels=image_channels, feature_dim=feature_dim,
            embed_dim=embed_dim, layers=layers, patch_size=patch_size,
            clip_length=clip_length, d_state=d_state
        )
        self.discriminator = PatchGANDiscriminator(image_channels, discriminator_channels)

    def predict(self, video):
        return self.generator(video)
