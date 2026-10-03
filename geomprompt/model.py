"""GeomPrompt: ViT-S/16 geometric prompts for a frozen RGB-D segmenter."""

import torch
from torch import nn
from torch.nn import functional as F
import timm


class CNNDecoder(nn.Module):
    def __init__(self, in_channels=384, hidden_channels=256, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


def make_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.outer(g)
    return kernel


class GaussianBlur(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        kernel = make_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", kernel)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)


class ProgressiveUpsampler(nn.Module):
    def __init__(
        self, channels: int = 1, blur_kernel: int = 5, blur_sigma: float = 1.0
    ):
        super().__init__()
        self.blur1 = GaussianBlur(channels, blur_kernel, blur_sigma)
        self.blur2 = GaussianBlur(channels, blur_kernel, blur_sigma)
        self.blur3 = GaussianBlur(channels, blur_kernel, blur_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.blur1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.blur2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.blur3(x)
        return x


class PromptAdapter(nn.Module):
    def __init__(self, channels: int = 3, hidden: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.conv1(x))
        residual = F.relu(self.conv2(residual))
        residual = self.conv3(residual)
        return x + residual


def low_pass_projection(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    _, _, H, W = x.shape
    x_down = F.avg_pool2d(x, factor)
    x_up = F.interpolate(x_down, size=(H, W), mode="bilinear", align_corners=False)
    return x_up


def compute_tv_loss(x: torch.Tensor) -> torch.Tensor:
    diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return diff_h.mean() + diff_w.mean()


def compute_magnitude_loss(delta: torch.Tensor) -> torch.Tensor:
    return torch.abs(delta).mean()


class GeomPrompt(nn.Module):
    def __init__(
        self,
        vit_model: str = "vit_small_patch16_224",
        pretrained: bool = True,
        freeze_vit: bool = False,
        image_size=480,
        residual_scale: float = 15.0,
        lowpass_factor: int = 2,
        use_adapter: bool = True,
        dynamic_img_size: bool = False,
    ):
        super().__init__()
        self.lowpass_factor = lowpass_factor
        self.use_adapter = use_adapter
        self.encoder = timm.create_model(
            vit_model,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=image_size,
            dynamic_img_size=dynamic_img_size,
        )
        encoder_channels = self.encoder.embed_dim
        self.decoder = CNNDecoder(
            in_channels=encoder_channels, hidden_channels=256, out_channels=1
        )
        self.upsampler = ProgressiveUpsampler(channels=1)
        self.register_buffer("gray_mean", torch.tensor(127.5))
        self.register_buffer("residual_scale", torch.tensor(residual_scale))
        self.register_buffer(
            "depth_mean", torch.tensor([0.48, 0.48, 0.48]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "depth_std", torch.tensor([0.28, 0.28, 0.28]).view(1, 3, 1, 1)
        )
        self.adapter = PromptAdapter(channels=3, hidden=16)
        if freeze_vit:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def set_residual_scale(self, scale: float):
        self.residual_scale.fill_(scale)

    def forward(self, rgb):
        delta = self.decoder(self.encode_rgb(rgb))
        residual = self.upsampler(delta)
        raw = (
            (self.gray_mean + self.residual_scale * torch.tanh(residual))
            .expand(-1, 3, -1, -1)
            .contiguous()
        )
        return (self.refine_prompt(raw), delta, raw)

    def encode_rgb(self, rgb):
        b, _, h, w = rgb.shape
        tokens = self.encoder.forward_features(rgb)[
            :, self.encoder.num_prefix_tokens :, :
        ]
        patch = self.encoder.patch_embed.patch_size[0]
        return tokens.transpose(1, 2).reshape(b, -1, h // patch, w // patch)

    def refine_prompt(self, raw):
        prompt = (raw / 255.0 - self.depth_mean) / self.depth_std
        if self.use_adapter:
            prompt = self.adapter(prompt)
        return low_pass_projection(prompt, self.lowpass_factor)


class DepthConditionEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 32, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden * 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeomPromptRecovery(GeomPrompt):
    """RGB and depth features fused to predict a bounded depth correction."""

    def __init__(self, *args, depth_feat_channels=64, **kwargs):
        super().__init__(*args, **kwargs)
        del self.gray_mean
        self.depth_encoder = DepthConditionEncoder(out_channels=depth_feat_channels)
        self.fuser = nn.Conv2d(
            self.encoder.embed_dim + depth_feat_channels, self.encoder.embed_dim, 1
        )
        nn.init.zeros_(self.decoder.conv3.weight)
        nn.init.zeros_(self.decoder.conv3.bias)

    def forward(self, rgb, depth):
        rgb_features = self.encode_rgb(rgb)
        depth_features = self.depth_encoder(depth)
        if depth_features.shape[-2:] != rgb_features.shape[-2:]:
            depth_features = F.interpolate(
                depth_features,
                size=rgb_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        fused = self.fuser(torch.cat([rgb_features, depth_features], dim=1))
        delta = self.decoder(fused)
        correction = self.residual_scale * torch.tanh(self.upsampler(delta))
        raw_depth = (depth * self.depth_std + self.depth_mean) * 255.0
        raw = (raw_depth + correction).clamp(0, 255)
        return (self.refine_prompt(raw), delta, raw)
