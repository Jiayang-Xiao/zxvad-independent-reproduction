"""Faithful PyTorch reimplementation of the zxVAD components.

This module implements the future-frame prediction backbone with a
MemAE-style memory module, the normalcy classifier, the untrained-CNN
anomaly synthesis module, and the four normalcy-relative losses described in
``Cross-Domain Video Anomaly Detection without Target Domain Adaptation``
(Aich et al., WACV 2023). The implementation is intentionally self-contained
and does not use target-domain statistics.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, resnet50


def _conv_bn_leaky(in_c: int, out_c: int, kernel: int = 4, stride: int = 2, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
    )


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator used in the zxVAD future-frame backbone."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _conv_bn_leaky(64, 128),
            _conv_bn_leaky(128, 256),
            _conv_bn_leaky(256, 512),
            nn.Conv2d(512, 1, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MemoryUnit(nn.Module):
    """Spatial memory read used at the U-Net bottleneck.

    The memory bank is learned jointly with the generator. Addressing uses
    cosine/linear similarity followed by softmax, ReLU hard-shrinkage with
    ``shrink`` and l1 normalization, as in MemAE and zxVAD.
    """

    def __init__(self, mem_dim: int, fea_dim: int, shrink: float = 0.0005) -> None:
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink = shrink
        self.weight = nn.Parameter(torch.empty(mem_dim, fea_dim))
        nn.init.uniform_(self.weight, -1.0 / math.sqrt(fea_dim), 1.0 / math.sqrt(fea_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, c, h, w = x.shape
        query = x.permute(0, 2, 3, 1).reshape(-1, c)
        att = F.linear(query, self.weight)  # (N*H*W) x K
        att = F.softmax(att, dim=1)
        if self.shrink > 0:
            pos = F.relu(att - self.shrink)
            att = (pos * att) / (torch.abs(att - self.shrink) + 1e-12)
            att = F.normalize(att, p=1, dim=1)
        output = F.linear(att, self.weight.t())  # (N*H*W) x C
        output = output.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()
        att_map = att.view(n, h, w, self.mem_dim).permute(0, 3, 1, 2).contiguous()
        return output, att_map

    def entropy_loss(self, att: torch.Tensor) -> torch.Tensor:
        # att: B x K x H x W
        return (-att * torch.log(att + 1e-12)).sum(dim=1).mean()


class _DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FutureFrameGenerator(nn.Module):
    """Four-layer U-Net future-frame generator with a memory bottleneck."""

    def __init__(self, input_frames: int = 4, mem_dim: int = 2000, shrink: float = 0.0005) -> None:
        super().__init__()
        in_c = input_frames * 3
        self.down1 = _DoubleConv(in_c, 64)
        self.down2 = _DoubleConv(64, 128)
        self.down3 = _DoubleConv(128, 256)
        self.down4 = _DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.memory = MemoryUnit(mem_dim, 512, shrink)

        self.up1 = nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1)
        self.upconv1 = _DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)
        self.upconv2 = _DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
        self.upconv3 = _DoubleConv(128, 64)
        self.out_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)
        mem_out, att = self.memory(d4)
        u1 = self.up1(mem_out)
        c1 = torch.cat([d3, u1], dim=1)
        u1 = self.upconv1(c1)
        u2 = self.up2(u1)
        c2 = torch.cat([d2, u2], dim=1)
        u2 = self.upconv2(c2)
        u3 = self.up3(u2)
        c3 = torch.cat([d1, u3], dim=1)
        u3 = self.upconv3(c3)
        return self.out_head(u3), att


class NormalcyClassifier(nn.Module):
    """CNN classifier with a scalar logit and a final conv feature map.

    A ResNet-18 trunk is used as a concrete small classifier. The paper does
    not fix the classifier architecture and only requires that the last
    convolutional feature maps are available for SCDA attention extraction.
    The final sigmoid is intentionally omitted.
    """

    def __init__(self) -> None:
        super().__init__()
        trunk = resnet18(weights=None)
        self.features = nn.Sequential(*list(trunk.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        logit = self.fc(self.pool(feat).flatten(1))
        return logit, feat


class UntrainedAnomalySynthesizer(nn.Module):
    """Pseudo-anomaly synthesis with an untrained randomly-initialized CNN."""

    def __init__(self, backbone: str = "resnet50") -> None:
        super().__init__()
        if backbone == "resnet50":
            trunk = resnet50(weights=None)
        elif backbone == "resnet18":
            trunk = resnet18(weights=None)
        else:
            raise ValueError(backbone)
        self.features = nn.Sequential(*list(trunk.children())[:-2])
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def attention_mask(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        a = torch.sum(torch.abs(feat), dim=1, keepdim=True)
        a = a / (a.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8))
        return (a > 0.1).float()


def paste_object(
    base: torch.Tensor, object_frame: torch.Tensor, mask_small: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paste localized object patches into base frames at random locations.

    Args:
        base: normal frames (B, 3, H, W).
        object_frame: frames from which object patches are extracted.
        mask_small: object masks from the untrained CNN (B, 1, h, w).

    Returns:
        Pseudo-abnormal frames and full-resolution binary masks.
    """
    b, _, h, w = base.shape
    mask_full = F.interpolate(mask_small, size=(h, w), mode="nearest")
    outs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    device = base.device
    for i in range(b):
        m = mask_full[i]  # 1,H,W
        obj = object_frame[i] * m
        beta = torch.rand((), device=device)
        bw = max(1, int(w * math.sqrt(1.0 - beta.item())))
        bh = max(1, int(h * math.sqrt(1.0 - beta.item())))
        cx = torch.randint(0, w + 1, (), device=device).item()
        cy = torch.randint(0, h + 1, (), device=device).item()
        left = min(max(cx - bw // 2, 0), w - bw)
        top = min(max(cy - bh // 2, 0), h - bh)
        obj_patch = F.interpolate(obj.unsqueeze(0), size=(bh, bw), mode="bilinear", align_corners=False)[0]
        mask_patch = F.interpolate(m.unsqueeze(0), size=(bh, bw), mode="nearest")[0]
        frame = base[i].clone()
        paste_mask = mask_patch > 0.5
        frame[:, top : top + bh, left : left + bw] = torch.where(
            paste_mask, obj_patch, frame[:, top : top + bh, left : left + bw]
        )
        full_mask = torch.zeros_like(m)
        full_mask[:, top : top + bh, left : left + bw] = paste_mask.float()
        outs.append(frame)
        masks.append(full_mask)
    return torch.stack(outs), torch.stack(masks)


def scda_attention(feat: torch.Tensor) -> torch.Tensor:
    """Channel-summed SCDA attention, normalized to [0, 1]."""
    a = torch.sum(torch.abs(feat), dim=1, keepdim=True)
    return a / (a.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8))


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    def grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return dx, dy

    px, py = grad(pred)
    tx, ty = grad(target)
    return torch.mean((px - tx).abs() ** alpha) + torch.mean((py - ty).abs() ** alpha)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Differentiable 1 - SSIM with a Gaussian window."""

    def gaussian(k: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        return g.unsqueeze(1) @ g.unsqueeze(0)

    if window_size % 2 == 0:
        window_size += 1
    window = gaussian(window_size, 1.5).to(pred.device).unsqueeze(0).unsqueeze(0)
    window = window.expand(pred.size(1), 1, window_size, window_size)
    c1, c2 = 0.01**2, 0.03**2
    groups = pred.size(1)
    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=groups)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=groups)
    sigma_x = F.conv2d(pred * pred, window, padding=window_size // 2, groups=groups) - mu_x**2
    sigma_y = F.conv2d(target * target, window, padding=window_size // 2, groups=groups) - mu_y**2
    sigma_xy = F.conv2d(pred * target, window, padding=window_size // 2, groups=groups) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - ssim.mean()


class NormalcyAugmentation:
    def __init__(self) -> None:
        self.aug = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomAffine(degrees=360),
                transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
            ]
        )

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # torchvision augmentation expects conventional image tensors.
        x01 = (x + 1.0) / 2.0
        out = []
        for i in range(x01.size(0)):
            out.append(self.aug(x01[i]))
        return torch.stack(out) * 2.0 - 1.0


class ArcFaceLoss(nn.Module):
    """ArcFace loss on normalized attention vectors."""

    def __init__(self, in_features: int, scale: float = 64.0, margin_degrees: float = 28.6) -> None:
        super().__init__()
        self.scale = scale
        self.margin = math.radians(margin_degrees)
        self.weight = nn.Parameter(torch.empty(2, in_features))
        nn.init.normal_(self.weight, 0, 0.01)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = F.normalize(embeddings, dim=1)
        w = F.normalize(self.weight, dim=1)
        cos = F.linear(x, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos)
        one_hot = F.one_hot(labels, num_classes=2).float()
        target_theta = theta + self.margin
        target_cos = torch.cos(target_theta)
        logits = self.scale * (cos * (1 - one_hot) + target_cos * one_hot)
        return F.cross_entropy(logits, labels)


def resample_attention(mask_256: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    """Resize a full-resolution mask to a feature-map attention size."""
    _, _, h, w = feat.shape
    return F.interpolate(mask_256, size=(h, w), mode="area")
