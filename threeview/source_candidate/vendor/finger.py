from __future__ import annotations
import math, os
import torch
from torch import nn
from torch.nn import functional as F

class _PoseMLPBlock(nn.Module):
    """E4a/E5mlp: k=1 逐帧仿射; act 仅隐藏层 (LeakyReLU 0.2, 同 E4 因果 AE)."""

    def __init__(self, cin: int, cout: int, act: bool):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.2) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.act(x) if self.act is not None else x

class PoseMLPFrameDecoder(nn.Module):
    """E4a/E5mlp: D_pose 逐帧 MLP (k=1 conv, RF=1 帧).

    输入 z (B, latent, T) @50Hz -> 输出绝对角度 (B, out, T) (线性, 无激活).
    in_channels 属性供 pose_modules._predict_pose 读取 z 维度 (同
    PoseCausalConvDecoder 契约).
    """

    def __init__(self, in_channels: int=32, hidden_channels: int=128, out_channels: int=20):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(_PoseMLPBlock(in_channels, hidden_channels, act=True), _PoseMLPBlock(hidden_channels, out_channels, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PoseFrameDecoderFingers(nn.Module):
    """分组 frame-MLP 解码器: z (B, G*latent, T) -> y (B, total, T).

    每组一个 PoseMLPFrameDecoder; 输出按绝对列 scatter 回 y (未覆盖列 = 0;
    单指/子集 run 的损失/指标只算覆盖列). in_channels 属性 = G*latent,
    供 pose_modules._predict_pose 取 z 通道数 (与单本 PoseMLPFrameDecoder
    契约一致); total_channels 默认 = max(group)+1.
    """

    def __init__(self, groups, hidden_channels: int=64, latent_channels: int=8, total_channels: int | None=None):
        super().__init__()
        self.groups = [list(map(int, g)) for g in groups]
        self.n_groups = len(self.groups)
        self.latent_channels = latent_channels
        self.in_channels = self.n_groups * latent_channels
        self.total_channels = total_channels or max((max(g) for g in self.groups)) + 1
        self.nets = nn.ModuleList([PoseMLPFrameDecoder(latent_channels, hidden_channels, len(g)) for g in self.groups])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, _, T = z.shape
        y = z.new_zeros(B, self.total_channels, T)
        for net, g, zf in zip(self.nets, self.groups, z.split(self.latent_channels, dim=1)):
            y[:, torch.tensor(g, device=z.device, dtype=torch.long)] = net(zf)
        return y
