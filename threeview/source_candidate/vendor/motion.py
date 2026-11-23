from __future__ import annotations
import math, os
import torch
from torch import nn
from torch.nn import functional as F

class CausalBlock(nn.Module):

    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.padding = 2 * dilation

    def forward(self, x):
        value = self.norm(x).transpose(1, 2)
        return x + F.silu(self.conv(F.pad(value, (self.padding, 0))).transpose(1, 2))

class MotionDecoder(nn.Module):

    def __init__(self, width=128, pose_channels=20, latent_channels=40):
        super().__init__()
        self.project = nn.Linear(latent_channels + pose_channels, width)
        self.blocks = nn.Sequential(CausalBlock(width, 1), CausalBlock(width, 2))
        self.delta_head = nn.Linear(width, pose_channels)
        self.gate_head = nn.Linear(width, pose_channels)

    def forward(self, q, r):
        with torch.autocast(device_type=q.device.type, enabled=False):
            h = self.blocks(F.silu(self.project(torch.cat([q.float(), r.float()], -1))))
            return (0.02 * self.delta_head(h), 0.98 * torch.sigmoid(self.gate_head(h)))

    def empty_cache(self, batch, device):
        return [torch.zeros(batch, 128, block.padding, device=device, dtype=torch.float32) for block in self.blocks]

    def step(self, q, r, cache, previous, first, *, tracking=False):
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, r = (q.float(), r.float())
            h = F.silu(self.project(torch.cat([q, r], -1)))
            for block, history in zip(self.blocks, cache):
                value = block.norm(h).unsqueeze(-1)
                window = torch.cat([history, value], -1)
                h = h + F.silu(block.conv(window).squeeze(-1))
                history.copy_(window[:, :, 1:])
            delta = 0.02 * self.delta_head(h)
            gate = 0.98 * torch.sigmoid(self.gate_head(h))
            initial = previous.float() if tracking else r
            return torch.where(first, initial, gate * (previous.float() + delta) + (1 - gate) * r)

def decode_sequence(motion, q, r, initial_pose=None):
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, r = (q.float(), r.float())
        delta, gate = motion(q, r)
        first = r[:, :1] if initial_pose is None else initial_pose.float()[:, None]
        a = torch.cat([torch.zeros_like(gate[:, :1]), gate[:, 1:]], 1)
        b = torch.cat([first, gate[:, 1:] * delta[:, 1:] + (1 - gate[:, 1:]) * r[:, 1:]], 1)
        offset = 1
        while offset < r.shape[1]:
            old_a, old_b = (a, b)
            a = torch.cat([old_a[:, :offset], old_a[:, offset:] * old_a[:, :-offset]], 1)
            b = torch.cat([old_b[:, :offset], old_b[:, offset:] + old_a[:, offset:] * old_b[:, :-offset]], 1)
            offset *= 2
        return b
