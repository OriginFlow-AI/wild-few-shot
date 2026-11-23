"""Four adaptations of the original A-loss W304 Affine model."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from adapter import PrefixMemory
from base_model import K2RingConfig, K2RingNetwork, diagonal_affine_scan
from train_data import HISTORY, POSITIONS

ROOT = Path(__file__).resolve().parent


class LowRank(nn.Module):
    def __init__(self, weight, rank=16):
        super().__init__()
        self.down = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.up = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.down, a=5 ** .5)

    def forward(self, weight):
        return weight + self.up @ self.down


class Arm(nn.Module):
    def __init__(self, name, device):
        super().__init__()
        self.name = name
        cfg = json.loads((ROOT / "MODEL_CONFIG.json").read_bytes())["config"]
        for field in ("patch_receptive_fields", "patch_left_paddings"):
            cfg[field] = tuple(cfg[field])
        self.base = K2RingNetwork(K2RingConfig(**cfg))
        state = torch.load(ROOT / "assets/A_ep1.pt", map_location="cpu", weights_only=True)
        self.base.load_state_dict(state, strict=True)
        self.base.eval().requires_grad_(False)
        scope = json.loads((ROOT / "PARAMETER_SCOPE.json").read_bytes())
        self.lora_targets = []
        self.memory = None
        if name == "B_D":
            for key in scope["full_decoder_keys"]:
                self.base.get_parameter(key).requires_grad_(True)
        elif name == "B_L":
            for key, shape in scope["targets"].items():
                path, field = key.rsplit(".", 1)
                module = self.base.get_submodule(path)
                weight = getattr(module, field)
                assert list(weight.shape) == shape
                parametrize.register_parametrization(module, field, LowRank(weight))
                self.lora_targets.append((path, field))
            self.base.output_head.requires_grad_(True)
        elif name in ("N", "F"):
            self.memory = PrefixMemory()
        else:
            raise ValueError(name)
        with np.load(ROOT / "assets/train_norm.npz", allow_pickle=False) as norm:
            for key, size in (("emg_mean", 16), ("emg_std", 16), ("force_mean", 5), ("force_std", 5)):
                value = np.asarray(norm[key], dtype=np.float32).reshape(-1)
                assert value.size == size and np.isfinite(value).all()
                if key == "emg_std":
                    value = np.maximum(value, 1e-6)
                elif key == "force_std":
                    assert (value > 0).all()
                self.register_buffer(key, torch.from_numpy(value), persistent=False)
        self.to(device)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        expected = {"B_D": 4921471, "B_L": 354783}
        if name in expected:
            assert trainable == expected[name], (name, trainable)

    def normalized(self, raw):
        return (raw - self.emg_mean) / self.emg_std

    def fused(self, x):
        with torch.no_grad():
            encoded = self.base.encode(x.transpose(1, 2).contiguous())
        return self.base.fuse_streams(self.base.decode_streams(encoded))[0]

    def features(self, x):
        fused = self.fused(x)
        offsets = torch.as_tensor(POSITIONS[:, None] + HISTORY, device=x.device)
        rms = x[:, offsets].square().mean(dim=2).sqrt()
        return torch.cat((fused, rms), dim=-1)

    def physical(self, output):
        return output * self.force_std[None, :, None] + self.force_mean[None, :, None]

    def forward(self, raw, banks=None, bank_index=None):
        x = self.normalized(raw)
        if self.memory is None or banks is None:
            return self.physical(self.base.affine_decode(self.fused(x))[0])
        features, force, amplitude, valid, coverage = banks
        key_value = self.memory.encode(features, force, amplitude, valid)
        selection = tuple(value.index_select(0, bank_index) for value in key_value)
        logits = self.memory(self.features(x), selection, coverage.index_select(0, bank_index),
                             self.base.output_head.weight, self.base.output_head.bias)
        p, v, gate = logits.chunk(3, dim=-1)
        p = self.base.config.output_scale * p
        v = self.base.config.output_scale * v
        gate = gate.sigmoid()
        n = self.base.config.position_anchor_tokens
        gate = torch.cat((torch.zeros_like(gate[:, :n]), gate[:, n:]), dim=1)
        bias = gate * v + (1.0 - gate) * p
        return self.physical(diagonal_affine_scan(gate, bias, bias.new_zeros(len(bias), 5)).transpose(1, 2))

    def optimizer_groups(self):
        if self.memory is not None:
            return [{"params": self.memory.parameters(), "lr": 3e-4}]
        head = list(self.base.output_head.parameters())
        head_ids = {id(p) for p in head}
        body = [p for p in self.base.parameters() if p.requires_grad and id(p) not in head_ids]
        return [{"params": body, "lr": 2e-5 if self.name == "B_D" else 3e-4},
                {"params": head, "lr": 3e-5}]

    def merge_error(self, raw):
        with torch.no_grad():
            before = self(raw)
            merged = Arm(self.name, raw.device)
            merged.load_state_dict(self.state_dict())
            for path, field in merged.lora_targets:
                parametrize.remove_parametrizations(merged.base.get_submodule(path), field,
                                                    leave_parametrized=True)
            after = merged(raw)
            torch.testing.assert_close(before, after, atol=1e-7, rtol=1e-7)
            return float((before - after).abs().max())


class Banks:
    def __init__(self, arm, train, dev, public, chunk):
        self.arm = arm
        self.train = train
        self.dev = dev
        self.public = public
        self.chunk = chunk
        self.device = arm.emg_mean.device

    @lru_cache(maxsize=256)
    @torch.no_grad()
    def one(self, role, rid, side):
        data = self.train if self.arm.name == "N" or role == "train" else self.dev
        base_finger = 5 * (side == "right")
        slots = []
        for local in range(5):
            finger = local + base_finger
            if self.arm.name == "N":
                refs = [(item["support"], item["slot_index"])
                        for item in self.public["finger_banks"][str(finger)]]
            else:
                entry = data.support(rid)["fingers"][finger]
                refs = [(rid, index) for index in range(len(entry["slots"]))] if entry["admitted"] else []
            slots.append([data.slot(support, finger, index) for support, index in refs])
        assert max(map(len, slots)) <= 128
        windows = sorted({item[0] for row in slots for item in row})
        features = {}
        for begin in range(0, len(windows), self.chunk):
            keys = windows[begin:begin + self.chunk]
            raw = torch.from_numpy(np.stack([data.emg(*key) for key in keys])).to(self.device)
            values = self.arm.features(self.arm.normalized(raw)).cpu()
            features.update(zip(keys, values))
        emg = torch.zeros(5, 128, 320)
        history = np.zeros((5, 128, 2, 256), dtype=np.float32)
        amplitude = np.zeros((5, 128, 4), dtype=np.float32)
        valid = np.zeros((5, 128), dtype=bool)
        coverage = np.zeros((5, 4), dtype=np.float32)
        mean = self.arm.force_mean.cpu().numpy()
        std = self.arm.force_std.cpu().numpy()
        for finger, row in enumerate(slots):
            for index, (key, token, force) in enumerate(row):
                emg[finger, index] = features[key][token]
                z = (force - mean[finger]) / std[finger]
                history[finger, index, 0] = z
                history[finger, index, 1] = 1
                amplitude[finger, index] = [z[-1], z.mean(), z.std(), z[-1] - z[0]]
                valid[finger, index] = True
            if row:
                endpoints = np.asarray([item[2][-1] for item in row])
                z = (endpoints - mean[finger]) / std[finger]
                coverage[finger] = [1, z.min(), z.max(), (abs(endpoints) <= .5).mean()]
        return emg, *(torch.from_numpy(x) for x in (history, amplitude, valid, coverage))

    def batch(self, role, samples):
        if self.arm.memory is None:
            return None, None
        data = self.train if role == "train" else self.dev
        keys = [(item[3], data.support_hand(item[3])) for item in samples]
        unique = list(dict.fromkeys(keys))
        banks = [self.one("train" if self.arm.name == "N" else role, *key) for key in unique]
        tensors = tuple(torch.stack(values).to(self.device) for values in zip(*banks))
        return tensors, torch.tensor([unique.index(key) for key in keys], device=self.device)
