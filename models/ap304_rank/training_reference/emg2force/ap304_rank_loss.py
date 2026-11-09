"""Differentiable raw-order supervision; each group stays within one trial."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(value, mask, dim):
    return (value * mask).sum(dim) / mask.sum(dim).clamp_min(1)


def ranking_objective(scores, batch, cfg):
    # scores [groups, two windows, five fingers, 89 tokens]
    scores = scores.float()
    if scores.shape[1:] != (2, 5, 89):
        raise ValueError("AP304 ranking expects [G,2,5,89] scores")
    flat = scores.transpose(1, 2).reshape(-1, 5, 178)
    index_i, index_j = batch["pair_i"].long(), batch["pair_j"].long()
    signs = batch["pair_sign"].float()
    mask = signs != 0
    left = flat.gather(2, index_i.flatten(2)).reshape_as(signs)
    right = flat.gather(2, index_j.flatten(2)).reshape_as(signs)
    margin = signs * (left - right) / cfg["objective"]["temperature"]
    pair_loss = F.softplus(-margin)
    component_count = mask.sum(-1)
    component_valid = component_count > 0
    component_loss = masked_mean(pair_loss, mask, -1)
    local = component_loss[:, :, :6].reshape(-1, 5, 2, 3)
    local_valid = component_valid[:, :, :6].reshape(-1, 5, 2, 3)
    lag_weights = scores.new_tensor(cfg["pair_supervision"]["within_window"]["bin_weights"])
    weights = lag_weights * local_valid
    finger_window = (local * weights).sum(-1) / weights.sum(-1).clamp_min(1e-12)
    fw_valid = local_valid.any(-1)
    per_window = masked_mean(finger_window, fw_valid, 1)
    window_valid = fw_valid.any(1)
    within = masked_mean(per_window, window_valid, -1)
    within_valid = window_valid.any(-1)
    cross = masked_mean(component_loss[:, :, 6], component_valid[:, :, 6], -1)
    cross_valid = component_valid[:, :, 6].any(-1)
    obj = cfg["objective"]
    w_local = obj["within_window_weight"] * within_valid
    w_cross = obj["across_window_weight"] * cross_valid
    per_group = (w_local * within + w_cross * cross) / (w_local + w_cross).clamp_min(1e-12)
    valid = within_valid | cross_valid
    real = (batch["starts"] >= 0)[:, :, None, None].expand_as(scores)
    penalty = masked_mean(scores.square().flatten(1), real.flatten(1), -1)
    per_group = (per_group + obj["score_l2_weight"] * penalty) * valid
    correct = (margin > 0).float() + 0.5 * (margin == 0).float()
    # Additive quantities, suitable for exact DDP/accumulation reductions.
    statistics = torch.stack([
        per_group.detach().sum(), valid.sum().float(),
        (correct[:, :, :6] * mask[:, :, :6]).sum(), mask[:, :, :6].sum().float(),
        (correct[:, :, 6] * mask[:, :, 6]).sum(), mask[:, :, 6].sum().float(),
        (pair_loss * mask).detach().sum(), mask.sum().float(),
        scores.detach().mul(real).sum(), scores.detach().square().mul(real).sum(), real.sum().float(),
        scores.new_tensor(float(len(scores))),
    ])
    return per_group.sum(), valid.sum(), statistics
