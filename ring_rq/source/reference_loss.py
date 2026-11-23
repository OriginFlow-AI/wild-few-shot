import torch
N_FORCE_DIM=5


def batch_finger_weights(target_raw: torch.Tensor) -> torch.Tensor:
    """w_i = mean_t |F_i(t)| after flattening batch and time."""
    return target_raw.abs().mean(dim=(0, 2))  # (5,)

def weighted_pearson_rw(pred: torch.Tensor, target: torch.Tensor,
                        weight_raw: torch.Tensor | None = None,
                        eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute r_w over a batch as if windows were concatenated in time.

    pred/target: (B, 5, T), normalised or raw. Pearson is scale invariant.
    weight_raw: raw target (B, 5, T) for w_i = mean |truth|. If None, target is used.
    Returns (r_w, r_i, w_i).
    """
    p = pred.transpose(0, 1).reshape(N_FORCE_DIM, -1)      # (5, B*T)
    t = target.transpose(0, 1).reshape(N_FORCE_DIM, -1)
    pm = p - p.mean(dim=1, keepdim=True)
    tm = t - t.mean(dim=1, keepdim=True)
    num = (pm * tm).sum(dim=1)
    p2 = (pm * pm).sum(dim=1)
    t2 = (tm * tm).sum(dim=1)
    valid = (p2 > eps) & (t2 > eps)
    r = num / torch.sqrt((p2 * t2).clamp(min=eps))
    wr_src = target if weight_raw is None else weight_raw
    w = batch_finger_weights(wr_src)
    eff_w = w * valid.float()
    denom = eff_w.sum()
    rw = (eff_w * r).sum() / denom.clamp(min=eps)
    rw = torch.where(denom > eps, rw, torch.zeros_like(rw))
    return rw, r, w

def weighted_corr_loss(pred_n: torch.Tensor, target_n: torch.Tensor,
                       target_raw: torch.Tensor) -> torch.Tensor:
    rw, _, _ = weighted_pearson_rw(pred_n, target_n, weight_raw=target_raw)
    return 1.0 - rw

def dynamic_fz_epsilon(target_raw: torch.Tensor) -> torch.Tensor:
    """epsilon = min(0.05 * max_{t,i} |F_i(t)|, 0.5N) for Fz-only."""
    peak = target_raw.abs().amax()
    return torch.minimum(0.05 * peak, target_raw.new_tensor(0.5)).clamp(min=1e-6)

def weighted_mape_loss(pred_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    eps = dynamic_fz_epsilon(target_raw)
    big = target_raw.abs() > eps
    rel = (pred_raw - target_raw).abs() / target_raw.abs().clamp(min=eps)
    n_big = big.sum(dim=(0, 2))                                  # (5,)
    per_finger = (rel * big.float()).sum(dim=(0, 2)) / n_big.clamp(min=1)
    valid = n_big > 0
    w = batch_finger_weights(target_raw) * valid.float()
    denom = w.sum()
    out = (w * per_finger).sum() / denom.clamp(min=1e-8)
    return torch.where(denom > 1e-8, out, torch.zeros_like(out))
