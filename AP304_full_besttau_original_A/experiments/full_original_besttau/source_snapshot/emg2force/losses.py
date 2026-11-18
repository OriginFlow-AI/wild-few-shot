from __future__ import annotations

import torch
from torch import Tensor


def force_sufficient_statistics(
    prediction_n: Tensor,
    target_n: Tensor,
    *,
    weight: Tensor | None = None,
    threshold_n: float = 0.5,
) -> Tensor:
    """Additive weighted statistics; merge windows before computing trial loss."""
    if prediction_n.shape != target_n.shape or prediction_n.ndim < 2:
        raise ValueError("statistics expect matching tensors with a leading batch dimension")
    if threshold_n <= 0:
        raise ValueError("threshold_n must be positive")
    x = prediction_n.double().flatten(1)
    y = target_n.double().flatten(1)
    if weight is None:
        sample_weight = torch.ones_like(y)
    else:
        if weight.shape != target_n.shape:
            raise ValueError(f"statistics weight/target mismatch: {weight.shape} vs {target_n.shape}")
        sample_weight = weight.double().flatten(1)
        if not bool(torch.isfinite(sample_weight).all()) or bool((sample_weight < 0).any()):
            raise ValueError("statistics weights must be finite and non-negative")
    error = (x - y).abs()
    active = y.abs() >= threshold_n
    active_weight = sample_weight * active
    return torch.stack(
        (
            sample_weight.sum(-1),
            (sample_weight * x).sum(-1),
            (sample_weight * y).sum(-1),
            (sample_weight * x.square()).sum(-1),
            (sample_weight * y.square()).sum(-1),
            (sample_weight * x * y).sum(-1),
            (sample_weight * error).sum(-1),
            (active_weight * error / y.abs().clamp_min(threshold_n)).sum(-1),
            active_weight.sum(-1),
        ),
        dim=-1,
    )


def force_objective_from_statistics(
    statistics: Tensor,
    *,
    corr_weight: float = 1.0,
    mape_weight: float = 0.3,
    mae_weight: float = 0.03,
) -> dict[str, Tensor]:
    """Original force objective for each complete trial (or merged shard).

    Constant signals use r=0; trials with no active elements use MAPE=0,
    matching force_objective's finite training convention. No trials are
    dropped from the loss average because their diagnostic r is undefined.
    """
    if statistics.ndim != 2 or statistics.shape[-1] != 9:
        raise ValueError("statistics must have shape [trials,9]")
    n, sx, sy, sxx, syy, sxy, error, mape_sum, active_count = statistics.double().unbind(-1)
    if bool((n <= 0).any()) or not bool(torch.isfinite(statistics).all()):
        raise ValueError("trial statistics must be finite and contain scored samples")
    covariance = sxy - sx * sy / n
    variance_x = (sxx - sx.square() / n).clamp_min(0)
    variance_y = (syy - sy.square() / n).clamp_min(0)
    correlation = covariance / (variance_x * variance_y).clamp_min(1e-16).sqrt()
    correlation = torch.where((variance_x > 0) & (variance_y > 0), correlation.clamp(-1, 1), 0.0)
    mae = error / n
    mape = mape_sum / active_count.clamp_min(1)
    return {
        "loss": corr_weight * (1 - correlation) + mape_weight * mape + mae_weight * mae,
        "correlation": correlation,
        "mae": mae,
        "mape": mape,
    }


def pearson_correlation(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    x = prediction.float().reshape(-1)
    y = target.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt((torch.sum(x * x) * torch.sum(y * y)).clamp_min(eps * eps))
    return torch.sum(x * y) / denominator


def force_objective(
    prediction_n: Tensor,
    target_n: Tensor,
    *,
    weight: Tensor | None = None,
    corr_weight: float = 1.0,
    mape_weight: float = 0.3,
    mae_weight: float = 0.03,
    mape_threshold_n: float = 0.5,
) -> dict[str, Tensor]:
    if prediction_n.shape != target_n.shape:
        raise ValueError(
            f"Force prediction/target mismatch: {prediction_n.shape} vs {target_n.shape}"
        )
    prediction_n = prediction_n.float()
    target_n = target_n.float()
    if weight is None:
        sample_weight = torch.ones_like(target_n)
    else:
        if weight.shape != target_n.shape:
            raise ValueError(f"Force weight/target mismatch: {weight.shape} vs {target_n.shape}")
        sample_weight = weight.float()
        if not bool(torch.isfinite(sample_weight).all()) or bool((sample_weight < 0).any()):
            raise ValueError("Force weights must be finite and non-negative")
    weight_sum = sample_weight.sum()
    if not bool(weight_sum > 0):
        raise ValueError("Force objective received no positively weighted targets")

    mean_prediction = (sample_weight * prediction_n).sum() / weight_sum
    mean_target = (sample_weight * target_n).sum() / weight_sum
    centered_prediction = prediction_n - mean_prediction
    centered_target = target_n - mean_target
    covariance = (sample_weight * centered_prediction * centered_target).sum()
    variance_prediction = (sample_weight * centered_prediction.square()).sum()
    variance_target = (sample_weight * centered_target.square()).sum()
    denominator = torch.sqrt((variance_prediction * variance_target).clamp_min(1e-16))
    correlation = covariance / denominator
    correlation = torch.where(
        (variance_prediction > 0) & (variance_target > 0),
        correlation.clamp(-1, 1),
        prediction_n.sum() * 0.0,
    )
    corr_loss = 1.0 - correlation
    absolute_error = (prediction_n - target_n).abs()
    active = target_n.abs() >= float(mape_threshold_n)
    ratios = absolute_error / target_n.abs().clamp_min(float(mape_threshold_n))
    active_weight = sample_weight * active
    active_weight_sum = active_weight.sum()
    if bool(active_weight_sum > 0):
        mape = (ratios * active_weight).sum() / active_weight_sum
    else:
        mape = prediction_n.sum() * 0.0
    mae = (absolute_error * sample_weight).sum() / weight_sum
    loss = float(corr_weight) * corr_loss + float(mape_weight) * mape + float(mae_weight) * mae
    return {
        "loss": loss,
        "corr_loss": corr_loss,
        "correlation": correlation,
        "mape": mape,
        "mae": mae,
        "active_count": active_weight_sum,
        "weight_sum": weight_sum,
    }
