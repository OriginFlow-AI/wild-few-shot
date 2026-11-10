from __future__ import annotations

import math

import torch
from torch import Tensor


def force_huber_loss(prediction_n: Tensor, target_n: Tensor, *, delta_n: float = 1.0,
                     weight: Tensor | None = None) -> Tensor:
    """Pure PyTorch Huber in calibrated N; delta is in N, loss in N squared.

    Small errors: 0.5 * error**2. Large errors: delta * (abs(error)-0.5*delta).
    Padding weights are normalized over valid elements only.
    """
    if not math.isfinite(float(delta_n)) or delta_n <= 0:
        raise ValueError('Huber delta_n must be finite and positive')
    if prediction_n.shape != target_n.shape:
        raise ValueError('Huber prediction/target shape mismatch')
    dtype = torch.float64 if torch.float64 in (prediction_n.dtype, target_n.dtype) else torch.float32
    prediction_n, target_n = prediction_n.to(dtype), target_n.to(dtype)
    penalty = torch.nn.functional.huber_loss(prediction_n, target_n, reduction='none', delta=float(delta_n))
    if weight is None:
        if penalty.numel() == 0:
            raise ValueError('Huber requires scored elements')
        return penalty.mean()
    if weight.shape != target_n.shape:
        raise ValueError('Huber weight/target shape mismatch')
    weight = weight.to(dtype)
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()) or not bool(weight.sum() > 0):
        raise ValueError('Huber weights must be finite, nonnegative and have positive sum')
    return (penalty * weight).sum() / weight.sum()


def high_force_mae_weights(target_n: Tensor, policy: dict) -> Tensor:
    """Bounded, target-only emphasis in calibrated N; thresholds fitted on Train.

    The five-finger axis is second-last, for both [B,5,T] and [5,T].
    This does not calibrate labels or estimate measurement confidence.
    """
    if target_n.ndim < 2 or target_n.shape[-2] != 5:
        raise ValueError('high-force weights expect [...,5,time]')
    if policy.get('domain') != 'calibrated_N' or policy.get('fit_split') != 'train':
        raise ValueError('high-force thresholds must be Train-only calibrated N')
    low = torch.as_tensor(policy['start_n'], device=target_n.device, dtype=target_n.dtype)
    high = torch.as_tensor(policy['full_n'], device=target_n.device, dtype=target_n.dtype)
    maximum = float(policy.get('max_weight', 3.0))
    if low.shape != (5,) or high.shape != (5,):
        raise ValueError('high-force thresholds need exactly five fingers')
    if not bool(torch.isfinite(low).all() & torch.isfinite(high).all()) or not bool(((high > low) & (low > 0)).all()):
        raise ValueError('require finite 0 < start_n < full_n')
    if not 1.0 <= maximum <= 5.0:
        raise ValueError('max_weight must be bounded between 1 and 5')
    u = ((target_n.detach().abs() - low[:, None]) / (high - low)[:, None]).clamp(0, 1)
    return 1.0 + (maximum - 1.0) * u.square() * (3.0 - 2.0 * u)


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
    mae_high_force: dict | None = None,
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
    mae_objective = mae
    if mae_high_force is not None:
        mae_sample_weight = sample_weight * high_force_mae_weights(target_n, mae_high_force)
        mae_objective = (absolute_error * mae_sample_weight).sum() / mae_sample_weight.sum()
    loss = float(corr_weight) * corr_loss + float(mape_weight) * mape + float(mae_weight) * mae_objective
    return {
        "loss": loss,
        "corr_loss": corr_loss,
        "correlation": correlation,
        "mape": mape,
        "mae": mae,
        "mae_objective": mae_objective,
        "active_count": active_weight_sum,
        "weight_sum": weight_sum,
    }
