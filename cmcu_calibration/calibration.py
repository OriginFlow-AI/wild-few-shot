"""The viewer's duplicate-AD median, weighted PAVA and in-domain interpolation."""

import numpy as np
from scipy.optimize import isotonic_regression


def fit_lookup(ad, reference, *, zero_anchor=True):
    """Fit a monotone lookup with an exact zero unless reproducing a legacy fit."""
    ad = np.asarray(ad, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if ad.ndim != 1 or ad.shape != reference.shape:
        raise ValueError("AD and BSQ must be equal-length one-dimensional arrays")
    valid = np.isfinite(ad) & np.isfinite(reference) & (reference >= 0)
    x, y = ad[valid], reference[valid]
    if len(x) < 8:
        return None
    order = np.lexsort((y, x))
    support, first, counts = np.unique(x[order], return_index=True, return_counts=True)
    if len(support) < 2:
        return None
    ordered_force = y[order]
    medians = (
        ordered_force[first + (counts - 1) // 2] + ordered_force[first + counts // 2]
    ) * 0.5
    if zero_anchor:
        # Nonnegative forces and monotonicity require f(x) = 0 for x <= 0.
        medians[support <= 0] = 0.0
    force = isotonic_regression(medians, weights=counts).x
    model = {"ad": support.tolist(), "force_N": force.tolist()}
    if zero_anchor:
        zero_index = int(np.searchsorted(support, 0.0))
        if zero_index == len(support) or support[zero_index] != 0:
            if zero_index in (0, len(support)):
                model["observed_ad_range"] = [float(support[0]), float(support[-1])]
            model["ad"].insert(zero_index, 0.0)
            model["force_N"].insert(zero_index, 0.0)
    return model


def predict_lookup(ad, model):
    values = np.asarray(ad, dtype=float)
    support = np.asarray(model["ad"], dtype=float)
    force = np.asarray(model["force_N"], dtype=float)
    if (
        support.ndim != 1
        or len(support) < 2
        or support.shape != force.shape
        or not np.isfinite(support).all()
        or not np.isfinite(force).all()
        or np.any(np.diff(support) <= 0)
        or np.any(np.diff(force) < 0)
    ):
        raise ValueError("Invalid monotone lookup model")
    if "observed_ad_range" in model:
        lower, upper = model["observed_ad_range"]
        outside = (values < lower) | (values > upper)
        values = np.where(outside & (values != 0), np.nan, values)
    return np.interp(values, support, force, left=np.nan, right=np.nan)


def slope_check(reference, default_force):
    """Fit default_force = slope * BSQ + intercept on 2 <= BSQ <= 25 N."""
    reference = np.asarray(reference, dtype=float)
    default_force = np.asarray(default_force, dtype=float)
    valid = (
        np.isfinite(reference)
        & np.isfinite(default_force)
        & (reference >= 2)
        & (reference <= 25)
    )
    x, y = reference[valid], default_force[valid]
    result = {
        "points": len(x),
        "slope": None,
        "intercept_N": None,
        "status": "insufficient",
    }
    if len(x) < 2 or np.ptp(x) == 0:
        return result
    centered = x - x.mean()
    slope = float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))
    result.update(
        slope=slope,
        intercept_N=float(y.mean() - slope * x.mean()),
        reference_span_N=float(np.ptp(x)),
        status="low" if slope < 0.3 else "high" if slope > 3 else "pass",
    )
    return result
