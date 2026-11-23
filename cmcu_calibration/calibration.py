"""The viewer's duplicate-AD median, weighted PAVA and in-domain interpolation."""

import numpy as np
from scipy.optimize import isotonic_regression


def fit_lookup(ad, reference):
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
    force = isotonic_regression(medians, weights=counts).x
    return {"ad": support.tolist(), "force_N": force.tolist()}


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
