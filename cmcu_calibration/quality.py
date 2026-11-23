"""Press routing, slope screening, native-frame sliding tests and optional legacy QC."""

from collections import Counter
import numpy as np
from scipy.stats import spearmanr

from .calibration import fit_lookup, predict_lookup, slope_check
from .io import pair_native
from .routing import fixed_bsq_zero, route_press_events, runs
from .sliding import WINDOW, FRAME_LIMIT, eligible_windows, best_frames, count_hand

CHANNELS = [side + str(i) for side in "LR" for i in range(5)]


def press_events(pair, zero, routing="ad_delta"):
    if routing == "ad_delta":
        return route_press_events(pair, bsq_zero_N=zero)
    reference = pair["bsq_force_N"] - zero
    smooth = np.median(
        np.lib.stride_tricks.sliding_window_view(np.pad(reference, 1, mode="edge"), 3),
        axis=1,
    )
    synchronized = np.repeat(
        pair["nearest_delta_ns"] <= 20_000_000, 5, axis=1
    ) & np.isfinite(pair["cmcu_force_N"])
    events = []
    for number, (start, end) in enumerate(runs(smooth > 0.3), 1):
        reason, finger = "", -1
        good = synchronized[start:end].all(axis=1)
        if start == 0 or end == len(reference):
            reason = "incomplete_press_at_boundary"
        elif np.max(smooth[start:end]) < 3:
            reason = "peak_below3N_cannot_confirm_finger"
        elif not good.any():
            reason = "missing_synchronized_fingers"
        else:
            peaks = pair["cmcu_force_N"][start:end][good].max(axis=0)
            finger = int(peaks.argmax())
            if peaks[finger] <= 0:
                reason = "zero_response"
            elif np.count_nonzero(peaks == peaks[finger]) > 1:
                reason = "tied_peak"
        events.append(
            dict(
                event=number,
                start_idx=start,
                end_idx=end,
                target_index=finger,
                eligible=not reason,
                reason=reason,
            )
        )
    return events


def sliding_audit(arrays, zero):
    """Check every complete BSQ press across all ten AD channels, independently of routing."""
    clock = arrays["bsq_t"]
    reference = arrays["bsq_force"] - zero
    smooth = np.median(
        np.lib.stride_tricks.sliding_window_view(np.pad(reference, 1, mode="edge"), 3),
        axis=1,
    )
    clocks = [arrays[side + "_t"] for side in ("left", "right")]
    ads = [arrays[side + "_ad"] for side in ("left", "right")]
    above = reference > 3
    hand_runs = [count_hand(c, a, clock, above) for c, a in zip(clocks, ads)]
    minimum = max(clock[0], min(c[0] for c in clocks))
    maximum = min(clock[-1], max(c[-1] for c in clocks)) - WINDOW
    events = {}
    for number, (start, end) in enumerate(runs(smooth > 0.3), 1):
        if start == 0 or end == len(clock) or smooth[start:end].max() < 3:
            continue
        spans = np.asarray(runs(above[start:end]), dtype=int).reshape(-1, 2) + start
        windows, coverage = eligible_windows(
            clock[spans[:, 0]],
            clock[spans[:, 1]],
            max(minimum, clock[start] - WINDOW),
            min(clock[end], maximum),
        )
        longest = np.zeros((2, 10), dtype=int)
        for side, (native_clock, rules) in enumerate(zip(clocks, hand_runs)):
            for rule, per_channel in enumerate(rules):
                for left, right in windows:
                    target = longest[rule, side * 5 : (side + 1) * 5]
                    np.maximum(
                        target,
                        best_frames(left, right, native_clock, per_channel),
                        out=target,
                    )
        best = int(longest[0].argmax())
        passed = bool(longest[0, best] >= FRAME_LIMIT)
        events[number] = dict(
            event=number,
            eligible=bool(windows),
            passed=passed,
            reason=(
                "pass"
                if passed
                else "AD_run_too_short" if windows else "BSQ_duration_too_short"
            ),
            bsq_coverage_s=float(coverage / 1e9),
            frames=int(longest[0, best]),
            channel=CHANNELS[best] if longest[0, best] else None,
            strict_passed=bool(longest[1].max() >= FRAME_LIMIT),
        )
    return events


def basic_quality(ad, reference, model):
    """Optional thresholds from the published 309a98e source-calibration QC."""
    if len(ad) < 20 or len(np.unique(ad)) < 8 or np.ptp(ad) < 100:
        return "insufficient_calibration_points_or_AD_range"
    correlation = float(spearmanr(ad, reference).statistic)
    if not np.isfinite(correlation) or correlation < 0.2:
        return "nonpositive_or_weak_calibration_monotonicity"
    span = float(np.ptp(model["force_N"]))
    if span < 1:
        return "collapsed_calibration_force_range"
    residual = float(np.quantile(abs(predict_lookup(ad, model) - reference), 0.9))
    if residual > 10 and residual > 0.5 * span:
        return "gross_source_fit_residual_P90_above_10N_and_half_range"
    return None


def prepare_source(
    arrays,
    *,
    routing="ad_delta",
    slope_scope="recording",
    sliding_scope="off",
    basic_qc=False,
    zero_N=None,
):
    pair = pair_native(arrays)
    zero = fixed_bsq_zero(pair["bsq_force_N"]) if zero_N is None else float(zero_N)
    if not np.isfinite(zero):
        raise ValueError("BSQ zero cannot be estimated")
    reference = pair["bsq_force_N"] - zero
    if np.ptp(reference) < 3:
        raise ValueError("BSQ has no press reaching the routing force range")
    events = press_events(pair, zero, routing)
    sliding = sliding_audit(arrays, zero)
    selected = [event for event in events if event["eligible"]]
    before_sliding = len(selected)
    if sliding_scope != "off":
        selected = [
            e for e in selected if sliding.get(e["event"], {}).get("passed", False)
        ]
    record_reasons = []
    if sliding_scope == "recording" and any(not e["passed"] for e in sliding.values()):
        record_reasons.append("sliding_recording_rejected")
    synchronized = np.repeat(pair["nearest_delta_ns"] <= 20_000_000, 5, axis=1)
    models, channels = {}, {}
    for finger in range(10):
        spans = [
            np.arange(e["start_idx"], e["end_idx"])
            for e in selected
            if e["target_index"] == finger
        ]
        if not spans:
            continue
        indices = np.concatenate(spans)
        valid = (
            synchronized[indices, finger]
            & (reference[indices] >= 0)
            & (reference[indices] <= 25)
        )
        indices = indices[valid]
        ad = pair["cmcu_ad"][indices, finger]
        force = pair["cmcu_force_N"][indices, finger]
        truth = reference[indices]
        slope = slope_check(truth, force)
        # Preserve the viewer's first-second zero anchors; do not subtract the AD baseline.
        quiet = (
            (pair["sync_ts_ns"] < pair["sync_ts_ns"][0] + 1_000_000_000)
            & (abs(reference) <= 0.05)
            & synchronized[:, finger]
        )
        x = np.r_[pair["cmcu_ad"][quiet, finger], ad]
        y = np.r_[np.zeros(int(quiet.sum())), truth]
        model = fit_lookup(x, y)
        reason = "insufficient_lookup_points" if model is None else None
        if model is not None and basic_qc:
            reason = basic_quality(x, y, model)
        slope_failed = slope["status"] in ("low", "high")
        if slope_failed and slope_scope == "finger":
            reason = "default_force_slope_outside_0.3_to_3"
        if slope_failed and slope_scope == "recording":
            record_reasons.append("default_force_slope_outside_0.3_to_3")
        channels[str(finger)] = dict(
            channel=CHANNELS[finger], slope=slope, points=len(x), rejected_reason=reason
        )
        if model is not None and reason is None:
            models[str(finger)] = model
    if not models:
        record_reasons.append("no_usable_calibration_finger")
    record_reasons = list(dict.fromkeys(record_reasons))
    if record_reasons:
        models = {}
    audit = dict(
        zero_N=zero,
        rejected=bool(record_reasons),
        record_reasons=record_reasons,
        channels=channels,
        routed_presses=before_sliding,
        used_presses=len(selected),
        routing_rejections=dict(
            Counter(e["reason"] for e in events if not e["eligible"])
        ),
        sliding=list(sliding.values()),
        accepted_press_intervals_ns=[
            [
                int(pair["sync_ts_ns"][e["start_idx"]]),
                int(pair["sync_ts_ns"][e["end_idx"]]),
            ]
            for e in selected
        ],
    )
    return models, audit
