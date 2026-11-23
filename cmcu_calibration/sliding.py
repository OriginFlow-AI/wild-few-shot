"""Historical native-frame sliding-window predicate, unchanged algorithm."""

import numpy as np
from .routing import runs

WINDOW = 1_500_000_000
MIN_BSQ = 1_200_000_000
FRAME_LIMIT = 135


def eligible_windows(begins, ends, lo, hi):
    if not len(begins):
        return [], 0
    boundaries = np.r_[begins, ends, begins - WINDOW, ends - WINDOW, lo, hi]
    grid = np.unique(boundaries[(boundaries >= lo) & (boundaries <= hi)])
    stop = grid + WINDOW
    covered = np.maximum(
        0, np.minimum(stop[:, None], ends) - np.maximum(grid[:, None], begins)
    ).sum(axis=1)
    intervals = [(s, s) for s, cover in zip(grid, covered) if cover >= MIN_BSQ]
    for a, b, ca, cb in zip(grid[:-1], grid[1:], covered[:-1], covered[1:]):
        if ca >= MIN_BSQ and cb >= MIN_BSQ:
            intervals.append((a, b))
        elif ca >= MIN_BSQ:
            intervals.append((a, a + ca - MIN_BSQ))
        elif cb >= MIN_BSQ:
            intervals.append((b - (cb - MIN_BSQ), b))
    merged = []
    for a, b in sorted(intervals):
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    return merged, int(covered.max(initial=0))


def best_frames(a, b, clock, runs_by_channel):
    best = np.zeros(5, dtype=int)
    i, j = np.searchsorted(clock, [a, b + 1])
    candidates = np.r_[a, clock[i:j], b]
    first = np.searchsorted(clock, candidates)
    past = np.searchsorted(clock, candidates + WINDOW)
    for f, (start, end) in enumerate(runs_by_channel):
        lhs = np.searchsorted(end, first.min(), side="right")
        rhs = np.searchsorted(start, past.max())
        if lhs < rhs:
            count = np.minimum(past[:, None], end[lhs:rhs]) - np.maximum(
                first[:, None], start[lhs:rhs]
            )
            best[f] = count.max(initial=0)
    return best


def count_hand(hand_clock, ad, bsq_clock, bsq_above):
    valid = ad > 45000
    latest = np.searchsorted(bsq_clock, hand_clock, side="right") - 1
    covered = (
        (latest >= 0) & (hand_clock <= bsq_clock[-1]) & bsq_above[np.maximum(latest, 0)]
    )
    return [
        [
            list(map(np.asarray, np.asarray(runs(v[:, k]), dtype=int).reshape(-1, 2).T))
            for k in range(5)
        ]
        for v in (valid, valid & covered[:, None])
    ]
