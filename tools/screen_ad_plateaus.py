"""Screen routed Test5 press windows for weak AD response during rising BSQ.

Input files contain aligned one-dimensional arrays from completed single-finger
press cores: t (seconds from recording start), x (zeroed BSQ N), ad (raw counts),
finger (0-9), press (positive chronological event ID). See the README.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DURATIONS_S = (0.3, 0.5, 1.0)
MIN_START_N = 4.0
MIN_RISE_N = 8.0
MAX_END_N = 25.0
MAX_BACKTRACK_N = 0.75
MAX_GAP_S = 0.06
MAX_AD_FRACTION = 0.10
PROFILE_CAPS = (2000, 3000, 4000)
REVIEW_CAP = 3000
MAX_WINDOW_AD_P90 = 50000
MIN_REPEAT_PRESSES = 2
MIN_WINDOW_SAMPLES = 8
FINGERS = tuple(
    f"{hand}_{finger}"
    for hand in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "little")
)
WINDOW_FIELDS = (
    "start_s",
    "end_s",
    "n",
    "bsq_start_n",
    "bsq_end_n",
    "bsq_rise_n",
    "ad_start",
    "ad_end",
    "ad_p90_minus_p10",
    "ad_p90",
)
FIELDS = (
    "record_id",
    "object_key",
    "finger",
    "press",
    *WINDOW_FIELDS,
    "finger_ad_p95_minus_p01",
    "ad_limit",
    "repeated_presses",
    "disposition",
)


def event_windows(t, bsq, ad):
    evidence = []
    indexes = np.arange(len(t))
    for duration in DURATIONS_S:
        ends = np.searchsorted(t, t + duration)
        starts = indexes[ends < len(t)]
        ends = ends[starts]
        widths = ends - starts + 1
        for width in np.unique(widths[widths >= MIN_WINDOW_SAMPLES]):
            i = starts[widths == width]
            refs = sliding_window_view(bsq, width)[i]
            start = np.median(refs[:, :3], axis=1)
            end = np.median(refs[:, -3:], axis=1)
            rising = (
                (start >= MIN_START_N)
                & (end - start >= MIN_RISE_N)
                & (end <= MAX_END_N)
            )
            i, refs, start, end = i[rising], refs[rising], start[rising], end[rising]
            if not len(i):
                continue
            time = sliding_window_view(t, width)[i]
            coverage = np.diff(time, axis=1).max(axis=1) <= MAX_GAP_S
            sections = np.stack(
                [np.median(part, axis=1) for part in np.array_split(refs, 3, axis=1)],
                axis=1,
            )
            smooth = np.median(sliding_window_view(refs, 3, axis=1), axis=-1)
            drawdown = (np.maximum.accumulate(smooth, axis=1) - smooth).max(axis=1)
            valid = (
                coverage
                & (np.diff(sections, axis=1) >= 0).all(axis=1)
                & (drawdown <= MAX_BACKTRACK_N)
            )
            i, start, end = i[valid], start[valid], end[valid]
            if not len(i):
                continue
            counts = sliding_window_view(ad, width)[i]
            p10, p90 = np.quantile(counts, [0.1, 0.9], axis=1)
            evidence.append(
                np.column_stack(
                    (
                        t[i],
                        t[i + width - 1],
                        np.full(len(i), width),
                        start,
                        end,
                        end - start,
                        np.median(counts[:, :3], axis=1),
                        np.median(counts[:, -3:], axis=1),
                        p90 - p10,
                        p90,
                    )
                )
            )
    return np.concatenate(evidence) if evidence else np.empty((0, len(WINDOW_FIELDS)))


def screen_recording(t, bsq, ad, finger, press, caps=PROFILE_CAPS):
    arrays = [np.asarray(a) for a in (t, bsq, ad, finger, press)]
    t, bsq, ad, finger, press = arrays
    if any(a.ndim != 1 or len(a) != len(t) for a in arrays):
        raise ValueError("Expected aligned one-dimensional arrays")
    if not all(np.isfinite(a).all() for a in arrays) or np.any(np.diff(t) <= 0):
        raise ValueError("Expected finite chronological samples")
    if (
        not np.isin(finger, np.arange(len(FINGERS))).all()
        or np.any(press <= 0)
        or np.any(press != np.floor(press))
        or np.any(np.diff(press) < 0)
    ):
        raise ValueError("Invalid finger index or press ID")
    raw_events = eligible_events = 0
    hits = {cap: [] for cap in caps}
    scales = {}
    for f in np.unique(finger):
        p01, p95 = np.quantile(ad[finger == f], [0.01, 0.95])
        scales[f] = float(p95 - p01)
    boundaries = np.r_[0, np.flatnonzero(np.diff(press)) + 1, len(t)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < MIN_WINDOW_SAMPLES:
            continue
        f = finger[start]
        if np.any(finger[start:end] != f):
            raise ValueError("A routed press must use one finger")
        raw_events += 1
        scale = scales[f]
        windows = event_windows(t[start:end], bsq[start:end], ad[start:end])
        if not len(windows):
            continue
        eligible_events += 1
        _, _, _, _, _, _, ad_start, ad_end, ad_span, ad_p90 = windows.T
        endpoint_delta = np.abs(ad_end - ad_start)
        for cap in caps:
            limit = min(float(cap), MAX_AD_FRACTION * scale)
            stable_ad = (
                (ad_p90 <= MAX_WINDOW_AD_P90)
                & (ad_span <= limit)
                & (endpoint_delta <= limit)
            )
            found = windows[stable_ad]
            if not len(found):
                continue
            # One row per press. Prefer the largest BSQ rise, then the flattest AD.
            start_s, _, _, _, _, rise_n, _, _, span, _ = found.T
            j = np.lexsort((start_s, span, -rise_n))[0]
            hits[cap].append((int(f), int(press[start]), found[j], scale, limit))
    return raw_events, eligible_events, hits


def scan_file(path):
    with np.load(path, allow_pickle=False) as data:
        result = screen_recording(
            *(data[k] for k in ("t", "x", "ad", "finger", "press"))
        )
    return path.stem, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="JSON array of recording metadata, using id and key fields",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    inventory = {r["id"]: r for r in json.loads(args.records.read_text())}
    files = sorted(args.aligned_dir.glob("*.npz"))
    if not files:
        parser.error("No aligned NPZ files found")
    if any(p.stem not in inventory for p in files):
        parser.error("Aligned NPZ filename does not appear in records inventory")
    counts = {
        cap: dict(
            presses=0,
            recordings=set(),
            recording_fingers=set(),
            repeated_presses=0,
            repeated_recordings=set(),
            repeated_recording_fingers=set(),
        )
        for cap in PROFILE_CAPS
    }
    rows = []
    raw = eligible = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rid, (events, rising, hits) in pool.map(scan_file, files, chunksize=8):
            raw += events
            eligible += rising
            for cap, found in hits.items():
                fingers = np.asarray([hit[0] for hit in found], dtype=int)
                repeat_counts = np.bincount(fingers, minlength=len(FINGERS))
                profile = counts[cap]
                profile["presses"] += len(found)
                if found:
                    profile["recordings"].add(rid)
                for f, event, window, scale, limit in found:
                    profile["recording_fingers"].add((rid, f))
                    repeats = int(repeat_counts[f])
                    if repeats < MIN_REPEAT_PRESSES:
                        continue
                    profile["repeated_presses"] += 1
                    profile["repeated_recordings"].add(rid)
                    profile["repeated_recording_fingers"].add((rid, f))
                    if cap == REVIEW_CAP:
                        values = [round(float(v), 3) for v in (*window, scale, limit)]
                        values[2] = int(values[2])
                        rows.append(
                            [
                                rid,
                                inventory[rid]["key"],
                                FINGERS[f],
                                event,
                                *values,
                                repeats,
                                "review",
                            ]
                        )
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "ad_plateau_review.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(rows)
    summary = dict(
        inventory_recordings=len(inventory),
        scanned_recordings=len(files),
        routed_presses=raw,
        presses_with_rising_window=eligible,
        thresholds=dict(
            durations_s=DURATIONS_S,
            min_start_n=MIN_START_N,
            min_rise_n=MIN_RISE_N,
            max_end_n=MAX_END_N,
            max_bsq_drawdown_n=MAX_BACKTRACK_N,
            max_sample_gap_s=MAX_GAP_S,
            max_ad_fraction=MAX_AD_FRACTION,
            max_window_ad_p90=MAX_WINDOW_AD_P90,
            review_cap=REVIEW_CAP,
            min_repeat_presses=MIN_REPEAT_PRESSES,
        ),
        profiles={
            cap: {k: len(v) if isinstance(v, set) else v for k, v in data.items()}
            for cap, data in counts.items()
        },
    )
    (args.out / "ad_plateau_screen_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
