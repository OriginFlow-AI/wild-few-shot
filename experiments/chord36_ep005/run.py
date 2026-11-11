"""Smoke-test Chord 3.6 or fit a frozen-encoder linear probe for five-finger Fz."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


WINDOW = 11_790
CONTEXT = 1_790
STRIDE = 40
FRAMES = (WINDOW - CONTEXT) // STRIDE


def smoke(regression, tracking):
    emg = np.random.default_rng(0).normal(0, 0.05, (1, WINDOW, 16)).astype(np.float32)
    anchor = np.full((1, 22), 0.1, dtype=np.float32)
    anchor_valid = np.ones_like(anchor, dtype=bool)
    anchor_valid[:, 20:] = False
    for name, pose in (
        ("regression", regression.predict22(emg)),
        ("tracking", tracking.predict_tracking22(emg, anchor, anchor_valid)),
    ):
        if pose.shape != (1, WINDOW, 22) or not np.isfinite(pose).all():
            raise RuntimeError(f"Unexpected {name} output: {pose.shape}")
        if np.any(pose[:, :CONTEXT]):
            raise RuntimeError(f"{name} context must contain only zero placeholders")
        if name == "tracking":
            np.testing.assert_allclose(pose[:, CONTEXT, :20], anchor[:, :20], atol=1e-6)
        print(f"{name}: finite {pose.shape}, {WINDOW - CONTEXT} scored samples")


@dataclass
class Split:
    emg: np.ndarray
    force: np.ndarray
    valid: np.ndarray
    subject: np.ndarray


def read_split(path):
    with np.load(path, allow_pickle=False) as data:
        missing = {"emg", "force_n", "subject_id"} - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
        emg, force, subject = (data[key] for key in ("emg", "force_n", "subject_id"))
        valid = data["valid"] if "valid" in data else None
    if emg.ndim != 3 or not len(emg) or emg.shape[1:] != (WINDOW, 16) or emg.dtype.kind != "f":
        raise ValueError(f"{path}: emg must be floating point [B, {WINDOW}, 16] with B > 0")
    if force.shape != (len(emg), WINDOW, 5) or force.dtype.kind not in "fiu":
        raise ValueError(f"{path}: force_n must be numeric [B, {WINDOW}, 5]")
    if subject.shape != (len(emg),) or subject.dtype.kind not in "US":
        raise ValueError(f"{path}: subject_id must be a string array [B]")
    subject = np.char.strip(subject.astype(str))
    if np.any(subject == ""):
        raise ValueError(f"{path}: subject_id must be nonempty")
    if valid is not None and (valid.shape != force.shape or valid.dtype != np.bool_):
        raise ValueError(f"{path}: valid must be boolean [B, {WINDOW}, 5]")
    emg = emg.astype(np.float32, copy=False)
    if not np.isfinite(emg).all():
        raise ValueError(f"{path}: emg must be finite as float32")

    # Same midpoint lattice as the published 2 kHz → 50 Hz target interpolation.
    left = CONTEXT + STRIDE // 2 - 1 + STRIDE * np.arange(FRAMES)
    right = left + 1
    first, second = force[:, left].astype(np.float64), force[:, right].astype(np.float64)
    scored = np.isfinite(first) & np.isfinite(second)
    if valid is not None:
        scored &= valid[:, left] & valid[:, right]
    targets = 0.5 * np.where(scored, first, 0) + 0.5 * np.where(scored, second, 0)
    return Split(emg, targets, scored, subject)


@torch.inference_mode()
def encode(encoder, emg, device, batch_size):
    output = []
    for start in range(0, len(emg), batch_size):
        windows = torch.from_numpy(emg[start:start + batch_size]).to(device).transpose(1, 2)
        tokens = encoder(windows)[:, encoder.context_tokens:]
        if tokens.shape[1:] != (FRAMES, 16, 224):
            raise RuntimeError(f"Unexpected encoder output: {tuple(tokens.shape)}")
        output.append(tokens.mean(dim=2).cpu().numpy())
    features = np.concatenate(output).astype(np.float64)
    if not np.isfinite(features).all():
        raise RuntimeError("Encoder produced nonfinite features")
    return features


def fit_head(features, train, ridge):
    width = features.shape[-1]
    weights = np.empty((width, 5), dtype=np.float64)
    bias = np.empty(5, dtype=np.float64)
    for finger in range(5):
        valid = train.valid[..., finger]
        x, y = features[valid], train.force[..., finger][valid]
        if len(y) < 2:
            raise ValueError(f"Train finger index {finger} needs at least two valid labels")
        mean, scale = x.mean(axis=0), x.std(axis=0).clip(min=1e-6)
        z = (x - mean) / scale
        gram = z.T @ z / len(z) + ridge * np.eye(width)
        weights[:, finger] = np.linalg.solve(gram, z.T @ (y - y.mean()) / len(z)) / scale
        bias[finger] = y.mean() - mean @ weights[:, finger]
    return weights, bias


def average(values):
    return float(np.mean(values)) if values else None


def evaluate(prediction, split):
    if not np.isfinite(prediction).all():
        raise RuntimeError("Linear head produced nonfinite predictions")
    subjects = np.unique(split.subject)
    by_subject = {name: [] for name in ("mae_n", "mape_at_0_5n_percent", "pearson_r")}
    for subject in subjects:
        rows = split.subject == subject
        pred, true, valid = prediction[rows], split.force[rows], split.valid[rows]
        mae, mape, corr = [], [], []
        for finger in range(5):
            mask = valid[..., finger]
            p, y = pred[..., finger][mask], true[..., finger][mask]
            if not len(y):
                continue
            err = np.abs(p - y)
            mae.append(err.mean())
            active = np.abs(y) > 0.5
            if active.any():
                mape.append(100 * np.mean(err[active] / np.abs(y[active])))
            if len(y) >= 2 and np.ptp(y) > 0 and np.ptp(p) > 0:
                corr.append(np.corrcoef(p, y)[0, 1])
        subject_metrics = {"mae_n": mae, "mape_at_0_5n_percent": mape, "pearson_r": corr}
        for metric, values in subject_metrics.items():
            if values:
                by_subject[metric].append(average(values))
    return {
        "windows": len(split.subject),
        "subjects": len(subjects),
        "valid_labels": int(split.valid.sum()),
        "subject_macro": {name: average(values) for name, values in by_subject.items()},
        "subjects_per_metric": {name: len(values) for name, values in by_subject.items()},
    }


def force_probe(encoder, args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    if not math.isfinite(args.ridge) or args.ridge <= 0 or args.batch_size < 1:
        raise ValueError("ridge must be positive and finite, batch-size must be at least 1")
    train = read_split(args.train)
    validation = {"val": read_split(args.val)}
    if args.test:
        validation["test"] = read_split(args.test)
    seen = set(train.subject)
    for name, split in validation.items():
        if seen.intersection(split.subject):
            raise ValueError(f"Subject IDs overlap between {name} and an earlier split")
        if not split.valid.any():
            raise ValueError(f"{name} contains no valid Fz labels at scoring times")
        seen.update(split.subject)

    x = encode(encoder, train.emg, args.device, args.batch_size)
    weight, bias = fit_head(x, train, args.ridge)
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise RuntimeError("Ridge fit produced nonfinite parameters")
    results = {}
    for name, split in validation.items():
        prediction = encode(encoder, split.emg, args.device, args.batch_size) @ weight + bias
        results[name] = (prediction, evaluate(prediction, split))

    metrics = {name: report for name, (_, report) in results.items()}
    report = json.dumps(metrics, indent=2, allow_nan=False)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "head.npz", weight=weight, bias=bias, ridge=args.ridge)
    offsets = CONTEXT + (STRIDE - 1) / 2 + STRIDE * np.arange(FRAMES)
    for name, (prediction, _) in results.items():
        np.savez_compressed(
            args.output / f"{name}_predictions.npz",
            force_n=prediction,
            valid=validation[name].valid,
            sample_offset=offsets,
        )
    (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".chord36-work"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("smoke", help="Run both pose modes on synthetic EMG")
    probe = modes.add_parser("force-probe", help="Fit ridge regression on frozen encoder features")
    probe.add_argument("--train", type=Path, required=True)
    probe.add_argument("--val", type=Path, required=True)
    probe.add_argument("--test", type=Path)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--ridge", type=float, default=0.01)
    probe.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    root = args.work_dir.expanduser().resolve()
    if not (root / "source/baselines/chord36_ep005/predictor.py").is_file():
        parser.error(f"Run prepare.py first for {root}")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be at least 1")
    torch.set_num_threads(args.cpu_threads)
    sys.path.insert(0, str(root / "source"))
    from baselines.chord36_ep005 import Chord36RQRegression, Chord36RQTracking

    regression = Chord36RQRegression(root / "bundle", device=args.device)
    if args.mode == "smoke":
        smoke(regression, Chord36RQTracking(root / "bundle", device=args.device))
    else:
        # The portable Engine constructs one encoder shared by all three pose readouts.
        encoder = regression.engine.streams[0].network
        force_probe(encoder, args)


if __name__ == "__main__":
    main()
