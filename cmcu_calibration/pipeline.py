"""Batch calibration with source provenance, valid masks, and quality receipts."""

import argparse
import fcntl
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile

import numpy as np

from .calibration import predict_lookup
from .io import load_native
from .quality import prepare_source


def json_native(value):
    if isinstance(value, dict):
        return {str(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(v) for v in value]
    if isinstance(value, np.generic):
        return json_native(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            json_native(value), stream, ensure_ascii=False, indent=2, allow_nan=False
        )
    temporary.replace(path)


def atomic_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def read_manifest(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["records"] if isinstance(payload, dict) else payload
    if not rows:
        raise ValueError("Manifest contains no records")
    identities = set()
    normalized = []
    for row in rows:
        row = dict(row)
        identity = str(row["id"])
        if (
            identity in identities
            or identity in (".", "..")
            or Path(identity).name != identity
        ):
            raise ValueError("Recording IDs must be unique safe filenames")
        identities.add(identity)
        start = datetime.fromisoformat(row["start"])
        if not row.get("subject"):
            raise ValueError("Every record needs its subject")
        day = start.date().isoformat()
        if row.get("date", day) != day:
            raise ValueError("Manifest date disagrees with recording start")
        source_path = Path(row["path"])
        row.update(
            id=identity,
            date=day,
            stage=row.get("stage", "mix-test-5"),
            path=str(
                source_path if source_path.is_absolute() else path.parent / source_path
            ),
        )
        if row.get("end"):
            end = datetime.fromisoformat(row["end"])
            if (end.tzinfo is None) != (start.tzinfo is None):
                raise ValueError("Recording start/end timezone formats must match")
            if end < start:
                raise ValueError("Recording end precedes start")
        normalized.append(row)
    aware = {datetime.fromisoformat(r["start"]).tzinfo is not None for r in normalized}
    if len(aware) != 1:
        raise ValueError("Use consistently local or timezone-aware recording times")
    return sorted(
        normalized, key=lambda row: (datetime.fromisoformat(row["start"]), row["id"])
    )


def select_source(row, sources, max_age_s=3600):
    now = datetime.fromisoformat(row["start"])
    matches = []
    for source in sources:
        same_group = all(
            row.get(key) == source["record"].get(key)
            for key in ("subject", "date", "pc", "boot")
        )
        age = (now - source["start"]).total_seconds()
        if same_group and 0 < age <= max_age_s and source["end"] <= now:
            matches.append(source)
    return (
        max(matches, key=lambda s: (s["start"], s["record"]["id"])) if matches else None
    )


def run(
    manifest,
    output,
    *,
    slope_scope="recording",
    sliding_scope="off",
    routing="ad_delta",
    basic_qc=False,
    max_age_s=3600,
):
    if not 0 < max_age_s <= 3600:
        raise ValueError("Source age must be positive and at most 3600 seconds")
    if slope_scope not in ("recording", "finger", "off"):
        raise ValueError("Invalid slope scope")
    if sliding_scope not in ("recording", "press", "off"):
        raise ValueError("Invalid sliding scope")
    if routing not in ("ad_delta", "default_force_peak"):
        raise ValueError("Invalid press routing")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # OS locks release on exit or crash; concurrent runs sharing an output serialize.
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        settings = dict(
            slope_scope=slope_scope,
            sliding_scope=sliding_scope,
            routing=routing,
            basic_qc=basic_qc,
            max_age_s=max_age_s,
        )
        atomic_json(output / "run.json", dict(status="running", settings=settings))
        try:
            summary = _run(manifest, output, **settings)
        except Exception:
            atomic_json(output / "run.json", dict(status="failed", settings=settings))
            raise
        atomic_json(output / "run.json", dict(status="complete", settings=settings))
        return summary


def _run(manifest, output, *, slope_scope, sliding_scope, routing, basic_qc, max_age_s):
    rows = read_manifest(Path(manifest).resolve())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources, audit_rows, model_rows = [], [], []
    settings = dict(
        slope_scope=slope_scope,
        sliding_scope=sliding_scope,
        routing=routing,
        basic_qc=basic_qc,
        max_age_s=max_age_s,
    )
    for row in rows:
        audit = dict(
            id=row["id"],
            subject=row["subject"],
            date=row["date"],
            stage=row["stage"],
            start=row["start"],
            status="error",
        )
        try:
            is_test5 = row["stage"] == "mix-test-5"
            arrays = load_native(row["path"], needs_reference=is_test5)
            source = select_source(row, sources, max_age_s)
            models, quality = {}, None
            if is_test5:
                models, quality = prepare_source(
                    arrays,
                    routing=routing,
                    slope_scope=slope_scope,
                    sliding_scope=sliding_scope,
                    basic_qc=basic_qc,
                    zero_N=row.get("zero_N"),
                )
                audit["quality"] = quality
            values = {}
            target_rejected = quality is not None and quality["rejected"]
            for hand, offset in (("left", 0), ("right", 5)):
                ad = arrays[hand + "_ad"]
                calibrated = np.full(ad.shape, np.nan)
                if source is not None and not target_rejected:
                    for finger in range(5):
                        curve = source["models"].get(str(offset + finger))
                        if curve is not None:
                            calibrated[:, finger] = predict_lookup(ad[:, finger], curve)
                if quality is not None:
                    for finger in range(5):
                        channel = quality["channels"].get(str(offset + finger))
                        if channel is not None and channel["rejected_reason"]:
                            calibrated[:, finger] = np.nan
                    if sliding_scope == "press":
                        clock = arrays[hand + "_t"]
                        accepted = np.zeros(len(clock), dtype=bool)
                        for start, end in quality["accepted_press_intervals_ns"]:
                            accepted |= (clock >= start) & (clock < end)
                        calibrated[~accepted] = np.nan
                values[hand + "_t"] = arrays[hand + "_t"]
                values[hand + "_force_N"] = calibrated
                values[hand + "_valid"] = np.isfinite(calibrated)
                values[hand + "_default_force_N"] = arrays[hand + "_force"]
            values["record_id"] = np.asarray(row["id"])
            values["source_id"] = np.asarray(source["record"]["id"] if source else "")
            values["target_rejected"] = np.asarray(target_rejected)
            atomic_npz(output / "force" / (row["id"] + ".npz"), values)
            audit.update(
                status=(
                    "quality_rejected"
                    if target_rejected
                    else "calibrated" if source else "no_source"
                ),
                source_id=source["record"]["id"] if source else None,
                source_age_s=(
                    (
                        datetime.fromisoformat(row["start"]) - source["start"]
                    ).total_seconds()
                    if source
                    else None
                ),
                valid_points=sum(
                    int(values[side + "_valid"].sum()) for side in ("left", "right")
                ),
                total_points=sum(
                    int(values[side + "_valid"].size) for side in ("left", "right")
                ),
            )
            # Append only after applying the previous source, so a recording never calibrates itself.
            if models:
                start = datetime.fromisoformat(row["start"])
                if row.get("end"):
                    end = datetime.fromisoformat(row["end"])
                else:
                    duration = (
                        max(
                            arrays[side + "_t"][-1] for side in ("bsq", "left", "right")
                        )
                        - min(
                            arrays[side + "_t"][0] for side in ("bsq", "left", "right")
                        )
                    ) / 1e9
                    end = start + timedelta(seconds=float(duration))
                sources.append(dict(record=row, start=start, end=end, models=models))
                model_rows.append(
                    dict(
                        id=row["id"],
                        subject=row["subject"],
                        date=row["date"],
                        start=row["start"],
                        end=end.isoformat(),
                        channels=models,
                    )
                )
        except Exception as error:
            audit.update(reason=f"{type(error).__name__}: {error}")
            # A retry must not leave an older successful output attached to a failed receipt.
            (output / "force" / (row["id"] + ".npz")).unlink(missing_ok=True)
        audit_rows.append(audit)
        atomic_json(output / "quality" / (row["id"] + ".json"), audit)
    summary = dict(
        settings=settings,
        records=len(rows),
        statuses=dict(Counter(row["status"] for row in audit_rows)),
        fitted_sources=len(model_rows),
        first_record_rejection=dict(
            Counter(
                row["quality"]["record_reasons"][0]
                for row in audit_rows
                if row.get("quality", {}).get("record_reasons")
            )
        ),
        recording_reasons=dict(
            Counter(
                reason
                for row in audit_rows
                for reason in row.get("quality", {}).get("record_reasons", [])
            )
        ),
    )
    atomic_json(output / "models.json", model_rows)
    atomic_json(output / "quality.json", audit_rows)
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--slope-scope", choices=("recording", "finger", "off"), default="recording"
    )
    parser.add_argument(
        "--sliding-scope", choices=("off", "press", "recording"), default="off"
    )
    parser.add_argument(
        "--routing", choices=("ad_delta", "default_force_peak"), default="ad_delta"
    )
    parser.add_argument("--basic-qc", action="store_true")
    parser.add_argument("--max-age-s", type=float, default=3600)
    args = parser.parse_args()
    if not 0 < args.max_age_s <= 3600:
        parser.error("max age must be positive and at most 3600 seconds")
    summary = run(
        args.manifest,
        args.out,
        slope_scope=args.slope_scope,
        sliding_scope=args.sliding_scope,
        routing=args.routing,
        basic_qc=args.basic_qc,
        max_age_s=args.max_age_s,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["statuses"].get("error"):
        raise SystemExit(1)
