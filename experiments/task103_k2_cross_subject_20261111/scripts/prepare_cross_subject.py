"""Repartition the frozen Task103 windows by person and fit Train-only norms."""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from pathlib import Path

import numpy as np

from common import EXPERIMENT, atomic, load, settings, sha, source_binding


SPLITS = ("train", "val", "test")
REFERENCE_STAGES = ("mix-test-5", "mix-test-7")


def subject_split(entries, seed):
    stages = collections.defaultdict(set)
    for entry in entries:
        stages[entry["subject"]].add(entry["stage"])
    strata = collections.defaultdict(list)
    for subject, present in stages.items():
        strata[tuple(stage in present for stage in REFERENCE_STAGES)].append(subject)
    assignment = {}
    for (has_test5, has_test7), people in sorted(strata.items()):
        ordered = sorted(people, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
        n = round(len(ordered) * 0.15) if has_test5 else 0
        for i, person in enumerate(ordered):
            assignment[person] = "val" if i < n else "test" if i < 2 * n else "train"
    return assignment


def record_split(stage, person_split):
    if stage == "mix-test-5" and person_split in ("val", "test"):
        return person_split
    if stage == "mix-test-7":
        return "test" if person_split == "test" else None
    if stage not in REFERENCE_STAGES and person_split == "train":
        return "train"
    return None


def entry_norm(job):
    source, entry = job
    record_id, side = entry["id"], entry["side"]
    base = source / "cache/emg" / record_id
    emg = np.load(base / f"{side}.npy", mmap_mode="r")
    mask = np.load(base / f"{side}_valid.npy", mmap_mode="r")
    valid = emg[mask]
    assert len(emg) == entry["samples"] and np.isfinite(valid).all()
    emg_stats = np.stack([np.full(16, len(valid)), valid.sum(axis=0, dtype=np.float64),
                          np.einsum("ij,ij->j", valid, valid, dtype=np.float64)])
    label_path = source / "cache/labels" / f"{record_id}.npz"
    assert sha(label_path) == entry["labels_sha256"]
    with np.load(label_path) as labels:
        valid = labels[f"{side}_valid"]
        force_stats = {}
        for kind in ("raw", "cal"):
            force = np.where(valid, labels[f"{side}_{kind}"], 0).astype(np.float64)
            force_stats[kind] = np.stack([valid.sum(axis=0), force.sum(axis=0),
                                          (force * force).sum(axis=0)])
    return emg_stats, force_stats


def normalized(stats):
    count, total, squares = stats
    assert np.all(count > 0)
    mean = total / count
    std = np.sqrt(np.maximum(0, squares / count - mean ** 2))
    assert np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std > 1e-6)
    return dict(mean=mean.tolist(), std=std.tolist(), count=count.astype(np.int64).tolist())


def compute_norm(source, entries, workers):
    emg = np.zeros((3, 16))
    force = {kind: np.zeros((3, 5)) for kind in ("raw", "cal")}
    jobs = [(source, e) for e in entries if e["split"] == "train"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (emg_stats, force_stats) in enumerate(pool.map(entry_norm, jobs, chunksize=8), 1):
            emg += emg_stats
            for kind in force:
                force[kind] += force_stats[kind]
            if i % 400 == 0:
                print(f"Train-only normalization: {i}/{len(jobs)} hands", flush=True)
    return normalized(emg), {kind: normalized(stats) for kind, stats in force.items()}


def main(source, root, workers):
    import torch
    from train import network

    config = settings()
    previous_path = source / "prepared/dataset.json"
    previous = load(previous_path)
    source_plan = load(source / "plan.json")
    plan_by_id = {row["id"]: row for row in source_plan["rows"]}
    people = subject_split(previous["entries"], config["split_seed"])
    entries = []
    reindex = {}
    for i, e in enumerate(previous["entries"]):
        person = e["subject"]
        split = record_split(e["stage"], people[person])
        if split is None:
            continue
        assert plan_by_id[e["source_id"]]["subject"] == person
        reindex[i] = (len(entries), split)
        entries.append({**e, "split": split})
    split_people = {s: sorted({e["subject"] for e in entries if e["split"] == s}) for s in SPLITS}
    assert set(people) == {p for group in split_people.values() for p in group}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert set(split_people[a]).isdisjoint(split_people[b])
    per_record = {e["id"]: e for e in entries}
    placeholder_etag = "00000000000000000000000000000000-1"
    for key in ("recording_id", "object_content_id"):
        identity = collections.defaultdict(set)
        for record_id, entry in per_record.items():
            meta = plan_by_id[record_id]
            if key == "object_content_id":
                if not meta["etag"] or meta["etag"] == placeholder_etag:
                    continue
                value = (meta["etag"], meta["size"])
            else:
                value = meta[key]
            identity[value].add(entry["split"])
        assert all(len(splits) == 1 for splits in identity.values()), f"Cross-split shared {key}"
    processed_hashes = collections.defaultdict(set)
    for entry in entries:
        processed_hashes[entry["EMG"]["sha256"]].add(entry["split"])
    assert all(len(splits) == 1 for splits in processed_hashes.values()), "Cross-split duplicate EMG"

    indices = {s: [] for s in SPLITS}
    source_index_sha = {}
    for old_split in SPLITS:
        path = source / "prepared" / f"{old_split}_windows.npy"
        source_index_sha[old_split] = sha(path)
        for old_entry, start in np.load(path):
            item = reindex.get(int(old_entry))
            if item is not None:
                entry, split = item
                indices[split].append((entry, int(start)))
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        pairs = np.asarray(sorted(indices[split]), dtype=np.int32)
        assert pairs.ndim == 2 and len(pairs) > 0
        assert len(np.unique(pairs, axis=0)) == len(pairs)
        assert set(pairs[:, 0]) == {i for i, e in enumerate(entries) if e["split"] == split}
        np.save(prepared / f"{split}_windows.npy", pairs)

    split_rows = []
    for record_id, entry in sorted(per_record.items()):
        original = plan_by_id[record_id]
        split_rows.append(dict(id=record_id, recording_id=original["recording_id"], subject=entry["subject"],
                               stage=entry["stage"], split=entry["split"], source_id=entry["source_id"]))
    with (prepared / "recording_split.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(split_rows)
    counts = {
        split: dict(people=len(split_people[split]), recordings=sum(r["split"] == split for r in split_rows),
                    hands=sum(e["split"] == split for e in entries), windows=len(indices[split]),
                    by_stage=dict(collections.Counter(r["stage"] for r in split_rows if r["split"] == split)))
        for split in SPLITS
    }
    split_manifest = dict(seed=config["split_seed"], policy="subject hash within test5/test7 availability strata",
                          subject_disjoint=True, source_record_identity_disjoint=True,
                          processed_emg_sha256_disjoint=True,
                          placeholder_etag_records=sum(plan_by_id[i]["etag"] == placeholder_etag for i in per_record),
                          calibration_source_same_subject=True, subjects=split_people, counts=counts,
                          source_dataset_sha256=sha(previous_path), source_window_indices_sha256=source_index_sha,
                          recording_split_sha256=sha(prepared / "recording_split.csv"))
    atomic(prepared / "subject_split.json", split_manifest)
    print(f"Frozen splits: {counts}", flush=True)

    plan_rows = []
    for r in source_plan["rows"]:
        person_split = people.get(r["subject"])
        split = record_split(r["stage"], person_split)
        plan_rows.append({**r, "split": split, "active": r["id"] in per_record})
    atomic(root / "plan.json", {**source_plan, "rows": plan_rows, "split_seed": config["split_seed"],
                                "evaluation_scope": config["evaluation_scope"]})
    emg_norm, force_norm = compute_norm(source, entries, workers)
    np.savez(prepared / "emg_train_norm.npz", emg_mean=np.asarray(emg_norm["mean"], dtype=np.float32),
             emg_std=np.asarray(emg_norm["std"], dtype=np.float32), emg_count=emg_norm["count"])
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    model = network()
    weights = root / "assets/common_stage2_initial_state_dict.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    data = dict(entries=entries, force_norm=force_norm, emg_norm=emg_norm,
                windows={s: len(indices[s]) for s in SPLITS}, split_recordings=counts,
                exclusions=previous["exclusions"], plan_sha256=sha(root / "plan.json"),
                config_sha256=sha(EXPERIMENT / "config.json"), source_code=source_binding(),
                subject_split_sha256=sha(prepared / "subject_split.json"),
                emg_norm_sha256=sha(prepared / "emg_train_norm.npz"),
                common_initial_sha256=sha(weights),
                window_indices_sha256={s: sha(prepared / f"{s}_windows.npy") for s in SPLITS})
    atomic(prepared / "dataset.json", data)
    atomic(root / "STATUS.json", dict(phase="prepared", counts=counts,
                                      dataset_sha256=sha(prepared / "dataset.json")))
    print("Prepared dataset, Train-only normalization and common random initialization.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", default=16, type=int)
    args = parser.parse_args()
    main(args.source, args.root, args.workers)
