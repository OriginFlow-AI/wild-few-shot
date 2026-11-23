"""Validate subject isolation on a complete proposed full-model split."""

import argparse
import gzip
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
EVAL_ROLES = ("val", "test")
ID_PATTERN = re.compile(r"([a-z]+)-?(\d{4})(?:\d+)?", re.I | re.ASCII)


def identity(subject):
    match = ID_PATTERN.fullmatch(subject.strip())
    if match is None:
        raise ValueError(f"Subject needs identity resolution: {subject!r}")
    return f"{match[1].lower()}-{match[2]}"


def subject_key(row):
    names = [row[key] for key in ("subject", "subject_id") if key in row]
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError("Every split row must carry a subject or subject_id")
    keys = {identity(name) for name in names}
    if len(keys) != 1:
        raise ValueError("Row subject and subject_id disagree")
    return next(iter(keys))


def require_disjoint(training_keys, excluded_keys):
    overlap = training_keys & excluded_keys
    if overlap:
        raise ValueError(f"Train contains {len(overlap)} held-out identities: {', '.join(sorted(overlap))}")


def validate_split(split, registry):
    for role in ("train", *EVAL_ROLES):
        if not isinstance(split.get(role), list):
            raise ValueError(f"Supply the complete train/val/test split; {role} must be a list")
    people = {role: {subject_key(row) for row in split[role]} for role in ("train", *EVAL_ROLES)}
    excluded = set(registry["excluded_identity_keys"]) | people["val"] | people["test"]
    require_disjoint(people["train"], excluded)
    registered_train = set(registry["registered_train_identity_keys"])
    unregistered = people["train"] - registered_train
    if unregistered:
        raise ValueError(f"{len(unregistered)} Train identities need new-cohort intake and evaluation reservation")
    if (people["val"] | people["test"]) & registered_train:
        raise ValueError("An existing Train identity was proposed for evaluation; check training exposure first")
    forbidden_stages = set(registry["excluded_train_stages"])
    allowed_stages = set(registry["eligible_train_stages"])
    blocked_sessions = {tuple(pair) for pair in registry["excluded_train_sessions"]}
    for row in split["train"]:
        stage = row.get("stage")
        session = row.get("session")
        if stage is None or not isinstance(session, str) or not session.strip():
            raise ValueError("Every Train row must carry stage and session")
        stage = str(stage)
        if "/" not in stage:
            stage = ("task83/" if stage.startswith("mix-") else "legacy/") + stage
        if stage in forbidden_stages:
            raise ValueError(f"Train contains held-out stage {stage}")
        if stage not in allowed_stages:
            raise ValueError(f"Train stage has no approved mapping: {stage}")
        if (subject_key(row), session) in blocked_sessions:
            raise ValueError("Train contains a held-out subject/session pair")
    test_exclusions = set(registry["excluded_test_recording_ids"])
    for row in split["test"]:
        if row.get("id") in test_exclusions or row.get("record_id") in test_exclusions or row.get("layout") == "1x5x3":
            raise ValueError("Test contains a Paxini/GEN3 recording")
    if people["val"] & people["test"]:
        raise ValueError("Val and Test share a subject identity")
    for role in EVAL_ROLES:
        opposite = "test" if role == "val" else "val"
        if set(registry["assigned_eval_identity_keys"][role]) & people[opposite]:
            raise ValueError(f"A {role} identity was assigned to {opposite}")
        if not set(registry["fine_required_identity_keys"][role]) <= people[role]:
            raise ValueError(f"The proposed {role} is missing identities from Force 3.0")
    return {"records": {role: len(split[role]) for role in people},
            "identities": {role: len(keys) for role, keys in people.items()},
            "training_exclusion_identities": len(excluded), "train_overlap": 0}


def read_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path, help="Complete proposed split, JSON or JSON.gz")
    parser.add_argument("--registry", type=Path, default=ROOT / "TRAIN_EXCLUSION.json")
    parser.add_argument("--export-registry", type=Path, help="Write a new registry carrying forward newly assigned evaluation subjects")
    args = parser.parse_args()
    registry = read_json(args.registry)
    split = read_json(args.split)
    try:
        report = validate_split(split, registry)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    if args.export_registry:
        assigned = {
            role: set(registry["assigned_eval_identity_keys"][role]) | {subject_key(row) for row in split[role]}
            for role in EVAL_ROLES
        }
        all_assigned = assigned["val"] | assigned["test"]
        next_registry = {
            **registry,
            "assigned_eval_identity_keys": {role: sorted(keys) for role, keys in assigned.items()},
            "excluded_identity_keys": sorted(set(registry["excluded_identity_keys"]) | all_assigned),
            "protected_without_active_eval_assignment": sorted(
                set(registry["protected_without_active_eval_assignment"]) - all_assigned),
            "pending_intake_identity_keys": sorted(
                set(registry.get("pending_intake_identity_keys", [])) - all_assigned),
        }
        with args.export_registry.open("x", encoding="utf-8") as handle:
            json.dump(next_registry, handle, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
