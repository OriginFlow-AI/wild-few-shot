"""Read-only model audit after an actual first full validation.

Only writes its JSON receipt under ROOT/checks. It does not run inference,
change training state, or read Test metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


BRANCHES = ("calibrated_huber", "raw_huber", "raw_original")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def audit(root, branch):
    root = Path(root)
    dataset_path = root / "prepared/dataset.json"
    dataset = json.loads(dataset_path.read_text())
    dataset_hash = sha(dataset_path)
    expected = {e["id"] for e in dataset["entries"] if e["split"] == "val"}
    validations = {}
    contracts = {}
    for name in BRANCHES:
        directory = root / "runs" / name
        val = json.loads((directory / "epoch001_val.json").read_text())
        rows = {row["id"]: row for row in val["rows"]}
        assert len(rows) == len(val["rows"]) == val["records"] == len(expected)
        assert set(rows) == expected
        assert np.isfinite(val["loss"])
        assert abs(float(np.mean([v["loss"] for v in rows.values()])) - val["loss"]) < 1e-10
        validations[name] = (val, rows)
        contracts[name] = json.loads((directory / "CONTRACT.json").read_text())
        assert contracts[name]["dataset_sha256"] == dataset_hash
    first = validations[BRANCHES[0]][1]
    for _, rows in validations.values():
        for record_id, row in rows.items():
            for target in ("raw", "cal", "bsq"):
                for key in ("points", "mape_points"):
                    assert row["metrics"][target][key] == first[record_id]["metrics"][target][key]
    assert len({c["common_initial_sha256"] for c in contracts.values()}) == 1
    assert contracts["raw_huber"]["force_norm"] == contracts["raw_original"]["force_norm"]

    candidate = root / "archive/epoch_candidates" / branch / dataset_hash[:16] / "epoch001.pt"
    best = root / "runs" / branch / "best.pt"
    before = torch.load(candidate, map_location="cpu", weights_only=True)
    after = torch.load(best, map_location="cpu", weights_only=True)
    assert before["state"]["history"] == []
    assert before["state"]["epoch"] == 0
    assert before["state"]["next_update"] == before["state"]["global_update"]
    assert after["state"]["epoch"] == 1 and after["state"]["next_update"] == 0
    assert after["state"]["global_update"] == before["state"]["global_update"]
    assert len(after["state"]["history"]) == 1
    assert after["state"]["history"][0]["val_loss"] == validations[branch][0]["loss"]
    assert after["state"]["best"] == validations[branch][0]["loss"]
    assert equal(before["binding"], after["binding"])
    expected_binding = {k: v for k, v in contracts[branch].items() if k != "configuration"}
    assert equal(after["binding"], expected_binding)
    for component in ("model", "optimizer", "torch_rng", "cuda_rng"):
        assert equal(before[component], after[component]), component
    assert all(torch.isfinite(t).all().item() for t in after["model"].values())
    result = dict(
        status="PASS_REAL_COMPLETE_VAL_AND_STATE_PRESERVATION",
        branch=branch,
        epoch=1,
        global_update=after["state"]["global_update"],
        Val_recordings=len(expected),
        prepared_Val_record_set_exactly_covered=True,
        trial_equal_validation_loss=validations[branch][0]["loss"],
        objective="(1-Corr)+0.3*MAPE+0.03*MAE" if branch == "raw_original" else "Huber_delta_1N",
        selected_best_epoch=1,
        selected_best_sha256=sha(best),
        pre_Val_candidate_sha256=sha(candidate),
        model_optimizer_CPU_CUDA_RNG_unchanged_by_full_Val=True,
        all_three_Val_record_sets_and_target_point_counts_identical=True,
        all_three_common_initial_sha256=contracts[branch]["common_initial_sha256"],
        dataset_sha256=dataset_hash,
        test_metrics_read=False,
        scoring_source_policy="self_test5_nearest_test5_for_other_stages_v1",
    )
    output = root / "checks" / f"{branch}_epoch001_Val_and_state_audit.json"
    temporary = output.with_suffix(".partial.json")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    args = parser.parse_args()
    audit(args.root, args.branch)
