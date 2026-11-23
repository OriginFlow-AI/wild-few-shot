"""Create an independent run from code and separately supplied private assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REQUIRED = [
    "data/FULL_SPLIT.json.gz", "data/TASK43_LOCAL_INVENTORY.json.gz",
    "data/CLOCK_BINDINGS.json", "data/LABEL_BINDINGS.json.gz",
    "LABEL_POLICY.json", "label_calibration/LABEL_POLICY.json",
    "label_calibration/calibration_assets/coefficients.json",
    "label_calibration/calibration_assets/tables.npz",
]
OPTIONAL = ["data/INDEX.json.gz", "data/norm.npz", "data/RQ_FROZEN.pt", "MCAP_FAST_QUALIFIED.json"]


def initialize(assets, destination):
    assets = Path(assets).resolve()
    destination = Path(destination).resolve()
    code = Path(__file__).resolve().parent
    if destination.exists():
        raise FileExistsError("Use a new independent run directory")
    if destination.is_relative_to(code) or assets.is_relative_to(code):
        raise ValueError("Private assets and run outputs must stay outside the source package")
    if assets == destination or destination.is_relative_to(assets) or assets.is_relative_to(destination):
        raise ValueError("Run and asset directories must be separate")
    for n in REQUIRED:
        if not (assets / n).is_file():
            raise FileNotFoundError(assets / n)
    if (assets / "data/norm.npz").exists() != (assets / "data/RQ_FROZEN.pt").exists():
        raise ValueError("Supply matched norm+RQ together, or omit both for Train-only fitting")
    # Select only source files and explicit configuration, never credentials,
    # source datasets, runtime outputs, checkpoints or arbitrary asset files.
    selected = [p for p in code.rglob("*.py") if "__pycache__" not in p.parts and "tests" not in p.relative_to(code).parts]
    selected += [code / n for n in ("PROTOCOL.json", "EXECUTION.json", "SOURCE_PROVENANCE.json", "README.md", "DATA_SCHEMA.md", "PORTABILITY_NOTES.md", "requirements.txt")]
    destination.mkdir(parents=True)
    for p in selected:
        q = destination / p.relative_to(code)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
    receipts = {}
    for n in REQUIRED + OPTIONAL:
        p = assets / n
        if not p.exists():
            continue
        q = destination / n
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        receipts[n] = hashlib.sha256(q.read_bytes()).hexdigest()
    clocks = json.loads((destination / "data/CLOCK_BINDINGS.json").read_bytes())["rows"]
    for binding in clocks.values():
        relative = Path(binding["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "data":
            raise ValueError("Clock paths must be relative artifacts beneath data/")
        q = destination / relative
        q.parent.mkdir(parents=True, exist_ok=True)
        if not q.exists():
            shutil.copyfile(assets / relative, q)
        actual = hashlib.sha256(q.read_bytes()).hexdigest()
        if actual != binding["sha256"]:
            raise ValueError(f"Clock binding mismatch: {relative}")
        receipts[relative.as_posix()] = actual
    for n in ("calibration", "qualification", "runs", "snapshots"):
        (destination / n).mkdir(exist_ok=True)
    receipt = {"status": "CREATED_NOT_TRAINING", "copied_private_assets": receipts, "Test_read": False}
    (destination / "INITIALIZED.json").write_text(json.dumps(receipt, separators=(",", ":")))
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", required=True)
    p.add_argument("--run", required=True)
    a = p.parse_args()
    print(json.dumps(initialize(a.assets, a.run)))
