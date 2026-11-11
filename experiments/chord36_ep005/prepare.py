"""Prepare the pinned Chord 3.6 ep005 source and verified inference bundle."""

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


SOURCE_URL = "https://github.com/OriginFlow-AI/FM_PULSE.git"
SOURCE_REV = "f6116893d894d7ca133e57445886072b11c75d26"
BASELINE = Path("baselines/chord36_ep005")
DELIVERY_URL = (
    "ssh://git@code.infra.originflowai.com:34575/ai-infra/pulse-software-delivery.git"
)
DELIVERY_REV = "eb66168b1adafa2ac15dab62306bdea8d0c61ab4"
ASSETS = Path("models/pose/Chord36_MuonWD005_ep005")


def run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True)


def checkout(directory, url, revision, *patterns):
    directory.mkdir()
    run("git", "init", "--quiet", cwd=directory)
    run("git", "remote", "add", "origin", url, cwd=directory)
    run("git", "config", "remote.origin.promisor", "true", cwd=directory)
    run("git", "config", "remote.origin.partialclonefilter", "blob:none", cwd=directory)
    run("git", "sparse-checkout", "set", "--no-cone", *patterns, cwd=directory)
    run("git", "fetch", "--depth=1", "--filter=blob:none", "origin", revision, cwd=directory)
    run("git", "checkout", "--quiet", "--detach", revision, cwd=directory)


def prepare(work_dir, asset_dir):
    if work_dir.exists() or work_dir.is_symlink():
        raise FileExistsError(f"Choose a new work directory: {work_dir}")
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="chord36-", dir=work_dir.parent) as scratch:
        scratch = Path(scratch)
        result = scratch / "result"
        result.mkdir()
        source = result / "source"
        checkout(source, SOURCE_URL, SOURCE_REV, f"/{BASELINE}/")
        run(sys.executable, BASELINE / "verify_release.py", cwd=source)

        if asset_dir is None:
            delivery = scratch / "delivery"
            checkout(
                delivery, DELIVERY_URL, DELIVERY_REV,
                f"/{ASSETS}/weights.pt", f"/{ASSETS}/tokenizer.pt",
            )
            asset_dir = delivery / ASSETS

        run(
            sys.executable, BASELINE / "build_bundle.py",
            "--weights", asset_dir / "weights.pt",
            "--tokenizer", asset_dir / "tokenizer.pt",
            "--output", result / "bundle",
            cwd=source,
        )
        result.rename(work_dir)
    print(f"Verified source and bundle: {work_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".chord36-work"))
    parser.add_argument(
        "--asset-dir", type=Path, help="Local directory containing release weights.pt and tokenizer.pt"
    )
    args = parser.parse_args()
    work_dir = args.work_dir.expanduser().absolute()
    asset_dir = args.asset_dir.expanduser().resolve() if args.asset_dir else None
    try:
        prepare(work_dir, asset_dir)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
