"""Explicit standalone two-stage run; no server assignment or automatic launch."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from io_local import ROOT, read, write, sha, closed_rows


def call(mode, phase, name=None, steps=None, resume=False):
    cmd = [sys.executable, "-B"]
    if mode in ("qualify", "formal"):
        cmd += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=8"]
    cmd += [str(ROOT / "train.py"), mode, "--phase", str(phase)]
    if name:
        cmd += ["--name", name]
    if steps:
        cmd += ["--steps", str(steps)]
    if resume:
        cmd += ["--resume"]
    tag = f"{mode}_p{phase}_{name or 'main'}{'_resume' if resume else ''}"
    with (ROOT / (tag + ".log")).open("ab", buffering=0) as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        write(ROOT / "ACTIVE_CHILD.json", dict(pid=child.pid, args=cmd, time=time.time()))
        code = child.wait()
    if code:
        raise RuntimeError(f"{tag}: return code {code}; inspect {tag}.log")


def check_assets():
    rows = closed_rows()
    subjects = {role: {r["subject"] for r in rs} for role, rs in rows.items()}
    roles = list(subjects)
    for i, a in enumerate(roles):
        for b in roles[i+1:]:
            assert not subjects[a] & subjects[b], ("Subject leakage", a, b)
    for role in ("train", "val"):
        assert rows[role], ("Empty role", role)
        assert all(r["split"] == role for r in rows[role])
    # Stage identity is explicit, never guessed from a task or session number.
    assert all(r["stage"] and r["stage"] not in ("legacy/4", "legacy/5", "mix/9", "mix/10", "mix-9", "mix-10") for r in rows["train"])
    for n in ("data/INDEX.json.gz", "data/CLOCK_BINDINGS.json", "data/LABEL_BINDINGS.json.gz", "data/norm.npz", "data/RQ_FROZEN.pt", "LABEL_POLICY.json"):
        assert (ROOT / n).is_file(), n


def qualify():
    records = []
    for phase in (1, 2):
        call("calibrate_qualification", phase)
        continuous, split = f"p{phase}_continuous", f"p{phase}_split"
        if (ROOT / "qualification" / continuous).exists() or (ROOT / "qualification" / split).exists():
            raise FileExistsError("Qualification outputs already exist; inspect them or use a fresh run")
        call("qualify", phase, continuous, 2)
        call("qualify", phase, split, 1)
        call("qualify", phase, split, 1, True)
        a, b = [read(ROOT / "qualification" / n / "EXIT_STATE.json") for n in (continuous, split)]
        assert a["global_step"] == b["global_step"] == 2 and a["fingerprints"] == b["fingerprints"]
        for rank in range(8):
            for prefix in ("INPUT", "GRAD"):
                paths = [ROOT / "qualification" / n / f"{prefix}_rank{rank}.jsonl" for n in (continuous, split)]
                assert paths[0].read_bytes() == paths[1].read_bytes(), (phase, rank, prefix)
        for n in (continuous, split):
            d = read(ROOT / "qualification" / n / "DEV_SMOKE.json")
            assert d["Test_read"] is False and d["complete"] is False
        records.append(dict(phase=phase, continuous=a, restored=b))
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob("*.py")}
    for n in ("PROTOCOL.json", "EXECUTION.json", "data/FULL_SPLIT.json.gz", "data/INDEX.json.gz", "data/TASK43_LOCAL_INVENTORY.json.gz", "data/LABEL_BINDINGS.json.gz", "data/CLOCK_BINDINGS.json", "LABEL_POLICY.json", "data/norm.npz", "data/RQ_FROZEN.pt"):
        files[n] = sha(ROOT / n)
    write(ROOT / "ADMISSION.json", dict(status="PASS_FULL_EIGHT_RANK_RESTORE_AND_BOTH_PHASES", records=records, files=files, Test_read=False))


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "formal", "all"))
    args = parser.parse_args()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_assets()
    if args.mode in ("qualify", "all"):
        qualify()
    if args.mode in ("formal", "all"):
        admission = read(ROOT / "ADMISSION.json")
        assert admission["status"] == "PASS_FULL_EIGHT_RANK_RESTORE_AND_BOTH_PHASES"
        for n, expected in admission["files"].items():
            assert sha(ROOT / n) == expected, ("Qualified code or input changed", n)
        write(ROOT / "TRAINING_STARTED.json", dict(time=time.time(), controller_pid=os.getpid(), Test_read=False))
        for phase in (1, 2):
            run = ROOT / "runs" / f"phase{phase}"
            if (run / "RUN_DONE.json").exists():
                continue
            # Refit the phase2 gradient coefficient on the actual phase1 best,
            # rather than reusing the qualification parent's coefficient.
            call("calibrate", phase)
            call("formal", phase, resume=(run / "last.pt").exists())
            assert read(run / "RUN_DONE.json")["status"] == "PASS_COMPLETE_VAL_BEST_RELOADED"
        write(ROOT / "RUN_DONE.json", dict(status="PASS_BOTH_STAGES_COMPLETE", Test_read=False))


if __name__ == "__main__":
    main()
