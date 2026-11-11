"""Prepare, train the three K2 models in parallel, and verify the four evaluations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "scripts"))
from common import BRANCHES, atomic, load, settings


def gpu_candidates():
    readings = {}
    for sample in range(3):
        result = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True)
        for line in result.splitlines():
            index, uuid, free, utilization = map(str.strip, line.split(","))
            readings.setdefault((int(index), uuid), []).append((int(free), int(utilization)))
        if sample < 2:
            time.sleep(3)
    candidates = []
    for (index, uuid), rows in readings.items():
        free_mib = min(row[0] for row in rows)
        limit = min(32, int((free_mib / 1024 - 6) // 4) * 4)
        if limit >= 20:
            candidates.append(dict(index=index, uuid=uuid, min_free_mib=free_mib,
                                   mean_utilization=sum(row[1] for row in rows) / len(rows),
                                   process_memory_limit_GiB=limit))
    return sorted(candidates, key=lambda row: (row["mean_utilization"], -row["min_free_mib"]))


def start(root, branch, gpu, smoke, attempt):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"],
               TASK103_GPU_MEMORY_GIB=str(gpu["process_memory_limit_GiB"]),
               TASK103_SMOKE_WINDOWS=str(settings()["global_batch"]))
    command = [sys.executable, "-B", str(CODE / "scripts/train.py"),
               "--root", str(root), "--branch", branch]
    if smoke:
        command.append("--smoke")
    kind = "smoke" if smoke else "train"
    logfile = root / "logs" / f"{kind}_{branch}_{attempt}.log"
    with logfile.open("ab") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    atomic(root / "checks" / f"{kind}_process_{branch}.json",
           dict(pid=process.pid, command=command, gpu=gpu, log=str(logfile), attempt=attempt))
    print(f"{kind} {branch}: PID {process.pid}, GPU {gpu['index']}, {gpu['process_memory_limit_GiB']} GiB", flush=True)
    return process, logfile


def invoke(path, *args):
    subprocess.run([sys.executable, "-B", str(CODE / path), *map(str, args)], check=True)


def train_all(root):
    remaining = [branch for branch in BRANCHES if not (root / "runs" / branch / "DONE.json").exists()]
    if not remaining:
        return
    cards = gpu_candidates()
    if len(cards) < len(remaining):
        raise RuntimeError(f"Only {len(cards)} GPUs with sufficient memory for {len(remaining)} branches")
    chosen = dict(zip(remaining, cards))
    smoke = {branch: start(root, branch, chosen[branch], True, 1) for branch in remaining}
    smoke_codes = {branch: child.wait() for branch, (child, _) in smoke.items()}
    if any(smoke_codes.values()):
        raise RuntimeError(f"GPU smoke failed: {smoke_codes}")

    running = {}
    failed = {}
    attempts = dict.fromkeys(remaining, 1)
    for branch in remaining:
        running[branch] = start(root, branch, chosen[branch], False, 1)
    while running:
        atomic(root / "STATUS.json", dict(phase="training_or_evaluating", active=list(running), failed=failed))
        for branch, (process, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[branch]
            if code == 0 and (root / "runs" / branch / "DONE.json").is_file():
                print(f"{branch} completed final evaluation.", flush=True)
                continue
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 32768))
                tail = handle.read().lower()
            if b"out of memory" in tail and attempts[branch] < 3:
                used = {chosen[b]["uuid"] for b in running}
                options = [g for g in gpu_candidates() if g["uuid"] not in used]
                if options:
                    chosen[branch] = options[0]
                    attempts[branch] += 1
                    running[branch] = start(root, branch, chosen[branch], False, attempts[branch])
                    continue
            failed[branch] = dict(exit_code=code, log=str(log))
        if running:
            time.sleep(10)
    if failed:
        raise RuntimeError(f"Failed branches: {failed}")


def main(root, source, prepare_only):
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with (root / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(root / "checks/launcher_process.json", dict(pid=os.getpid(), argv=sys.argv))
        try:
            if not (root / "prepared/dataset.json").is_file():
                atomic(root / "STATUS.json", dict(phase="preparing_cross_subject_split_and_normalization"))
                invoke("scripts/prepare_cross_subject.py", "--source", source, "--root", root)
            if prepare_only:
                return
            train_all(root)
            atomic(root / "STATUS.json", dict(phase="generating_report"))
            invoke("scripts/report.py", "--root", root)
            atomic(root / "STATUS.json", dict(phase="report_ready"))
            invoke("analysis/audit_final_run.py", "--root", root)
            invoke("analysis/score_force_thresholds.py", "--root", root, "--output", root / "results")
            invoke("analysis/export_models.py", "--root", root)
            atomic(root / "STATUS.json", dict(phase="complete", model=settings()["model"],
                                               training_branches=BRANCHES, experiments=[1, 2, 3, 4]))
        except Exception as exc:
            atomic(root / "STATUS.json", dict(phase="failed", error=str(exc)))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    main(args.root, args.source, args.prepare_only)
