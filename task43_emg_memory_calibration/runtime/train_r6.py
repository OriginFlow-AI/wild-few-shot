"""JSON-driven A-loss training, full Dev, and exact checkpoint continuation."""
import argparse
import collections
import fcntl
import json
import math
import os
import random
import subprocess
import time
import traceback
import gc
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from arms import Arm, Banks
from force_loss import force_objective
from train_data import Dataset, read

ROOT = Path(__file__).resolve().parent
ARMS = ("F", "B_L")
LOSS = {"corr_weight": .5, "mape_weight": .1, "mae_weight": .15}


def atomic(path, value, tensor=False):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        if tensor:
            torch.save(value, f)
        else:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False,separators=(",",":")).encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    if tensor and path.name == "last.pt" and path.exists():
        os.replace(path,path.with_name("previous.pt"))
    os.replace(tmp, path)


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def setup(config):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        min(.9, config["gpu_memory_GiB"] * (1 << 30) / torch.cuda.get_device_properties(0).total_memory)
    )
    return device


def hold_lock(path):
    stream = path.open("a")
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def gpu_lock(config):
    uuid = config["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must match gpu_uuid in the JSON")
    raw = subprocess.check_output([
        "nvidia-smi", "-i", uuid, "--query-gpu=memory.free", "--format=csv,noheader,nounits",
    ], text=True)
    if int(raw.strip()) < config["gpu_memory_GiB"] * 1024:
        raise RuntimeError("GPU free memory below the configured requirement")
    return hold_lock(ROOT / (uuid + ".lock"))


def checked_inputs(config):
    formal = config["mode"] == "formal"
    expected=config.get("plan_sha256")
    if expected and hashlib.sha256(Path(config["train"]["manifest"]).read_bytes()).hexdigest()!=expected:
        raise ValueError("Manifest binding mismatch")
    train = Dataset(config["train"], "train", formal)
    dev = Dataset(config["dev"], "dev", formal)
    if train.identities & dev.identities:
        raise ValueError("Train and Dev identities overlap")
    return train, dev, None


def metrics(pred, target, mask):
    x, y = pred[mask].astype(np.float64), target[mask].astype(np.float64)
    if not len(y):
        return None
    err = abs(x - y)
    active = abs(y) >= .5
    low = abs(y) <= .5
    vx = np.sum((x - x.mean()) ** 2)
    vy = np.sum((y - y.mean()) ** 2)
    corr = float(np.clip(np.sum((x-x.mean())*(y-y.mean())) / np.sqrt(vx*vy), -1, 1)) if vx > 0 and vy > 0 else None
    mape = float(np.mean(err[active] / abs(y[active]))) if active.any() else None
    return {
        "L_A": .5*(1-(corr or 0)) + .1*(mape or 0) + .15*float(err.mean()),
        "L_ref": 1-(corr or 0) + .3*(mape or 0),
        "MAE": float(err.mean()), "MAPE": None if mape is None else 100*mape,
        "Corr": corr, "R2": float(1 - np.sum((x-y)**2)/vy) if vy > 0 else None,
        "low_MAE": float(err[low].mean()) if low.any() else None,
        "low_FP1": float((x[low] > 1).mean()) if low.any() else None,
    }


def average(rows):
    keys = rows[0].keys()
    result = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def forward_chunk(arm, banks, role, samples, device, backward):
    x = torch.from_numpy(np.stack([item[0] for item in samples])).to(device)
    memory, index = banks.batch(role, samples)
    if backward:
        pred = checkpoint(arm, x, memory, index, use_reentrant=False)
    else:
        pred = arm(x, memory, index)
    y = torch.from_numpy(np.stack([item[1] for item in samples])).to(device)
    mask = torch.from_numpy(np.stack([item[2] for item in samples])).to(device)
    return pred, y, mask


@torch.no_grad()
def evaluate(arm, banks, data, config, device):
    # A group is one recording and hand. Metrics are computed on its complete
    # windows, then averaged by finger, stage and identity.
    from val_bands_r6 import ForceBands
    bands = ForceBands()
    trials = collections.defaultdict(list)
    windows = points = 0
    for group, (query, hand) in enumerate(data.groups):
        lo, hi = data.ends[group:group + 2]
        if config["mode"] == "smoke":
            hi = min(hi, lo + config["dev_windows_per_hand"])
        pieces = []
        for begin in range(lo, hi, config.get("val_chunk",config["physical_chunk"])):
            samples = [data.sample(i) for i in range(begin, min(hi, begin+config.get("val_chunk",config["physical_chunk"])))]
            pieces.append(tuple(x.cpu().numpy() for x in forward_chunk(
                arm, banks, data.role, samples, device, backward=False)))
        p, y, m = [np.concatenate(part) for part in zip(*pieces)]
        bands.add(query, hand, p, y, m)
        windows += len(p)
        points += int(m.sum())
        for finger in range(5):
            row = metrics(p[:, finger], y[:, finger], m[:, finger])
            if row is not None:
                trials[query["identity_group"], query["stage"], hand["hand"], finger].append(row)
    stages = collections.defaultdict(list)
    for (person, stage, _, _), rows in trials.items():
        stages[person, stage].append(average(rows))
    people = collections.defaultdict(list)
    for (person, stage), rows in stages.items():
        people[person].append(average(rows))
    per_person = [average(rows) for rows in people.values()]
    report = average(per_person)
    valid_r2 = [row["R2"] for row in per_person if row["R2"] is not None]
    report.update(windows=windows, points=points, people=len(people),
                  median_R2=float(np.median(valid_r2)) if valid_r2 else None)
    report['per_person']={p:average(rows) for p,rows in people.items()}
    report['per_stage']={str(st):average([average(rows) for (person,s),rows in stages.items() if s==st]) for st in sorted({s for _,s in stages})}
    primary={p:average([average(rows) for (person,st),rows in stages.items() if person==p and st not in (4,5)]) for p in people if any(person==p and st not in (4,5) for person,st in stages)}
    report['without_stage45']=average(list(primary.values())) if primary else None
    report["force_band_details"] = bands.finish()
    return report


def manual_stop(output):
    return any((root / name).exists()
               for root in (ROOT.parent, ROOT, output)
               for name in ("STOP_REQUEST", "PAUSED_BY_USER", "MAINTENANCE.json"))


def check_ram():
    pss_kib = next(int(s.split()[1]) for s in Path("/proc/self/smaps_rollup").read_text().splitlines()
                   if s.startswith("Pss:"))
    if pss_kib >= 48 * (1 << 20):
        raise RuntimeError("48 GiB self PSS limit")


def run(config, name, output, resume, stop_after):
    if config["mode"] not in ("formal", "smoke"):
        raise ValueError("Expected mode formal or smoke")
    formal = config["mode"] == "formal"
    if formal and (config["micro"], config["accumulation"], config["epochs"]) != (256, 4, 24):
        raise ValueError("Formal budget must use micro256 x4 for 24 epochs")
    if formal and stop_after is not None:
        raise ValueError("A fixed step stop is available in smoke only")
    if manual_stop(output):
        raise RuntimeError("Manual stop marker present")
    if formal:
        for marker in ("TARGET_READY.json","BANK_READY.json","QUALIFIED.json","RAW_BANK_READY_R6.json"):
            if not (ROOT/marker).exists():raise RuntimeError("Preparation/qualification incomplete: "+marker)
    train, dev, public = checked_inputs(config)
    device = setup(config)
    arm = Arm(name, device)
    groups = arm.optimizer_groups()
    peaks = [g["lr"] for g in groups]
    for group in groups:
        group["lr"] *= .5
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4, fused=False, foreach=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=2, cooldown=1, min_lr=[v/30 for v in peaks])
    from support_cache_raw_r6 import CachedBanks
    bank = CachedBanks(arm, train, dev, public, config["physical_chunk"])
    loader=ThreadPoolExecutor(max_workers=config.get("input_workers",16))
    prefetcher=ThreadPoolExecutor(max_workers=1)
    named = dict(arm.named_parameters())
    frozen = {k: p.detach().cpu().clone() for k, p in named.items() if not p.requires_grad}
    initial = {k: p.detach().cpu().clone() for k, p in named.items() if p.requires_grad}
    count = sum(p.numel() for p in named.values() if p.requires_grad)
    epoch = cursor = step = 0
    history, loss_history = [], []
    best, plan = None, None
    total = 752*1024 if formal else config["smoke_windows"]
    batch_size = config["micro"] * config["accumulation"]
    init_probe = {}
    last = output / "last.pt"
    if resume:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["arm"] != name or saved["windows"] != len(train):
            raise ValueError("Checkpoint belongs to a different configuration")
        arm.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["adam"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        epoch, cursor, step = saved["epoch"], saved["cursor"], saved["step"]
        history, loss_history, best, plan = (saved[k] for k in ("history", "losses", "best", "plan"))
        init_probe = saved["init_probe"]
    elif last.exists():
        raise FileExistsError("Existing run requires --resume")

    def save(destination=last):
        atomic(destination, dict(config=config, arm=name, windows=len(train), model=arm.state_dict(),
                          adam=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          rng=capture_rng(), epoch=epoch, cursor=cursor, step=step, history=history,
                          losses=loss_history, best=best, plan=plan, init_probe=init_probe), tensor=True)

    if not resume and not formal:
        raw = torch.from_numpy(train.sample(0)[0][None]).to(device)
        with torch.no_grad():
            sample = [train.sample(0)]
            mem, idx = bank.batch("train", sample)
            base = arm.physical(arm.base(arm.normalized(raw).transpose(1, 2).contiguous()))
            out = arm(raw, mem, idx)
            torch.testing.assert_close(base, out, atol=1e-7, rtol=1e-7)
            init_probe["initial_delta"] = float((base - out).abs().max())
    resume_start_step = step
    while epoch < config["epochs"]:
        if plan is None:
            if formal:
                from sampler import plan as make_plan
                folder=ROOT/'sample_plans';folder.mkdir(exist_ok=True)
                file=folder/f'epoch{epoch+1:02}.npy'
                if file.exists():plan=np.load(file,allow_pickle=False)
                else:
                    plan,stats=make_plan(train,epoch,total,config['seed'])
                    temp=file.with_suffix('.'+name+'.tmp')
                    with temp.open('wb') as f:np.save(f,plan,allow_pickle=False)
                    if file.exists() and not np.array_equal(np.load(file,allow_pickle=False),plan):raise ValueError('Shared sampler disagreement')
                    os.replace(temp,file);atomic(folder/f'epoch{epoch+1:02}.json',stats)
            else:plan=np.random.default_rng(config['seed']+epoch).integers(0,len(train),size=total,dtype=np.int64)
            if len(plan)!=total:raise ValueError('Wrong epoch draw budget')
            if epoch == 1:
                for group, peak in zip(optimizer.param_groups, peaks):
                    group["lr"] = peak
        pending=None
        while cursor < total:
            check_ram()
            if manual_stop(output):
                save()
                atomic(output / "PAUSED.json", {"step": step})
                return
            tick=time.monotonic()
            ids = plan[cursor:cursor+batch_size]
            loaded=list(loader.map(train.sample,ids)) if pending is None else pending.result()
            pending=None
            next_cursor=cursor+len(ids)
            if next_cursor<total:
                next_ids=plan[next_cursor:next_cursor+batch_size].copy()
                pending=prefetcher.submit(lambda values=next_ids:list(loader.map(train.sample,values)))
            input_seconds=time.monotonic()-tick
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.
            for begin in range(0, len(ids), config["micro"]):
                micro = ids[begin:begin+config["micro"]]
                parts = []
                for first in range(0, len(micro), config["physical_chunk"]):
                    selected = micro[first:first+config["physical_chunk"]]
                    samples = loaded[begin+first:begin+first+len(selected)]
                    parts.append(forward_chunk(arm, bank, "train", samples, device, backward=True))
                pred, target, valid = [torch.cat(values) for values in zip(*parts)]
                loss = force_objective(pred, target, weight=valid, **LOSS)["loss"] * (len(micro)/len(ids))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training objective")
                loss.backward()
                weighted_loss += float(loss.detach())
            grads = [p for p in named.values() if p.requires_grad]
            if any(p.grad is None for p in grads):
                raise RuntimeError("A trainable parameter did not participate in the forward")
            torch.nn.utils.clip_grad_norm_(grads, 1., error_if_nonfinite=True)
            optimizer.step()
            del parts,pred,target,valid,loss,samples,loaded
            optimizer.zero_grad(set_to_none=True)
            cursor += len(ids)
            step += 1
            if not formal:
                loss_history.append(weighted_loss)
            if (formal and step == resume_start_step + 1) or step % config["save_every"] == 0 or step == stop_after:
                save()
            if formal:
                progress = {
                    "arm": name, "epoch": epoch + 1, "step": step,"phase":"TRAIN",
                    "input_seconds":input_seconds,"update_seconds":time.monotonic()-tick,
                    "window_cursor": cursor, "epoch_windows": total,
                    "epoch_updates": math.ceil(total / batch_size),
                    "train_loss": weighted_loss, "time": time.time(),
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "input_workers":config.get("input_workers",16),
                    "ordered_window_ids_sha256":hashlib.sha256(ids.tobytes()).hexdigest(),
                }
                atomic(output / "PROGRESS.json", progress)
                with (output / "TIMING.jsonl").open("a") as timing:
                    timing.write(json.dumps(progress,separators=(",",":"))+"\n")
            if step == stop_after and cursor < total:
                return
        save()
        gc.collect();torch.cuda.empty_cache()
        atomic(output/'PHASE.json',{'phase':'VAL','step':step,'epoch':epoch+1,'time':time.time()})
        val_start=time.time()
        validation = evaluate(arm, bank, dev, config, device)
        details=validation.pop('force_band_details')
        detail_path=output/f'VAL_BANDS_EP{epoch+1:03}.json'
        atomic(detail_path,details)
        validation['force_bands']={k:v['summary'] for k,v in details['bands'].items()}
        validation['force_band_details']={'path':detail_path.name,'sha256':hashlib.sha256(detail_path.read_bytes()).hexdigest()}
        validation['seconds']=time.time()-val_start
        for key,before in frozen.items():
            if not torch.equal(before,named[key].detach().cpu()):raise ValueError('Frozen parameters changed: '+key)
        improved = best is None or validation["L_A"] < best["metrics"]["L_A"]
        if improved:
            best = {"epoch": epoch + 1, "step": step, "metrics": validation}
        history.append({
            "epoch": epoch+1, "step": step, "dev": validation, "best": best,
            "selection_metric": "L_A", "selection_direction": "min",
            "best_epoch_so_far": best["epoch"], "best_value_so_far": best["metrics"]["L_A"],
        })
        scheduler.step(validation["L_A"])
        epoch += 1
        cursor, plan = 0, None
        if improved:
            # Publish the best checkpoint before the cursor advances past this epoch.
            save(output / "best.pt")
        save()
        atomic(output / "HISTORY.json", history)
        if step == stop_after and epoch < config["epochs"]:
            return
        if epoch-best["epoch"]>=config.get("early_stop_patience",10):break
    for key, before in frozen.items():
        if not torch.equal(before, named[key].detach().cpu()):
            raise ValueError(f"Frozen parameter changed: {key}")
    if not formal:
        if len(optimizer.state) != sum(p.requires_grad for p in named.values()):
            raise ValueError("Adam state does not cover every trainable parameter")
        changed = {k for k, p in initial.items() if not torch.equal(p, named[k].detach().cpu())}
        if not changed:
            raise ValueError("No trainable parameter changed")
        init_probe["changed_parameter_tensors"] = len(changed)
        if name == "B_L":
            if sum(key.endswith(".up") for key in changed) != len(arm.lora_targets):
                raise ValueError("Not all LoRA target matrices updated")
            raw = torch.from_numpy(train.sample(0)[0][None]).to(device)
            init_probe["merged_max_error"] = arm.merge_error(raw)
    prefetcher.shutdown(wait=True)
    loader.shutdown(wait=True)
    atomic(output / "DONE.json", {
        "arm": name, "updates": step, "trainable_parameters": count,
        "frozen_parameters_unchanged": True, "best": best, "mode": config["mode"], **init_probe,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    config = read(args.config)
    if args.check_inputs:
        train, dev, _ = checked_inputs(config)
        print(json.dumps({"train_windows": len(train), "dev_windows": len(dev)}))
        return
    if not args.arm or not args.run or not args.run.replace("_", "").isalnum():
        parser.error("Provide --arm and --run using letters, numbers or underscores")
    output = ROOT / "runs" / args.run / args.arm
    output.mkdir(parents=True, exist_ok=True)
    with hold_lock(output / "run.lock"), gpu_lock(config):
        if (output / "DONE.json").exists():
            raise FileExistsError("This arm has already finished")
        proc = Path("/proc/self")
        atomic(output / "OWNER.json", {
            "pid": os.getpid(), "uid": os.getuid(), "time": time.time(),
            "start_ticks": int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19]),
            "command": (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
            "cwd": os.getcwd(),
        })
        try:
            run(config, args.arm, output, args.resume, args.stop_after)
        except BaseException:
            atomic(output / "FAILED.json", {"trace": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
