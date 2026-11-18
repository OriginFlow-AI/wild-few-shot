"""Raw-only AP304 windows, reproducible pairs, and resumable train-only statistics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **values)
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest(), "little")


def load_spec(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if "extends" in cfg:
        from .config import _deep_merge
        parent = yaml.safe_load((path.parent / cfg["extends"]).read_text())
        cfg = _deep_merge(parent, cfg)
    if cfg["schema"] != "ap304_rank_pretrain_spec_v1":
        raise ValueError("unsupported AP304 ranking schema")
    root = Path(cfg.get("workspace", path.parents[2])).resolve()
    cfg["workspace"] = str(root)
    cfg["config_path"] = str(path)
    if cfg["experiment"]["initialization"] != "random" or not cfg["experiment"]["train_all_parameters"]:
        raise ValueError("this runner implements random-init, fully trainable AP304 only")
    ds = cfg["dataset"]
    if ds["stage_filter"] is not None or ds["train_fraction"] != 1.0 or ds["clean_supervision_root"]:
        raise ValueError("full raw data without cleaning is required")
    if sha(root / ds["index"]) != ds["index_sha256"]:
        raise ValueError("index hash mismatch")
    if cfg["windows"]["input_samples"] != 4000 or cfg["windows"]["native_target_first_sample"] != 470:
        raise ValueError("unsupported AP304 time alignment")
    return cfg


def load_index(cfg):
    rows = json.loads((Path(cfg["workspace"]) / cfg["dataset"]["index"]).read_text())
    subjects = [{r["subject"] for r in rows[s]} for s in ("train", "val", "test")]
    if any(subjects[i] & subjects[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("subject leakage across splits")
    trial_sets = [{r["trial_key"] for r in rows[s]} for s in ("train", "val", "test")]
    if any(trial_sets[i] & trial_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("trial leakage across splits")
    for s in ("train", "val", "test"):
        if len(trial_sets[("train", "val", "test").index(s)]) != len(rows[s]):
            raise ValueError("duplicate trial in " + s)
    return rows


def paths_for(row, cache):
    source = Path(row["path"])
    base = Path(cache) / source.parent.parent.name / source.parent.name / source.stem
    return Path(str(base) + ".emg.npy"), Path(str(base) + ".fz.npy")


def window_starts(length):
    if length < 4000:
        raise ValueError("raw trial shorter than AP304 window")
    starts = np.arange(0, length - 3999, 4000, dtype=np.int64)
    if starts[-1] != length - 4000:
        starts = np.append(starts, length - 4000)
    return starts


def open_arrays(row, cache):
    ep, fp = paths_for(row, cache)
    emg = np.load(ep, mmap_mode="r", allow_pickle=False)
    force = np.load(fp, mmap_mode="r", allow_pickle=False)
    n = int(row["n_samples"])
    if emg.shape != (n, 16) or force.shape != (n, 5) or emg.dtype != np.float32 or force.dtype != np.float32:
        raise ValueError(f"invalid raw arrays: {row['trial_key']}: {emg.shape}, {force.shape}")
    return emg, force


def rank_span(force):
    ref = np.asarray(force[470::40], dtype=np.float64)
    q = np.percentile(ref, [5, 95], axis=0)
    span = q[1] - q[0]
    eps = 1e-6 * np.abs(ref).max(axis=0)
    return span, eps


def _scan_trial(job):
    row, cache, fit_norm = job
    emg, force = open_arrays(row, cache)
    total = np.zeros(16, np.float64)
    squares = np.zeros(16, np.float64)
    for begin in range(0, len(emg), 65536):
        x = np.asarray(emg[begin:begin + 65536], np.float64)
        y = force[begin:begin + 65536]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("nonfinite raw data: " + row["trial_key"])
        if fit_norm:
            total += x.sum(0)
            squares += np.square(x).sum(0)
    span, eps = rank_span(force)
    ep, fp = paths_for(row, cache)
    stats = [ep.stat().st_size, ep.stat().st_mtime_ns, fp.stat().st_size, fp.stat().st_mtime_ns]
    return total, squares, len(emg), span, eps, stats


def prepare(cfg, workers=8):
    import fcntl
    out = Path(cfg["workspace"]) / cfg["artifacts"]["root"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete=out/'prepared/COMPLETE.json'
        splits=['train','val']+(['test'] if cfg['execution'].get('prepare_test_in_background') else [])
        if complete.exists() and all((out/f'prepared/{s}_stats.npz').exists() for s in splits):
            ready=json.loads(complete.read_text())
            if (ready['index_sha256']==cfg['dataset']['index_sha256'] and
                    ready['norm_sha256']==sha(Path(cfg['workspace'])/cfg['normalization']['output'])):
                mark_train_ready(cfg)
                print('RAW_PREPARATION_ALREADY_COMPLETE',flush=True)
                return
        _prepare(cfg, workers)


def mark_train_ready(cfg):
    """Full Train readiness is independent of Val/Test scanning progress."""
    root=Path(cfg['workspace']);out=root/cfg['artifacts']['root']
    prep=out/'prepared';stats=prep/'train_stats.npz';norm=root/cfg['normalization']['output']
    if not stats.exists() or not norm.exists():
        return False
    with np.load(stats,allow_pickle=False) as z:
        if (str(z['index_sha256'])!=cfg['dataset']['index_sha256'] or
                len(z['span'])!=cfg['dataset']['expected_train_trials']):
            raise ValueError('full Train span readiness mismatch')
    with np.load(norm,allow_pickle=False) as z:
        if (str(z['source_split_sha256'])!=cfg['dataset']['index_sha256'] or
                int(z['n_trials'])!=cfg['dataset']['expected_train_trials']):
            raise ValueError('normalization is not fitted on the complete Train split')
    ready=dict(index_sha256=cfg['dataset']['index_sha256'],norm_sha256=sha(norm),
               train_trials=cfg['dataset']['expected_train_trials'],normalization_scope='full_train_only')
    path=prep/'TRAIN_READY.json'
    if path.exists():
        if json.loads(path.read_text())!=ready:raise ValueError('Train readiness binding changed')
    else:
        atomic_json(path,ready)
    return True


def _prepare(cfg, workers=8):
    """Scan Train/Val once; completed 512-trial shards survive interruptions."""
    root = Path(cfg["workspace"])
    out = root / cfg["artifacts"]["root"]
    out.mkdir(parents=True, exist_ok=True)
    rows = load_index(cfg)
    prep = out / "prepared"
    prep.mkdir(exist_ok=True)
    combined = {}
    splits=['train','val']+(['test'] if cfg['execution'].get('prepare_test_in_background') else [])
    for split in splits:
        split_rows = rows[split]
        if len(split_rows) != cfg["dataset"].get("expected_" + split + "_trials",len(split_rows)):
            raise ValueError("unexpected trial count")
        parts = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for begin in range(0, len(split_rows), 512):
                stop = min(begin + 512, len(split_rows))
                shard = prep / split / f"{begin:06d}.npz"
                keys = np.asarray([r["trial_key"] for r in split_rows[begin:stop]])
                if shard.exists():
                    with np.load(shard, allow_pickle=False) as z:
                        part = {k: z[k].copy() for k in z.files}
                    if str(part["index_sha256"]) != cfg["dataset"]["index_sha256"] or not np.array_equal(keys, part["keys"]):
                        raise ValueError("preparation shard identity mismatch")
                else:
                    jobs = [(r, cfg["dataset"]["cache_root"], split == "train") for r in split_rows[begin:stop]]
                    scanned = list(pool.map(_scan_trial, jobs, chunksize=4))
                    part = dict(keys=keys, total=np.stack([x[0] for x in scanned]).sum(0),
                                squares=np.stack([x[1] for x in scanned]).sum(0),
                                frames=np.asarray(sum(x[2] for x in scanned), np.int64),
                                span=np.stack([x[3] for x in scanned]), eps=np.stack([x[4] for x in scanned]),
                                file_stats=np.asarray([x[5] for x in scanned], np.int64),
                                index_sha256=np.asarray(cfg["dataset"]["index_sha256"]))
                    atomic_npz(shard, **part)
                parts.append(part)
                progress=dict(phase="preparing_raw", split=split,completed_trials=stop,
                              total_trials=len(split_rows),workers=workers)
                atomic_json(out / "preparation_status.json",progress)
                if not (cfg['execution'].get('overlap_validation_preparation') and
                        (prep/'TRAIN_READY.json').exists()):
                    atomic_json(out / "status.json",progress)
                print(json.dumps(dict(phase="prepare", split=split, trials=stop, total=len(split_rows))), flush=True)
        combined[split] = dict(span=np.concatenate([p["span"] for p in parts]),
                               eps=np.concatenate([p["eps"] for p in parts]),
                               file_stats=np.concatenate([p["file_stats"] for p in parts]),
                               index_sha256=np.asarray(cfg["dataset"]["index_sha256"]))
        atomic_npz(prep / f"{split}_stats.npz", **combined[split])
        if split == "train":
            frames = sum(int(p["frames"]) for p in parts)
            mean = sum(p["total"] for p in parts) / frames
            variance = sum(p["squares"] for p in parts) / frames - mean ** 2
            std = np.sqrt(np.maximum(variance, 0))
            if not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("invalid full Train EMG normalization")
            norm_path = root / cfg["normalization"]["output"]
            atomic_npz(norm_path, emg_mean=mean.astype(np.float32), emg_std=std.astype(np.float32),
                       n_samples=np.asarray(frames), n_trials=np.asarray(len(split_rows)),
                       source_split_sha256=np.asarray(cfg["dataset"]["index_sha256"]), fit_role=np.asarray("train_only"))
            mark_train_ready(cfg)
    manifest = {}
    for s in ("train", "val", "test"):
        counts = [len(window_starts(r["n_samples"])) for r in rows[s]]
        manifest[s] = dict(trials=len(rows[s]), subjects=len({r["subject"] for r in rows[s]}),
                           windows=sum(counts), groups=sum((n + 1) // 2 for n in counts))
    atomic_json(out / "split_and_window_manifest.json", manifest)
    atomic_json(prep / "COMPLETE.json", dict(index_sha256=cfg["dataset"]["index_sha256"],
                norm_sha256=sha(root / cfg["normalization"]["output"]), manifest=manifest))
    print("RAW_PREPARATION_COMPLETE", flush=True)


def build_model(cfg):
    source = Path(cfg["workspace"]) / cfg["architecture"]["source"]
    if sha(source) != cfg["architecture"]["source_sha256"]:
        raise ValueError("AP304 model source changed")
    spec = importlib.util.spec_from_file_location("ap304_rank_model_source", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    net = module.K2RingNetwork(module.K2RingConfig())
    n = sum(p.numel() for p in net.parameters())
    if n != cfg["architecture"]["expected_parameters"] or any(not p.requires_grad for p in net.parameters()):
        raise ValueError("AP304 architecture/freeze mismatch")
    return net


def make_plan(rows, seed, epoch, world=8, micro=4, accumulate=4):
    n = sum((len(window_starts(r["n_samples"])) + 1) // 2 for r in rows)
    batch = world * micro * accumulate
    plan = np.full((math.ceil(n / batch) * batch, 3), -1, dtype=np.int32)
    pos = 0
    for trial, row in enumerate(rows):
        size = len(window_starts(row["n_samples"]))
        order = np.random.default_rng(seed_for(seed, epoch, row["trial_key"], "windows")).permutation(size)
        if size % 2:
            order = np.append(order, -1)
        groups = order.reshape(-1, 2)
        plan[pos:pos + len(groups), 0] = trial
        plan[pos:pos + len(groups), 1:] = groups
        pos += len(groups)
    np.random.default_rng(seed_for(seed, epoch, "group_shuffle")).shuffle(plan)
    return plan


_TRIU = np.triu_indices(89, 1)


def make_pairs(y, starts, span, eps, rng, cfg):
    """Five fingers, six within-window lag bins and one cross-window component."""
    ii = np.zeros((5, 7, 128), np.int64)
    jj = np.zeros_like(ii)
    ss = np.zeros(ii.shape, np.float32)
    active = span > eps
    threshold = np.maximum(cfg["pair_supervision"]["min_gap"]["fraction_of_reference_span"] * span, eps)
    def select(finger, component, a, b, diff, ok, quota):
        eligible = np.flatnonzero(ok & (np.abs(diff) > threshold[finger]))
        if not len(eligible) or not active[finger]:
            return
        pick = rng.choice(eligible, size=min(quota, len(eligible)), replace=False)
        n = len(pick)
        ii[finger, component, :n] = a[pick]
        jj[finger, component, :n] = b[pick]
        ss[finger, component, :n] = np.sign(diff[pick])
    a, b = _TRIU
    lag = b - a
    within = cfg["pair_supervision"]["within_window"]
    for w in range(2):
        if starts[w] < 0:
            continue
        for f in range(5):
            diff = y[w, f, a] - y[w, f, b]
            for k, ((lo, hi), quota) in enumerate(zip(within["lag_token_bins"], within["pair_quotas"])):
                select(f, w * 3 + k, a + w * 89, b + w * 89, diff, (lag >= lo) & (lag <= hi), quota)
    if min(starts) >= 0:
        a, b = np.indices((89, 89)).reshape(2, -1)
        gap = np.abs(starts[0] + 470 + 40 * a - starts[1] - 470 - 40 * b)
        for f in range(5):
            select(f, 6, a, 89 + b, y[0, f, a] - y[1, f, b],
                   gap >= cfg["pair_supervision"]["across_windows"]["min_absolute_time_gap_samples"],
                   cfg["pair_supervision"]["across_windows"]["max_pairs_per_finger_per_group"])
    return dict(pair_i=ii, pair_j=jj, pair_sign=ss)


class RawReader:
    def __init__(self, cfg, rows, split):
        self.cfg, self.rows, self.split = cfg, rows, split
        self.arrays = OrderedDict()
        root = Path(cfg["workspace"])
        with np.load(root / cfg["normalization"]["output"], allow_pickle=False) as z:
            if str(z["source_split_sha256"]) != cfg["dataset"]["index_sha256"]:
                raise ValueError("normalization provenance mismatch")
            self.mean, self.std = z["emg_mean"].copy(), z["emg_std"].copy()
        stats = root / cfg["artifacts"]["root"] / "prepared" / f"{split}_stats.npz"
        self.stats = None
        if stats.exists():
            with np.load(stats, allow_pickle=False) as z:
                if str(z["index_sha256"]) != cfg["dataset"]["index_sha256"]:
                    raise ValueError("span provenance mismatch")
                self.stats = {k: z[k].copy() for k in ("span", "eps", "file_stats")}

    def open(self, trial):
        if trial in self.arrays:
            self.arrays.move_to_end(trial)
            return self.arrays[trial]
        row = self.rows[trial]
        emg, force = open_arrays(row, self.cfg["dataset"]["cache_root"])
        if self.stats is not None:
            ep, fp = paths_for(row, self.cfg["dataset"]["cache_root"])
            actual = [ep.stat().st_size, ep.stat().st_mtime_ns, fp.stat().st_size, fp.stat().st_mtime_ns]
            if not np.array_equal(actual, self.stats["file_stats"][trial]):
                raise ValueError("raw file changed after scan: " + row["trial_key"])
            span, eps = self.stats["span"][trial], self.stats["eps"][trial]
        else:
            if not np.isfinite(emg).all() or not np.isfinite(force).all():
                raise ValueError("nonfinite raw trial: " + row["trial_key"])
            span, eps = rank_span(force)
        self.arrays[trial] = (emg, force, span, eps)
        while len(self.arrays) > self.cfg["execution"]["max_open_trials_per_worker"]:
            self.arrays.popitem(last=False)
        return self.arrays[trial]

    def group(self, trial, window_ids, epoch):
        x = np.zeros((2, 16, 4000), np.float32)
        y = np.zeros((2, 5, 89), np.float32)
        starts = np.full(2, -1, np.int64)
        span = np.zeros(5)
        eps = np.zeros(5)
        key = "dummy"
        if trial >= 0:
            row = self.rows[trial]
            key = row["trial_key"]
            emg, force, span, eps = self.open(trial)
            all_starts = window_starts(len(emg))
            for w, wid in enumerate(window_ids):
                if wid < 0:
                    continue
                start = int(all_starts[wid])
                starts[w] = start
                x[w] = ((emg[start:start + 4000] - self.mean) / self.std).T
                y[w] = force[start + 470:start + 3991:40].T
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("nonfinite group: " + key)
        rng = np.random.default_rng(seed_for(self.cfg["experiment"]["seed"], epoch, key, *starts, "pairs"))
        pairs = make_pairs(y, starts, span, eps, rng, self.cfg)
        return dict(x=x, y=y, starts=starts, **pairs)


class GroupDataset(Dataset):
    def __init__(self, cfg, rows, plan_path, epoch, rank, world, start_update=0):
        self.cfg, self.rows, self.plan_path = cfg, rows, str(plan_path)
        self.epoch, self.rank, self.world = epoch, rank, world
        self.micro = cfg["execution"]["groups_per_device_microbatch"]
        self.accumulate = cfg["execution"]["accumulate_microbatches"]
        self.start_update = start_update
        self.reader = None
        self.plan = None
        plan = np.load(plan_path, mmap_mode="r", allow_pickle=False)
        self.size = len(plan) // world - start_update * self.accumulate * self.micro

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.reader is None:
            self.reader = RawReader(self.cfg, self.rows, "train")
            self.plan = np.load(self.plan_path, mmap_mode="r", allow_pickle=False)
        i += self.start_update * self.accumulate * self.micro
        step, rem = divmod(i, self.accumulate * self.micro)
        accum, offset = divmod(rem, self.micro)
        global_i = step * self.accumulate * self.world * self.micro + accum * self.world * self.micro + self.rank * self.micro + offset
        trial, a, b = map(int, self.plan[global_i])
        return self.reader.group(trial, (a, b), self.epoch)


class TrialDataset(Dataset):
    """One trial per item; evaluator can stitch chronological native outputs."""
    def __init__(self, cfg, rows, split, rank, world, limit=None):
        self.cfg, self.rows, self.split = cfg, rows, split
        self.ids = list(range(len(rows)))[rank::world]
        if limit is not None:
            self.ids = self.ids[:limit]
        self.reader = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.reader is None:
            self.reader = RawReader(self.cfg, self.rows, self.split)
        trial = self.ids[i]
        row = self.rows[trial]
        n = len(window_starts(row["n_samples"]))
        order = np.random.default_rng(seed_for(self.cfg["experiment"]["seed"], 0, row["trial_key"], "windows")).permutation(n)
        if n % 2:
            order = np.append(order, -1)
        groups = [self.reader.group(trial, pair, 0) for pair in order.reshape(-1, 2)]
        return dict(trial=trial, groups={key: np.stack([g[key] for g in groups]) for key in groups[0]})
