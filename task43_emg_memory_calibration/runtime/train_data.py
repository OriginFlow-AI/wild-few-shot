"""Read the frozen Stage3 cohort from the existing read-only Force cache."""
import bisect
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from calibration.calibration_tools import Calibration

ROOT = Path(__file__).resolve().parent
POSITIONS = 470 + 40 * np.arange(89, dtype=np.int64)
HISTORY = np.arange(-255, 1, dtype=np.int64)
SUPPORT_WINDOWS = 30


def read(path):
    return json.loads(Path(path).read_bytes())


@lru_cache(maxsize=2)
def load_plan(path):
    plan = read(path)
    if plan["schema"] != "stage3-prefix60-crossstage-v1":
        raise ValueError("Invalid or incomplete cohort plan")
    return plan


class Dataset:
    def __init__(self, spec, role, formal):
        plan = load_plan(str(Path(spec["manifest"]).resolve()))
        source_role = "val" if role == "dev" else role
        if not formal and spec.get("role", source_role) != source_role:
            raise ValueError("Smoke manifest role mismatch")
        cohort = plan["roles"][source_role]
        self.role = role
        self.cache = ROOT / "data"
        self.records = cohort["records"]
        self.supports = cohort["supports"]
        self.calibration = Calibration(ROOT / "calibration")
        self.groups, self.ends = [], [0]
        selected = set(spec.get("smoke_query_keys", [])) if not formal else set()
        for row in cohort["queries"]:
            if selected and row["query"] not in selected:
                continue
            q = self.records[row["query"]]
            if q["stage"] == 3 or (source_role == "train" and q["stage"] in (4,5)):
                raise ValueError("Support-only or forbidden Train stage entered queries")
            query = dict(rid=row["query"], support=row["support"],
                         identity_group=__import__("identity_rule").identity(q["subject"]), stage=q["stage"])
            hand = dict(hand=q["hand"], start=row["start"], windows=row["windows"])
            self.groups.append((query, hand))
            self.ends.append(self.ends[-1] + row["windows"])
        if not self.groups or (selected and len(selected) != len(self.groups)):
            raise ValueError(f"Empty or incomplete selected input: {source_role}")
        if formal and len(self) != plan["summary"][source_role]["windows"]:
            raise ValueError("Frozen window population changed")
        self.identities = {q["identity_group"] for q, _ in self.groups}

    def __len__(self):
        return self.ends[-1]

    @lru_cache(maxsize=1024)
    def target(self,key):
        with np.load(self.targets_path(key),allow_pickle=False) as z:return z["target"]

    def targets_path(self,key):
        return ROOT/'targets'/('val' if self.role=='dev' else self.role)/(key+'.npz')

    def emg(self, key, hand, origin, start):
        if hand != self.records[key]["hand"] or origin != 0:
            raise ValueError("Wrong cache hand or sample origin")
        x = __import__("source_reader").rows(self.records[key], "emg", start, start+4000)
        if x.shape != (4000, 16) or not np.isfinite(x).all():
            raise ValueError(f"Invalid EMG window: {key}")
        return x

    def force(self, key, indices):
        lo,hi=int(indices.min()),int(indices.max())+1
        raw = __import__("source_reader").rows(self.records[key],"fz",lo,hi)[indices-lo]
        return self.calibration.forward(raw, layout=self.records[key]["layout"])

    def sample(self, index):
        group = bisect.bisect_right(self.ends, int(index)) - 1
        local = int(index) - self.ends[group]
        query, hand = self.groups[group]
        start = hand["start"] + 4000 * local
        key, side = query["rid"], hand["hand"]
        x = self.emg(key, side, 0, start)
        target=self.target(key)[local].T.copy()
        return x, target, np.ones_like(target, dtype=bool), query["support"], side

    def support_hand(self, rid):
        return self.supports[rid]["hand"]

    @lru_cache(maxsize=256)
    def support(self, rid):
        entry = self.supports[rid]
        key, hand = entry["record"], entry["hand"]
        __import__("source_reader").ensure_prefix(self.records[key])
        starts = 4000 * np.arange(entry["stop"] // 4000, dtype=np.int64)
        end = starts[:, None] + POSITIONS
        force = self.force(key, end)
        bins = np.digitize(abs(force), [.5, 5.])
        diversity = sum((bins == j).any(axis=1).sum(axis=1) for j in range(3))
        chosen = []
        # One window per equal-time segment, prioritizing force-band coverage.
        for group in np.array_split(np.arange(len(starts)), min(SUPPORT_WINDOWS, len(starts))):
            chosen.append(max(group, key=lambda i: (int(diversity[i]), -abs(i-group.mean()), -i)))
        end = end[chosen].ravel()
        bands = bins[chosen].reshape(-1, 5)
        fingers = [dict(admitted=False, slots=[]) for _ in range(10)]
        offset = 5 * (hand == "right")
        for finger in range(5):
            # One endpoint in each of the 60 one-second bins first; then
            # add coverage across low/mid/high labels from support alone.
            slots = []
            seconds=end//2000
            for second in range(60):
                ix=np.flatnonzero(seconds==second)
                if len(ix): slots.append(int(ix[len(ix)//2]))
            for band in range(3):
                ix=np.setdiff1d(np.flatnonzero(bands[:,finger]==band),slots)
                if len(ix):slots.extend(ix[np.linspace(0,len(ix)-1,min(22,len(ix))).astype(int)].tolist())
            remaining = np.setdiff1d(np.arange(len(end)), slots)
            count = min(128-len(slots), len(remaining))
            if count:
                slots.extend(remaining[np.linspace(0, len(remaining)-1, count).astype(int)])
            fingers[offset+finger] = dict(admitted=bool(slots), slots=end[sorted(slots)].tolist())
        unique = sorted({t for finger in fingers for t in finger["slots"]})
        histories = self.force(key, np.asarray(unique)[:, None] + HISTORY)
        return {"fingers": fingers, "histories": dict(zip(unique, histories))}

    def slot(self, rid, global_finger, slot_index):
        source = self.supports[rid]
        key, side = source["record"], source["hand"]
        bank = self.support(rid)
        endpoint = bank["fingers"][global_finger]["slots"][slot_index]
        start, offset = divmod(endpoint, 4000)
        token = (offset-470)//40
        if not (0 <= token < 89 and POSITIONS[token] == offset and endpoint < source["stop"]):
            raise ValueError("Support token or prefix boundary mismatch")
        force = bank["histories"][endpoint][:, global_finger % 5]
        return (key, side, 0, start*4000), token, force
