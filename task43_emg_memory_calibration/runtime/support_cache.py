from functools import lru_cache
import json, lzma, os, shutil, time
from pathlib import Path
import numpy as np
import torch
from arms import Banks
ROOT=Path(__file__).resolve().parent
FIELDS = (
    ((5, 128, 320), np.dtype("<f4")),
    ((5, 128, 2, 256), np.dtype("<f4")),
    ((5, 128, 4), np.dtype("<f4")),
    ((5, 128), np.dtype("?")),
    ((5, 4), np.dtype("<f4")),
)
SIZES = tuple(int(np.prod(shape)) * dtype.itemsize for shape, dtype in FIELDS)
RAW_BYTES = sum(SIZES)
DISK_RESERVE = 5 << 30
MAX_CACHE_BYTES = 8 << 30


def keys(data):
    return sorted({(data.role, query["support"], data.support_hand(query["support"]))
                   for query, _ in data.groups})


def bank_path(directory, item):
    role, rid, hand = item
    if role not in ("train", "dev") or hand not in ("left", "right") or Path(rid).name != rid or rid in ("", ".", ".."):
        raise ValueError(f"Invalid Stage3 support key: {item}")
    return directory / role / f"{rid}.{hand}.xz"


def pack(values):
    arrays = [tensor.detach().numpy() for tensor in values]
    if len(arrays) != len(FIELDS):
        raise ValueError("Unexpected number of static fields")
    for value, (shape, dtype) in zip(arrays, FIELDS):
        if value.shape != shape or value.dtype != dtype or not value.flags.c_contiguous:
            raise ValueError("Unexpected static bank layout")
    raw = b"".join(
        value.view(np.uint8).reshape(-1, dtype.itemsize).T.tobytes()
        for value, (_, dtype) in zip(arrays, FIELDS)
    )
    return lzma.compress(raw, filters=[{"id": lzma.FILTER_LZMA2, "preset": 0, "dict_size": 1 << 21}])


def unpack(payload):
    raw = bytearray(lzma.decompress(payload))
    if len(raw) != RAW_BYTES:
        raise ValueError("Truncated static bank")
    values, offset = [], 0
    for (shape, dtype), size in zip(FIELDS, SIZES):
        planes = np.frombuffer(raw, dtype=np.uint8, count=size, offset=offset).reshape(dtype.itemsize, -1)
        value = planes.T.copy().reshape(-1).view(dtype).reshape(shape)
        values.append(torch.from_numpy(value))
        offset += size
    return tuple(values)


def save_bank(directory, item, values):
    destination = bank_path(directory, item)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    payload = pack(values)
    before = shutil.disk_usage(directory).free
    if before - len(payload) < DISK_RESERVE:
        raise OSError("Insufficient experiment filesystem space for static bank")
    temporary = destination.with_suffix(".partial")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    for a, b in zip(values, unpack(temporary.read_bytes())):
        if a.view(torch.uint8).numpy().tobytes() != b.view(torch.uint8).numpy().tobytes():
            raise ValueError("Static bank changed during write/read")
    os.replace(temporary, destination)
    return len(payload)


class CachedBanks(Banks):
    directory=ROOT/'static_banks'
    @lru_cache(maxsize=256)
    def one(self,role,rid,side):
        return unpack(bank_path(self.directory,(role,rid,side)).read_bytes())

@torch.no_grad()
def build(train,dev,config,limit=0):
    from train import setup, atomic
    from arms import Arm
    device=setup(config);arm=Arm('F',device);banks=Banks(arm,train,dev,None,config['physical_chunk'])
    jobs=keys(train)+keys(dev)
    if limit:jobs=jobs[:limit]
    start=time.time();made=0;total=0
    for i,item in enumerate(jobs):
        path=bank_path(CachedBanks.directory,item)
        if path.exists():unpack(path.read_bytes())
        else:
            value=banks.one(*item);total+=save_bank(CachedBanks.directory,item,value);made+=1
        if (i+1)%20==0:atomic(ROOT/'BANK_PROGRESS.json',{'completed':i+1,'total':len(jobs),'new':made,'bytes':total,'seconds':time.time()-start})
    atomic(ROOT/('BANK_SMOKE_READY.json' if limit else 'BANK_READY.json'),{'completed':len(jobs),'new':made,'seconds':time.time()-start,'schema':'prefix60_128_slots_per_finger'})
