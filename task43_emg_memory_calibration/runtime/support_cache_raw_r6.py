"""Lossless, uncompressed static banks; trainable K/V remain dynamic."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import torch
from arms import Banks
from support_cache import FIELDS, SIZES, bank_path

ROOT=Path(__file__).resolve().parent
DIRECTORY=ROOT/'static_banks_raw_r6'
RAW_BYTES=sum(SIZES)


def path_for(role,rid,side):
    # Reuse strict role/rid/hand validation, with an independent destination.
    return bank_path(DIRECTORY,(role,rid,side)).with_suffix('.raw')


def unpack_raw(raw):
    if len(raw)!=RAW_BYTES:raise ValueError('Truncated raw static bank')
    result=[];offset=0
    for (shape,dtype),size in zip(FIELDS,SIZES):
        value=np.frombuffer(raw,dtype=dtype,count=size//dtype.itemsize,offset=offset).reshape(shape).copy()
        result.append(torch.from_numpy(value));offset+=size
    return tuple(result)


class CachedBanks(Banks):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        ready=json.loads((ROOT/'RAW_BANK_READY_R6.json').read_text())
        plan=json.loads((ROOT/'PLAN.json').read_bytes())
        expected=sum(len({(q['support'],c['supports'][q['support']]['hand']) for q in c['queries']}) for c in plan['roles'].values())
        if ready['status']!='PASS' or ready['count']!=expected or ready['raw_bytes_per_bank']!=RAW_BYTES:
            raise ValueError('Unqualified raw bank cache')

    @lru_cache(maxsize=256)
    def one(self,role,rid,side):
        return unpack_raw(path_for(role,rid,side).read_bytes())
