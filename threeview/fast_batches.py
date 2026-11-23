"""Same windows/augmentation; read immutable metadata once before worker launch."""
from collections import OrderedDict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
if __package__:
    from .batches import Batcher as Reference
    from . import io_local as data
else:
    from batches import Batcher as Reference
    import io_local as data
@lru_cache(maxsize=1)
def metadata():
    return data.closed_rows(),data.read(data.ROOT/'data/INDEX.json.gz'),data.read(data.ROOT/'data/CLOCK_BINDINGS.json')['rows']
class SharedReader(data.LocalReader):
    def __init__(self,max_bytes=512<<20):
        self.cache=OrderedDict();self.bytes=0;self.max_bytes=max_bytes;self.clocks=metadata()[2]
class Batcher(Reference):
    def __init__(self,role,norm,workers=8):
        assert role in ('train','val'),'Test closed'
        # Populate immutable label routing/LUT on the main thread, avoiding
        # duplicate JSON parses or lazy-module races in the eight I/O workers.
        if __package__:
            from .label_policy import bindings,mapping
        else:
            from label_policy import bindings,mapping
        bindings();mapping()
        rows,index,_=metadata();data.task43_inventory()
        self.rows=rows[role];self.role=role;self.norm=norm;self.index=data.Index(self.rows,None,prepared=index[role]);self.pool=ThreadPoolExecutor(max_workers=workers);self.local=threading.local()
    def _group(self,args):
        if not hasattr(self.local,'reader'):self.local.reader=SharedReader()
        return super()._group(args)
