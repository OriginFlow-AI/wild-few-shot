"""Bounded S3 range reads and dynamic protobuf decoding; credentials stay outside Git."""
import io
from collections import OrderedDict
import numpy as np
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from google.protobuf import descriptor_pool, message_factory


class RangeReader(io.BufferedIOBase):
    def __init__(self, client, row, bucket):
        self.client, self.row, self.bucket = client, row, bucket
        self.pos, self.size, self.bytes_read = 0, row['size'], 0
        self.cache = OrderedDict()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError('Negative seek')
        return self.pos

    def _range(self, lo, hi):
        response = self.client.get_object(Bucket=self.bucket, Key=self.row['key'],
                                         Range=f'bytes={lo}-{hi-1}', IfMatch=self.row['etag'])
        body = response['Body']
        try: data = body.read()
        finally: body.close()
        if len(data) != hi - lo: raise OSError('Short ranged read')
        self.bytes_read += len(data)
        return data

    def read(self, size=-1):
        end = self.size if size is None or size < 0 else min(self.size, self.pos + size)
        if end <= self.pos: return b''
        if end - self.pos > 64 << 20: raise ValueError('Ranged read exceeds 64 MiB')
        if end - self.pos > 131072:
            data = self._range(self.pos, end); self.pos = end; return data
        parts = []
        while self.pos < end:
            start = self.pos // 65536 * 65536
            if start not in self.cache:
                self.cache[start] = self._range(start, min(start + 65536, self.size))
                if len(self.cache) > 8: self.cache.popitem(last=False)
            block = self.cache[start]
            count = min(end - self.pos, len(block) - (self.pos - start))
            parts.append(block[self.pos-start:self.pos-start+count]); self.pos += count
        return b''.join(parts)

    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data; return len(data)


def decoder(schema):
    descriptors = FileDescriptorSet(); descriptors.ParseFromString(schema.data)
    pool = descriptor_pool.DescriptorPool(); pending = list(descriptors.file)
    while pending:
        later = []
        for item in pending:
            try: pool.AddSerializedFile(item.SerializeToString())
            except (TypeError, ValueError): later.append(item)
        if len(later) == len(pending): raise ValueError('Unresolved protobuf dependency')
        pending = later
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name))


def unique(t, *arrays):
    t = np.asarray(t, np.int64); order = np.argsort(t, kind='stable'); sorted_t = t[order]
    keep = np.r_[True, np.diff(sorted_t) > 0] if len(t) else np.empty(0, bool)
    return (sorted_t[keep], *[np.asarray(a)[order][keep] for a in arrays])
