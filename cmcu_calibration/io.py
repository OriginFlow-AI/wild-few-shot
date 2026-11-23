"""Load local native NPZ or MCAP, without credentials or cloud-specific paths."""

from pathlib import Path
import numpy as np

TOPICS = ("/bsq/middle/force", "/cmcu05b/left/cmcu05b", "/cmcu05b/right/cmcu05b")


def nearest(clock, query):
    high = np.searchsorted(clock, query).clip(0, len(clock) - 1)
    low = (high - 1).clip(0)
    index = np.where(abs(clock[low] - query) <= abs(clock[high] - query), low, high)
    return index, abs(clock[index] - query)


def decode_class(schema):
    from google.protobuf.descriptor_pb2 import FileDescriptorSet
    from google.protobuf import descriptor_pool, message_factory

    descriptors = FileDescriptorSet()
    descriptors.ParseFromString(schema.data)
    pool = descriptor_pool.DescriptorPool()
    pending = list(descriptors.file)
    while pending:
        unresolved = []
        for item in pending:
            try:
                pool.AddSerializedFile(item.SerializeToString())
            except (TypeError, ValueError):
                unresolved.append(item)
        if len(unresolved) == len(pending):
            raise ValueError("Unresolved protobuf schema dependencies")
        pending = unresolved
    descriptor = pool.FindMessageTypeByName(schema.name)
    if hasattr(message_factory, "GetMessageClass"):
        return message_factory.GetMessageClass(descriptor)
    return message_factory.MessageFactory(pool).GetPrototype(descriptor)


def load_mcap(path):
    from mcap.reader import make_reader

    groups = {topic: [] for topic in TOPICS}
    classes = {}
    with path.open("rb") as stream:
        for schema, channel, message in make_reader(stream).iter_messages(
            topics=TOPICS, log_time_order=False
        ):
            if schema is None:
                raise ValueError("Force channel is missing its protobuf schema")
            if schema.id not in classes:
                classes[schema.id] = decode_class(schema)
            frame = classes[schema.id].FromString(message.data)
            tick = int(frame.sync_ts_ns or message.log_time)
            received = int(frame.recv_ts_ns)
            if channel.topic == TOPICS[0]:
                values = (float(frame.force_n),)
            else:
                if not hasattr(frame, "ad_values") or len(frame.ad_values) < 5:
                    raise ValueError("CMCU schema or message has no five-channel AD")
                if len(frame.pressures) < 5:
                    raise ValueError("CMCU message has fewer than five force channels")
                values = (
                    list(frame.ad_values[:5]),
                    np.asarray(frame.pressures[:5], dtype=float) * 0.001,
                )
            groups[channel.topic].append((tick, received, *values))
    arrays, offsets = {}, []
    for side, topic in zip(("bsq", "left", "right"), TOPICS):
        rows = groups[topic]
        if not rows:
            continue
        columns = list(zip(*rows))
        clock = np.asarray(columns[0], dtype=np.int64)
        order = np.argsort(clock, kind="stable")
        clock = clock[order]
        unique = np.r_[True, np.diff(clock) > 0]
        arrays[side + "_t"] = clock[unique]
        received = np.asarray(columns[1], dtype=np.int64)[order][unique]
        offsets.append(float(np.median(received - clock[unique])))
        if side == "bsq":
            arrays["bsq_force"] = np.asarray(columns[2], dtype=float)[order][unique]
        else:
            arrays[side + "_ad"] = np.asarray(columns[2], dtype=float)[order][unique]
            arrays[side + "_force"] = np.asarray(columns[3], dtype=float)[order][unique]
    arrays["clock_ok"] = bool(offsets) and max(offsets) - min(offsets) <= 50_000_000
    return arrays


def load_native(path, needs_reference=False):
    path = Path(path)
    if path.suffix.lower() == ".mcap":
        arrays = load_mcap(path)
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
    if not bool(arrays.get("clock_ok", True)):
        raise ValueError("Force stream clock offset check failed")
    for side in ("bsq", "left", "right"):
        if side == "bsq" and "bsq_t" not in arrays and not needs_reference:
            continue
        clock = np.asarray(arrays[side + "_t"])
        if clock.ndim != 1 or len(clock) < 2 or clock.dtype.kind not in "iu":
            raise ValueError(side + " timestamps must be integer nanoseconds")
        if np.any(np.diff(clock) <= 0):
            raise ValueError(side + " timestamps must increase strictly")
        arrays[side + "_t"] = clock.astype(np.int64)
        fields = ("force",) if side == "bsq" else ("ad", "force")
        for field in fields:
            key = side + "_" + field
            shape = (len(clock),) if side == "bsq" else (len(clock), 5)
            value = np.asarray(arrays[key], dtype=float)
            if side != "bsq" and value.ndim == 2 and value.shape[1] >= 5:
                value = value[:, :5]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(key + " has an invalid shape or nonfinite values")
            arrays[key] = value
    return arrays


def pair_native(arrays):
    clock = arrays["bsq_t"]
    aligned = [nearest(arrays[side + "_t"], clock) for side in ("left", "right")]
    return dict(
        sync_ts_ns=clock,
        bsq_force_N=arrays["bsq_force"],
        cmcu_ad=np.column_stack(
            [
                arrays[side + "_ad"][index]
                for side, (index, _) in zip(("left", "right"), aligned)
            ]
        ),
        cmcu_force_N=np.column_stack(
            [
                arrays[side + "_force"][index]
                for side, (index, _) in zip(("left", "right"), aligned)
            ]
        ),
        nearest_delta_ns=np.column_stack([delta for _, delta in aligned]),
    )
