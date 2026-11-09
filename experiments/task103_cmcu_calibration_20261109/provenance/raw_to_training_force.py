#!/usr/bin/env python3
"""原始力 NPZ/MCAP → 力训练 NPZ 处理器(processor,EMG 处理与 pose 对齐)。

把上游原始力录制转换为 emg2force 训练格式。EMG 处理链**逐位复用** pose 的
raw_to_training:重采样到 2kHz → 带通 40–850Hz + 50Hz 及谐波陷波(读 ``emg_raw``,
**只滤一次**)。旧 NPZ 的 ``force`` 仅兼容透传；MCAP CMCU05B 以
``cmcu05b_<hand>_force_mn`` 为权威标签，按独立 sync timestamp 插值到 EMG 时间轴，
不合成已弃用的通用 ``force``。

与 pose(raw_to_training.convert_one)的异同:
    - EMG 链相同:``_to_session_relative`` + 同一重采样分支 + ``_apply_semg_filter``(复用其函数,
      非复制),滤波器(SOS 陷波谐波 + 40–850 带通 + steady-state 初始化)完全一致。
    - 标签不同:pose 是 manus 插值 + 15Hz 平滑 + deg→rad + 裁剪重叠；力按同轴
      或独立时间戳重采样，**不平滑**(力动态与关节角不同,15Hz 低通不适用)。
    - 输出 ``emg`` 和实际存在的传感器标签；CMCU05B 输出
      ``cmcu05b_<hand>_force_mn [T,9]``。

处理步骤:
    1. 读 ``emg_raw``(真原始,未滤波;缺失退回 ``emg`` 并标注)
    2. EMG 重采样到统一 2kHz(应对不规则 / 低采样率录制)
    3. EMG 带通 40–850Hz + 50Hz 及谐波陷波(只滤一次)
    4. 各传感器标签优先按独立时间戳重采样；无时间戳且与 EMG 等长才视为同轴
    5. 裁剪到 EMG 与全部标签的共同覆盖区间，不对未采集区间做端点填充
    6. 输出 EMG + 传感器原名标签；不构造不存在的 ``force``

CLI:
    python -m src.data_prep.processors.raw_to_training_force \\
        --input-dir <raw_root> --output-dir <out> [--n-jobs 8]
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from tqdm import tqdm

from src.data_prep.processors.raw_recording import (
    available_force_recording_sides,
    is_raw_recording_path,
    load_force_recording,
    side_output_path,
)

# EMG 处理链与 pose 完全一致:直接复用 raw_to_training 的滤波器与时基归一(不复制,保证 byte 对齐)
from src.data_prep.processors.raw_to_training import (
    TARGET_FS,
    _apply_semg_filter,
    _to_session_relative,
)
from src.data_prep.processors.timestamps import (
    select_emg_timestamps,
    validate_timestamps,
)

_FORCE_TS_CANDIDATES = ("force_timestamps", "force_timestamps_raw")

# gen3 字段命名 + 列布局([Fx,Fy,Fz] × 5 指, Fz 列 = 2,5,8,11,14)
_GEN3_FIELD_CANDIDATES = (
    "gen3_resultant",
    "gen3_right_resultant",
    "gen3_left_resultant",
)
_GEN3_TS_FIELDS = {
    "gen3_resultant": "gen3_timestamps_raw",
    "gen3_right_resultant": "gen3_right_timestamps_raw",
    "gen3_left_resultant": "gen3_left_timestamps_raw",
}
_GEN3_FZ_COLS = (2, 5, 8, 11, 14)  # Fz of thumb / index / middle / ring / pinky

# cmcu05 / cmcu05b 字段命名: cmcu05[b]_{finger,left,right}_force_mn
# 1D per-finger 或 1D/2D per-hand；低频流按自己的 timestamps_raw 重采样
_CMCU_FIELD_RE = re.compile(
    r"^cmcu05b?_(?:thumb|index|middle|ring|pinky|left|right)_force_mn$"
)


def _resample_to_target(
    arr: np.ndarray, n_out: int, duration: float, actual_fs: float
) -> np.ndarray:
    """把 emg-rate 数组重采样到 2kHz 时间轴(分支与 raw_to_training 的 EMG 重采样逐字一致)。

    actual_fs 已接近 2kHz(|Δ|≤50)→ 截断/末值填充到 n_out(不插值);否则线性插值。
    EMG 与 force 共用本函数 + 同一 (n_out, duration, actual_fs),故重采样后仍逐样本对齐。
    """
    if abs(actual_fs - TARGET_FS) > 50:
        src_ts = np.linspace(0.0, duration, len(arr))
        dst_ts = np.linspace(0.0, duration, n_out)
        return interp1d(src_ts, arr, axis=0, kind="linear")(dst_ts).astype(np.float32)
    out = arr[:n_out].astype(np.float32)
    if len(out) < n_out:
        pad = np.repeat(out[-1:], n_out - len(out), axis=0)
        out = np.concatenate([out, pad], axis=0)
    return out


def _interpolate_stream(
    arr: np.ndarray,
    src_ts: np.ndarray,
    dst_ts: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    return interp1d(
        src_ts,
        arr,
        axis=0,
        kind="linear",
        bounds_error=True,
    )(dst_ts).astype(np.float32)


def _normalize_timed_stream(
    arr: np.ndarray, src_ts: np.ndarray, *, label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one independent stream and deterministically remove duplicates."""
    if arr.ndim < 1:
        raise ValueError(f"{label} data must have a sample dimension")
    if len(src_ts) != len(arr):
        raise ValueError(f"{label} timestamp len={len(src_ts)} != data len={len(arr)}")
    timestamps = validate_timestamps(
        src_ts,
        label=label,
        allow_duplicates=True,
    )
    keep = np.concatenate(([True], np.diff(timestamps) > 0))
    timestamps = timestamps[keep]
    arr = arr[keep]
    if len(timestamps) < 2:
        raise ValueError(f"{label} has too few unique timestamps ({len(timestamps)})")
    return arr, timestamps


def _prepare_stream(
    arr: np.ndarray,
    data,
    *,
    label: str,
    timestamp_candidates: tuple[str, ...],
    emg_len: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resolve an explicit timeline, falling back to sample alignment only if absent."""
    timestamp_key = next(
        (key for key in timestamp_candidates if key in data.files), None
    )
    if timestamp_key is not None:
        timestamps = _to_session_relative(
            np.asarray(data[timestamp_key], dtype=np.float64), data
        )
        return _normalize_timed_stream(arr, timestamps, label=timestamp_key)
    if len(arr) != emg_len:
        raise ValueError(
            f"{label} len={len(arr)} != emg len={emg_len}, but no timestamps "
            f"{timestamp_candidates}"
        )
    return arr, None


def preprocess_emg_stream(
    data,
    *,
    label: str,
    require_raw: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run the canonical force-training EMG preprocessing chain.

    Returns ``(emg_filtered, output_grid, source_grid, metadata)``.  The public
    helper keeps online fine-tuning and the offline converter on one
    implementation instead of duplicating the resampling/filter coefficients.
    ``emg_raw`` recordings are already stored in physical units, so no ADC
    scaling is applied here.
    """
    if "emg_raw" in data.files:
        source_key = "emg_raw"
    elif require_raw:
        raise KeyError(f"{label}: missing emg_raw required by canonical preprocessing")
    elif "emg" in data.files:
        source_key = "emg"
    else:
        raise KeyError(f"{label}: missing emg_raw/emg")

    emg = np.asarray(data[source_key], dtype=np.float32)
    if emg.ndim != 2 or emg.shape[1] != 16:
        raise ValueError(f"{label}: {source_key} shape={emg.shape}, expected (N, 16)")
    if len(emg) < 2:
        raise ValueError(f"{label}: {source_key} has too few samples ({len(emg)})")
    if not np.isfinite(emg).all():
        raise ValueError(f"{label}: {source_key} contains NaN/Inf")

    ts_key, selected_emg_ts = select_emg_timestamps(data, label=label)
    if len(selected_emg_ts) != len(emg):
        raise ValueError(
            f"{ts_key} len={len(selected_emg_ts)} != {source_key} len={len(emg)}"
        )
    source_grid = _to_session_relative(selected_emg_ts, data)
    t_start, t_end = float(source_grid[0]), float(source_grid[-1])
    duration = t_end - t_start
    if duration <= 0:
        raise ValueError(f"{ts_key} has non-positive duration {duration}")
    actual_fs = (len(emg) - 1) / duration
    n_out = int(duration * TARGET_FS) + 1

    emg_resampled = _resample_to_target(emg, n_out, duration, actual_fs)
    emg_filtered = _apply_semg_filter(emg_resampled, float(TARGET_FS)).astype(
        np.float32
    )
    output_grid = np.linspace(t_start, t_end, n_out)
    return emg_filtered, output_grid, source_grid, {
        "source_key": source_key,
        "timestamp_key": ts_key,
        "src_fs": float(actual_fs),
        "src_samples": int(len(emg)),
        "out_samples": int(n_out),
        "target_fs": int(TARGET_FS),
    }


def convert_one(src: Path, dst: Path, *, side: str | None = None) -> dict:
    """Convert one raw force recording without synthesizing deprecated labels."""
    data = load_force_recording(src, side=side)
    try:
        # EMG 链由可复用 helper 唯一定义，供离线转换和在线微调共用。
        emg_filtered, emg_grid, emg_ts, emg_meta = preprocess_emg_stream(
            data, label=src.name
        )
        ts_key = str(emg_meta["timestamp_key"])
        actual_fs = float(emg_meta["src_fs"])
        n_out = int(emg_meta["out_samples"])
        duration = float(emg_grid[-1] - emg_grid[0])
        emg_len = int(emg_meta["src_samples"])

        # 每个输出标签保存 (原始值, 显式时间轴或 None)。只要显式时间戳存在，
        # 即使标签恰好与 EMG 等长也必须按时间对齐，不能按样本序号假定同轴。
        streams: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

        # --- force: 显式时间戳优先；确无时间戳且等长才按 EMG 同轴处理 ---
        if "force" in data.files:
            force = np.asarray(data["force"], dtype=np.float32)
            streams["force"] = _prepare_stream(
                force,
                data,
                label="force",
                timestamp_candidates=_FORCE_TS_CANDIDATES,
                emg_len=emg_len,
            )

        # --- gen3 per-finger Fz: (T, 5) 多通道, 避免 emg2force 采集端的 5 指 Fz 求和损失 ---
        # 输出 force_per_finger 字段, 列顺序 = thumb / index / middle / ring / pinky。
        # 找不到 gen3 字段时(非 gen3 模式), 不写该字段, 训练侧用 1D `force` 即可。
        # 保留 gen3 完整 (T, 15) Fx/Fy/Fz × 5 指, 字段名与 raw 一致;
        # 既不压成 1D, 也不只留 Fz —— 训练侧自由取列。
        gen3_key = next((k for k in _GEN3_FIELD_CANDIDATES if k in data.files), None)
        if gen3_key is not None:
            gen3_arr = np.asarray(data[gen3_key], dtype=np.float32)
            if gen3_arr.ndim != 2 or gen3_arr.shape[1] < max(_GEN3_FZ_COLS) + 1:
                raise ValueError(
                    f"{gen3_key} shape={gen3_arr.shape}, expected (T, >=15)"
                )
            gen3_timestamp = _GEN3_TS_FIELDS[gen3_key]
            streams[gen3_key] = _prepare_stream(
                gen3_arr,
                data,
                label=gen3_key,
                timestamp_candidates=(gen3_timestamp,),
                emg_len=emg_len,
            )

        # --- cmcu05 / cmcu05b 原字段保留: 重采样到 2kHz EMG 时间轴, 字段名与 raw 一致 ---
        # 显式 timestamps_raw 始终优先；仅旧数据没有时间戳且与 EMG 等长时按同轴处理。
        cmcu_keys: list[str] = []
        for key in data.files:
            if not _CMCU_FIELD_RE.match(key):
                continue
            arr = np.asarray(data[key], dtype=np.float32)
            if arr.ndim not in (1, 2):
                raise ValueError(f"{key} shape={arr.shape}, expected (T,) or (T, N)")
            timestamp_key = key.replace("_force_mn", "_timestamps_raw")
            streams[key] = _prepare_stream(
                arr,
                data,
                label=key,
                timestamp_candidates=(timestamp_key,),
                emg_len=emg_len,
            )
            cmcu_keys.append(key)

        if not streams:
            raise KeyError(
                f"no supported force label found in {src.name}; expected force, "
                "gen3 resultant, or cmcu05/cmcu05b *_force_mn"
            )

        # 所有输出共享 EMG 与标签的共同覆盖区间。禁止端点填充覆盖未采集区间。
        overlap_start = max(
            [float(t_start)]
            + [float(ts[0]) for _arr, ts in streams.values() if ts is not None]
        )
        overlap_end = min(
            [float(t_end)]
            + [float(ts[-1]) for _arr, ts in streams.values() if ts is not None]
        )
        if overlap_end <= overlap_start:
            raise ValueError(
                "EMG and force time ranges do not overlap: "
                f"emg=[{t_start}, {t_end}], common_force=[{overlap_start}, {overlap_end}]"
            )
        overlap_mask = (emg_grid >= overlap_start) & (emg_grid <= overlap_end)
        if np.count_nonzero(overlap_mask) < 2:
            raise ValueError(
                "EMG/force overlap contains fewer than two 2kHz samples: "
                f"overlap=[{overlap_start}, {overlap_end}]"
            )
        output_grid = emg_grid[overlap_mask]
        emg_out = emg_filtered[overlap_mask]

        label_out: dict[str, np.ndarray] = {}
        for key, (arr, timestamps) in streams.items():
            if timestamps is None:
                aligned = _resample_to_target(arr, n_out, duration, actual_fs)
                label_out[key] = aligned[overlap_mask].astype(np.float32)
            else:
                label_out[key] = _interpolate_stream(
                    arr,
                    timestamps,
                    output_grid,
                    label=key,
                )

        force_out = label_out.get("force")
        gen3_out = label_out.get(gen3_key) if gen3_key is not None else None
        cmcu_out = {key: label_out[key] for key in cmcu_keys}

        # 落盘前兜底:输出绝不能含 NaN/Inf(把损坏数据挡在训练集外的最后一道关)。
        if not np.isfinite(emg_out).all():
            raise ValueError(f"non-finite emg after processing in {src.name}")
        if force_out is not None and not np.isfinite(force_out).all():
            raise ValueError(f"non-finite force after processing in {src.name}")
        if gen3_out is not None and not np.isfinite(gen3_out).all():
            raise ValueError(f"non-finite {gen3_key} after processing in {src.name}")
        for k, v in cmcu_out.items():
            if not np.isfinite(v).all():
                raise ValueError(f"non-finite {k} after processing in {src.name}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        save_kw = {"emg": emg_out, **label_out}
        np.savez(dst, **save_kw)

        return {
            "src_fs": actual_fs,
            "src_samples": emg_len,
            "out_samples": len(emg_out),
            "out_duration": len(emg_out) / TARGET_FS,
            "alignment": {
                "source": "raw_to_training_force",
                "timebase": "session_relative",
                "target_fs": TARGET_FS,
                "emg_timestamp_key": ts_key,
                "emg_start_s": float(emg_grid[0]),
                "emg_end_s": float(emg_grid[-1]),
                "overlap_start_s": float(overlap_start),
                "overlap_end_s": float(overlap_end),
                "canonical_start_s": float(output_grid[0]),
                "canonical_end_s": float(output_grid[-1]),
                "canonical_samples": int(len(emg_out)),
                "pre_crop_samples": int(len(emg_grid)),
            },
            "force_range": (
                (float(force_out.min()), float(force_out.max()))
                if force_out is not None
                else None
            ),
            "gen3_field": gen3_key,
            "gen3_shape": list(gen3_out.shape) if gen3_out is not None else None,
            "cmcu_fields": sorted(cmcu_out.keys()),
        }
    finally:
        data.close()


def _safe_convert(src: Path, dst: Path, side: str | None = None) -> tuple[str, str]:
    if dst.exists():
        return src.name, "skipped"
    try:
        info = convert_one(src, dst, side=side)
        return src.name, (
            f"ok: {info['src_fs']:.0f}Hz->2kHz, {info['out_samples']} samples "
            f"({info['out_duration']:.1f}s)"
        )
    except Exception as exc:
        return src.name, f"error: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="原始力 NPZ/MCAP → emg2force 训练 NPZ 处理器"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="原始力数据根目录(递归)"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="训练 NPZ 输出目录"
    )
    parser.add_argument("--n-jobs", type=int, default=8, help="并行进程数")
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        help="只处理指定手；默认自动检测，双手 MCAP 输出左右两条 NPZ",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Path] = []
    for root, _dirs, files in os.walk(args.input_dir, followlinks=True):
        for fn in files:
            if is_raw_recording_path(fn) and not fn.startswith("norm_stats"):
                candidates.append(Path(root) / fn)
    raw_files: list[Path] = []
    for f in candidates:
        try:
            if f.suffix.lower() == ".mcap":
                if available_force_recording_sides(f):
                    raw_files.append(f)
            else:
                with np.load(f, allow_pickle=True) as d:
                    has_label = (
                        "force" in d.files
                        or any(_CMCU_FIELD_RE.match(key) for key in d.files)
                        or any(key in d.files for key in _GEN3_FIELD_CANDIDATES)
                    )
                    if has_label and ("emg_raw" in d.files or "emg" in d.files):
                        raw_files.append(f)
        except Exception as e:
            print(f"  [SKIP-CORRUPT] {f}: {e}")
    raw_files.sort()
    print(f"Found {len(raw_files)} raw force NPZ/MCAP files")

    tasks: list[tuple[Path, Path, str | None]] = []
    for src in raw_files:
        detected = available_force_recording_sides(src)
        sides: tuple[str | None, ...]
        if args.side and detected:
            sides = tuple(side for side in detected if side == args.side)
        elif args.side:
            sides = (args.side,) if src.suffix.lower() == ".npz" else ()
        elif detected:
            sides = detected
        else:
            sides = (None,)
        split = len(sides) > 1
        base_dst = (
            args.output_dir
            / f"{src.parent.parent.name}__{src.parent.name}__{src.stem}.npz"
        )
        for side in sides:
            dst = (
                side_output_path(base_dst, side, split=split)
                if side is not None
                else base_dst
            )
            tasks.append((src, dst, side))

    if args.n_jobs > 1 and len(tasks) > 1:
        results = Parallel(
            n_jobs=args.n_jobs, prefer="processes", return_as="generator_unordered"
        )(delayed(_safe_convert)(s, d, side) for s, d, side in tasks)
    else:
        results = (_safe_convert(s, d, side) for s, d, side in tasks)

    ok = skipped = fail = 0
    for name, status in tqdm(results, total=len(tasks), desc="Processing"):
        if status == "skipped":
            skipped += 1
        elif status.startswith("ok"):
            ok += 1
        else:
            fail += 1
            print(f"  [ERROR] {name}: {status}")

    print(f"\nDone: {ok} ok, {skipped} skipped, {fail} failed")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
