#!/usr/bin/env python3
"""原始 NPZ/MCAP → 训练 NPZ 处理器(processor)。

把上游原始录制转换为 emg2pose 训练格式。本模块是 PROCESSOR(只转换、不做质量
判断);质量检查由 data_prep/checkers/ 下的 checker 负责。

处理步骤(沿用 process_format_v3_to_training 的 SOP):
    1. EMG 重采样到统一 2kHz(应对不规则 / 低采样率录制)
    2. EMG 带通 40-850Hz + 50Hz 及谐波陷波
    3. Manus skeleton 映射到 StandardHandModel 后线性插值到 EMG 2kHz 时间轴
    4. StandardHandModel 输出 23 维 theta, 再映射为 emg2pose 20 维 joint_angles
    5. 裁剪到 EMG-Manus skeleton 时间重叠段
    6. 输出 {emg [T,16], joint_angles [T,20]} NPZ

与 process_format_v3_to_training 的关系(整合替代 + 一处有意修正):
    - SEMGFilter 内联为 scipy SOS(已验证与 prometheus 版 bit-exact),去掉 vendor 依赖。
    - **修正二次滤波 bug**:v3 读的是 `emg`(采集软件已带通滤波的字段),再过一次
      SEMGFilter = 二次滤波。本模块改读 **`emg_raw`(真原始信号)**,SEMGFilter 因此
      只滤一次。输出因此与 v3 **不再 byte-exact**(这是有意的修正)。
      尺度无突变:SEMGFilter(emg_raw) 的 std 与旧 emg 量级一致(滤波线性 + 高通去 DC)。
    - 仍 **未追求与推理逐位对齐**(推理是 emg_raw→×scale→median(5)→SEMGFilter);
      ×EMG_VALUE_SCALE、median(5)、clip 等训练/推理对齐项留待后续。
    - 标签链已改为 Manus skeleton -> StandardHandModel -> 20 维 joint_angles, 不再直接使用 Manus ergonomics。

CLI(与原预处理参数兼容):
    python -m src.data_prep.processors.raw_to_training \\
        --input-dir <raw_root> --output-dir <out> [--subject S] [--side left] [--n-jobs 8]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from src.data_prep.processors.raw_recording import (
    RAW_RECORDING_SUFFIXES,
    available_recording_sides,
    is_raw_recording_path,
    load_raw_recording,
    side_output_path,
)
from src.data_prep.processors.standard_hand_helper import (
    standard_joint_angles_from_skeleton,
)
from src.data_prep.processors.timestamps import (
    select_emg_timestamps,
    to_session_relative,
)

try:
    from joblib import Parallel, delayed
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal local envs.
    Parallel = None
    delayed = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal local envs.

    def tqdm(iterable, *args, **kwargs):
        return iterable


TARGET_FS = 2000  # 目标采样率 Hz
LABEL_PIPELINE_ID = "shm_skeleton_v1"

# EMG 滤波默认参数(移植自 prometheus.filtering.SEMGFilter.make_default_2kHz)
EMG_FILTER_LOWCUT = 40.0
EMG_FILTER_HIGHCUT = 850.0
EMG_FILTER_LINE_FREQ = 50.0
EMG_FILTER_NOTCH_Q = 30.0
EMG_FILTER_ORDER = 4


def processor_fingerprint() -> str:
    """Return the deterministic fingerprint of the raw-to-training processor.

    This deliberately excludes checker/gate parameters.  It can therefore be
    used by artifact-only migrations without claiming that quality checks were
    rerun.
    """
    return (
        f"fs{TARGET_FS}_bp{EMG_FILTER_LOWCUT:g}-{EMG_FILTER_HIGHCUT:g}"
        f"_notch{EMG_FILTER_LINE_FREQ:g}q{EMG_FILTER_NOTCH_Q:g}"
        f"_ord{EMG_FILTER_ORDER}_label{LABEL_PIPELINE_ID}"
    )


# ──────────────────────────────────────────────────────────────────────
# 内联 EMG 滤波(等价 SEMGFilter.make_default_2kHz, bit-exact)
# ──────────────────────────────────────────────────────────────────────
def _build_semg_sos(fs: float) -> np.ndarray:
    nyquist = fs / 2.0
    sos_list: list[np.ndarray] = []
    harmonic = EMG_FILTER_LINE_FREQ
    while harmonic < nyquist:
        b, a = iirnotch(harmonic, EMG_FILTER_NOTCH_Q, fs=fs)
        sos_list.append(tf2sos(b, a))
        harmonic += EMG_FILTER_LINE_FREQ
    sos_list.append(
        butter(
            EMG_FILTER_ORDER,
            (EMG_FILTER_LOWCUT, EMG_FILTER_HIGHCUT),
            btype="bandpass",
            fs=fs,
            output="sos",
        )
    )
    return np.vstack(sos_list)


def _apply_semg_filter(emg: np.ndarray, fs: float) -> np.ndarray:
    """SOS 滤波 + 首段均值初始化 zi(等价 SEMGFilter.initialize_steady)。"""
    sos = _build_semg_sos(fs)
    n_warmup = min(1000, emg.shape[0])
    data_mean = np.mean(emg[:n_warmup], axis=0, keepdims=True)
    zi = sosfilt_zi(sos)[:, :, None] * data_mean
    zi = np.ascontiguousarray(zi)
    out, _ = sosfilt(sos, emg, zi=zi, axis=0)
    return out.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# 时基归一(沿用 v3 _to_session_relative)
# ──────────────────────────────────────────────────────────────────────
def _to_session_relative(emg_sys_ts: np.ndarray, data) -> np.ndarray:
    """把 emg_sys_timestamps 归一到 session 相对时间。

    上游存在两种约定:已相对(首值 ~1s 的硬件缓冲延迟)与系统单调(首值与
    record_t0 同量级)。用与 record_t0 的偏差区分(阈值 10s)。
    """
    return to_session_relative(emg_sys_ts, data)


# ──────────────────────────────────────────────────────────────────────
# 单文件转换
# ──────────────────────────────────────────────────────────────────────
def convert_one(
    src: Path,
    dst: Path,
    side: str,
    *,
    hand_path: Path | None = None,
    device: str = "cpu",
) -> dict:
    data = load_raw_recording(src, side=side)
    try:
        # 读 emg_raw(真原始, 未滤波): 下游 SEMGFilter 因此只滤一次, 修掉 v3 的二次滤波。
        # 个别旧文件可能只有 emg(已滤波), 退回读它(等价旧行为, 会二次滤波)并标注。
        if "emg_raw" in data.files:
            emg = data["emg_raw"].astype(np.float32)  # [T_emg, 16]
        else:
            emg = data["emg"].astype(np.float32)
        ts_key, emg_timestamps = select_emg_timestamps(data, label=src.name)
        if len(emg_timestamps) != len(emg):
            raise ValueError(
                f"{ts_key} len={len(emg_timestamps)} != emg len={len(emg)}"
            )
        emg_sys_ts = _to_session_relative(emg_timestamps, data)

        # --- EMG: 构建统一 2kHz 时间轴并重采样 ---
        t_start, t_end = emg_sys_ts[0], emg_sys_ts[-1]
        duration = t_end - t_start
        actual_fs = (len(emg) - 1) / duration
        n_out = int(duration * TARGET_FS) + 1

        if abs(actual_fs - TARGET_FS) > 50:
            src_ts = np.linspace(0.0, duration, len(emg))
            dst_ts = np.linspace(0.0, duration, n_out)
            emg_resampled = interp1d(src_ts, emg, axis=0, kind="linear")(dst_ts).astype(
                np.float32
            )
        else:
            emg_resampled = emg[:n_out].astype(np.float32)
            if len(emg_resampled) < n_out:
                pad = np.repeat(emg_resampled[-1:], n_out - len(emg_resampled), axis=0)
                emg_resampled = np.concatenate([emg_resampled, pad], axis=0)

        # --- EMG: 带通 + 陷波(作用在 emg_raw 上, 即唯一一次滤波)---
        emg_filtered = _apply_semg_filter(emg_resampled, float(TARGET_FS))

        # --- Manus skeleton -> StandardHandModel -> 20 维 joint_angles ---
        skel_abs, standard_joint_angles = standard_joint_angles_from_skeleton(
            data,
            side,
            hand_path=hand_path,
            device=device,
        )

        # EMG session-absolute 时间轴(保留 ~1s 硬件延迟偏移)
        emg_abs = np.linspace(t_start, t_end, n_out)

        # --- 裁剪到 EMG-Manus skeleton 时间重叠段 ---
        t_overlap_start = max(skel_abs[0], emg_abs[0])
        t_overlap_end = min(skel_abs[-1], emg_abs[-1])
        if t_overlap_start > t_overlap_end:
            raise ValueError(
                "EMG and skeleton time ranges do not overlap: "
                f"emg=[{emg_abs[0]}, {emg_abs[-1]}], "
                f"skeleton=[{skel_abs[0]}, {skel_abs[-1]}]"
            )

        mask = (emg_abs >= t_overlap_start) & (emg_abs <= t_overlap_end)
        if not mask.any():
            raise ValueError(
                "EMG/skeleton overlap contains no EMG samples: "
                f"overlap=[{t_overlap_start}, {t_overlap_end}]"
            )

        # --- 标准手 joint_angles 插值到 EMG 2kHz 时间轴 ---
        joint_interp = interp1d(
            skel_abs,
            standard_joint_angles,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )(emg_abs).astype(np.float64)

        emg_out = emg_filtered[mask].astype(np.float32)
        joint_out = joint_interp[mask].astype(np.float32)

        # 落盘前兜底:输出绝不能含 NaN/Inf。任一非有限 → 报错不产出
        # (processor 阶段无条件落 processed_data,这是把损坏数据挡在训练集外的最后一道关)。
        if not np.isfinite(joint_out).all():
            raise ValueError(
                f"non-finite joint_angles after processing in {src.name} "
                f"({np.isnan(joint_out).mean():.0%} NaN/Inf) — 疑似 Manus 时间戳/帧损坏"
            )
        if not np.isfinite(emg_out).all():
            raise ValueError(f"non-finite emg after processing in {src.name}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, emg=emg_out, joint_angles=joint_out)

        return {
            "src_fs": actual_fs,
            "src_samples": len(emg),
            "out_samples": len(emg_out),
            "out_duration": len(emg_out) / TARGET_FS,
            "alignment": {
                "source": "raw_to_training",
                "timebase": "session_relative",
                "target_fs": TARGET_FS,
                "emg_timestamp_key": ts_key,
                "emg_start_s": float(emg_abs[0]),
                "emg_end_s": float(emg_abs[-1]),
                "overlap_start_s": float(t_overlap_start),
                "overlap_end_s": float(t_overlap_end),
                "canonical_start_s": float(emg_abs[mask][0]),
                "canonical_end_s": float(emg_abs[mask][-1]),
                "canonical_samples": int(len(emg_out)),
                "pre_crop_samples": int(len(emg_abs)),
            },
            "joint_range_deg": (
                np.rad2deg(joint_out.min()),
                np.rad2deg(joint_out.max()),
            ),
        }
    finally:
        data.close()


def _safe_convert(
    src: Path,
    dst: Path,
    side: str,
    hand_path: Path | None = None,
    device: str = "cpu",
) -> tuple[str, str]:
    if dst.exists():
        return src.name, "skipped"
    try:
        info = convert_one(src, dst, side.lower(), hand_path=hand_path, device=device)
        return src.name, (
            f"ok: {info['src_fs']:.0f}Hz->2kHz, {info['out_samples']} samples "
            f"({info['out_duration']:.1f}s)"
        )
    except Exception as exc:
        return src.name, f"error: {exc}"


# ──────────────────────────────────────────────────────────────────────
# CLI(与原预处理参数兼容)
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="原始 NPZ/MCAP → emg2pose 训练 NPZ 处理器"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="原始 .npz/.mcap 文件或数据根目录",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="训练 NPZ 输出目录"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="可选过滤:仅处理 <input-dir>/<subject>/*",
    )
    parser.add_argument(
        "--side",
        type=str,
        default=None,
        choices=["left", "right"],
        help="仅处理指定手；默认自动检测，双手录制拆成 left/right 两条",
    )
    parser.add_argument(
        "--hand-path",
        type=Path,
        default=None,
        help="标准手 .npy 路径；默认使用已安装 StandardHandModel 的内置模型",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="StandardHandModel 求解设备, 如 cpu 或 cuda:0",
    )
    parser.add_argument("--n-jobs", type=int, default=8, help="并行进程数")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    walk_root = args.input_dir
    if args.subject:
        if args.input_dir.is_file():
            raise SystemExit("--subject cannot be used when --input-dir is a file")
        walk_root = args.input_dir / args.subject
        if not walk_root.is_dir():
            raise SystemExit(
                f"--subject given but {walk_root} is not a directory. "
                f"Expected layout: <input-dir>/<subject>/<record_id>/*.npz"
            )
        print(f"[subject] filtering to {walk_root}")

    skip_names = {"norm_stats.npz"}
    candidates: list[Path] = []
    if walk_root.is_file():
        if is_raw_recording_path(walk_root) and walk_root.name not in skip_names:
            candidates.append(walk_root)
    else:
        for root, _dirs, files in os.walk(walk_root, followlinks=True):
            for fn in files:
                if is_raw_recording_path(fn) and fn not in skip_names:
                    candidates.append(Path(root) / fn)
    raw_files: list[Path] = []
    for f in candidates:
        if f.suffix.lower() == ".mcap":
            raw_files.append(f)
            continue
        try:
            with np.load(f, allow_pickle=True) as d:
                if "emg_raw" in d.files or "emg" in d.files:
                    raw_files.append(f)
        except Exception as e:
            print(f"  [SKIP-CORRUPT] {f}: {e}")
    raw_files.sort()
    suffixes = "/".join(sorted(RAW_RECORDING_SUFFIXES))
    print(f"Found {len(raw_files)} recording files ({suffixes})")

    tasks: list[tuple[Path, Path, str]] = []
    for src in raw_files:
        base_dst = (
            args.output_dir
            / f"{src.parent.parent.name}__{src.parent.name}__{src.stem}.npz"
        )
        try:
            sides = (args.side,) if args.side else available_recording_sides(src)
        except Exception as exc:
            print(f"  [SKIP-CORRUPT] {src}: {exc}")
            continue
        if not sides:
            print(f"  [SKIP-NO-POSE] {src}: no matching EMG + skeleton side")
            continue
        split = len(sides) > 1
        tasks.extend(
            (src, side_output_path(base_dst, side, split=split), side) for side in sides
        )

    if (
        args.n_jobs > 1
        and len(tasks) > 1
        and Parallel is not None
        and delayed is not None
    ):
        results = Parallel(
            n_jobs=args.n_jobs, prefer="processes", return_as="generator_unordered"
        )(
            delayed(_safe_convert)(s, d, side, args.hand_path, args.device)
            for s, d, side in tasks
        )
    else:
        results = (
            _safe_convert(s, d, side, args.hand_path, args.device)
            for s, d, side in tasks
        )

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
