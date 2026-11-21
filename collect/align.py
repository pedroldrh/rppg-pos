"""
Align Apple Watch HR labels to video frame timestamps.

Primary output:  windows.csv  — window-level HR labels for evaluation
Secondary output: frame_hr.csv — frame-level interpolated HR for visualization only

Usage:
    py collect/align.py --session data/raw/session_20260407_143022 --hr data/processed/hr_parsed.csv
    py collect/align.py --test
"""

import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Core functions ───────────────────────────────────────────────────

def load_session_metadata(session_dir):
    """Load metadata.json from a session directory.

    Returns dict with at least 'start_time_utc' as a timezone-aware datetime.
    """
    meta_path = os.path.join(session_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["start_time_utc"] = pd.to_datetime(meta["start_time_utc"], utc=True).to_pydatetime()
    return meta


def build_windows(frame_timestamps_df, hr_df, video_start_utc,
                  window_sec=None, stride_sec=None, min_samples=None):
    """Build evaluation windows aligning video time to Apple Watch HR.

    Uses video_start_utc + frame elapsed time to compute absolute UTC
    timestamps per window, then bins Apple Watch HR samples into each window.

    Args:
        frame_timestamps_df: DataFrame with columns [frame_idx, elapsed_ms, system_time_ms]
        hr_df: DataFrame with columns [timestamp_utc, hr_bpm] (from parse_apple_health)
        video_start_utc: timezone-aware datetime — source of truth from metadata.json
        window_sec: window duration (default from config)
        stride_sec: window stride (default from config)
        min_samples: minimum HR samples for a valid window (default from config)

    Returns:
        DataFrame with columns:
            window_idx, window_start_sec, window_end_sec,
            window_start_utc, window_end_utc,
            hr_bpm_median, hr_bpm_mean, hr_sample_count, valid
    """
    window_sec = window_sec or config.EVAL_WINDOW_SEC
    stride_sec = stride_sec or config.EVAL_STRIDE_SEC
    min_samples = min_samples or config.MIN_HR_SAMPLES_PER_WINDOW

    # Total video duration from frame timestamps
    max_elapsed_ms = frame_timestamps_df["elapsed_ms"].max()
    total_sec = max_elapsed_ms / 1000.0

    # Ensure hr_df timestamps are UTC-aware
    hr_times = pd.to_datetime(hr_df["timestamp_utc"], utc=True)
    hr_bpm = hr_df["hr_bpm"].values

    windows = []
    window_idx = 0
    start_sec = 0.0

    while start_sec + window_sec <= total_sec + 0.01:
        end_sec = start_sec + window_sec

        # Convert to absolute UTC
        win_start_utc = video_start_utc + timedelta(seconds=start_sec)
        win_end_utc = video_start_utc + timedelta(seconds=end_sec)

        # Find HR samples within this window
        mask = (hr_times >= pd.Timestamp(win_start_utc)) & (hr_times < pd.Timestamp(win_end_utc))
        samples_in_window = hr_bpm[mask.values]
        sample_count = len(samples_in_window)

        valid = sample_count >= min_samples
        if valid:
            median_hr = float(np.median(samples_in_window))
            mean_hr = float(np.mean(samples_in_window))
        else:
            median_hr = float("nan")
            mean_hr = float("nan")

        windows.append({
            "window_idx": window_idx,
            "window_start_sec": round(start_sec, 2),
            "window_end_sec": round(end_sec, 2),
            "window_start_utc": win_start_utc.isoformat(),
            "window_end_utc": win_end_utc.isoformat(),
            "hr_bpm_median": median_hr,
            "hr_bpm_mean": round(mean_hr, 2) if not math.isnan(mean_hr) else mean_hr,
            "hr_sample_count": sample_count,
            "valid": valid,
        })

        window_idx += 1
        start_sec += stride_sec

    return pd.DataFrame(windows)


def interpolate_frame_hr(frame_timestamps_df, hr_df, video_start_utc):
    """Interpolate HR to every video frame. For visualization/debug only.

    Uses video_start_utc + elapsed_ms to compute absolute UTC per frame,
    then linearly interpolates Apple Watch HR samples.

    Returns DataFrame with columns:
        frame_idx, elapsed_ms, system_time_ms, timestamp_utc, hr_bpm_interpolated
    """
    # Compute absolute UTC per frame
    frame_elapsed_sec = frame_timestamps_df["elapsed_ms"].values / 1000.0
    frame_utc = [video_start_utc + timedelta(seconds=float(e)) for e in frame_elapsed_sec]

    # Convert HR timestamps to seconds relative to video start
    hr_times = pd.to_datetime(hr_df["timestamp_utc"], utc=True)
    hr_offsets_sec = [(t.to_pydatetime() - video_start_utc).total_seconds() for t in hr_times]
    hr_bpm = hr_df["hr_bpm"].values

    if len(hr_offsets_sec) < 2:
        # Not enough data to interpolate
        result = frame_timestamps_df.copy()
        result["timestamp_utc"] = [t.isoformat() for t in frame_utc]
        result["hr_bpm_interpolated"] = float("nan")
        return result

    # Linear interpolation, NaN outside the HR data range
    interp_fn = interp1d(hr_offsets_sec, hr_bpm, kind="linear",
                         bounds_error=False, fill_value=float("nan"))
    interpolated = interp_fn(frame_elapsed_sec)

    result = frame_timestamps_df.copy()
    result["timestamp_utc"] = [t.isoformat() for t in frame_utc]
    result["hr_bpm_interpolated"] = np.round(interpolated, 1)
    return result[["frame_idx", "elapsed_ms", "system_time_ms", "timestamp_utc", "hr_bpm_interpolated"]]


def align_session(session_dir, hr_path, output_dir=None):
    """End-to-end alignment for a single session.

    Reads metadata.json and frame_timestamps.csv from session_dir,
    loads parsed HR from hr_path, produces windows.csv and frame_hr.csv.
    """
    meta = load_session_metadata(session_dir)
    video_start_utc = meta["start_time_utc"]

    frame_ts = pd.read_csv(os.path.join(session_dir, "frame_timestamps.csv"))
    hr_df = pd.read_csv(hr_path)

    windows = build_windows(frame_ts, hr_df, video_start_utc)
    frame_hr = interpolate_frame_hr(frame_ts, hr_df, video_start_utc)

    out = output_dir or session_dir
    os.makedirs(out, exist_ok=True)

    windows_path = os.path.join(out, "windows.csv")
    frame_hr_path = os.path.join(out, "frame_hr.csv")
    windows.to_csv(windows_path, index=False)
    frame_hr.to_csv(frame_hr_path, index=False)

    valid_count = windows["valid"].sum()
    total_count = len(windows)
    print(f"Alignment complete:")
    print(f"  Windows: {valid_count}/{total_count} valid")
    print(f"  Saved: {windows_path}")
    print(f"  Saved: {frame_hr_path}")

    return windows, frame_hr


# ── Tests ────────────────────────────────────────────────────────────

def run_tests():
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            failed += 1

    video_start = datetime(2026, 4, 7, 18, 30, 0, tzinfo=timezone.utc)

    # ── Test 1: Basic window alignment ───────────────────────────
    print("\nTest 1: Basic window alignment (30s video, HR every 5s at 72 BPM)")
    # 30 seconds at 30 fps = 901 frames (0 through 30s inclusive)
    n_frames = 901
    frame_ts = pd.DataFrame({
        "frame_idx": range(n_frames),
        "elapsed_ms": [i * (1000.0 / 30) for i in range(n_frames)],
        "system_time_ms": [1.0] * n_frames,  # not used for primary alignment
    })
    # HR every 5 seconds = 7 samples at 0, 5, 10, 15, 20, 25, 30
    hr_df = pd.DataFrame({
        "timestamp_utc": [
            (video_start + timedelta(seconds=s)).isoformat()
            for s in [0, 5, 10, 15, 20, 25, 30]
        ],
        "hr_bpm": [72.0] * 7,
    })

    windows = build_windows(frame_ts, hr_df, video_start)
    check("Produces windows", len(windows) > 0)
    # 10-sec window, 5-sec stride, 30s video: windows at 0-10, 5-15, 10-20, 15-25, 20-30 = 5
    check("Correct window count (5)", len(windows) == 5)
    check("All windows valid", windows["valid"].all())
    check("All medians are 72", (windows["hr_bpm_median"] == 72.0).all())
    check("All means are 72", (windows["hr_bpm_mean"] == 72.0).all())
    check("Has UTC columns", "window_start_utc" in windows.columns and "window_end_utc" in windows.columns)

    # ── Test 2: Invalid windows (insufficient samples) ───────────
    print("\nTest 2: Invalid windows with sparse HR data")
    # Only 2 HR samples at 0s and 25s — most windows will have < 2 samples
    hr_sparse = pd.DataFrame({
        "timestamp_utc": [
            (video_start + timedelta(seconds=s)).isoformat()
            for s in [0, 25]
        ],
        "hr_bpm": [70.0, 80.0],
    })
    windows_sparse = build_windows(frame_ts, hr_sparse, video_start)
    invalid_windows = windows_sparse[~windows_sparse["valid"]]
    check("Some windows are invalid", len(invalid_windows) > 0)
    check("Invalid windows have NaN median", invalid_windows["hr_bpm_median"].isna().all())
    check("Invalid windows have NaN mean", invalid_windows["hr_bpm_mean"].isna().all())

    # ── Test 3: Irregular Apple Watch sampling ───────────────────
    print("\nTest 3: Irregular sampling (0s, 4s, 9s, 15s, 21s)")
    hr_irregular = pd.DataFrame({
        "timestamp_utc": [
            (video_start + timedelta(seconds=s)).isoformat()
            for s in [0, 4, 9, 15, 21]
        ],
        "hr_bpm": [68.0, 72.0, 74.0, 70.0, 76.0],
    })
    windows_irreg = build_windows(frame_ts, hr_irregular, video_start)

    # Window 0-10: samples at 0, 4, 9 → 3 samples, median=72
    w0 = windows_irreg[windows_irreg["window_idx"] == 0].iloc[0]
    check("Window 0: 3 samples", w0["hr_sample_count"] == 3)
    check("Window 0: valid", w0["valid"] == True)
    check("Window 0: median=72", w0["hr_bpm_median"] == 72.0)

    # Window 5-15: samples at 9, → wait, also need 15? 15 is at boundary.
    # [5, 15) → samples at 9 = 1 sample... let me recalculate
    # Actually [5, 15) includes samples at 9 only (15 is excluded by < boundary)
    # So window 1 has 1 sample → invalid
    w1 = windows_irreg[windows_irreg["window_idx"] == 1].iloc[0]
    check("Window 1 (5-15): sample count correct", w1["hr_sample_count"] == 1)
    check("Window 1: invalid (only 1 sample)", w1["valid"] == False)

    # Window 10-20: samples at 15 → 1 sample → invalid
    w2 = windows_irreg[windows_irreg["window_idx"] == 2].iloc[0]
    check("Window 2 (10-20): sample count correct", w2["hr_sample_count"] == 1)
    check("Window 2: invalid", w2["valid"] == False)

    # Window 15-25: samples at 15, 21 → 2 samples, median=73
    w3 = windows_irreg[windows_irreg["window_idx"] == 3].iloc[0]
    check("Window 3 (15-25): 2 samples", w3["hr_sample_count"] == 2)
    check("Window 3: valid", w3["valid"] == True)
    check("Window 3: median=73", w3["hr_bpm_median"] == 73.0)

    # ── Test 4: Frame-level interpolation ────────────────────────
    print("\nTest 4: Frame-level interpolation")
    frame_hr = interpolate_frame_hr(frame_ts, hr_irregular, video_start)
    check("Row count matches frames", len(frame_hr) == n_frames)
    check("Has timestamp_utc column", "timestamp_utc" in frame_hr.columns)
    check("Has hr_bpm_interpolated column", "hr_bpm_interpolated" in frame_hr.columns)
    # Frame at 0s should be ~68 BPM (first HR sample)
    check("Frame 0 HR ~= 68", abs(frame_hr.iloc[0]["hr_bpm_interpolated"] - 68.0) < 0.5)

    # ── Test 5: Output schema ────────────────────────────────────
    print("\nTest 5: Output schema")
    expected_win_cols = [
        "window_idx", "window_start_sec", "window_end_sec",
        "window_start_utc", "window_end_utc",
        "hr_bpm_median", "hr_bpm_mean", "hr_sample_count", "valid"
    ]
    expected_frame_cols = [
        "frame_idx", "elapsed_ms", "system_time_ms", "timestamp_utc", "hr_bpm_interpolated"
    ]
    check("Window columns correct", list(windows.columns) == expected_win_cols)
    check("Frame HR columns correct", list(frame_hr.columns) == expected_frame_cols)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Align Apple Watch HR to video frames")
    parser.add_argument("--session", help="Path to session directory")
    parser.add_argument("--hr", help="Path to parsed HR CSV")
    parser.add_argument("--output", help="Output directory (default: session dir)")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    if not args.session or not args.hr:
        parser.error("--session and --hr are required (or use --test)")

    align_session(args.session, args.hr, output_dir=args.output)


if __name__ == "__main__":
    main()
