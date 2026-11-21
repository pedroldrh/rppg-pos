"""
Signal processing for rPPG: bandpass filtering and windowed HR estimation.

Takes the raw POS pulse signal, filters it to the cardiac band, and
estimates heart rate per evaluation window via Hann-windowed FFT peak detection.

Uses the same window grid as collect/align.py (config.EVAL_WINDOW_SEC / EVAL_STRIDE_SEC).

Usage:
    py rppg/signal_processing.py --test
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Bandpass filter ──────────────────────────────────────────────────

def bandpass_filter(signal, fps, low_hz=None, high_hz=None, order=None):
    """Apply a zero-phase Butterworth bandpass filter to the signal.

    Handles NaN by filtering only contiguous valid segments independently.
    NaN regions in the input remain NaN in the output.

    Args:
        signal: (N,) NumPy array, may contain NaN
        fps: sampling rate in Hz
        low_hz: low cutoff (default config.HR_BAND_HZ[0])
        high_hz: high cutoff (default config.HR_BAND_HZ[1])
        order: filter order (default config.BANDPASS_ORDER)

    Returns:
        (N,) filtered signal with NaN preserved
    """
    low_hz = low_hz if low_hz is not None else config.HR_BAND_HZ[0]
    high_hz = high_hz if high_hz is not None else config.HR_BAND_HZ[1]
    order = order if order is not None else config.BANDPASS_ORDER

    nyq = fps / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")

    output = np.full_like(signal, float("nan"), dtype=np.float64)

    # Find contiguous non-NaN segments
    is_valid = ~np.isnan(signal)
    segments = _contiguous_segments(is_valid)

    # Minimum segment length for filtfilt: 3 * max(len(a), len(b)) per scipy docs
    min_len = 3 * max(len(a), len(b))

    for start, end in segments:
        seg = signal[start:end]
        if len(seg) >= min_len:
            output[start:end] = filtfilt(b, a, seg)
        # Segments too short to filter are left as NaN

    return output


def _contiguous_segments(mask):
    """Find start/end indices of contiguous True runs in a boolean array.

    Returns list of (start, end) tuples (end is exclusive).
    """
    segments = []
    in_segment = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_segment:
            start = i
            in_segment = True
        elif not v and in_segment:
            segments.append((start, i))
            in_segment = False
    if in_segment:
        segments.append((start, len(mask)))
    return segments


# ── FFT-based HR estimation per window ───────────────────────────────

def compute_window_fft_hr(segment, fps, low_hz=None, high_hz=None):
    """Estimate HR from a single window of filtered pulse signal using FFT.

    Applies a Hann window before FFT. Finds the dominant spectral peak
    within the cardiac band and computes SNR.

    Args:
        segment: (L,) NumPy array — filtered pulse for one window
        fps: sampling rate
        low_hz, high_hz: cardiac band limits

    Returns:
        dict with keys:
            hr_bpm, peak_freq_hz, peak_power, cardiac_band_power_mean, snr
        All values are float. Returns NaN values if no valid peak found.
    """
    low_hz = low_hz if low_hz is not None else config.HR_BAND_HZ[0]
    high_hz = high_hz if high_hz is not None else config.HR_BAND_HZ[1]

    nan_result = {
        "hr_bpm": float("nan"),
        "peak_freq_hz": float("nan"),
        "peak_power": float("nan"),
        "cardiac_band_power_mean": float("nan"),
        "snr": float("nan"),
    }

    n = len(segment)
    if n < 2:
        return nan_result

    # Apply Hann window
    window = np.hanning(n)
    windowed = (segment - segment.mean()) * window

    # FFT
    fft_vals = np.abs(np.fft.rfft(windowed)) ** 2  # power spectrum
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    # Restrict to cardiac band
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not band_mask.any():
        return nan_result

    band_power = fft_vals[band_mask]
    band_freqs = freqs[band_mask]

    if len(band_power) == 0:
        return nan_result

    # Find peak
    peak_idx = np.argmax(band_power)
    peak_freq = band_freqs[peak_idx]
    peak_pow = band_power[peak_idx]

    # ── Harmonic detection ───────────────────────────────────────
    # If the dominant peak could be a harmonic (2x the real frequency),
    # check if there's a subharmonic at half the frequency with reasonable
    # power. If so, prefer the subharmonic — it's likely the true cardiac
    # frequency. This fixes the common failure mode where noise or harmonics
    # at 130-160 BPM are picked over the real ~65-80 BPM signal.
    if config.HARMONIC_CHECK:
        sub_freq = peak_freq / 2.0
        if sub_freq >= low_hz:
            # Find the closest bin to the subharmonic frequency
            sub_idx = np.argmin(np.abs(band_freqs - sub_freq))
            sub_pow = band_power[sub_idx]
            # If the subharmonic has meaningful power, prefer it
            if sub_pow >= config.HARMONIC_POWER_RATIO * peak_pow:
                peak_idx = sub_idx
                peak_freq = band_freqs[peak_idx]
                peak_pow = sub_pow

    # SNR: peak power / mean power of non-peak bins in cardiac band
    # If only 1 bin in band, SNR is undefined
    if len(band_power) > 1:
        non_peak_mask = np.ones(len(band_power), dtype=bool)
        non_peak_mask[peak_idx] = False
        mean_non_peak = band_power[non_peak_mask].mean()
        snr = peak_pow / mean_non_peak if mean_non_peak > 0 else float("nan")
    else:
        snr = float("nan")

    return {
        "hr_bpm": peak_freq * 60.0,
        "peak_freq_hz": float(peak_freq),
        "peak_power": float(peak_pow),
        "cardiac_band_power_mean": float(band_power.mean()),
        "snr": float(snr),
    }


def compute_snr(peak_power, band_power_array, peak_idx):
    """Compute SNR: peak power / mean non-peak power in the cardiac band.

    Args:
        peak_power: power at the dominant frequency
        band_power_array: power values across the cardiac band
        peak_idx: index of the peak in band_power_array

    Returns:
        float SNR value, or NaN if not computable
    """
    if len(band_power_array) < 2:
        return float("nan")
    non_peak = np.delete(band_power_array, peak_idx)
    mean_non_peak = non_peak.mean()
    if mean_non_peak <= 0:
        return float("nan")
    return float(peak_power / mean_non_peak)


# ── Windowed HR estimation ───────────────────────────────────────────

def estimate_hr_windowed(pulse, fps, window_sec=None, stride_sec=None,
                         max_nan_fraction=None, min_snr=None,
                         return_debug=False):
    """Estimate HR per evaluation window from filtered pulse signal.

    Uses the same window grid as collect/align.py:
        config.EVAL_WINDOW_SEC / config.EVAL_STRIDE_SEC

    Args:
        pulse: (N,) filtered pulse signal (may contain NaN)
        fps: sampling rate
        window_sec, stride_sec: override config values
        max_nan_fraction: skip window if NaN fraction exceeds this
        min_snr: mark window invalid if SNR below this
        return_debug: include spectral debug columns

    Returns:
        DataFrame with columns:
            window_idx, window_start_sec, window_end_sec,
            hr_bpm_estimated, snr, valid
        If return_debug=True, also includes:
            peak_freq_hz, peak_power, cardiac_band_power_mean
    """
    window_sec = window_sec if window_sec is not None else config.EVAL_WINDOW_SEC
    stride_sec = stride_sec if stride_sec is not None else config.EVAL_STRIDE_SEC
    max_nan_fraction = max_nan_fraction if max_nan_fraction is not None else config.HR_MAX_NAN_FRACTION
    min_snr = min_snr if min_snr is not None else config.HR_MIN_SNR

    n_frames = len(pulse)
    total_sec = n_frames / fps

    rows = []
    window_idx = 0
    start_sec = 0.0

    while start_sec + window_sec <= total_sec + 0.01:
        end_sec = start_sec + window_sec

        # Convert seconds to frame indices
        start_sample = int(round(start_sec * fps))
        end_sample = int(round(end_sec * fps))
        end_sample = min(end_sample, n_frames)
        segment = pulse[start_sample:end_sample]

        # Check NaN fraction
        nan_count = np.isnan(segment).sum()
        nan_frac = nan_count / len(segment)

        if nan_frac > max_nan_fraction:
            row = {
                "window_idx": window_idx,
                "window_start_sec": round(start_sec, 2),
                "window_end_sec": round(end_sec, 2),
                "hr_bpm_estimated": float("nan"),
                "snr": float("nan"),
                "valid": False,
            }
            if return_debug:
                row.update({
                    "peak_freq_hz": float("nan"),
                    "peak_power": float("nan"),
                    "cardiac_band_power_mean": float("nan"),
                })
        else:
            # Replace NaNs with zero for FFT (they are a small fraction)
            clean_segment = segment.copy()
            clean_segment[np.isnan(clean_segment)] = 0.0

            fft_result = compute_window_fft_hr(clean_segment, fps)

            snr_val = fft_result["snr"]
            hr_val = fft_result["hr_bpm"]

            # Validity: need a real HR estimate and sufficient SNR
            valid = (not np.isnan(hr_val) and
                     not np.isnan(snr_val) and
                     snr_val >= min_snr)

            if not valid:
                hr_val = float("nan")

            row = {
                "window_idx": window_idx,
                "window_start_sec": round(start_sec, 2),
                "window_end_sec": round(end_sec, 2),
                "hr_bpm_estimated": round(hr_val, 1) if not np.isnan(hr_val) else hr_val,
                "snr": round(snr_val, 2) if not np.isnan(snr_val) else snr_val,
                "valid": valid,
            }
            if return_debug:
                row.update({
                    "peak_freq_hz": fft_result["peak_freq_hz"],
                    "peak_power": fft_result["peak_power"],
                    "cardiac_band_power_mean": fft_result["cardiac_band_power_mean"],
                })

        rows.append(row)
        window_idx += 1
        start_sec += stride_sec

    return pd.DataFrame(rows)


# ── Temporal smoothing ───────────────────────────────────────────────

def smooth_hr_temporal(df, max_jump_bpm=None):
    """Smooth HR estimates by replacing temporal outliers with neighbor median.

    A window is considered an outlier if its HR deviates by more than
    max_jump_bpm from the median of its valid neighbors. Heart rate doesn't
    jump from 65 to 160 BPM between adjacent 10-second windows — when the
    estimate says it does, the estimate is wrong.

    Only modifies valid windows. Invalid windows stay invalid.

    Args:
        df: DataFrame from estimate_hr_windowed with columns
            [window_idx, hr_bpm_estimated, valid, ...]
        max_jump_bpm: maximum allowed deviation from neighbor median
                      (default config.TEMPORAL_MAX_JUMP_BPM)

    Returns:
        DataFrame with hr_bpm_estimated corrected for outliers.
        Adds columns: hr_bpm_raw (original), was_smoothed (bool).
    """
    max_jump = max_jump_bpm if max_jump_bpm is not None else config.TEMPORAL_MAX_JUMP_BPM

    df = df.copy()
    df["hr_bpm_raw"] = df["hr_bpm_estimated"].copy()
    df["was_smoothed"] = False

    valid_mask = df["valid"] == True
    valid_hrs = df.loc[valid_mask, "hr_bpm_estimated"].values.copy()
    valid_indices = df.index[valid_mask].tolist()

    if len(valid_hrs) < 3:
        return df

    # Two-pass approach:
    # Pass 1: identify outliers using the global median of all valid windows
    global_median = np.median(valid_hrs)
    is_outlier = np.abs(valid_hrs - global_median) > max_jump

    # Pass 2: replace outliers with median of their non-outlier neighbors
    smoothed = valid_hrs.copy()
    smoothed_flags = [False] * len(valid_hrs)

    for i in range(len(valid_hrs)):
        if not is_outlier[i]:
            continue

        # Gather non-outlier neighbors (up to 2 on each side)
        neighbors = []
        for j in range(max(0, i - 2), min(len(valid_hrs), i + 3)):
            if j != i and not is_outlier[j]:
                neighbors.append(valid_hrs[j])

        if neighbors:
            smoothed[i] = np.median(neighbors)
        else:
            # All neighbors are also outliers — use global median
            smoothed[i] = global_median
        smoothed_flags[i] = True

    # Write back
    for i, idx in enumerate(valid_indices):
        df.loc[idx, "hr_bpm_estimated"] = round(smoothed[i], 1)
        df.loc[idx, "was_smoothed"] = smoothed_flags[i]

    n_smoothed = sum(smoothed_flags)
    if n_smoothed > 0:
        print(f"  Temporal smoothing: corrected {n_smoothed} outlier window(s)")

    return df


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

    fps = 30

    # ── Test 1: 75 BPM sinusoid + noise ──────────────────────────
    print("\nTest 1: 75 BPM (1.25 Hz) sinusoid + noise")
    n = fps * 30  # 30 seconds
    t = np.arange(n) / fps
    target_hr = 75.0
    target_freq = target_hr / 60.0
    signal_clean = np.sin(2 * np.pi * target_freq * t)
    noise = np.random.randn(n) * 0.1
    signal_noisy = signal_clean + noise

    # Filter it
    filtered = bandpass_filter(signal_noisy, fps)
    check("Filtered signal has no NaN (clean input)", not np.isnan(filtered).any())

    # Estimate HR
    df = estimate_hr_windowed(filtered, fps, return_debug=True)
    valid_rows = df[df["valid"]]
    check("At least some valid windows", len(valid_rows) > 0)

    if len(valid_rows) > 0:
        mean_hr = valid_rows["hr_bpm_estimated"].mean()
        check(f"Mean estimated HR within 3 BPM of 75 (got {mean_hr:.1f})",
              abs(mean_hr - target_hr) < 3.0)

    # ── Test 2: Bandpass removes out-of-band frequencies ─────────
    print("\nTest 2: Out-of-band suppression")
    # Create signal with in-band (1.5 Hz) and out-of-band (0.1 Hz, 10 Hz)
    in_band = np.sin(2 * np.pi * 1.5 * t)
    out_low = 5.0 * np.sin(2 * np.pi * 0.1 * t)  # strong low-freq
    out_high = 5.0 * np.sin(2 * np.pi * 10.0 * t)  # strong high-freq
    mixed = in_band + out_low + out_high

    filtered_mixed = bandpass_filter(mixed, fps)

    # FFT of filtered signal — peak should be near 1.5 Hz
    fft_vals = np.abs(np.fft.rfft(filtered_mixed))
    freqs = np.fft.rfftfreq(len(filtered_mixed), d=1.0 / fps)
    band_mask = (freqs >= 0.5) & (freqs <= 4.0)
    peak_freq = freqs[band_mask][np.argmax(fft_vals[band_mask])]
    check(f"Peak after filtering near 1.5 Hz (got {peak_freq:.2f})",
          abs(peak_freq - 1.5) < 0.15)

    # Out-of-band power should be much reduced
    low_mask = freqs < 0.3
    low_power_before = np.abs(np.fft.rfft(mixed))[low_mask].sum()
    low_power_after = fft_vals[low_mask].sum()
    check("Low-freq power reduced by filtering",
          low_power_after < low_power_before * 0.1)

    # ── Test 3: NaN-heavy window invalidation ────────────────────
    print("\nTest 3: NaN-heavy window handling")
    pulse_nan = np.sin(2 * np.pi * target_freq * t)
    # Make first 10 seconds (first window) mostly NaN
    pulse_nan[:int(fps * 8)] = float("nan")  # 80% of a 10-sec window

    df_nan = estimate_hr_windowed(pulse_nan, fps)
    first_win = df_nan[df_nan["window_idx"] == 0].iloc[0]
    check("First window (80% NaN) is invalid", first_win["valid"] == False)
    check("First window HR is NaN", np.isnan(first_win["hr_bpm_estimated"]))

    # Later windows should still be valid
    later_valid = df_nan[df_nan["window_start_sec"] >= 10.0]
    check("Later windows have some valid results", later_valid["valid"].any())

    # ── Test 4: Weak peak / low SNR invalidation ─────────────────
    print("\nTest 4: Low SNR invalidation")
    # White noise — no clear peak
    noise_only = np.random.randn(n) * 1.0
    filtered_noise = bandpass_filter(noise_only, fps)

    # Use a high SNR threshold to make sure noisy windows are invalid
    df_noise = estimate_hr_windowed(filtered_noise, fps, min_snr=50.0)
    check("Noisy signal with high SNR threshold: no valid windows",
          df_noise["valid"].sum() == 0)

    # ── Test 5: Output schema ────────────────────────────────────
    print("\nTest 5: Output schema")
    df_schema = estimate_hr_windowed(filtered, fps, return_debug=False)
    expected_cols = ["window_idx", "window_start_sec", "window_end_sec",
                     "hr_bpm_estimated", "snr", "valid"]
    check("Standard columns correct", list(df_schema.columns) == expected_cols)

    df_debug = estimate_hr_windowed(filtered, fps, return_debug=True)
    expected_debug_cols = expected_cols + [
        "peak_freq_hz", "peak_power", "cardiac_band_power_mean"
    ]
    check("Debug columns correct", list(df_debug.columns) == expected_debug_cols)

    # ── Test 6: Window grid matches align.py ─────────────────────
    print("\nTest 6: Window grid alignment")
    # Verify window boundaries match config
    check("First window starts at 0.0", df_schema.iloc[0]["window_start_sec"] == 0.0)
    check("First window ends at EVAL_WINDOW_SEC",
          df_schema.iloc[0]["window_end_sec"] == config.EVAL_WINDOW_SEC)
    if len(df_schema) > 1:
        stride = df_schema.iloc[1]["window_start_sec"] - df_schema.iloc[0]["window_start_sec"]
        check(f"Window stride = {config.EVAL_STRIDE_SEC}s", stride == config.EVAL_STRIDE_SEC)

    # ── Test 7: Harmonic detection ─────────────────────────────────
    print("\nTest 7: Harmonic detection (prefers subharmonic)")
    # Create a signal with a strong 2.5 Hz component (150 BPM — a harmonic)
    # and a weaker 1.25 Hz component (75 BPM — the real cardiac frequency)
    n_harm = fps * 10
    t_harm = np.arange(n_harm) / fps
    # The harmonic is stronger than the fundamental
    fundamental = 1.0 * np.sin(2 * np.pi * 1.25 * t_harm)
    harmonic = 1.5 * np.sin(2 * np.pi * 2.5 * t_harm)
    sig_harm = fundamental + harmonic

    result_harm = compute_window_fft_hr(sig_harm, fps)
    detected_hr = result_harm["hr_bpm"]
    # With harmonic check, should prefer ~75 BPM over ~150 BPM
    check(f"Harmonic detection prefers 75 BPM over 150 (got {detected_hr:.0f})",
          abs(detected_hr - 75) < 15)

    # ── Test 8: Temporal smoothing ───────────────────────────────
    print("\nTest 8: Temporal smoothing")
    # Simulate 6 windows: mostly ~72 BPM with one catastrophic outlier
    smooth_df = pd.DataFrame({
        "window_idx": [0, 1, 2, 3, 4, 5],
        "window_start_sec": [0, 5, 10, 15, 20, 25],
        "window_end_sec": [10, 15, 20, 25, 30, 35],
        "hr_bpm_estimated": [72.0, 74.0, 150.0, 71.0, 73.0, 72.0],
        "snr": [5.0] * 6,
        "valid": [True] * 6,
    })
    smoothed = smooth_hr_temporal(smooth_df, max_jump_bpm=25.0)
    check("Outlier at window 2 was smoothed",
          smoothed.loc[2, "was_smoothed"] == True)
    check("Smoothed value near neighbors (not 150)",
          abs(smoothed.loc[2, "hr_bpm_estimated"] - 73.0) < 5)
    check("Non-outlier windows unchanged",
          smoothed.loc[0, "was_smoothed"] == False)
    check("Has hr_bpm_raw column with original",
          smoothed.loc[2, "hr_bpm_raw"] == 150.0)

    # Test with multiple outliers
    smooth_df2 = pd.DataFrame({
        "window_idx": range(8),
        "window_start_sec": [i*5 for i in range(8)],
        "window_end_sec": [i*5+10 for i in range(8)],
        "hr_bpm_estimated": [70, 72, 140, 135, 71, 73, 70, 72],
        "snr": [5.0] * 8,
        "valid": [True] * 8,
    })
    smoothed2 = smooth_hr_temporal(smooth_df2, max_jump_bpm=25.0)
    check("Both consecutive outliers smoothed",
          smoothed2.loc[2, "was_smoothed"] and smoothed2.loc[3, "was_smoothed"])

    # Test with invalid windows — should be left alone
    smooth_df3 = pd.DataFrame({
        "window_idx": [0, 1, 2],
        "window_start_sec": [0, 5, 10],
        "window_end_sec": [10, 15, 20],
        "hr_bpm_estimated": [72.0, float("nan"), 74.0],
        "snr": [5.0, float("nan"), 5.0],
        "valid": [True, False, True],
    })
    smoothed3 = smooth_hr_temporal(smooth_df3, max_jump_bpm=25.0)
    check("Invalid window stays invalid",
          smoothed3.loc[1, "valid"] == False)

    # ── Test 9: Bandpass filter NaN safety ────────────────────────
    print("\nTest 9: Bandpass filter NaN safety")
    pulse_gap = np.sin(2 * np.pi * 1.0 * t)
    pulse_gap[200:250] = float("nan")  # ~1.7 sec gap
    filtered_gap = bandpass_filter(pulse_gap, fps)
    check("NaN region preserved after filtering",
          np.isnan(filtered_gap[200:250]).all())
    check("Valid region before gap is filtered",
          not np.isnan(filtered_gap[0:190]).any())
    check("Valid region after gap is filtered",
          not np.isnan(filtered_gap[260:]).any())

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="rPPG signal processing")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    print("Use bandpass_filter() and estimate_hr_windowed() from Python.")
    print("Run with --test to verify the implementation.")


if __name__ == "__main__":
    main()
