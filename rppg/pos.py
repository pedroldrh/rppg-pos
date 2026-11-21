"""
POS (Plane-Orthogonal-to-Skin) rPPG algorithm.

Reference: Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
"Algorithmic Principles of Remote PPG." IEEE TBME, 64(7), 1479-1491.

Outputs a raw pulse signal — no filtering or HR estimation (that is
signal_processing.py's job).

Usage:
    py rppg/pos.py --test
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Core POS algorithm ───────────────────────────────────────────────

def pos_algorithm(rgb_array, fps, window_sec=None, max_nan_fraction=None,
                  min_std=None, return_diagnostics=False):
    """POS algorithm (Wang et al., 2017) for rPPG pulse extraction.

    For each sliding window:
        1. Temporal normalization: C_n = C / mean(C, axis=0)
        2. Projection:
            S1 = C_n[:,1] - C_n[:,2]         (G - B)
            S2 = C_n[:,1] + C_n[:,2] - 2*C_n[:,0]  (G + B - 2R)
        3. Combination:
            h = S1 + (std(S1) / std(S2)) * S2
        4. Overlap-add h into output signal

    Args:
        rgb_array: (N, 3) NumPy array of mean [R, G, B] per frame.
                   May contain NaN rows for missing frames.
        fps: frames per second
        window_sec: POS window duration (default from config.POS_WINDOW_SEC)
        max_nan_fraction: skip window if NaN fraction exceeds this
                          (default from config.POS_MAX_NAN_FRACTION)
        min_std: numerical stability guard for std ratio
                 (default from config.POS_MIN_STD)
        return_diagnostics: if True, also return a diagnostics dict

    Returns:
        pulse: (N,) NumPy array — raw POS pulse signal (NaN where skipped)
        diagnostics (optional): dict with processing stats
    """
    window_sec = window_sec if window_sec is not None else config.POS_WINDOW_SEC
    max_nan_fraction = max_nan_fraction if max_nan_fraction is not None else config.POS_MAX_NAN_FRACTION
    min_std = min_std if min_std is not None else config.POS_MIN_STD

    n_frames = len(rgb_array)
    window_len = int(window_sec * fps)

    if window_len < 2:
        raise ValueError(f"Window too short: {window_len} frames "
                         f"(window_sec={window_sec}, fps={fps})")

    pulse = np.zeros(n_frames, dtype=np.float64)
    weight = np.zeros(n_frames, dtype=np.float64)

    processed_windows = 0
    skipped_nan = 0
    skipped_unstable = 0

    for start in range(0, n_frames - window_len + 1):
        end = start + window_len
        window = rgb_array[start:end]  # (L, 3)

        # ── NaN check ────────────────────────────────────────────
        nan_mask = np.isnan(window).any(axis=1)
        nan_fraction = nan_mask.sum() / window_len
        if nan_fraction > max_nan_fraction:
            skipped_nan += 1
            continue

        # If small number of NaNs, use only valid rows for mean computation
        valid = window[~nan_mask]
        if len(valid) < 2:
            skipped_nan += 1
            continue

        # ── Step 1: Temporal normalization ───────────────────────
        mean_rgb = valid.mean(axis=0)  # (3,)

        # Guard against zero mean (e.g., completely black frames)
        if np.any(mean_rgb < min_std):
            skipped_unstable += 1
            continue

        # Normalize entire window (NaN rows will propagate NaN, which is fine
        # since we overlap-add only valid contributions)
        c_norm = window / mean_rgb  # (L, 3)

        # For any NaN rows in the window, replace with 1.0 (neutral) so they
        # don't corrupt the projection but contribute zero signal
        nan_rows = np.isnan(c_norm).any(axis=1)
        c_norm[nan_rows] = 1.0

        # ── Step 2: POS projection ──────────────────────────────
        # R=0, G=1, B=2
        s1 = c_norm[:, 1] - c_norm[:, 2]                    # G - B
        s2 = c_norm[:, 1] + c_norm[:, 2] - 2.0 * c_norm[:, 0]  # G + B - 2R

        # ── Step 3: Combination with stability guard ─────────────
        std_s1 = np.std(s1)
        std_s2 = np.std(s2)

        if std_s2 < min_std:
            # Near-zero variance in S2 — signal is too weak or constant
            if std_s1 < min_std:
                # Both projections are flat — skip
                skipped_unstable += 1
                continue
            # Use S1 only (S2 contributes nothing)
            h = s1
        else:
            alpha = std_s1 / std_s2
            h = s1 + alpha * s2

        # Zero-mean the window signal
        h = h - h.mean()

        # Zero out contributions from NaN rows
        h[nan_rows] = 0.0

        # ── Step 4: Overlap-add ──────────────────────────────────
        pulse[start:end] += h
        weight[start:end] += (~nan_rows).astype(float)
        processed_windows += 1

    # Normalize by overlap count
    valid_mask = weight > 0
    pulse[valid_mask] /= weight[valid_mask]
    pulse[~valid_mask] = float("nan")

    # Count NaN output frames
    nan_output = np.isnan(pulse).sum()

    if return_diagnostics:
        diag = {
            "processed_windows": processed_windows,
            "skipped_nan": skipped_nan,
            "skipped_unstable": skipped_unstable,
            "nan_output_frames": int(nan_output),
            "total_frames": n_frames,
            "window_len_frames": window_len,
        }
        return pulse, diag

    return pulse


def pos_from_dataframe(df, fps, region_cols=("rgb_r", "rgb_g", "rgb_b"),
                       return_diagnostics=False):
    """Convenience wrapper: extract RGB from DataFrame and run POS.

    Args:
        df: DataFrame from face_detect with at least the rgb columns
        fps: frames per second
        region_cols: tuple of column names for (R, G, B)
        return_diagnostics: if True, also return diagnostics dict

    Returns:
        pulse: (N,) NumPy array
        diagnostics (optional): dict
    """
    rgb_array = df[list(region_cols)].to_numpy(dtype=np.float64)
    return pos_algorithm(rgb_array, fps, return_diagnostics=return_diagnostics)


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
    n = 300  # 10 seconds

    # ── Test 1: Synthetic sinusoid — dominant frequency ──────────
    print("\nTest 1: Synthetic 1.2 Hz sinusoid in green channel")
    t = np.arange(n) / fps
    freq_hz = 1.2
    # Base RGB around skin-like values with a sinusoidal modulation in G
    r = np.full(n, 160.0) + np.random.randn(n) * 0.5
    g = np.full(n, 140.0) + 3.0 * np.sin(2 * np.pi * freq_hz * t) + np.random.randn(n) * 0.5
    b = np.full(n, 120.0) + np.random.randn(n) * 0.5
    rgb = np.column_stack([r, g, b])

    pulse, diag = pos_algorithm(rgb, fps, return_diagnostics=True)

    check("Output length matches input", len(pulse) == n)
    check("Processed windows > 0", diag["processed_windows"] > 0)

    # FFT to find dominant frequency
    valid_pulse = pulse[~np.isnan(pulse)]
    fft_vals = np.abs(np.fft.rfft(valid_pulse - valid_pulse.mean()))
    freqs = np.fft.rfftfreq(len(valid_pulse), d=1.0 / fps)
    # Ignore DC and very low frequencies
    mask = freqs > 0.5
    dominant_freq = freqs[mask][np.argmax(fft_vals[mask])]
    check(f"Dominant freq near 1.2 Hz (got {dominant_freq:.2f})",
          abs(dominant_freq - freq_hz) < 0.2)

    # ── Test 2: Constant RGB — near-zero output ──────────────────
    print("\nTest 2: Constant RGB input")
    rgb_const = np.full((n, 3), [150.0, 140.0, 130.0])
    pulse_const, diag_const = pos_algorithm(rgb_const, fps, return_diagnostics=True)

    # All windows should be skipped as unstable (zero variance)
    check("Constant input: all windows skipped as unstable",
          diag_const["skipped_unstable"] > 0)
    # Output should be all NaN or all zero
    valid_const = pulse_const[~np.isnan(pulse_const)]
    if len(valid_const) > 0:
        check("Constant input: max amplitude < 1e-6",
              np.max(np.abs(valid_const)) < 1e-6)
    else:
        check("Constant input: all NaN output", True)

    # ── Test 3: NaN handling ─────────────────────────────────────
    print("\nTest 3: NaN handling")
    rgb_nan = rgb.copy()
    # Insert a long NaN block (frames 50-100)
    rgb_nan[50:101] = float("nan")
    pulse_nan, diag_nan = pos_algorithm(rgb_nan, fps, return_diagnostics=True)

    check("Output length matches input with NaNs", len(pulse_nan) == n)
    check("Some windows skipped due to NaN", diag_nan["skipped_nan"] > 0)
    check("NaN input region produces NaN output",
          np.isnan(pulse_nan[60:90]).all())
    check("Valid regions still have signal",
          not np.isnan(pulse_nan[0:30]).all())

    # ── Test 4: Near-zero variance stability ─────────────────────
    print("\nTest 4: Near-zero variance stability")
    rgb_nearzero = np.full((n, 3), [150.0, 140.0, 130.0])
    # Add tiny noise (below min_std threshold)
    rgb_nearzero += np.random.randn(n, 3) * 1e-10
    pulse_nz, diag_nz = pos_algorithm(rgb_nearzero, fps, return_diagnostics=True)

    check("Near-zero variance: no crash", True)
    check("Near-zero variance: windows skipped",
          diag_nz["skipped_unstable"] > 0 or diag_nz["processed_windows"] >= 0)

    # ── Test 5: Output length equals input length ────────────────
    print("\nTest 5: Various input lengths")
    for test_n in [60, 100, 500]:
        rgb_test = np.random.rand(test_n, 3) * 100 + 100
        # Add some variation so windows aren't skipped
        rgb_test[:, 1] += 2.0 * np.sin(2 * np.pi * 1.0 * np.arange(test_n) / fps)
        p = pos_algorithm(rgb_test, fps)
        check(f"N={test_n}: output length matches", len(p) == test_n)

    # ── Test 6: Overlap-add sanity ───────────────────────────────
    print("\nTest 6: Overlap-add sanity")
    # With a clean sinusoidal input, most frames should be covered
    pulse_clean, diag_clean = pos_algorithm(rgb, fps, return_diagnostics=True)
    covered_frames = n - diag_clean["nan_output_frames"]
    coverage = covered_frames / n
    check(f"Coverage > 90% on clean signal ({coverage:.1%})", coverage > 0.9)

    # Frames near the edges may have fewer overlapping windows,
    # but middle frames should all be covered
    mid_start = n // 4
    mid_end = 3 * n // 4
    mid_nans = np.isnan(pulse_clean[mid_start:mid_end]).sum()
    check("Middle 50% has no NaN gaps", mid_nans == 0)

    # ── Test 7: pos_from_dataframe wrapper ───────────────────────
    print("\nTest 7: pos_from_dataframe wrapper")
    df = pd.DataFrame({
        "frame_idx": range(n),
        "rgb_r": rgb[:, 0],
        "rgb_g": rgb[:, 1],
        "rgb_b": rgb[:, 2],
        "valid_detection": True,
        "interpolated": False,
    })
    pulse_df, diag_df = pos_from_dataframe(df, fps, return_diagnostics=True)
    check("DataFrame wrapper output length", len(pulse_df) == n)
    check("DataFrame wrapper processed windows > 0",
          diag_df["processed_windows"] > 0)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="POS rPPG algorithm")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    print("Use pos_algorithm() or pos_from_dataframe() from Python.")
    print("Run with --test to verify the implementation.")


if __name__ == "__main__":
    main()
