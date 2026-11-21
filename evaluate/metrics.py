"""
Evaluation metrics for rPPG HR estimation vs Apple Watch ground truth.

Computes MAE, RMSE, Median Absolute Error, and Pearson correlation
on mutually valid evaluation windows.

Usage:
    py evaluate/metrics.py --test
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Individual metrics ───────────────────────────────────────────────

def compute_mae(estimated, ground_truth):
    """Mean Absolute Error between two arrays.

    Args:
        estimated, ground_truth: array-like of same length, no NaNs expected

    Returns:
        float MAE
    """
    est = np.asarray(estimated, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    return float(np.mean(np.abs(est - gt)))


def compute_rmse(estimated, ground_truth):
    """Root Mean Squared Error between two arrays.

    Args:
        estimated, ground_truth: array-like of same length, no NaNs expected

    Returns:
        float RMSE
    """
    est = np.asarray(estimated, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    return float(np.sqrt(np.mean((est - gt) ** 2)))


def compute_median_ae(estimated, ground_truth):
    """Median Absolute Error between two arrays.

    More robust to outlier windows than MAE or RMSE.

    Args:
        estimated, ground_truth: array-like of same length, no NaNs expected

    Returns:
        float Median AE
    """
    est = np.asarray(estimated, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    return float(np.median(np.abs(est - gt)))


def compute_pearson(estimated, ground_truth):
    """Pearson correlation coefficient and p-value (from scipy.stats.pearsonr).

    Returns (NaN, NaN) if:
        - fewer than 2 data points
        - either series is constant (zero variance)

    Args:
        estimated, ground_truth: array-like of same length, no NaNs expected

    Returns:
        (pearson_r, pearson_p) tuple of floats
    """
    est = np.asarray(estimated, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)

    if len(est) < 2:
        return (float("nan"), float("nan"))

    if np.std(est) == 0 or np.std(gt) == 0:
        warnings.warn("Pearson: constant series detected, returning NaN")
        return (float("nan"), float("nan"))

    r, p = pearsonr(est, gt)
    return (float(r), float(p))


# ── Window merging ───────────────────────────────────────────────────

def merge_estimate_and_ground_truth(est_df, gt_df):
    """Merge estimated HR windows with ground-truth HR windows.

    Joins on window_idx and validates that window boundaries match.

    Args:
        est_df: DataFrame from signal_processing.estimate_hr_windowed()
            Required columns: window_idx, window_start_sec, window_end_sec,
                              hr_bpm_estimated, valid
        gt_df: DataFrame from align.build_windows()
            Required columns: window_idx, window_start_sec, window_end_sec,
                              hr_bpm_median, valid

    Returns:
        Merged DataFrame with columns:
            window_idx, window_start_sec, window_end_sec,
            hr_bpm_estimated, est_valid, hr_bpm_gt, gt_valid

    Raises:
        ValueError if window grids do not match.
    """
    # Find common window indices
    common_idx = set(est_df["window_idx"]) & set(gt_df["window_idx"])
    if len(common_idx) == 0:
        raise ValueError("No overlapping window indices between estimate and ground truth")

    est = est_df[est_df["window_idx"].isin(common_idx)].copy()
    gt = gt_df[gt_df["window_idx"].isin(common_idx)].copy()

    # Sort both by window_idx for alignment
    est = est.sort_values("window_idx").reset_index(drop=True)
    gt = gt.sort_values("window_idx").reset_index(drop=True)

    # Validate that window boundaries match
    start_match = np.allclose(
        est["window_start_sec"].values, gt["window_start_sec"].values, atol=0.1
    )
    end_match = np.allclose(
        est["window_end_sec"].values, gt["window_end_sec"].values, atol=0.1
    )
    if not start_match or not end_match:
        raise ValueError(
            "Window boundaries do not match between estimate and ground truth. "
            "Ensure both use the same config.EVAL_WINDOW_SEC and config.EVAL_STRIDE_SEC."
        )

    merged = pd.DataFrame({
        "window_idx": est["window_idx"].values,
        "window_start_sec": est["window_start_sec"].values,
        "window_end_sec": est["window_end_sec"].values,
        "hr_bpm_estimated": est["hr_bpm_estimated"].values,
        "est_valid": est["valid"].values,
        "hr_bpm_gt": gt["hr_bpm_median"].values,
        "gt_valid": gt["valid"].values,
    })

    return merged


# ── Session evaluation ───────────────────────────────────────────────

def evaluate_session(est_df, gt_df, return_merged=False):
    """Evaluate estimated HR against ground truth for one session.

    Only evaluates windows where both estimate and ground truth are valid
    and both HR values are non-null.

    Args:
        est_df: DataFrame from signal_processing.estimate_hr_windowed()
        gt_df: DataFrame from align.build_windows()
        return_merged: if True, also return the merged per-window DataFrame

    Returns:
        metrics: dict with keys:
            mae, rmse, median_ae, pearson_r, pearson_p,
            n_total_windows, n_mutually_valid_windows,
            n_skipped_gt_invalid, n_skipped_est_invalid,
            n_skipped_missing_values
        merged_df (optional): the per-window DataFrame used for evaluation
    """
    merged = merge_estimate_and_ground_truth(est_df, gt_df)

    n_total = len(merged)
    n_gt_invalid = (~merged["gt_valid"]).sum()
    n_est_invalid = (~merged["est_valid"]).sum()

    # Filter to mutually valid
    both_valid = merged["gt_valid"] & merged["est_valid"]
    valid_subset = merged[both_valid].copy()

    # Further filter out any remaining NaN values
    has_values = (
        valid_subset["hr_bpm_estimated"].notna() &
        valid_subset["hr_bpm_gt"].notna()
    )
    n_missing_values = (~has_values).sum()
    eval_subset = valid_subset[has_values]

    n_mutually_valid = len(eval_subset)

    if n_mutually_valid < 2:
        metrics = {
            "mae": float("nan"),
            "rmse": float("nan"),
            "median_ae": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "n_total_windows": int(n_total),
            "n_mutually_valid_windows": int(n_mutually_valid),
            "n_skipped_gt_invalid": int(n_gt_invalid),
            "n_skipped_est_invalid": int(n_est_invalid),
            "n_skipped_missing_values": int(n_missing_values),
        }
        if n_mutually_valid < 2:
            warnings.warn(
                f"Only {n_mutually_valid} mutually valid windows — "
                f"not enough for reliable metrics"
            )
    else:
        est_vals = eval_subset["hr_bpm_estimated"].values
        gt_vals = eval_subset["hr_bpm_gt"].values

        r, p = compute_pearson(est_vals, gt_vals)

        metrics = {
            "mae": round(compute_mae(est_vals, gt_vals), 2),
            "rmse": round(compute_rmse(est_vals, gt_vals), 2),
            "median_ae": round(compute_median_ae(est_vals, gt_vals), 2),
            "pearson_r": round(r, 4) if not np.isnan(r) else r,
            "pearson_p": round(p, 6) if not np.isnan(p) else p,
            "n_total_windows": int(n_total),
            "n_mutually_valid_windows": int(n_mutually_valid),
            "n_skipped_gt_invalid": int(n_gt_invalid),
            "n_skipped_est_invalid": int(n_est_invalid),
            "n_skipped_missing_values": int(n_missing_values),
        }

    if return_merged:
        return metrics, merged
    return metrics


# ── Tests ────────────────────────────────────────────────────────────

def _make_gt_df(hr_values, valid=None):
    """Helper: create a ground-truth-like DataFrame."""
    n = len(hr_values)
    if valid is None:
        valid = [True] * n
    return pd.DataFrame({
        "window_idx": range(n),
        "window_start_sec": [i * 5.0 for i in range(n)],
        "window_end_sec": [i * 5.0 + 10.0 for i in range(n)],
        "hr_bpm_median": hr_values,
        "hr_sample_count": [3] * n,
        "valid": valid,
    })


def _make_est_df(hr_values, valid=None):
    """Helper: create an estimate-like DataFrame."""
    n = len(hr_values)
    if valid is None:
        valid = [True] * n
    return pd.DataFrame({
        "window_idx": range(n),
        "window_start_sec": [i * 5.0 for i in range(n)],
        "window_end_sec": [i * 5.0 + 10.0 for i in range(n)],
        "hr_bpm_estimated": hr_values,
        "snr": [5.0] * n,
        "valid": valid,
    })


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

    # ── Test 1: Perfect prediction ───────────────────────────────
    print("\nTest 1: Perfect prediction")
    gt = _make_gt_df([70, 72, 74, 76, 78])
    est = _make_est_df([70, 72, 74, 76, 78])
    m = evaluate_session(est, gt)
    check("MAE = 0", m["mae"] == 0.0)
    check("RMSE = 0", m["rmse"] == 0.0)
    check("Median AE = 0", m["median_ae"] == 0.0)
    check("Pearson r = 1.0", m["pearson_r"] == 1.0)
    check("All 5 windows valid", m["n_mutually_valid_windows"] == 5)

    # ── Test 2: Constant offset ──────────────────────────────────
    print("\nTest 2: Constant +5 BPM offset")
    est_off = _make_est_df([75, 77, 79, 81, 83])
    m2 = evaluate_session(est_off, gt)
    check("MAE = 5.0", m2["mae"] == 5.0)
    check("RMSE = 5.0", m2["rmse"] == 5.0)
    check("Median AE = 5.0", m2["median_ae"] == 5.0)
    check("Pearson r = 1.0 (perfect correlation despite offset)",
          m2["pearson_r"] == 1.0)

    # ── Test 3: Random prediction ────────────────────────────────
    print("\nTest 3: Random prediction")
    np.random.seed(42)
    gt_rand = _make_gt_df([70, 72, 74, 76, 78, 70, 72, 74, 76, 78])
    est_rand = _make_est_df(list(np.random.uniform(50, 100, 10)))
    m3 = evaluate_session(est_rand, gt_rand)
    check("MAE > 0 for random", m3["mae"] > 0)
    check("RMSE >= MAE", m3["rmse"] >= m3["mae"])
    check("Pearson r is finite", not np.isnan(m3["pearson_r"]))

    # ── Test 4: Partial validity counting ────────────────────────
    print("\nTest 4: Partial validity")
    gt_part = _make_gt_df(
        [70, 72, float("nan"), 76, 78],
        valid=[True, True, False, True, True]
    )
    est_part = _make_est_df(
        [70, float("nan"), 74, 76, 78],
        valid=[True, False, True, True, True]
    )
    m4, merged4 = evaluate_session(est_part, gt_part, return_merged=True)
    check("Total windows = 5", m4["n_total_windows"] == 5)
    check("GT invalid = 1", m4["n_skipped_gt_invalid"] == 1)
    check("Est invalid = 1", m4["n_skipped_est_invalid"] == 1)
    # Mutually valid: windows 0, 3, 4 (window 1 est invalid, window 2 gt invalid)
    check("Mutually valid = 3", m4["n_mutually_valid_windows"] == 3)
    check("Merged DataFrame returned", len(merged4) == 5)

    # ── Test 5: Insufficient windows ─────────────────────────────
    print("\nTest 5: Insufficient windows (< 2 valid)")
    gt_one = _make_gt_df([70], valid=[True])
    est_one = _make_est_df([72], valid=[True])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m5 = evaluate_session(est_one, gt_one)
        check("Warning issued for insufficient windows", len(w) > 0)
    check("MAE is NaN", np.isnan(m5["mae"]))
    check("RMSE is NaN", np.isnan(m5["rmse"]))
    check("Pearson r is NaN", np.isnan(m5["pearson_r"]))
    check("Mutually valid = 1", m5["n_mutually_valid_windows"] == 1)

    # ── Test 6: Constant series Pearson edge case ────────────────
    print("\nTest 6: Constant series Pearson")
    gt_const = _make_gt_df([72, 72, 72, 72, 72])
    est_const = _make_est_df([75, 75, 75, 75, 75])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m6 = evaluate_session(est_const, gt_const)
        check("Warning for constant series", len(w) > 0)
    check("Pearson r is NaN for constant series", np.isnan(m6["pearson_r"]))
    check("MAE still computed (= 3.0)", m6["mae"] == 3.0)

    # ── Test 7: Mismatched window grid ───────────────────────────
    print("\nTest 7: Mismatched window grid")
    gt_ok = _make_gt_df([70, 72, 74])
    est_bad = pd.DataFrame({
        "window_idx": [0, 1, 2],
        "window_start_sec": [0.0, 7.0, 14.0],  # wrong stride
        "window_end_sec": [10.0, 17.0, 24.0],
        "hr_bpm_estimated": [70, 72, 74],
        "snr": [5.0, 5.0, 5.0],
        "valid": [True, True, True],
    })
    try:
        evaluate_session(est_bad, gt_ok)
        check("Mismatched grid raises error", False)
    except ValueError:
        check("Mismatched grid raises error", True)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="rPPG evaluation metrics")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    print("Use evaluate_session() from Python.")
    print("Run with --test to verify the implementation.")


if __name__ == "__main__":
    main()
