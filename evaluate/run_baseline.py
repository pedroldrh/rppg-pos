"""
Baseline evaluation orchestrator.

Discovers sessions, runs the full rPPG pipeline on each, and produces
a JSON report + flat CSV summary grouped by lighting condition.

Usage:
    py evaluate/run_baseline.py --data-dir data/raw
    py evaluate/run_baseline.py --data-dir data/raw --region combined --save-intermediates
    py evaluate/run_baseline.py --data-dir data/raw --list-sessions
    py evaluate/run_baseline.py --test
"""

import argparse
import json
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from rppg.face_detect import extract_rgb_signals
from rppg.pos import pos_from_dataframe
from rppg.signal_processing import bandpass_filter, estimate_hr_windowed, smooth_hr_temporal
from evaluate.metrics import evaluate_session


# ── Session discovery and validation ─────────────────────────────────

def discover_sessions(data_dir):
    """Find all session directories under data_dir.

    A session directory is any folder matching session_* that contains
    metadata.json.

    Returns:
        List of absolute paths to session directories, sorted by name.
    """
    sessions = []
    if not os.path.isdir(data_dir):
        return sessions
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path) and name.startswith("session_"):
            sessions.append(path)
    return sessions


def find_video_file(session_dir):
    """Find the video file in a session directory (video.avi or video.mp4).

    Returns the path if found, None otherwise.
    """
    for ext in ["video.avi", "video.mp4"]:
        path = os.path.join(session_dir, ext)
        if os.path.isfile(path):
            return path
    return None


def validate_session_files(session_dir):
    """Check that a session directory has all required files.

    Required: metadata.json, video file (.avi or .mp4), windows.csv

    Returns:
        (is_valid, missing_files) tuple
    """
    missing = []
    if not os.path.isfile(os.path.join(session_dir, "metadata.json")):
        missing.append("metadata.json")
    if find_video_file(session_dir) is None:
        missing.append("video.avi/.mp4")
    if not os.path.isfile(os.path.join(session_dir, "windows.csv")):
        missing.append("windows.csv")
    return (len(missing) == 0, missing)


# ── Per-session pipeline ─────────────────────────────────────────────

def run_session_pipeline(session_dir, region="forehead", save_intermediates=False):
    """Run the full rPPG pipeline on one session.

    Steps:
        1. Extract RGB signals from video (face_detect)
        2. Run POS algorithm (pos)
        3. Bandpass filter + windowed HR estimation (signal_processing)
        4. Load ground-truth windows.csv
        5. Evaluate metrics

    Args:
        session_dir: path to session directory
        region: ROI region for face detection
        save_intermediates: if True, save RGB/pulse/HR CSVs to data/processed/

    Returns:
        dict with keys:
            session, condition, region, metrics, pos_diagnostics,
            detection_rate, intermediate_paths, success, error
    """
    session_name = os.path.basename(session_dir)
    result = {
        "session": session_name,
        "condition": None,
        "region": region,
        "metrics": None,
        "pos_diagnostics": None,
        "detection_rate": None,
        "intermediate_paths": {},
        "success": False,
        "error": None,
    }

    # Load metadata
    with open(os.path.join(session_dir, "metadata.json")) as f:
        meta = json.load(f)
    result["condition"] = meta.get("condition", "unknown")
    fps = meta.get("fps_actual", config.FPS)

    # Step 1: Face detection + RGB extraction
    video_path = find_video_file(session_dir)
    rgb_df = extract_rgb_signals(video_path, region=region)

    detection_rate = rgb_df["valid_detection"].mean()
    result["detection_rate"] = round(float(detection_rate), 4)

    # Step 2: POS
    pulse, pos_diag = pos_from_dataframe(rgb_df, fps, return_diagnostics=True)
    result["pos_diagnostics"] = pos_diag

    # Step 3: Bandpass filter + windowed HR
    filtered = bandpass_filter(pulse, fps)
    est_df = estimate_hr_windowed(filtered, fps)

    # Step 3b: Temporal smoothing (fixes catastrophic outlier windows)
    if config.TEMPORAL_SMOOTH:
        est_df = smooth_hr_temporal(est_df)

    # Step 4: Load ground-truth windows
    gt_df = pd.read_csv(os.path.join(session_dir, "windows.csv"))

    # Step 5: Evaluate
    metrics = evaluate_session(est_df, gt_df)
    result["metrics"] = metrics
    result["success"] = True

    # Save intermediates if requested
    if save_intermediates:
        out_dir = os.path.join(config.DATA_PROCESSED, session_name)
        os.makedirs(out_dir, exist_ok=True)

        rgb_path = os.path.join(out_dir, "rgb_signals.csv")
        rgb_df.to_csv(rgb_path, index=False)
        result["intermediate_paths"]["rgb_signals"] = rgb_path

        pulse_path = os.path.join(out_dir, "pulse.csv")
        pd.DataFrame({"pulse_raw": pulse, "pulse_filtered": filtered}).to_csv(
            pulse_path, index=False
        )
        result["intermediate_paths"]["pulse"] = pulse_path

        est_path = os.path.join(out_dir, "hr_estimated.csv")
        est_df.to_csv(est_path, index=False)
        result["intermediate_paths"]["hr_estimated"] = est_path

    return result


# ── Aggregation ──────────────────────────────────────────────────────

def aggregate_results(session_results):
    """Aggregate per-session results by condition.

    Returns:
        dict keyed by condition, each with:
            n_sessions, total_mutually_valid_windows,
            mae_mean, mae_median, rmse_mean, rmse_median,
            median_ae_mean, median_ae_median,
            pearson_r_mean, pearson_r_median
    """
    by_condition = {}
    for r in session_results:
        if not r["success"] or r["metrics"] is None:
            continue
        cond = r["condition"]
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(r["metrics"])

    summary = {}
    for cond, metrics_list in by_condition.items():
        maes = [m["mae"] for m in metrics_list if not np.isnan(m["mae"])]
        rmses = [m["rmse"] for m in metrics_list if not np.isnan(m["rmse"])]
        med_aes = [m["median_ae"] for m in metrics_list if not np.isnan(m["median_ae"])]
        rs = [m["pearson_r"] for m in metrics_list if not np.isnan(m["pearson_r"])]
        total_valid = sum(m["n_mutually_valid_windows"] for m in metrics_list)

        summary[cond] = {
            "n_sessions": len(metrics_list),
            "total_mutually_valid_windows": total_valid,
            "mae_mean": round(float(np.mean(maes)), 2) if maes else float("nan"),
            "mae_median": round(float(np.median(maes)), 2) if maes else float("nan"),
            "rmse_mean": round(float(np.mean(rmses)), 2) if rmses else float("nan"),
            "rmse_median": round(float(np.median(rmses)), 2) if rmses else float("nan"),
            "median_ae_mean": round(float(np.mean(med_aes)), 2) if med_aes else float("nan"),
            "median_ae_median": round(float(np.median(med_aes)), 2) if med_aes else float("nan"),
            "pearson_r_mean": round(float(np.mean(rs)), 4) if rs else float("nan"),
            "pearson_r_median": round(float(np.median(rs)), 4) if rs else float("nan"),
        }

    return summary


def results_to_csv_rows(session_results):
    """Flatten session results into rows for the summary CSV.

    Returns list of dicts ready for pd.DataFrame.
    """
    rows = []
    for r in session_results:
        m = r["metrics"] or {}
        pd_diag = r["pos_diagnostics"] or {}
        rows.append({
            "session": r["session"],
            "condition": r["condition"],
            "region": r["region"],
            "mae": m.get("mae", float("nan")),
            "rmse": m.get("rmse", float("nan")),
            "median_ae": m.get("median_ae", float("nan")),
            "pearson_r": m.get("pearson_r", float("nan")),
            "n_mutually_valid_windows": m.get("n_mutually_valid_windows", 0),
            "detection_rate": r["detection_rate"],
            "pos_processed_windows": pd_diag.get("processed_windows", 0),
            "pos_skipped_windows": (
                pd_diag.get("skipped_nan", 0) + pd_diag.get("skipped_unstable", 0)
            ),
            "success": r["success"],
            "error": r["error"],
        })
    return rows


# ── Report saving ────────────────────────────────────────────────────

def save_reports(session_results, summary, output_dir):
    """Save JSON report and CSV summary to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON report
    report = {
        "sessions": session_results,
        "summary": summary,
    }
    json_path = os.path.join(output_dir, "baseline_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  JSON report: {json_path}")

    # CSV summary
    csv_rows = results_to_csv_rows(session_results)
    csv_df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(output_dir, "baseline_summary.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"  CSV summary: {csv_path}")


# ── Console output ───────────────────────────────────────────────────

def print_summary(session_results, summary):
    """Print a clean summary table to console."""
    n_total = len(session_results)
    n_success = sum(1 for r in session_results if r["success"])
    n_failed = n_total - n_success

    print(f"\n{'='*60}")
    print(f"Baseline Evaluation Summary")
    print(f"{'='*60}")
    print(f"Sessions: {n_total} discovered, {n_success} succeeded, {n_failed} failed")

    if n_failed > 0:
        print(f"\nFailed sessions:")
        for r in session_results:
            if not r["success"]:
                print(f"  {r['session']}: {r['error']}")

    for cond, s in sorted(summary.items()):
        print(f"\n--- {cond.upper()} ({s['n_sessions']} sessions, "
              f"{s['total_mutually_valid_windows']} valid windows) ---")
        print(f"  MAE:       mean={s['mae_mean']:.2f}  median={s['mae_median']:.2f}")
        print(f"  RMSE:      mean={s['rmse_mean']:.2f}  median={s['rmse_median']:.2f}")
        print(f"  Median AE: mean={s['median_ae_mean']:.2f}  median={s['median_ae_median']:.2f}")
        print(f"  Pearson r: mean={s['pearson_r_mean']:.4f}  median={s['pearson_r_median']:.4f}")

    print(f"\n{'='*60}")


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

    # ── Test 1: Session discovery ────────────────────────────────
    print("\nTest 1: Session discovery")
    with tempfile.TemporaryDirectory() as tmp:
        # Create mock sessions
        for name in ["session_001", "session_002", "not_a_session", "session_003"]:
            d = os.path.join(tmp, name)
            os.makedirs(d)
        sessions = discover_sessions(tmp)
        check("Finds 3 session dirs", len(sessions) == 3)
        check("Sorted order", os.path.basename(sessions[0]) == "session_001")
        check("Ignores non-session dirs", all("session_" in os.path.basename(s) for s in sessions))

    # ── Test 2: File validation ──────────────────────────────────
    print("\nTest 2: File validation")
    with tempfile.TemporaryDirectory() as tmp:
        sess = os.path.join(tmp, "session_test")
        os.makedirs(sess)

        # Missing all files
        valid, missing = validate_session_files(sess)
        check("Missing all 3 files", not valid and len(missing) == 3)

        # Add some files
        for f in ["metadata.json", "video.avi"]:
            open(os.path.join(sess, f), "w").close()
        valid2, missing2 = validate_session_files(sess)
        check("Missing windows.csv only", not valid2 and "windows.csv" in missing2)

        # Add last file
        open(os.path.join(sess, "windows.csv"), "w").close()
        valid3, missing3 = validate_session_files(sess)
        check("All files present", valid3 and len(missing3) == 0)

    # ── Test 3: Empty discovery ──────────────────────────────────
    print("\nTest 3: Edge cases")
    check("Non-existent dir returns empty", discover_sessions("/nonexistent/path") == [])

    # ── Test 4: Aggregation logic ────────────────────────────────
    print("\nTest 4: Aggregation")
    mock_results = [
        {"success": True, "condition": "normal", "metrics": {
            "mae": 3.0, "rmse": 4.0, "median_ae": 2.5, "pearson_r": 0.9,
            "n_mutually_valid_windows": 5,
        }},
        {"success": True, "condition": "normal", "metrics": {
            "mae": 5.0, "rmse": 6.0, "median_ae": 4.0, "pearson_r": 0.8,
            "n_mutually_valid_windows": 4,
        }},
        {"success": True, "condition": "low_light", "metrics": {
            "mae": 10.0, "rmse": 12.0, "median_ae": 9.0, "pearson_r": 0.5,
            "n_mutually_valid_windows": 3,
        }},
        {"success": False, "condition": "low_light", "metrics": None},
    ]
    summary = aggregate_results(mock_results)
    check("Two conditions in summary", len(summary) == 2)
    check("Normal: 2 sessions", summary["normal"]["n_sessions"] == 2)
    check("Normal: MAE mean = 4.0", summary["normal"]["mae_mean"] == 4.0)
    check("Normal: MAE median = 4.0", summary["normal"]["mae_median"] == 4.0)
    check("Normal: total valid windows = 9", summary["normal"]["total_mutually_valid_windows"] == 9)
    check("Low light: 1 session (failed one excluded)", summary["low_light"]["n_sessions"] == 1)

    # ── Test 5: CSV row flattening ───────────────────────────────
    print("\nTest 5: CSV row flattening")
    mock_full = [
        {
            "session": "session_001", "condition": "normal", "region": "forehead",
            "metrics": {"mae": 3.0, "rmse": 4.0, "median_ae": 2.5,
                        "pearson_r": 0.9, "n_mutually_valid_windows": 5},
            "pos_diagnostics": {"processed_windows": 250, "skipped_nan": 2, "skipped_unstable": 1},
            "detection_rate": 0.95, "success": True, "error": None,
        },
    ]
    rows = results_to_csv_rows(mock_full)
    check("One CSV row", len(rows) == 1)
    check("CSV row has mae", rows[0]["mae"] == 3.0)
    check("CSV row has pos_skipped_windows", rows[0]["pos_skipped_windows"] == 3)
    check("CSV row has detection_rate", rows[0]["detection_rate"] == 0.95)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run baseline rPPG evaluation")
    parser.add_argument("--data-dir", default=config.DATA_RAW,
                        help="Directory containing session folders")
    parser.add_argument("--region", choices=["forehead", "cheeks", "combined"],
                        default="forehead", help="ROI region (default: forehead)")
    parser.add_argument("--output", default=config.DATA_RESULTS,
                        help="Output directory for reports")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Save RGB, pulse, and HR CSVs per session")
    parser.add_argument("--list-sessions", action="store_true",
                        help="Discover sessions and check files, then exit")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    # Discover sessions
    sessions = discover_sessions(args.data_dir)
    print(f"Discovered {len(sessions)} session(s) in {args.data_dir}")

    if len(sessions) == 0:
        print("No sessions found. Record data first with collect/record_video.py")
        return

    # List mode
    if args.list_sessions:
        for s in sessions:
            valid, missing = validate_session_files(s)
            status = "READY" if valid else f"MISSING: {', '.join(missing)}"
            name = os.path.basename(s)
            # Load condition if metadata exists
            meta_path = os.path.join(s, "metadata.json")
            cond = "?"
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    cond = json.load(f).get("condition", "?")
            print(f"  {name}  [{cond}]  {status}")
        return

    # Run pipeline on each session
    all_results = []
    for s in sessions:
        name = os.path.basename(s)
        valid, missing = validate_session_files(s)

        if not valid:
            print(f"\n[SKIP] {name} — missing: {', '.join(missing)}")
            all_results.append({
                "session": name,
                "condition": None,
                "region": args.region,
                "metrics": None,
                "pos_diagnostics": None,
                "detection_rate": None,
                "intermediate_paths": {},
                "success": False,
                "error": f"Missing files: {', '.join(missing)}",
            })
            continue

        print(f"\n[RUN] {name}")
        try:
            result = run_session_pipeline(
                s, region=args.region,
                save_intermediates=args.save_intermediates,
            )
            m = result["metrics"]
            print(f"  Detection rate: {result['detection_rate']:.1%}")
            print(f"  Valid windows:  {m['n_mutually_valid_windows']}/{m['n_total_windows']}")
            if not np.isnan(m["mae"]):
                print(f"  MAE: {m['mae']:.1f}  RMSE: {m['rmse']:.1f}  r: {m['pearson_r']}")
            else:
                print(f"  Insufficient valid windows for metrics")
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "session": name,
                "condition": None,
                "region": args.region,
                "metrics": None,
                "pos_diagnostics": None,
                "detection_rate": None,
                "intermediate_paths": {},
                "success": False,
                "error": str(e),
            })

    # Aggregate and report
    summary = aggregate_results(all_results)
    print_summary(all_results, summary)

    # Save reports
    print(f"\nSaving reports to {args.output}")
    save_reports(all_results, summary, args.output)


if __name__ == "__main__":
    main()
