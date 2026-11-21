"""
Enhanced evaluation orchestrator.

Runs preprocessing pipelines on low-light sessions, then evaluates the
rPPG pipeline on preprocessed videos. Compares against baseline if available.

Usage:
    py enhance/run_enhanced.py --data-dir data/raw
    py enhance/run_enhanced.py --data-dir data/raw --pipelines clahe denoise_clahe_gamma
    py enhance/run_enhanced.py --data-dir data/raw --condition low_light --force-preprocess
    py enhance/run_enhanced.py --test
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from evaluate.run_baseline import (
    discover_sessions, validate_session_files, run_session_pipeline,
)
from enhance.preprocess import preprocess_video


# ── Session filtering ────────────────────────────────────────────────

def discover_target_sessions(data_dir, condition="low_light"):
    """Discover sessions filtered by lighting condition.

    Args:
        data_dir: directory containing session folders
        condition: filter to this condition (None for all)

    Returns:
        List of (session_path, condition_str) tuples
    """
    sessions = discover_sessions(data_dir)
    results = []
    for s in sessions:
        meta_path = os.path.join(s, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        cond = meta.get("condition", "unknown")
        if condition is None or cond == condition:
            results.append((s, cond))
    return results


# ── Preprocessing with caching ───────────────────────────────────────

def maybe_preprocess_video(session_dir, pipeline_name, force=False):
    """Preprocess a session's video, reusing cached output if available.

    Preprocessed videos are stored at:
        data/processed/{session_name}/video_{pipeline_name}.avi

    Args:
        session_dir: path to session directory
        pipeline_name: key into config.PREPROCESS_PIPELINES
        force: if True, regenerate even if cached file exists

    Returns:
        (output_path, stats_dict, was_cached) tuple
    """
    session_name = os.path.basename(session_dir)
    out_dir = os.path.join(config.DATA_PROCESSED, session_name)
    out_path = os.path.join(out_dir, f"video_{pipeline_name}.avi")

    if os.path.isfile(out_path) and not force:
        return (out_path, None, True)

    from evaluate.run_baseline import find_video_file
    video_path = find_video_file(session_dir)
    stats = preprocess_video(video_path, out_path, pipeline_name)
    return (out_path, stats, False)


# ── Per-session enhanced pipeline ────────────────────────────────────

def run_enhanced_session(session_dir, pipeline_name, region="forehead",
                         force_preprocess=False):
    """Run rPPG pipeline on a preprocessed version of a session.

    Steps:
        1. Preprocess video (or reuse cached)
        2. Run baseline pipeline on preprocessed video
           (reuses run_session_pipeline by temporarily substituting video)

    Returns:
        dict with keys:
            session, condition, region, pipeline, metrics,
            pos_diagnostics, detection_rate, preprocess_stats,
            was_cached, success, error
    """
    session_name = os.path.basename(session_dir)
    result = {
        "session": session_name,
        "condition": None,
        "region": region,
        "pipeline": pipeline_name,
        "metrics": None,
        "pos_diagnostics": None,
        "detection_rate": None,
        "preprocess_stats": None,
        "was_cached": False,
        "success": False,
        "error": None,
    }

    # Load condition from metadata
    with open(os.path.join(session_dir, "metadata.json")) as f:
        meta = json.load(f)
    result["condition"] = meta.get("condition", "unknown")
    fps = meta.get("fps_actual", config.FPS)

    # Step 1: Preprocess
    pp_path, pp_stats, was_cached = maybe_preprocess_video(
        session_dir, pipeline_name, force=force_preprocess
    )
    result["preprocess_stats"] = pp_stats
    result["was_cached"] = was_cached

    # Step 2: Run rPPG pipeline on preprocessed video
    # We reuse the baseline pipeline imports directly rather than
    # calling run_session_pipeline (which hardcodes video.avi path)
    from rppg.face_detect import extract_rgb_signals
    from rppg.pos import pos_from_dataframe
    from rppg.signal_processing import bandpass_filter, estimate_hr_windowed, smooth_hr_temporal
    from evaluate.metrics import evaluate_session

    rgb_df = extract_rgb_signals(pp_path, region=region)
    result["detection_rate"] = round(float(rgb_df["valid_detection"].mean()), 4)

    pulse, pos_diag = pos_from_dataframe(rgb_df, fps, return_diagnostics=True)
    result["pos_diagnostics"] = pos_diag

    filtered = bandpass_filter(pulse, fps)
    est_df = estimate_hr_windowed(filtered, fps)

    if config.TEMPORAL_SMOOTH:
        est_df = smooth_hr_temporal(est_df)

    gt_df = pd.read_csv(os.path.join(session_dir, "windows.csv"))
    metrics = evaluate_session(est_df, gt_df)
    result["metrics"] = metrics
    result["success"] = True

    return result


# ── Aggregation ──────────────────────────────────────────────────────

def aggregate_pipeline_results(all_results):
    """Aggregate results by pipeline name.

    Returns:
        dict keyed by pipeline, each with:
            n_sessions, n_successful, total_mutually_valid_windows,
            mae_mean, mae_median, rmse_mean, rmse_median
    """
    by_pipeline = {}
    for r in all_results:
        if not r["success"] or r["metrics"] is None:
            pl = r["pipeline"]
            if pl not in by_pipeline:
                by_pipeline[pl] = {"successful": [], "failed": 0}
            by_pipeline[pl]["failed"] += 1
            continue

        pl = r["pipeline"]
        if pl not in by_pipeline:
            by_pipeline[pl] = {"successful": [], "failed": 0}
        by_pipeline[pl]["successful"].append(r["metrics"])

    summary = {}
    for pl, data in by_pipeline.items():
        metrics_list = data["successful"]
        maes = [m["mae"] for m in metrics_list if not np.isnan(m["mae"])]
        rmses = [m["rmse"] for m in metrics_list if not np.isnan(m["rmse"])]
        med_aes = [m["median_ae"] for m in metrics_list if not np.isnan(m["median_ae"])]
        total_valid = sum(m["n_mutually_valid_windows"] for m in metrics_list)

        summary[pl] = {
            "n_sessions": len(metrics_list) + data["failed"],
            "n_successful": len(metrics_list),
            "n_failed": data["failed"],
            "total_mutually_valid_windows": total_valid,
            "mae_mean": round(float(np.mean(maes)), 2) if maes else float("nan"),
            "mae_median": round(float(np.median(maes)), 2) if maes else float("nan"),
            "rmse_mean": round(float(np.mean(rmses)), 2) if rmses else float("nan"),
            "rmse_median": round(float(np.median(rmses)), 2) if rmses else float("nan"),
            "median_ae_mean": round(float(np.mean(med_aes)), 2) if med_aes else float("nan"),
            "median_ae_median": round(float(np.median(med_aes)), 2) if med_aes else float("nan"),
        }

    return summary


def build_comparison_rows(all_results, baseline_lookup):
    """Build flat CSV rows with baseline comparison columns.

    Args:
        all_results: list of enhanced result dicts
        baseline_lookup: dict mapping session_name -> baseline metrics dict
                         (or empty dict if no baseline available)

    Returns:
        list of row dicts for pd.DataFrame
    """
    rows = []
    for r in all_results:
        m = r["metrics"] or {}
        pd_diag = r["pos_diagnostics"] or {}

        enhanced_mae = m.get("mae", float("nan"))
        enhanced_rmse = m.get("rmse", float("nan"))

        # Baseline comparison
        bl = baseline_lookup.get(r["session"], {})
        baseline_mae = bl.get("mae", float("nan"))
        baseline_rmse = bl.get("rmse", float("nan"))

        mae_delta = float("nan")
        mae_pct = float("nan")
        rmse_delta = float("nan")

        if not np.isnan(baseline_mae) and not np.isnan(enhanced_mae):
            mae_delta = round(enhanced_mae - baseline_mae, 2)
            if baseline_mae > 0:
                mae_pct = round((baseline_mae - enhanced_mae) / baseline_mae * 100, 1)

        if not np.isnan(baseline_rmse) and not np.isnan(enhanced_rmse):
            rmse_delta = round(enhanced_rmse - baseline_rmse, 2)

        rows.append({
            "session": r["session"],
            "condition": r["condition"],
            "region": r["region"],
            "pipeline": r["pipeline"],
            "baseline_mae": baseline_mae,
            "enhanced_mae": enhanced_mae,
            "mae_delta": mae_delta,
            "mae_pct_improvement": mae_pct,
            "baseline_rmse": baseline_rmse,
            "enhanced_rmse": enhanced_rmse,
            "rmse_delta": rmse_delta,
            "median_ae": m.get("median_ae", float("nan")),
            "pearson_r": m.get("pearson_r", float("nan")),
            "n_mutually_valid_windows": m.get("n_mutually_valid_windows", 0),
            "detection_rate": r["detection_rate"],
            "pos_processed_windows": pd_diag.get("processed_windows", 0),
            "pos_skipped_windows": (
                pd_diag.get("skipped_nan", 0) + pd_diag.get("skipped_unstable", 0)
            ),
            "was_cached": r["was_cached"],
            "success": r["success"],
            "error": r["error"],
        })

    return rows


# ── Baseline loading ─────────────────────────────────────────────────

def load_baseline_lookup(results_dir):
    """Load baseline metrics keyed by session name.

    Returns empty dict if baseline report doesn't exist.
    """
    report_path = os.path.join(results_dir, "baseline_report.json")
    if not os.path.isfile(report_path):
        return {}
    with open(report_path) as f:
        report = json.load(f)
    lookup = {}
    for s in report.get("sessions", []):
        if s.get("success") and s.get("metrics"):
            lookup[s["session"]] = s["metrics"]
    return lookup


# ── Report saving ────────────────────────────────────────────────────

def save_reports(all_results, pipeline_summary, comparison_rows, output_dir):
    """Save JSON report and CSV summary."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON report
    report = {
        "sessions": all_results,
        "pipeline_summary": pipeline_summary,
    }
    json_path = os.path.join(output_dir, "enhanced_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  JSON report: {json_path}")

    # CSV summary
    csv_df = pd.DataFrame(comparison_rows)
    csv_path = os.path.join(output_dir, "enhanced_summary.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"  CSV summary: {csv_path}")


# ── Console output ───────────────────────────────────────────────────

def print_summary(all_results, pipeline_summary):
    """Print concise summary to console."""
    n_total = len(all_results)
    n_success = sum(1 for r in all_results if r["success"])
    n_failed = n_total - n_success

    print(f"\n{'='*60}")
    print(f"Enhanced Evaluation Summary")
    print(f"{'='*60}")
    print(f"Runs: {n_total} total, {n_success} succeeded, {n_failed} failed")

    if n_failed > 0:
        print(f"\nFailed runs:")
        for r in all_results:
            if not r["success"]:
                print(f"  {r['session']} [{r['pipeline']}]: {r['error']}")

    # Per-pipeline summary
    print(f"\nPer-pipeline results:")
    best_pipeline = None
    best_mae = float("inf")

    for pl, s in sorted(pipeline_summary.items()):
        print(f"\n  {pl} ({s['n_successful']}/{s['n_sessions']} sessions, "
              f"{s['total_mutually_valid_windows']} valid windows)")
        print(f"    MAE:  mean={s['mae_mean']:.2f}  median={s['mae_median']:.2f}")
        print(f"    RMSE: mean={s['rmse_mean']:.2f}  median={s['rmse_median']:.2f}")

        if not np.isnan(s["mae_mean"]) and s["mae_mean"] < best_mae:
            best_mae = s["mae_mean"]
            best_pipeline = pl

    if best_pipeline:
        print(f"\nBest pipeline by mean MAE: {best_pipeline} ({best_mae:.2f})")

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

    # ── Test 1: Session filtering ────────────────────────────────
    print("\nTest 1: Session filtering by condition")
    with tempfile.TemporaryDirectory() as tmp:
        for name, cond in [("session_001", "normal"), ("session_002", "low_light"),
                           ("session_003", "low_light"), ("session_004", "normal")]:
            d = os.path.join(tmp, name)
            os.makedirs(d)
            with open(os.path.join(d, "metadata.json"), "w") as f:
                json.dump({"condition": cond}, f)

        all_sess = discover_target_sessions(tmp, condition=None)
        check("All sessions: 4", len(all_sess) == 4)

        low_light = discover_target_sessions(tmp, condition="low_light")
        check("Low light only: 2", len(low_light) == 2)
        check("Both are low_light", all(c == "low_light" for _, c in low_light))

        normal = discover_target_sessions(tmp, condition="normal")
        check("Normal only: 2", len(normal) == 2)

    # ── Test 2: Pipeline selection ───────────────────────────────
    print("\nTest 2: Pipeline selection")
    available = list(config.PREPROCESS_PIPELINES.keys())
    check("At least 4 pipelines defined", len(available) >= 4)
    check("clahe in pipelines", "clahe" in available)
    check("denoise_clahe_gamma in pipelines", "denoise_clahe_gamma" in available)

    # ── Test 3: Baseline lookup ──────────────────────────────────
    print("\nTest 3: Baseline lookup")
    with tempfile.TemporaryDirectory() as tmp:
        # No baseline file
        check("Missing baseline returns empty dict",
              load_baseline_lookup(tmp) == {})

        # With baseline file
        report = {
            "sessions": [
                {"session": "session_001", "success": True,
                 "metrics": {"mae": 5.0, "rmse": 6.0}},
                {"session": "session_002", "success": False, "metrics": None},
            ]
        }
        with open(os.path.join(tmp, "baseline_report.json"), "w") as f:
            json.dump(report, f)

        lookup = load_baseline_lookup(tmp)
        check("Loads successful session", "session_001" in lookup)
        check("Skips failed session", "session_002" not in lookup)
        check("Correct MAE value", lookup["session_001"]["mae"] == 5.0)

    # ── Test 4: Comparison row building ──────────────────────────
    print("\nTest 4: Comparison rows with improvement columns")
    mock_results = [
        {
            "session": "session_001", "condition": "low_light",
            "region": "forehead", "pipeline": "clahe",
            "metrics": {"mae": 4.0, "rmse": 5.0, "median_ae": 3.5,
                        "pearson_r": 0.85, "n_mutually_valid_windows": 5},
            "pos_diagnostics": {"processed_windows": 200,
                                "skipped_nan": 3, "skipped_unstable": 1},
            "detection_rate": 0.92, "was_cached": False,
            "success": True, "error": None,
        },
    ]
    baseline = {"session_001": {"mae": 8.0, "rmse": 10.0}}
    rows = build_comparison_rows(mock_results, baseline)

    check("One comparison row", len(rows) == 1)
    r = rows[0]
    check("baseline_mae = 8.0", r["baseline_mae"] == 8.0)
    check("enhanced_mae = 4.0", r["enhanced_mae"] == 4.0)
    check("mae_delta = -4.0", r["mae_delta"] == -4.0)
    check("mae_pct_improvement = 50.0%", r["mae_pct_improvement"] == 50.0)
    check("rmse_delta = -5.0", r["rmse_delta"] == -5.0)
    check("pipeline recorded", r["pipeline"] == "clahe")
    check("region recorded", r["region"] == "forehead")

    # Without baseline
    rows_no_bl = build_comparison_rows(mock_results, {})
    check("No baseline: mae_delta is NaN", np.isnan(rows_no_bl[0]["mae_delta"]))

    # ── Test 5: Pipeline aggregation ─────────────────────────────
    print("\nTest 5: Pipeline aggregation")
    mock_all = [
        {"success": True, "pipeline": "clahe", "metrics": {
            "mae": 4.0, "rmse": 5.0, "median_ae": 3.0,
            "n_mutually_valid_windows": 5}},
        {"success": True, "pipeline": "clahe", "metrics": {
            "mae": 6.0, "rmse": 7.0, "median_ae": 5.0,
            "n_mutually_valid_windows": 4}},
        {"success": True, "pipeline": "gamma_07", "metrics": {
            "mae": 7.0, "rmse": 8.0, "median_ae": 6.0,
            "n_mutually_valid_windows": 5}},
        {"success": False, "pipeline": "gamma_07", "metrics": None},
    ]
    agg = aggregate_pipeline_results(mock_all)
    check("Two pipelines in summary", len(agg) == 2)
    check("clahe: 2 successful", agg["clahe"]["n_successful"] == 2)
    check("clahe: MAE mean = 5.0", agg["clahe"]["mae_mean"] == 5.0)
    check("clahe: total valid windows = 9",
          agg["clahe"]["total_mutually_valid_windows"] == 9)
    check("gamma_07: 1 successful + 1 failed",
          agg["gamma_07"]["n_successful"] == 1 and agg["gamma_07"]["n_failed"] == 1)

    # ── Test 6: Output schema ────────────────────────────────────
    print("\nTest 6: Output schema")
    expected_cols = [
        "session", "condition", "region", "pipeline",
        "baseline_mae", "enhanced_mae", "mae_delta", "mae_pct_improvement",
        "baseline_rmse", "enhanced_rmse", "rmse_delta",
        "median_ae", "pearson_r", "n_mutually_valid_windows",
        "detection_rate", "pos_processed_windows", "pos_skipped_windows",
        "was_cached", "success", "error",
    ]
    check("CSV columns match expected schema",
          list(rows[0].keys()) == expected_cols)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run enhanced rPPG evaluation")
    parser.add_argument("--data-dir", default=config.DATA_RAW,
                        help="Directory containing session folders")
    parser.add_argument("--condition", default="low_light",
                        help="Filter sessions by condition (default: low_light, 'all' for no filter)")
    parser.add_argument("--pipelines", nargs="+",
                        default=list(config.PREPROCESS_PIPELINES.keys()),
                        help="Preprocessing pipelines to evaluate")
    parser.add_argument("--region", choices=["forehead", "cheeks", "combined"],
                        default="forehead", help="ROI region")
    parser.add_argument("--output", default=config.DATA_RESULTS,
                        help="Output directory for reports")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="Regenerate preprocessed videos even if cached")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    # Discover sessions
    condition_filter = None if args.condition == "all" else args.condition
    target_sessions = discover_target_sessions(args.data_dir, condition=condition_filter)
    print(f"Discovered {len(target_sessions)} {args.condition} session(s)")
    print(f"Pipelines: {args.pipelines}")

    if len(target_sessions) == 0:
        print("No matching sessions found.")
        return

    # Validate pipelines
    for pl in args.pipelines:
        if pl not in config.PREPROCESS_PIPELINES:
            print(f"ERROR: Unknown pipeline '{pl}'. "
                  f"Available: {list(config.PREPROCESS_PIPELINES.keys())}")
            return

    # Load baseline for comparison
    baseline_lookup = load_baseline_lookup(args.output)
    if baseline_lookup:
        print(f"Loaded baseline results for {len(baseline_lookup)} session(s)")
    else:
        print("No baseline report found — comparison columns will be NaN")

    # Run pipeline sweep
    all_results = []
    for session_dir, cond in target_sessions:
        session_name = os.path.basename(session_dir)
        valid, missing = validate_session_files(session_dir)

        if not valid:
            print(f"\n[SKIP] {session_name} — missing: {', '.join(missing)}")
            for pl in args.pipelines:
                all_results.append({
                    "session": session_name, "condition": cond,
                    "region": args.region, "pipeline": pl,
                    "metrics": None, "pos_diagnostics": None,
                    "detection_rate": None, "preprocess_stats": None,
                    "was_cached": False, "success": False,
                    "error": f"Missing files: {', '.join(missing)}",
                })
            continue

        for pl in args.pipelines:
            print(f"\n[RUN] {session_name} + {pl}")
            try:
                result = run_enhanced_session(
                    session_dir, pl,
                    region=args.region,
                    force_preprocess=args.force_preprocess,
                )
                m = result["metrics"]
                cached_str = " (cached)" if result["was_cached"] else ""
                print(f"  Preprocessed{cached_str}  Detection: {result['detection_rate']:.1%}")
                print(f"  Valid windows: {m['n_mutually_valid_windows']}/{m['n_total_windows']}")
                if not np.isnan(m["mae"]):
                    bl_mae = baseline_lookup.get(session_name, {}).get("mae")
                    bl_str = f"  (baseline: {bl_mae:.1f})" if bl_mae else ""
                    print(f"  MAE: {m['mae']:.1f}{bl_str}  RMSE: {m['rmse']:.1f}")
                else:
                    print(f"  Insufficient valid windows for metrics")
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    "session": session_name, "condition": cond,
                    "region": args.region, "pipeline": pl,
                    "metrics": None, "pos_diagnostics": None,
                    "detection_rate": None, "preprocess_stats": None,
                    "was_cached": False, "success": False,
                    "error": str(e),
                })

    # Aggregate and report
    pipeline_summary = aggregate_pipeline_results(all_results)
    comparison_rows = build_comparison_rows(all_results, baseline_lookup)
    print_summary(all_results, pipeline_summary)

    # Save
    print(f"\nSaving reports to {args.output}")
    save_reports(all_results, pipeline_summary, comparison_rows, args.output)


if __name__ == "__main__":
    main()
