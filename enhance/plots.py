"""
Before/after comparison plots for rPPG enhancement evaluation.

Reads saved CSV summaries and JSON reports. Generates publication-ready
PNG figures. Strictly read-only — no pipeline execution or metric recomputation.

Usage:
    py enhance/plots.py --results-dir data/results
    py enhance/plots.py --results-dir data/results --output-dir data/results/figures
    py enhance/plots.py --test
"""

import argparse
import json
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DPI = 150


# ── Data loading ─────────────────────────────────────────────────────

def load_results(results_dir):
    """Load baseline and enhanced CSV summaries and JSON reports.

    Returns dict with keys (each may be None if file missing):
        baseline_csv, enhanced_csv, baseline_json, enhanced_json
    """
    data = {
        "baseline_csv": None,
        "enhanced_csv": None,
        "baseline_json": None,
        "enhanced_json": None,
    }

    bl_csv = os.path.join(results_dir, "baseline_summary.csv")
    if os.path.isfile(bl_csv):
        data["baseline_csv"] = pd.read_csv(bl_csv)

    en_csv = os.path.join(results_dir, "enhanced_summary.csv")
    if os.path.isfile(en_csv):
        data["enhanced_csv"] = pd.read_csv(en_csv)

    bl_json = os.path.join(results_dir, "baseline_report.json")
    if os.path.isfile(bl_json):
        with open(bl_json) as f:
            data["baseline_json"] = json.load(f)

    en_json = os.path.join(results_dir, "enhanced_report.json")
    if os.path.isfile(en_json):
        with open(en_json) as f:
            data["enhanced_json"] = json.load(f)

    return data


def find_best_pipeline_per_session(enhanced_csv):
    """Find the pipeline with the lowest MAE for each session.

    Ties broken by pipeline name (alphabetical, deterministic).

    Args:
        enhanced_csv: DataFrame with columns [session, pipeline, enhanced_mae, success]

    Returns:
        DataFrame with one row per session: session, best_pipeline, best_mae
    """
    df = enhanced_csv[
        (enhanced_csv["success"] == True) &
        (enhanced_csv["enhanced_mae"].notna())
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=["session", "best_pipeline", "best_mae"])

    # Sort by MAE then pipeline name for deterministic tie-breaking
    df = df.sort_values(["enhanced_mae", "pipeline"])
    best = df.groupby("session").first().reset_index()
    return best[["session", "pipeline", "enhanced_mae"]].rename(
        columns={"pipeline": "best_pipeline", "enhanced_mae": "best_mae"}
    )


# ── Plot functions ───────────────────────────────────────────────────

def plot_mae_by_pipeline(enhanced_csv, output_path):
    """Bar chart: mean MAE per pipeline for low-light sessions.

    Args:
        enhanced_csv: DataFrame from enhanced_summary.csv
        output_path: path to save PNG
    """
    df = enhanced_csv[
        (enhanced_csv["success"] == True) &
        (enhanced_csv["enhanced_mae"].notna())
    ].copy()

    if df.empty:
        print("  SKIP: mae_by_pipeline — no valid enhanced results")
        return False

    # Filter to low_light if condition column exists
    if "condition" in df.columns:
        low_light = df[df["condition"] == "low_light"]
        if not low_light.empty:
            df = low_light

    pipeline_stats = df.groupby("pipeline")["enhanced_mae"].agg(["mean", "median", "count"])
    pipeline_stats = pipeline_stats.sort_values("mean")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(pipeline_stats))
    bars = ax.bar(x, pipeline_stats["mean"], color="#4A90D9", edgecolor="white")

    # Add value labels
    for i, (_, row) in enumerate(pipeline_stats.iterrows()):
        ax.text(i, row["mean"] + 0.2, f"{row['mean']:.1f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(pipeline_stats.index, rotation=30, ha="right")
    ax.set_ylabel("Mean MAE (BPM)")
    ax.set_title("Low-Light MAE by Preprocessing Pipeline")
    ax.set_ylim(0, max(pipeline_stats["mean"]) * 1.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return True


def plot_mae_distribution_by_pipeline(enhanced_csv, output_path):
    """Box/strip plot: MAE distribution per pipeline across low-light sessions.

    Args:
        enhanced_csv: DataFrame from enhanced_summary.csv
        output_path: path to save PNG
    """
    df = enhanced_csv[
        (enhanced_csv["success"] == True) &
        (enhanced_csv["enhanced_mae"].notna())
    ].copy()

    if df.empty:
        print("  SKIP: mae_distribution — no valid enhanced results")
        return False

    if "condition" in df.columns:
        low_light = df[df["condition"] == "low_light"]
        if not low_light.empty:
            df = low_light

    pipelines = sorted(df["pipeline"].unique())
    if len(pipelines) == 0:
        return False

    data_by_pipeline = [
        df[df["pipeline"] == pl]["enhanced_mae"].dropna().values
        for pl in pipelines
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data_by_pipeline, tick_labels=pipelines, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("#B0D4F1")
        patch.set_edgecolor("#4A90D9")

    # Overlay individual points
    for i, vals in enumerate(data_by_pipeline):
        jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   color="#4A90D9", alpha=0.6, s=30, zorder=3)

    ax.set_ylabel("MAE (BPM)")
    ax.set_title("Low-Light MAE Distribution by Pipeline")
    ax.set_xticklabels(pipelines, rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return True


def plot_per_session_best_vs_baseline(enhanced_csv, output_path):
    """Grouped bar chart: baseline MAE vs best-enhanced MAE per session.

    Args:
        enhanced_csv: DataFrame from enhanced_summary.csv (must have baseline_mae)
        output_path: path to save PNG
    """
    best = find_best_pipeline_per_session(enhanced_csv)
    if best.empty:
        print("  SKIP: best_vs_baseline — no best pipeline data")
        return False

    # Merge with baseline MAE
    bl_col = "baseline_mae"
    if bl_col not in enhanced_csv.columns:
        print("  SKIP: best_vs_baseline — no baseline_mae column")
        return False

    # Get baseline MAE per session (take first non-NaN)
    bl_per_session = (
        enhanced_csv[enhanced_csv[bl_col].notna()]
        .groupby("session")[bl_col].first().reset_index()
    )

    merged = best.merge(bl_per_session, on="session", how="inner")
    if merged.empty:
        print("  SKIP: best_vs_baseline — no matching sessions with baseline")
        return False

    merged = merged.sort_values("session")

    fig, ax = plt.subplots(figsize=(max(8, len(merged) * 1.2), 5))
    x = np.arange(len(merged))
    width = 0.35

    bars1 = ax.bar(x - width / 2, merged[bl_col], width,
                   label="Baseline", color="#E74C3C", alpha=0.8)
    bars2 = ax.bar(x + width / 2, merged["best_mae"], width,
                   label="Best Enhanced", color="#2ECC71", alpha=0.8)

    ax.set_xticks(x)
    labels = [f"{s}\n({p})" for s, p in zip(merged["session"], merged["best_pipeline"])]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (BPM)")
    ax.set_title("Baseline vs Best Enhancement (per session)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return True


def plot_condition_comparison(baseline_csv, output_path):
    """Bar chart: normal vs low-light MAE from baseline (shows degradation).

    Args:
        baseline_csv: DataFrame from baseline_summary.csv
        output_path: path to save PNG
    """
    if baseline_csv is None or baseline_csv.empty:
        print("  SKIP: condition_comparison — no baseline data")
        return False

    df = baseline_csv[
        (baseline_csv["success"] == True) &
        (baseline_csv["mae"].notna())
    ].copy()

    if "condition" not in df.columns:
        print("  SKIP: condition_comparison — no condition column")
        return False

    cond_stats = df.groupby("condition")["mae"].agg(["mean", "median", "count"])
    if len(cond_stats) < 2:
        print("  SKIP: condition_comparison — need both normal and low_light")
        return False

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"normal": "#2ECC71", "low_light": "#E74C3C"}
    x = range(len(cond_stats))
    bars = ax.bar(x, cond_stats["mean"],
                  color=[colors.get(c, "#999") for c in cond_stats.index],
                  edgecolor="white")

    for i, (cond, row) in enumerate(cond_stats.iterrows()):
        ax.text(i, row["mean"] + 0.3,
                f"{row['mean']:.1f}\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in cond_stats.index])
    ax.set_ylabel("Mean MAE (BPM)")
    ax.set_title("Baseline MAE: Normal vs Low Light")
    ax.set_ylim(0, max(cond_stats["mean"]) * 1.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return True


def plot_bland_altman(results_dir, best_pipeline, output_path):
    """Bland-Altman plot for the best pipeline using per-window merged data.

    Only generated if per-window merged DataFrames are available in
    session intermediate outputs. Skipped cleanly otherwise.

    Args:
        results_dir: directory containing reports
        best_pipeline: name of the best pipeline
        output_path: path to save PNG

    Returns:
        True if plot was generated, False if skipped.
    """
    # Look for per-window merged data in enhanced report
    en_json_path = os.path.join(results_dir, "enhanced_report.json")
    if not os.path.isfile(en_json_path):
        print("  SKIP: bland_altman — no enhanced report")
        return False

    with open(en_json_path) as f:
        report = json.load(f)

    # Collect per-window data from sessions that used the best pipeline
    all_estimated = []
    all_gt = []

    for s in report.get("sessions", []):
        if not s.get("success") or s.get("pipeline") != best_pipeline:
            continue
        m = s.get("metrics", {})
        # We don't have per-window data in the report by default
        # This plot requires intermediate merged CSVs
        # Check if they exist in data/processed
        session_name = s["session"]
        merged_path = os.path.join(
            "data", "processed", session_name, "merged_evaluation.csv"
        )
        if os.path.isfile(merged_path):
            merged = pd.read_csv(merged_path)
            valid = merged[
                (merged.get("est_valid", True) == True) &
                (merged.get("gt_valid", True) == True) &
                (merged["hr_bpm_estimated"].notna()) &
                (merged["hr_bpm_gt"].notna())
            ]
            all_estimated.extend(valid["hr_bpm_estimated"].tolist())
            all_gt.extend(valid["hr_bpm_gt"].tolist())

    if len(all_estimated) < 3:
        print("  SKIP: bland_altman — insufficient per-window data "
              "(need merged_evaluation.csv in data/processed/)")
        return False

    est = np.array(all_estimated)
    gt = np.array(all_gt)
    mean_vals = (est + gt) / 2
    diff_vals = est - gt
    mean_diff = diff_vals.mean()
    std_diff = diff_vals.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_vals, diff_vals, alpha=0.5, s=30, color="#4A90D9")
    ax.axhline(mean_diff, color="red", linestyle="--", label=f"Mean: {mean_diff:.1f}")
    ax.axhline(mean_diff + 1.96 * std_diff, color="gray", linestyle=":",
               label=f"+1.96 SD: {mean_diff + 1.96 * std_diff:.1f}")
    ax.axhline(mean_diff - 1.96 * std_diff, color="gray", linestyle=":",
               label=f"-1.96 SD: {mean_diff - 1.96 * std_diff:.1f}")

    ax.set_xlabel("Mean of Estimated and Ground Truth HR (BPM)")
    ax.set_ylabel("Estimated - Ground Truth (BPM)")
    ax.set_title(f"Bland-Altman: {best_pipeline}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return True


# ── Summary artifact ─────────────────────────────────────────────────

def save_plot_summary(output_dir, plots_generated, enhanced_csv):
    """Save a small text summary describing what was plotted."""
    summary_path = os.path.join(output_dir, "plots_summary.txt")

    lines = ["Circadify rPPG — Plots Summary", "=" * 40, ""]

    # What was generated
    lines.append(f"Plots generated: {len(plots_generated)}")
    for name, generated in plots_generated.items():
        status = "OK" if generated else "SKIPPED"
        lines.append(f"  {name}: {status}")

    # Best pipeline
    if enhanced_csv is not None and not enhanced_csv.empty:
        best = find_best_pipeline_per_session(enhanced_csv)
        if not best.empty:
            lines.append("")

            # Sessions plotted
            sessions = enhanced_csv[enhanced_csv["success"] == True]["session"].nunique()
            pipelines = enhanced_csv[enhanced_csv["success"] == True]["pipeline"].unique()
            lines.append(f"Sessions with valid results: {sessions}")
            lines.append(f"Pipelines evaluated: {', '.join(sorted(pipelines))}")

            # Best pipeline by mean MAE
            df = enhanced_csv[
                (enhanced_csv["success"] == True) &
                (enhanced_csv["enhanced_mae"].notna())
            ]
            if "condition" in df.columns:
                ll = df[df["condition"] == "low_light"]
                if not ll.empty:
                    df = ll

            if not df.empty:
                pl_means = df.groupby("pipeline")["enhanced_mae"].mean()
                best_pl = pl_means.idxmin()
                lines.append(f"Best pipeline (lowest mean low-light MAE): "
                             f"{best_pl} ({pl_means[best_pl]:.2f})")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Summary: {summary_path}")


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

    # ── Mock data ────────────────────────────────────────────────
    baseline_csv = pd.DataFrame({
        "session": ["s1", "s2", "s3", "s4"],
        "condition": ["normal", "normal", "low_light", "low_light"],
        "mae": [3.0, 4.0, 8.0, 10.0],
        "rmse": [4.0, 5.0, 10.0, 12.0],
        "success": [True, True, True, True],
    })

    enhanced_csv = pd.DataFrame({
        "session": ["s3", "s3", "s3", "s4", "s4", "s4"],
        "condition": ["low_light"] * 6,
        "region": ["forehead"] * 6,
        "pipeline": ["clahe", "gamma_07", "denoise_clahe_gamma"] * 2,
        "baseline_mae": [8.0, 8.0, 8.0, 10.0, 10.0, 10.0],
        "enhanced_mae": [5.0, 6.0, 4.5, 7.0, 8.0, 6.0],
        "mae_delta": [-3.0, -2.0, -3.5, -3.0, -2.0, -4.0],
        "mae_pct_improvement": [37.5, 25.0, 43.75, 30.0, 20.0, 40.0],
        "baseline_rmse": [10.0, 10.0, 10.0, 12.0, 12.0, 12.0],
        "enhanced_rmse": [6.0, 7.0, 5.5, 8.0, 9.0, 7.0],
        "rmse_delta": [-4.0, -3.0, -4.5, -4.0, -3.0, -5.0],
        "success": [True] * 6,
    })

    # ── Test 1: Best pipeline selection ──────────────────────────
    print("\nTest 1: Best pipeline selection")
    best = find_best_pipeline_per_session(enhanced_csv)
    check("Returns 2 sessions", len(best) == 2)

    s3_best = best[best["session"] == "s3"].iloc[0]
    check("s3 best pipeline = denoise_clahe_gamma", s3_best["best_pipeline"] == "denoise_clahe_gamma")
    check("s3 best MAE = 4.5", s3_best["best_mae"] == 4.5)

    s4_best = best[best["session"] == "s4"].iloc[0]
    check("s4 best pipeline = denoise_clahe_gamma", s4_best["best_pipeline"] == "denoise_clahe_gamma")

    # ── Test 2: Plots generate without crashing ──────────────────
    print("\nTest 2: Plot generation (mock data)")
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "mae_by_pipeline_low_light.png")
        ok1 = plot_mae_by_pipeline(enhanced_csv, p1)
        check("mae_by_pipeline generated", ok1)
        check("mae_by_pipeline file exists and non-empty",
              os.path.isfile(p1) and os.path.getsize(p1) > 0)

        p2 = os.path.join(tmp, "mae_distribution_by_pipeline.png")
        ok2 = plot_mae_distribution_by_pipeline(enhanced_csv, p2)
        check("mae_distribution generated", ok2)
        check("mae_distribution file non-empty",
              os.path.isfile(p2) and os.path.getsize(p2) > 0)

        p3 = os.path.join(tmp, "mae_best_vs_baseline_per_session.png")
        ok3 = plot_per_session_best_vs_baseline(enhanced_csv, p3)
        check("best_vs_baseline generated", ok3)
        check("best_vs_baseline file non-empty",
              os.path.isfile(p3) and os.path.getsize(p3) > 0)

        p4 = os.path.join(tmp, "baseline_condition_comparison.png")
        ok4 = plot_condition_comparison(baseline_csv, p4)
        check("condition_comparison generated", ok4)
        check("condition_comparison file non-empty",
              os.path.isfile(p4) and os.path.getsize(p4) > 0)

    # ── Test 3: Missing data handling ────────────────────────────
    print("\nTest 3: Missing data handling")
    with tempfile.TemporaryDirectory() as tmp:
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=enhanced_csv.columns)
        ok_empty = plot_mae_by_pipeline(empty_df, os.path.join(tmp, "empty.png"))
        check("Empty data: skips gracefully", not ok_empty)

        # None baseline
        ok_none = plot_condition_comparison(None, os.path.join(tmp, "none.png"))
        check("None baseline: skips gracefully", not ok_none)

        # Bland-Altman without per-window data
        ok_ba = plot_bland_altman(tmp, "clahe", os.path.join(tmp, "ba.png"))
        check("Bland-Altman without data: skips gracefully", not ok_ba)

    # ── Test 4: Plot summary artifact ────────────────────────────
    print("\nTest 4: Plot summary")
    with tempfile.TemporaryDirectory() as tmp:
        plots_status = {"plot1": True, "plot2": False}
        save_plot_summary(tmp, plots_status, enhanced_csv)
        summary_path = os.path.join(tmp, "plots_summary.txt")
        check("Summary file created", os.path.isfile(summary_path))
        content = open(summary_path).read()
        check("Summary mentions best pipeline", "denoise_clahe_gamma" in content)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots")
    parser.add_argument("--results-dir", default="data/results",
                        help="Directory containing report files")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save figures (default: results-dir/figures)")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    # Load data
    data = load_results(args.results_dir)

    has_baseline = data["baseline_csv"] is not None
    has_enhanced = data["enhanced_csv"] is not None

    if not has_baseline and not has_enhanced:
        print("No results found. Run baseline and/or enhanced evaluation first.")
        return

    print(f"Loaded: baseline={'yes' if has_baseline else 'no'}, "
          f"enhanced={'yes' if has_enhanced else 'no'}")

    # Output directory
    out_dir = args.output_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Generate plots
    plots = {}

    if has_enhanced:
        plots["mae_by_pipeline_low_light"] = plot_mae_by_pipeline(
            data["enhanced_csv"],
            os.path.join(out_dir, "mae_by_pipeline_low_light.png"),
        )
        plots["mae_distribution_by_pipeline"] = plot_mae_distribution_by_pipeline(
            data["enhanced_csv"],
            os.path.join(out_dir, "mae_distribution_by_pipeline.png"),
        )
        plots["mae_best_vs_baseline_per_session"] = plot_per_session_best_vs_baseline(
            data["enhanced_csv"],
            os.path.join(out_dir, "mae_best_vs_baseline_per_session.png"),
        )

        # Bland-Altman for best pipeline
        best_per_session = find_best_pipeline_per_session(data["enhanced_csv"])
        if not best_per_session.empty:
            # Most common best pipeline
            best_pl = best_per_session["best_pipeline"].mode().iloc[0]
            plots["bland_altman_best_pipeline"] = plot_bland_altman(
                args.results_dir, best_pl,
                os.path.join(out_dir, f"bland_altman_{best_pl}.png"),
            )

    if has_baseline:
        plots["baseline_condition_comparison"] = plot_condition_comparison(
            data["baseline_csv"],
            os.path.join(out_dir, "baseline_condition_comparison.png"),
        )

    # Summary
    save_plot_summary(out_dir, plots, data.get("enhanced_csv"))

    generated = sum(1 for v in plots.values() if v)
    skipped = sum(1 for v in plots.values() if not v)
    print(f"\nDone: {generated} plots generated, {skipped} skipped")


if __name__ == "__main__":
    main()
