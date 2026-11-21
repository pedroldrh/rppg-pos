"""
Session batch processor — parse Apple Health HR and align all sessions in one pass.

Reads the Apple Health export XML once, routes HR records into each session's
time window, saves hr_labels.csv, and runs alignment to produce windows.csv.

Usage:
    py collect/batch_process.py --export data/export.xml --data-dir data/raw
    py collect/batch_process.py --export data/export.xml --data-dir data/raw --dry-run
    py collect/batch_process.py --export data/export.xml --data-dir data/raw --force
    py collect/batch_process.py --test
"""

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from collect.parse_apple_health import (
    normalize_hr_dataframe, _parse_apple_timestamp, HR_TYPE,
)
from collect.align import build_windows, interpolate_frame_hr


# ── Session discovery ────────────────────────────────────────────────

def discover_sessions_with_metadata(data_dir):
    """Find session directories and load their metadata.

    Returns list of dicts:
        {path, name, start_utc, end_utc, condition, subject_id}
    Skips sessions missing metadata.json or required time fields.
    """
    sessions = []
    if not os.path.isdir(data_dir):
        return sessions

    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isdir(path) or not name.startswith("session_"):
            continue

        meta_path = os.path.join(path, "metadata.json")
        if not os.path.isfile(meta_path):
            print(f"  WARN: {name} — missing metadata.json, skipping")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        start_str = meta.get("start_time_utc")
        end_str = meta.get("end_time_utc")

        if not start_str or not end_str:
            print(f"  WARN: {name} — missing start_time_utc or end_time_utc, skipping")
            continue

        start_utc = pd.to_datetime(start_str, utc=True).to_pydatetime()
        end_utc = pd.to_datetime(end_str, utc=True).to_pydatetime()

        sessions.append({
            "path": path,
            "name": name,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "condition": meta.get("condition", "unknown"),
            "subject_id": meta.get("subject_id", "unknown"),
        })

    return sessions


# ── Single-pass XML parsing with session bucketing ───────────────────

def parse_xml_to_sessions(xml_path, sessions):
    """Parse Apple Health XML once, routing HR records into session time windows.

    For each HR record in the XML, checks which session(s) it falls into
    (by timestamp) and appends it to that session's record list.

    Args:
        xml_path: path to Apple Health export.xml
        sessions: list of session dicts from discover_sessions_with_metadata()

    Returns:
        dict mapping session name -> list of {timestamp_utc, hr_bpm} dicts
    """
    buckets = {s["name"]: [] for s in sessions}

    # Build a sorted list of (start, end, name) for efficient bucketing
    windows = [(s["start_utc"], s["end_utc"], s["name"]) for s in sessions]

    # Find the overall time range to skip records outside all sessions
    if not windows:
        return buckets
    global_start = min(w[0] for w in windows)
    global_end = max(w[1] for w in windows)

    records_scanned = 0
    records_matched = 0

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "Record" or elem.get("type") != HR_TYPE:
            elem.clear()
            continue

        date_str = elem.get("startDate", "")
        value_str = elem.get("value", "")
        elem.clear()

        if not date_str or not value_str:
            continue

        ts = _parse_apple_timestamp(date_str)
        if ts is None:
            continue

        records_scanned += 1

        # Quick skip if outside all sessions
        if ts < global_start or ts > global_end:
            continue

        try:
            hr = float(value_str)
        except ValueError:
            continue

        # Route to matching session(s)
        for win_start, win_end, win_name in windows:
            if win_start <= ts <= win_end:
                buckets[win_name].append({"timestamp_utc": ts, "hr_bpm": hr})
                records_matched += 1

    print(f"  XML parse: {records_scanned} HR records scanned, "
          f"{records_matched} matched to sessions")

    return buckets


# ── Per-session processing ───────────────────────────────────────────

def save_session_hr(session_path, records, source_file):
    """Save parsed HR records as hr_labels.csv in the session directory.

    Returns (n_records, output_path).
    """
    df = normalize_hr_dataframe(records, "apple_health_xml", source_file)
    output_path = os.path.join(session_path, "hr_labels.csv")
    df.to_csv(output_path, index=False)
    return len(df), output_path


def run_alignment(session_path):
    """Run alignment on a session that has frame_timestamps.csv and hr_labels.csv.

    Returns (n_total_windows, n_valid_windows) or None on failure.
    """
    ts_path = os.path.join(session_path, "frame_timestamps.csv")
    hr_path = os.path.join(session_path, "hr_labels.csv")
    meta_path = os.path.join(session_path, "metadata.json")

    if not os.path.isfile(ts_path):
        return None, "missing frame_timestamps.csv"
    if not os.path.isfile(hr_path):
        return None, "missing hr_labels.csv"

    with open(meta_path) as f:
        meta = json.load(f)
    video_start_utc = pd.to_datetime(meta["start_time_utc"], utc=True).to_pydatetime()

    frame_ts = pd.read_csv(ts_path)
    hr_df = pd.read_csv(hr_path)

    windows = build_windows(frame_ts, hr_df, video_start_utc)
    frame_hr = interpolate_frame_hr(frame_ts, hr_df, video_start_utc)

    windows.to_csv(os.path.join(session_path, "windows.csv"), index=False)
    frame_hr.to_csv(os.path.join(session_path, "frame_hr.csv"), index=False)

    n_total = len(windows)
    n_valid = int(windows["valid"].sum())
    return (n_total, n_valid), None


# ── Main orchestration ───────────────────────────────────────────────

def batch_process(export_path, data_dir, force=False, dry_run=False):
    """Parse HR data and align all sessions in one pass.

    Returns list of result dicts for summary.
    """
    # Discover sessions
    sessions = discover_sessions_with_metadata(data_dir)
    print(f"\nDiscovered {len(sessions)} session(s) in {data_dir}")

    if not sessions:
        print("No sessions found.")
        return []

    # Check which sessions need processing
    to_process = []
    already_done = []
    for s in sessions:
        hr_exists = os.path.isfile(os.path.join(s["path"], "hr_labels.csv"))
        win_exists = os.path.isfile(os.path.join(s["path"], "windows.csv"))

        if hr_exists and win_exists and not force:
            already_done.append(s)
        else:
            to_process.append(s)

    if already_done:
        print(f"Already processed: {len(already_done)} session(s) (use --force to redo)")
    print(f"To process: {len(to_process)} session(s)")

    if dry_run:
        print(f"\n--- DRY RUN ---")
        for s in sessions:
            hr_exists = os.path.isfile(os.path.join(s["path"], "hr_labels.csv"))
            win_exists = os.path.isfile(os.path.join(s["path"], "windows.csv"))
            status = "DONE" if (hr_exists and win_exists) else "NEEDS PROCESSING"
            print(f"  {s['name']}  [{s['condition']}]  {s['subject_id']}  "
                  f"{s['start_utc'].strftime('%H:%M:%S')}-{s['end_utc'].strftime('%H:%M:%S')}  "
                  f"{status}")
        return []

    if not to_process and not force:
        print("Nothing to do.")
        return []

    # Determine which sessions to parse HR for
    sessions_for_hr = to_process if not force else sessions

    # Single-pass XML parse
    print(f"\nParsing {export_path} ...")
    buckets = parse_xml_to_sessions(export_path, sessions_for_hr)

    # Process each session
    results = []
    for s in sessions_for_hr:
        name = s["name"]
        records = buckets.get(name, [])

        result = {
            "session": name,
            "condition": s["condition"],
            "subject_id": s["subject_id"],
            "hr_records": len(records),
            "hr_saved": False,
            "windows_total": 0,
            "windows_valid": 0,
            "aligned": False,
            "error": None,
        }

        # Save HR labels
        if len(records) == 0:
            result["error"] = "no HR records in time window"
            print(f"\n  {name}: NO HR records found — check Apple Watch was logging")
            results.append(result)
            continue

        n_saved, hr_path = save_session_hr(s["path"], records, export_path)
        result["hr_records"] = n_saved
        result["hr_saved"] = True

        # Run alignment
        align_result, align_error = run_alignment(s["path"])
        if align_error:
            result["error"] = f"alignment failed: {align_error}"
            print(f"\n  {name}: HR saved ({n_saved} records), alignment failed: {align_error}")
        else:
            n_total, n_valid = align_result
            result["windows_total"] = n_total
            result["windows_valid"] = n_valid
            result["aligned"] = True
            print(f"\n  {name} [{s['condition']}]: "
                  f"{n_saved} HR records, {n_valid}/{n_total} valid windows")

        results.append(result)

    # Summary
    print(f"\n{'='*50}")
    print(f"Batch Processing Summary")
    print(f"{'='*50}")
    n_ok = sum(1 for r in results if r["aligned"])
    n_no_hr = sum(1 for r in results if r["hr_records"] == 0)
    n_fail = sum(1 for r in results if r["error"] and r["hr_records"] > 0)
    total_hr = sum(r["hr_records"] for r in results)
    total_valid_win = sum(r["windows_valid"] for r in results)

    print(f"Sessions processed: {len(results)}")
    print(f"  Successfully aligned: {n_ok}")
    print(f"  No HR data: {n_no_hr}")
    print(f"  Alignment failed: {n_fail}")
    print(f"Total HR records: {total_hr}")
    print(f"Total valid windows: {total_valid_win}")

    if n_no_hr > 0:
        print(f"\nSessions with no HR data:")
        for r in results:
            if r["hr_records"] == 0:
                print(f"  {r['session']}: {r['error']}")

    if n_fail > 0:
        print(f"\nSessions with alignment failures:")
        for r in results:
            if r["error"] and r["hr_records"] > 0:
                print(f"  {r['session']}: {r['error']}")

    return results


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

    # ── Test 1: Session discovery with metadata ──────────────────
    print("\nTest 1: Session discovery")
    with tempfile.TemporaryDirectory() as tmp:
        # Good session
        s1 = os.path.join(tmp, "session_001")
        os.makedirs(s1)
        with open(os.path.join(s1, "metadata.json"), "w") as f:
            json.dump({
                "start_time_utc": "2026-04-07T18:30:00+00:00",
                "end_time_utc": "2026-04-07T18:31:00+00:00",
                "condition": "normal",
                "subject_id": "pedro",
            }, f)

        # Missing metadata
        s2 = os.path.join(tmp, "session_002")
        os.makedirs(s2)

        # Missing end_time
        s3 = os.path.join(tmp, "session_003")
        os.makedirs(s3)
        with open(os.path.join(s3, "metadata.json"), "w") as f:
            json.dump({"start_time_utc": "2026-04-07T18:32:00+00:00"}, f)

        sessions = discover_sessions_with_metadata(tmp)
        check("Finds 1 valid session (skips 2 bad ones)", len(sessions) == 1)
        check("Correct session name", sessions[0]["name"] == "session_001")
        check("Has start_utc", sessions[0]["start_utc"] is not None)
        check("Has condition", sessions[0]["condition"] == "normal")

    # ── Test 2: Single-pass XML bucketing ────────────────────────
    print("\nTest 2: Single-pass XML bucketing")
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
 <Record type="HKQuantityTypeIdentifierHeartRate" startDate="2026-04-07 14:30:05 -0400" value="72"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" startDate="2026-04-07 14:30:15 -0400" value="75"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" startDate="2026-04-07 14:30:25 -0400" value="71"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" startDate="2026-04-07 14:31:30 -0400" value="80"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" startDate="2026-04-07 14:31:40 -0400" value="82"/>
 <Record type="HKQuantityTypeIdentifierStepCount" startDate="2026-04-07 14:30:05 -0400" value="10"/>
</HealthData>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        xml_path = f.name

    try:
        # Session 1: 18:30:00 - 18:31:00 UTC (= 14:30 - 14:31 ET)
        # Session 2: 18:31:20 - 18:32:00 UTC (= 14:31:20 - 14:32 ET)
        mock_sessions = [
            {
                "name": "session_A",
                "start_utc": datetime(2026, 4, 7, 18, 30, 0, tzinfo=timezone.utc),
                "end_utc": datetime(2026, 4, 7, 18, 31, 0, tzinfo=timezone.utc),
            },
            {
                "name": "session_B",
                "start_utc": datetime(2026, 4, 7, 18, 31, 20, tzinfo=timezone.utc),
                "end_utc": datetime(2026, 4, 7, 18, 32, 0, tzinfo=timezone.utc),
            },
        ]

        buckets = parse_xml_to_sessions(xml_path, mock_sessions)
        check("Session A gets 3 records", len(buckets["session_A"]) == 3)
        check("Session B gets 2 records", len(buckets["session_B"]) == 2)
        check("Session A first HR is 72", buckets["session_A"][0]["hr_bpm"] == 72.0)
        check("Session B first HR is 80", buckets["session_B"][0]["hr_bpm"] == 80.0)
        check("Step count ignored", sum(len(v) for v in buckets.values()) == 5)
    finally:
        os.unlink(xml_path)

    # ── Test 3: Non-overlapping sessions routed correctly ────────
    print("\nTest 3: No cross-contamination between sessions")
    # Session A and B don't overlap, so no record should appear in both
    all_a_times = {r["timestamp_utc"] for r in buckets["session_A"]}
    all_b_times = {r["timestamp_utc"] for r in buckets["session_B"]}
    check("No overlap between session buckets", len(all_a_times & all_b_times) == 0)

    # ── Test 4: Empty bucket handling ────────────────────────────
    print("\nTest 4: Session with no matching HR data")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        xml_path = f.name
    try:
        # Session outside the XML data range
        far_session = [{
            "name": "session_far",
            "start_utc": datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "end_utc": datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
        }]
        far_buckets = parse_xml_to_sessions(xml_path, far_session)
        check("Out-of-range session gets 0 records", len(far_buckets["session_far"]) == 0)
    finally:
        os.unlink(xml_path)

    # ── Test 5: Dry run doesn't modify anything ──────────────────
    print("\nTest 5: Dry run is non-destructive")
    with tempfile.TemporaryDirectory() as tmp:
        s = os.path.join(tmp, "session_test")
        os.makedirs(s)
        with open(os.path.join(s, "metadata.json"), "w") as f:
            json.dump({
                "start_time_utc": "2026-04-07T18:30:00+00:00",
                "end_time_utc": "2026-04-07T18:31:00+00:00",
                "condition": "normal",
            }, f)

        results = batch_process("fake.xml", tmp, dry_run=True)
        check("Dry run returns empty results", len(results) == 0)
        check("No hr_labels.csv created",
              not os.path.isfile(os.path.join(s, "hr_labels.csv")))

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch parse Apple Health HR and align all sessions"
    )
    parser.add_argument("--export", help="Path to Apple Health export.xml")
    parser.add_argument("--data-dir", default=config.DATA_RAW,
                        help="Directory containing session folders")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess sessions that already have hr_labels.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without doing it")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    if not args.export:
        parser.error("--export is required (path to Apple Health export.xml)")

    if not os.path.isfile(args.export):
        print(f"ERROR: Export file not found: {args.export}")
        sys.exit(1)

    batch_process(args.export, args.data_dir, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
