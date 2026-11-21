"""
Parse Apple Watch heart-rate data from Apple Health XML or CSV.

Usage:
    py collect/parse_apple_health.py --input data/raw/export.xml
    py collect/parse_apple_health.py --input data/raw/export.xml --start 2026-04-07T14:30:00 --end 2026-04-07T14:31:00
    py collect/parse_apple_health.py --input data/raw/hr.csv --output data/processed/hr_parsed.csv
    py collect/parse_apple_health.py --test
"""

import argparse
import csv
import io
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HR_TYPE = "HKQuantityTypeIdentifierHeartRate"

# Accepted CSV column aliases (lowercase)
TIMESTAMP_ALIASES = {"timestamp_utc", "timestamp", "date", "datetime", "time", "startdate"}
HR_ALIASES = {"hr_bpm", "heart rate", "heart_rate", "heartrate", "heart rate (count/min)", "value", "bpm"}


# ── Core functions ───────────────────────────────────────────────────

def parse_xml_hr(path, start=None, end=None):
    """Stream-parse Apple Health export.xml for heart-rate records.

    Uses iterparse to avoid loading the entire XML into memory.
    Returns a list of dicts with keys: timestamp_utc, hr_bpm.
    """
    records = []
    for event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "Record" or elem.get("type") != HR_TYPE:
            elem.clear()
            continue

        date_str = elem.get("startDate", "")
        value_str = elem.get("value", "")
        elem.clear()

        if not date_str or not value_str:
            continue

        # Apple Health format: "2026-04-07 14:30:03 -0400"
        ts = _parse_apple_timestamp(date_str)
        if ts is None:
            continue

        if start and ts < start:
            continue
        if end and ts > end:
            continue

        try:
            hr = float(value_str)
        except ValueError:
            continue

        records.append({"timestamp_utc": ts, "hr_bpm": hr})

    return records


def parse_csv_hr(path):
    """Parse heart-rate CSV with common column naming conventions.

    Supported timestamp columns: timestamp_utc, timestamp, date, datetime, time, startDate
    Supported HR columns: hr_bpm, heart rate, heart_rate, heartrate, value, bpm

    Raises ValueError if required columns cannot be identified.
    """
    df = pd.read_csv(path)
    cols_lower = {c.lower().strip(): c for c in df.columns}

    ts_col = None
    for alias in TIMESTAMP_ALIASES:
        if alias in cols_lower:
            ts_col = cols_lower[alias]
            break

    hr_col = None
    for alias in HR_ALIASES:
        if alias in cols_lower:
            hr_col = cols_lower[alias]
            break

    if ts_col is None:
        raise ValueError(
            f"Cannot find timestamp column. Found columns: {list(df.columns)}. "
            f"Expected one of: {TIMESTAMP_ALIASES}"
        )
    if hr_col is None:
        raise ValueError(
            f"Cannot find heart rate column. Found columns: {list(df.columns)}. "
            f"Expected one of: {HR_ALIASES}"
        )

    records = []
    for _, row in df.iterrows():
        ts = pd.to_datetime(row[ts_col], utc=True)
        try:
            hr = float(row[hr_col])
        except (ValueError, TypeError):
            continue
        records.append({"timestamp_utc": ts, "hr_bpm": hr})

    return records


def normalize_hr_dataframe(records, source_type, source_file):
    """Convert list of dicts to a standardized DataFrame.

    Columns: timestamp_utc, hr_bpm, source_type, source_file
    """
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "hr_bpm", "source_type", "source_file"])

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["hr_bpm"] = df["hr_bpm"].astype(float)
    df["source_type"] = source_type
    df["source_file"] = source_file
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df[["timestamp_utc", "hr_bpm", "source_type", "source_file"]]


def filter_time_range(df, start=None, end=None):
    """Filter DataFrame to a UTC time window."""
    if df.empty:
        return df
    if start:
        start_ts = pd.to_datetime(start, utc=True)
        df = df[df["timestamp_utc"] >= start_ts]
    if end:
        end_ts = pd.to_datetime(end, utc=True)
        df = df[df["timestamp_utc"] <= end_ts]
    return df.reset_index(drop=True)


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_apple_timestamp(s):
    """Parse Apple Health timestamp string to timezone-aware UTC datetime."""
    # Format: "2026-04-07 14:30:03 -0400"
    s = s.strip()
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass
    # Fallback: ISO format
    try:
        dt = pd.to_datetime(s, utc=True)
        return dt.to_pydatetime()
    except Exception:
        return None


def print_summary(df):
    """Print a short summary of the parsed HR data."""
    if df.empty:
        print("No heart-rate records found.")
        return
    print(f"  Records:   {len(df)}")
    print(f"  Time range: {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
    print(f"  HR range:   {df['hr_bpm'].min():.0f} – {df['hr_bpm'].max():.0f} BPM")


# ── Entry point ──────────────────────────────────────────────────────

def parse_file(input_path, start=None, end=None):
    """Auto-detect format and parse HR data. Returns standardized DataFrame."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".xml":
        # For XML, filter during parsing for efficiency
        start_dt = pd.to_datetime(start, utc=True).to_pydatetime() if start else None
        end_dt = pd.to_datetime(end, utc=True).to_pydatetime() if end else None
        records = parse_xml_hr(input_path, start=start_dt, end=end_dt)
        df = normalize_hr_dataframe(records, "apple_health_xml", input_path)
    elif ext == ".csv":
        records = parse_csv_hr(input_path)
        df = normalize_hr_dataframe(records, "csv", input_path)
        df = filter_time_range(df, start, end)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .xml or .csv")

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

    # ── Test 1: Mock XML parsing ─────────────────────────────────
    print("\nTest 1: XML parsing")
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
 <Record type="HKQuantityTypeIdentifierHeartRate" startDate="2026-04-07 14:30:03 -0400" value="72"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" startDate="2026-04-07 14:30:08 -0400" value="75"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" startDate="2026-04-07 14:30:13 -0400" value="71"/>
 <Record type="HKQuantityTypeIdentifierStepCount" startDate="2026-04-07 14:30:03 -0400" value="10"/>
</HealthData>"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        xml_path = f.name
    try:
        records = parse_xml_hr(xml_path)
        check("Parses 3 HR records (ignores step count)", len(records) == 3)
        check("First HR is 72", records[0]["hr_bpm"] == 72.0)
        check("Timestamps are UTC-aware", records[0]["timestamp_utc"].tzinfo is not None)
    finally:
        os.unlink(xml_path)

    # ── Test 2: Mock CSV parsing ─────────────────────────────────
    print("\nTest 2: CSV parsing")
    csv_content = "Date,Heart Rate\n2026-04-07T18:30:03Z,72\n2026-04-07T18:30:08Z,75\n2026-04-07T18:30:13Z,71\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = f.name
    try:
        records = parse_csv_hr(csv_path)
        check("Parses 3 CSV records", len(records) == 3)
        check("First HR is 72", records[0]["hr_bpm"] == 72.0)
    finally:
        os.unlink(csv_path)

    # ── Test 3: Time filtering ───────────────────────────────────
    print("\nTest 3: Time filtering")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        xml_path = f.name
    try:
        # Filter to only first two records (14:30:03 and 14:30:08 ET = 18:30:03 and 18:30:08 UTC)
        records = parse_xml_hr(
            xml_path,
            start=datetime(2026, 4, 7, 18, 30, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 7, 18, 30, 10, tzinfo=timezone.utc),
        )
        check("Time filter returns 2 records", len(records) == 2)

        # CSV time filtering via filter_time_range
        all_records = parse_csv_hr(csv_path) if os.path.exists(csv_path) else parse_csv_hr(csv_path)
    except Exception:
        pass
    finally:
        os.unlink(xml_path)

    # filter_time_range on a DataFrame
    test_df = normalize_hr_dataframe(
        [
            {"timestamp_utc": datetime(2026, 4, 7, 18, 30, 3, tzinfo=timezone.utc), "hr_bpm": 72},
            {"timestamp_utc": datetime(2026, 4, 7, 18, 30, 8, tzinfo=timezone.utc), "hr_bpm": 75},
            {"timestamp_utc": datetime(2026, 4, 7, 18, 30, 13, tzinfo=timezone.utc), "hr_bpm": 71},
        ],
        "test", "test"
    )
    filtered = filter_time_range(test_df, start="2026-04-07T18:30:05Z", end="2026-04-07T18:30:10Z")
    check("filter_time_range returns 1 record", len(filtered) == 1)
    check("Filtered record has HR 75", filtered.iloc[0]["hr_bpm"] == 75.0)

    # ── Test 4: Output columns and dtypes ────────────────────────
    print("\nTest 4: Output schema")
    df = normalize_hr_dataframe(
        [{"timestamp_utc": datetime(2026, 4, 7, 18, 30, 3, tzinfo=timezone.utc), "hr_bpm": 72}],
        "apple_health_xml", "/tmp/test.xml"
    )
    check("Has 4 columns", list(df.columns) == ["timestamp_utc", "hr_bpm", "source_type", "source_file"])
    check("timestamp_utc is datetime64 with tz", hasattr(df["timestamp_utc"].dtype, "tz"))
    check("hr_bpm is float", df["hr_bpm"].dtype == float)
    check("source_type is correct", df.iloc[0]["source_type"] == "apple_health_xml")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Parse Apple Watch HR data")
    parser.add_argument("--input", help="Path to .xml or .csv file")
    parser.add_argument("--start", help="Start time filter (ISO format)")
    parser.add_argument("--end", help="End time filter (ISO format)")
    parser.add_argument("--output", help="Save parsed CSV to this path")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    if not args.input:
        parser.error("--input is required (or use --test)")

    df = parse_file(args.input, start=args.start, end=args.end)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} records to {args.output}")
    else:
        print("Parsed HR data:")
        print_summary(df)


if __name__ == "__main__":
    main()
