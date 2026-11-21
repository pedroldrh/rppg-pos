"""
Webcam video recorder for rPPG data collection.

Usage:
    py collect/record_video.py --duration 60 --condition normal --subject pedro
    py collect/record_video.py --duration 30 --condition low_light --subject pedro --notes "desk lamp off"

Flow:
    1. Opens webcam, shows live preview
    2. Press SPACE to begin
    3. 3-2-1 countdown (sync cue for Apple Watch)
    4. Records for --duration seconds (press Q to stop early)
    5. Saves video.avi, frame_timestamps.csv, metadata.json
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import cv2

# Allow imports when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def record_session(duration_sec, condition, subject_id, notes=""):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Preview phase ────────────────────────────────────────────────
    print("Preview is live. Press SPACE to begin recording, Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preview = frame.copy()
        cv2.putText(preview, "PREVIEW - press SPACE to start",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, f"Condition: {condition}  Subject: {subject_id}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Circadify Recorder", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("Cancelled.")
            return

    # ── Countdown 3-2-1 ──────────────────────────────────────────────
    for i in range(3, 0, -1):
        start_tick = time.time()
        while time.time() - start_tick < 1.0:
            ret, frame = cap.read()
            if not ret:
                break
            overlay = frame.copy()
            cv2.putText(overlay, str(i), (actual_w // 2 - 30, actual_h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
            cv2.putText(overlay, "Start Apple Watch HR logging now!",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Circadify Recorder", overlay)
            cv2.waitKey(1)

    # ── Prepare output directory ─────────────────────────────────────
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = os.path.join(config.DATA_RAW, session_name)
    os.makedirs(session_dir, exist_ok=True)

    # Choose codec — try MJPG first (most compatible), fall back to avc1 on Mac
    video_path = os.path.join(session_dir, "video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(video_path, fourcc, config.FPS, (actual_w, actual_h))
    if not writer.isOpened():
        writer.release()
        video_path = os.path.join(session_dir, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(video_path, fourcc, config.FPS, (actual_w, actual_h))

    timestamps_path = os.path.join(session_dir, "frame_timestamps.csv")
    ts_file = open(timestamps_path, "w", newline="")
    ts_writer = csv.writer(ts_file)
    ts_writer.writerow(["frame_idx", "elapsed_ms", "system_time_ms"])

    # ── Record ───────────────────────────────────────────────────────
    start_time_utc = datetime.now(timezone.utc)
    start_mono = time.monotonic()
    start_system = time.time()
    frame_idx = 0
    stopped_early = False

    print(f"Recording for {duration_sec}s ... press Q to stop early.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now_mono = time.monotonic()
        elapsed_ms = (now_mono - start_mono) * 1000.0
        system_time_ms = time.time() * 1000.0

        writer.write(frame)
        ts_writer.writerow([frame_idx, f"{elapsed_ms:.1f}", f"{system_time_ms:.1f}"])
        frame_idx += 1

        # Show recording indicator
        display = frame.copy()
        elapsed_sec = elapsed_ms / 1000.0
        cv2.circle(display, (20, 25), 8, (0, 0, 255), -1)
        cv2.putText(display, f"REC {elapsed_sec:.1f}s / {duration_sec}s",
                    (35, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Circadify Recorder", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stopped_early = True
            break
        if elapsed_sec >= duration_sec:
            break

    end_time_utc = datetime.now(timezone.utc)
    duration_actual = (time.monotonic() - start_mono)

    # ── Cleanup ──────────────────────────────────────────────────────
    ts_file.close()
    writer.release()
    cap.release()
    cv2.destroyAllWindows()

    # ── Compute actual FPS ───────────────────────────────────────────
    fps_actual = frame_idx / duration_actual if duration_actual > 0 else 0

    # ── Save metadata ────────────────────────────────────────────────
    metadata = {
        "start_time_utc": start_time_utc.isoformat(),
        "end_time_utc": end_time_utc.isoformat(),
        "duration_sec_requested": duration_sec,
        "duration_sec_actual": round(duration_actual, 2),
        "fps_requested": config.FPS,
        "fps_actual": round(fps_actual, 2),
        "resolution": [actual_w, actual_h],
        "condition": condition,
        "subject_id": subject_id,
        "notes": notes,
        "frame_count": frame_idx,
        "stopped_early": stopped_early,
    }
    metadata_path = os.path.join(session_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSession saved to: {session_dir}")
    print(f"  Frames: {frame_idx}")
    print(f"  Duration: {duration_actual:.1f}s (requested {duration_sec}s)")
    print(f"  FPS: {fps_actual:.1f} (requested {config.FPS})")
    return session_dir


def main():
    parser = argparse.ArgumentParser(description="Record face video for rPPG")
    parser.add_argument("--duration", type=int, default=config.DURATION_SEC,
                        help="Recording duration in seconds")
    parser.add_argument("--condition", choices=["normal", "low_light"],
                        default="normal", help="Lighting condition")
    parser.add_argument("--subject", default="subject_01", help="Subject ID")
    parser.add_argument("--notes", default="", help="Optional session notes")
    parser.add_argument("--test", action="store_true",
                        help="Quick 3-second test recording")
    args = parser.parse_args()

    if args.test:
        args.duration = 3

    record_session(args.duration, args.condition, args.subject, args.notes)


if __name__ == "__main__":
    main()
