"""
Face detection and ROI-based RGB signal extraction using MediaPipe Face Mesh.

Usage:
    py rppg/face_detect.py --video data/raw/session_.../video.avi
    py rppg/face_detect.py --video data/raw/session_.../video.avi --region combined
    py rppg/face_detect.py --video data/raw/session_.../video.avi --save-debug-frames debug_out/
    py rppg/face_detect.py --test
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Path to the downloaded face landmarker model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")

# MediaPipe Face Mesh landmark indices for ROI regions
# These define polygons on the face mesh for forehead and cheek areas.
FOREHEAD_LANDMARKS = [10, 67, 109, 108, 151, 337, 338, 297]
LEFT_CHEEK_LANDMARKS = [116, 123, 147, 187, 205, 36, 50]
RIGHT_CHEEK_LANDMARKS = [345, 352, 376, 411, 425, 266, 280]

MAX_SHORT_GAP = 5  # frames — gaps this size or smaller get interpolated


# ── ROI mask extraction ──────────────────────────────────────────────

def get_roi_mask(landmarks, frame_shape, region="forehead"):
    """Build a binary mask for the specified face region.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList
        frame_shape: (height, width, channels)
        region: "forehead", "cheeks", or "combined"

    Returns:
        Binary mask (h, w) as np.uint8, or None if region is invalid.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    def _add_polygon(indices):
        pts = np.array([
            [int(landmarks[i].x * w), int(landmarks[i].y * h)]
            for i in indices
        ], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)

    if region == "forehead":
        _add_polygon(FOREHEAD_LANDMARKS)
    elif region == "cheeks":
        _add_polygon(LEFT_CHEEK_LANDMARKS)
        _add_polygon(RIGHT_CHEEK_LANDMARKS)
    elif region == "combined":
        _add_polygon(FOREHEAD_LANDMARKS)
        _add_polygon(LEFT_CHEEK_LANDMARKS)
        _add_polygon(RIGHT_CHEEK_LANDMARKS)
    else:
        return None

    return mask


def extract_frame_rgb(frame, mask):
    """Compute mean RGB over masked pixels in a single frame.

    Args:
        frame: BGR image (h, w, 3) as np.uint8
        mask: binary mask (h, w) as np.uint8 (255 = included)

    Returns:
        (r, g, b) tuple of floats, or (NaN, NaN, NaN) if mask is empty.
    """
    pixels = frame[mask == 255]
    if len(pixels) == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean_bgr = pixels.mean(axis=0)
    return (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))


# ── Gap interpolation ────────────────────────────────────────────────

def interpolate_short_gaps(df, max_gap=MAX_SHORT_GAP):
    """Linearly interpolate short NaN gaps in RGB columns.

    Only fills consecutive NaN runs of length <= max_gap.
    Longer gaps are left as NaN.

    Adds a column 'interpolated' (bool) marking which frames were filled.

    Args:
        df: DataFrame with columns [rgb_r, rgb_g, rgb_b, valid_detection]
        max_gap: maximum gap length to interpolate

    Returns:
        DataFrame with interpolated values and 'interpolated' column.
    """
    df = df.copy()
    df["interpolated"] = False

    # Identify NaN runs
    is_nan = df["valid_detection"] == False
    run_id = (is_nan != is_nan.shift()).cumsum()

    for rid in run_id[is_nan].unique():
        run_mask = (run_id == rid) & is_nan
        run_len = run_mask.sum()
        if run_len <= max_gap:
            # Mark these frames as interpolated
            df.loc[run_mask, "interpolated"] = True

    # Interpolate only the short-gap frames
    # First, temporarily fill long gaps with a sentinel so interpolate skips them
    for col in ["rgb_r", "rgb_g", "rgb_b"]:
        # Only interpolate where we marked as short gap
        series = df[col].copy()
        long_gap_mask = is_nan & ~df["interpolated"]
        sentinel_vals = series[long_gap_mask].copy()

        # Set long gaps to a value that won't be interpolated
        series[long_gap_mask] = -9999
        series = series.replace(-9999, float("nan"))

        # Now all NaNs are short gaps — interpolate them
        interp = series.interpolate(method="linear", limit_direction="both")

        # Restore long gaps as NaN
        interp[long_gap_mask] = float("nan")
        df[col] = interp

    return df


# ── Main extraction ──────────────────────────────────────────────────

def extract_rgb_signals(video_path, region="forehead"):
    """Extract per-frame mean RGB signals from a video using face mesh ROI.

    Args:
        video_path: path to video file
        region: "forehead", "cheeks", or "combined"

    Returns:
        DataFrame with columns:
            frame_idx, rgb_r, rgb_g, rgb_b, valid_detection, interpolated
        Also prints detection stats.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = FaceLandmarker.create_from_options(options)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame_idx * 1000 / fps)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            mask = get_roi_mask(landmarks, frame.shape, region)
            r, g, b = extract_frame_rgb(frame, mask)
            valid = True
        else:
            r, g, b = float("nan"), float("nan"), float("nan")
            valid = False

        rows.append({
            "frame_idx": frame_idx,
            "rgb_r": r,
            "rgb_g": g,
            "rgb_b": b,
            "valid_detection": valid,
        })
        frame_idx += 1

    cap.release()
    landmarker.close()

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: No frames read from video.")
        return df

    # Interpolate short gaps
    df = interpolate_short_gaps(df)

    # Print stats
    total = len(df)
    detected = df["valid_detection"].sum()
    interpolated_count = df["interpolated"].sum()
    long_gap_missing = ((~df["valid_detection"]) & (~df["interpolated"])).sum()

    print(f"Face detection stats:")
    print(f"  Total frames:       {total}")
    print(f"  Detected:           {detected} ({100*detected/total:.1f}%)")
    print(f"  Interpolated (short gap): {interpolated_count}")
    print(f"  Missing (long gap):       {long_gap_missing}")

    return df


def rgb_signals_to_numpy(df):
    """Convert DataFrame RGB columns to (N, 3) NumPy array.

    NaN rows are preserved as NaN in the array.
    """
    return df[["rgb_r", "rgb_g", "rgb_b"]].to_numpy()


# ── Debug visualization ──────────────────────────────────────────────

def save_debug_frames(video_path, output_dir, region="forehead", sample_interval=30):
    """Save sampled frames with ROI overlay for visual verification.

    Draws the ROI polygon on every sample_interval-th frame and saves as PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = FaceLandmarker.create_from_options(options)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_idx * 1000 / fps)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            display = frame.copy()

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                mask = get_roi_mask(landmarks, frame.shape, region)
                # Overlay mask as semi-transparent green
                overlay = display.copy()
                overlay[mask == 255] = [0, 255, 0]
                display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)
                label = f"Frame {frame_idx} - {region} ROI"
            else:
                label = f"Frame {frame_idx} - NO FACE DETECTED"

            cv2.putText(display, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(out_path, display)
            saved += 1

        frame_idx += 1

    cap.release()
    landmarker.close()
    print(f"Saved {saved} debug frames to {output_dir}")


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

    # ── Test 1: ROI mask sanity ──────────────────────────────────
    print("\nTest 1: ROI mask sanity")

    # Create a mock landmarks object with known positions
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # Place landmarks in a known pattern (normalized 0-1 coords)
    mock_lm = [MockLandmark(0.5, 0.5)] * 500  # fill with center points

    # Set forehead landmarks to form a small polygon in upper-center
    for i in FOREHEAD_LANDMARKS:
        mock_lm[i] = MockLandmark(0.4 + 0.02 * (i % 4), 0.2 + 0.01 * (i % 3))

    frame_shape = (480, 640, 3)
    mask = get_roi_mask(mock_lm, frame_shape, "forehead")
    check("Forehead mask is not None", mask is not None)
    check("Forehead mask has correct shape", mask.shape == (480, 640))
    check("Forehead mask has nonzero pixels", np.sum(mask > 0) > 0)
    check("Forehead mask is not entire frame", np.sum(mask > 0) < 480 * 640)

    # Cheeks mask
    for i in LEFT_CHEEK_LANDMARKS + RIGHT_CHEEK_LANDMARKS:
        mock_lm[i] = MockLandmark(0.3 + 0.03 * (i % 5), 0.5 + 0.01 * (i % 3))
    mask_cheeks = get_roi_mask(mock_lm, frame_shape, "cheeks")
    check("Cheeks mask has nonzero pixels", np.sum(mask_cheeks > 0) > 0)

    # Combined mask
    mask_combined = get_roi_mask(mock_lm, frame_shape, "combined")
    check("Combined mask >= forehead mask area",
          np.sum(mask_combined > 0) >= np.sum(mask > 0))

    # ── Test 2: extract_frame_rgb ────────────────────────────────
    print("\nTest 2: extract_frame_rgb with known color")
    # Create a solid red frame (BGR: 0, 0, 255)
    red_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    red_frame[:, :] = [0, 0, 255]  # BGR
    # Full-frame mask
    full_mask = np.ones((480, 640), dtype=np.uint8) * 255
    r, g, b = extract_frame_rgb(red_frame, full_mask)
    check("Red channel = 255", r == 255.0)
    check("Green channel = 0", g == 0.0)
    check("Blue channel = 0", b == 0.0)

    # Empty mask
    empty_mask = np.zeros((480, 640), dtype=np.uint8)
    r2, g2, b2 = extract_frame_rgb(red_frame, empty_mask)
    check("Empty mask returns NaN", np.isnan(r2) and np.isnan(g2) and np.isnan(b2))

    # ── Test 3: DataFrame schema and frame order ─────────────────
    print("\nTest 3: Output schema and frame order")
    # Simulate what extract_rgb_signals returns
    rows = []
    for i in range(20):
        rows.append({
            "frame_idx": i,
            "rgb_r": 100.0 + i,
            "rgb_g": 150.0,
            "rgb_b": 200.0,
            "valid_detection": True,
        })
    df = pd.DataFrame(rows)
    df = interpolate_short_gaps(df)

    expected_cols = ["frame_idx", "rgb_r", "rgb_g", "rgb_b",
                     "valid_detection", "interpolated"]
    check("Has correct columns", list(df.columns) == expected_cols)
    check("Frame order preserved", list(df["frame_idx"]) == list(range(20)))
    check("No interpolation needed (all valid)", df["interpolated"].sum() == 0)

    # NumPy conversion
    arr = rgb_signals_to_numpy(df)
    check("NumPy shape is (20, 3)", arr.shape == (20, 3))

    # ── Test 4: Detection failure + short gap interpolation ──────
    print("\nTest 4: Short gap interpolation")
    rows2 = []
    for i in range(20):
        if 5 <= i <= 7:  # 3-frame gap (short, should be interpolated)
            rows2.append({
                "frame_idx": i, "rgb_r": float("nan"), "rgb_g": float("nan"),
                "rgb_b": float("nan"), "valid_detection": False,
            })
        elif 12 <= i <= 18:  # 7-frame gap (long, should stay NaN)
            rows2.append({
                "frame_idx": i, "rgb_r": float("nan"), "rgb_g": float("nan"),
                "rgb_b": float("nan"), "valid_detection": False,
            })
        else:
            rows2.append({
                "frame_idx": i, "rgb_r": 100.0, "rgb_g": 150.0,
                "rgb_b": 200.0, "valid_detection": True,
            })
    df2 = pd.DataFrame(rows2)
    df2 = interpolate_short_gaps(df2)

    # Short gap (frames 5-7) should be interpolated
    short_gap = df2[(df2["frame_idx"] >= 5) & (df2["frame_idx"] <= 7)]
    check("Short gap frames are interpolated", short_gap["interpolated"].all())
    check("Short gap rgb_r is not NaN", short_gap["rgb_r"].notna().all())

    # Long gap (frames 12-18) should remain NaN
    long_gap = df2[(df2["frame_idx"] >= 12) & (df2["frame_idx"] <= 18)]
    check("Long gap frames are NOT interpolated", (~long_gap["interpolated"]).all())
    check("Long gap rgb_r is NaN", long_gap["rgb_r"].isna().all())

    # Valid detection column is preserved (not overwritten by interpolation)
    check("valid_detection unchanged for short gap",
          (~short_gap["valid_detection"]).all())

    # ── Test 5: Interpolation only fills short gaps ──────────────
    print("\nTest 5: Mixed gaps boundary check")
    rows3 = []
    for i in range(15):
        if i in [3, 4]:  # 2-frame gap (short)
            rows3.append({
                "frame_idx": i, "rgb_r": float("nan"), "rgb_g": float("nan"),
                "rgb_b": float("nan"), "valid_detection": False,
            })
        elif i in [8, 9, 10, 11, 12, 13]:  # 6-frame gap (long, > MAX_SHORT_GAP=5)
            rows3.append({
                "frame_idx": i, "rgb_r": float("nan"), "rgb_g": float("nan"),
                "rgb_b": float("nan"), "valid_detection": False,
            })
        else:
            rows3.append({
                "frame_idx": i, "rgb_r": 80.0, "rgb_g": 120.0,
                "rgb_b": 160.0, "valid_detection": True,
            })
    df3 = pd.DataFrame(rows3)
    df3 = interpolate_short_gaps(df3)

    check("2-frame gap interpolated", df3.loc[3, "interpolated"] and df3.loc[4, "interpolated"])
    check("6-frame gap NOT interpolated", not df3.loc[8, "interpolated"] and not df3.loc[13, "interpolated"])
    check("6-frame gap still NaN", df3.loc[8:13, "rgb_r"].isna().all())

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Extract face ROI RGB signals")
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--region", choices=["forehead", "cheeks", "combined"],
                        default="forehead", help="ROI region (default: forehead)")
    parser.add_argument("--output", help="Save RGB signals CSV to this path")
    parser.add_argument("--save-debug-frames", metavar="DIR",
                        help="Save sampled frames with ROI overlay to directory")
    parser.add_argument("--debug-interval", type=int, default=30,
                        help="Sample every N frames for debug output (default: 30)")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    if not args.video:
        parser.error("--video is required (or use --test)")

    if args.save_debug_frames:
        save_debug_frames(args.video, args.save_debug_frames,
                          region=args.region, sample_interval=args.debug_interval)

    df = extract_rgb_signals(args.video, region=args.region)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} frames to {args.output}")
    else:
        print(f"\nFirst 5 rows:")
        print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
