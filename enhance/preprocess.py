"""
Low-light video preprocessing for rPPG improvement.

Applies targeted frame-level preprocessing before rPPG extraction.
The rPPG pipeline stays unchanged — only the input video is modified.

Operations:
    - CLAHE: adaptive local contrast enhancement on L channel (LAB space)
    - Gamma correction: nonlinear brightness mapping (gamma < 1 brightens)
    - Brightness normalization: scale mean luminance toward a target
    - Denoising: non-local means denoising (reduces photon noise)

Usage:
    py enhance/preprocess.py --video data/raw/session_.../video.avi --pipeline clahe
    py enhance/preprocess.py --video data/raw/session_.../video.avi --pipeline denoise_clahe_gamma
    py enhance/preprocess.py --video data/raw/session_.../video.avi --inspect 5
    py enhance/preprocess.py --test
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Individual operations ────────────────────────────────────────────

def apply_clahe(frame, clip_limit=None, tile_grid_size=None):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Operates on the L channel in LAB color space. Improves local contrast
    without globally shifting brightness. Particularly useful when face
    features are lost in low-contrast dark frames.

    Args:
        frame: BGR image (h, w, 3) as np.uint8
        clip_limit: contrast limit (default config.CLAHE_CLIP)
        tile_grid_size: grid size tuple (default config.CLAHE_GRID)

    Returns:
        Processed BGR frame as np.uint8
    """
    clip_limit = clip_limit if clip_limit is not None else config.CLAHE_CLIP
    tile_grid_size = tile_grid_size if tile_grid_size is not None else config.CLAHE_GRID

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gamma(frame, gamma=0.7):
    """Apply gamma correction.

    gamma < 1 brightens the image (lifts shadows).
    gamma > 1 darkens the image (compresses highlights).
    gamma = 1 is identity.

    Args:
        frame: BGR image (h, w, 3) as np.uint8
        gamma: exponent value

    Returns:
        Processed BGR frame as np.uint8
    """
    table = np.array([
        np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(frame, table)


def apply_brightness_norm(frame, target_mean=None, max_scale=None):
    """Normalize mean brightness toward a target value.

    Operates on the L channel in LAB space. Scales luminance so that
    the mean L value approaches target_mean. Clipping prevents highlight
    blowout.

    Args:
        frame: BGR image (h, w, 3) as np.uint8
        target_mean: desired mean luminance 0-255 (default config.BRIGHTNESS_NORM_TARGET)
        max_scale: maximum scale factor to prevent blowout
                   (default config.BRIGHTNESS_NORM_MAX_SCALE)

    Returns:
        Processed BGR frame as np.uint8
    """
    target_mean = target_mean if target_mean is not None else config.BRIGHTNESS_NORM_TARGET
    max_scale = max_scale if max_scale is not None else config.BRIGHTNESS_NORM_MAX_SCALE

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    current_mean = l_ch.mean()
    if current_mean < 1.0:
        # Nearly black frame — scaling won't help, just return as-is
        return frame

    scale = target_mean / current_mean
    scale = min(scale, max_scale)  # safety cap

    l_scaled = np.clip(l_ch.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    lab = cv2.merge([l_scaled, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_adaptive_gamma(frame, brightness_threshold=None, gamma=None):
    """Conditionally apply gamma correction based on frame brightness.

    Only enhances frames that are below the brightness threshold.
    Frames that are already bright enough pass through unchanged.
    This avoids degrading well-lit frames while recovering dark ones.

    Args:
        frame: BGR image (h, w, 3) as np.uint8
        brightness_threshold: L-channel mean below which gamma is applied
                              (default config.ADAPTIVE_BRIGHTNESS_THRESHOLD)
        gamma: gamma value to apply on dark frames
               (default config.ADAPTIVE_GAMMA)

    Returns:
        Processed BGR frame as np.uint8 (unchanged if bright enough)
    """
    brightness_threshold = brightness_threshold if brightness_threshold is not None else config.ADAPTIVE_BRIGHTNESS_THRESHOLD
    gamma = gamma if gamma is not None else config.ADAPTIVE_GAMMA

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mean_l = lab[:, :, 0].mean()

    if mean_l < brightness_threshold:
        return apply_gamma(frame, gamma=gamma)
    return frame


def apply_denoise(frame, h=None, h_color=None,
                  template_window_size=None, search_window_size=None):
    """Apply non-local means denoising.

    Reduces photon noise common in low-light conditions. Should generally
    be applied before contrast enhancement to avoid amplifying noise.

    Args:
        frame: BGR image (h, w, 3) as np.uint8
        h: luminance filter strength (default config.DENOISE_H)
        h_color: color filter strength (default config.DENOISE_H_COLOR)
        template_window_size: patch size (default config.DENOISE_TEMPLATE_WINDOW)
        search_window_size: search area (default config.DENOISE_SEARCH_WINDOW)

    Returns:
        Processed BGR frame as np.uint8
    """
    h = h if h is not None else config.DENOISE_H
    h_color = h_color if h_color is not None else config.DENOISE_H_COLOR
    template_window_size = template_window_size or config.DENOISE_TEMPLATE_WINDOW
    search_window_size = search_window_size or config.DENOISE_SEARCH_WINDOW

    return cv2.fastNlMeansDenoisingColored(
        frame, None, h, h_color, template_window_size, search_window_size
    )


# ── Pipeline execution ───────────────────────────────────────────────

# Map operation names to functions
_OP_MAP = {
    "clahe": apply_clahe,
    "gamma": apply_gamma,
    "brightness_norm": apply_brightness_norm,
    "denoise": apply_denoise,
    "adaptive_gamma": apply_adaptive_gamma,
}


def apply_pipeline(frame, pipeline_steps):
    """Apply a sequence of preprocessing operations to a single frame.

    Args:
        frame: BGR image (h, w, 3) as np.uint8
        pipeline_steps: list of (op_name, params_dict) tuples
                        e.g. [("clahe", {}), ("gamma", {"gamma": 0.7})]

    Returns:
        Processed BGR frame as np.uint8
    """
    result = frame
    for op_name, params in pipeline_steps:
        fn = _OP_MAP.get(op_name)
        if fn is None:
            raise ValueError(f"Unknown preprocessing operation: {op_name}")
        result = fn(result, **params)
    return result


def preprocess_video(video_path, output_path, pipeline_name):
    """Apply a named preprocessing pipeline to all frames of a video.

    Preserves frame count, fps, and frame size. Reports brightness stats.

    Args:
        video_path: input video path
        output_path: output video path
        pipeline_name: key into config.PREPROCESS_PIPELINES

    Returns:
        dict with summary stats:
            input_mean_brightness, output_mean_brightness,
            frame_count, fps, resolution, pipeline
    """
    if pipeline_name not in config.PREPROCESS_PIPELINES:
        raise ValueError(
            f"Unknown pipeline: {pipeline_name}. "
            f"Available: {list(config.PREPROCESS_PIPELINES.keys())}"
        )

    pipeline_steps = config.PREPROCESS_PIPELINES[pipeline_name]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*config.CODEC)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    input_brightness_sum = 0.0
    output_brightness_sum = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Measure input brightness (L channel mean)
        lab_in = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        input_brightness_sum += lab_in[:, :, 0].mean()

        # Apply pipeline
        processed = apply_pipeline(frame, pipeline_steps)

        # Measure output brightness
        lab_out = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        output_brightness_sum += lab_out[:, :, 0].mean()

        writer.write(processed)
        frame_count += 1

    cap.release()
    writer.release()

    stats = {
        "input_mean_brightness": round(input_brightness_sum / max(frame_count, 1), 2),
        "output_mean_brightness": round(output_brightness_sum / max(frame_count, 1), 2),
        "frame_count": frame_count,
        "fps": fps,
        "resolution": [w, h],
        "pipeline": pipeline_name,
    }

    print(f"Preprocessed {frame_count} frames with '{pipeline_name}':")
    print(f"  Brightness: {stats['input_mean_brightness']:.1f} -> {stats['output_mean_brightness']:.1f}")
    print(f"  Saved: {output_path}")

    return stats


def inspect_pipeline(video_path, pipeline_name, n_frames=5, output_dir=None):
    """Save before/after comparison frames for quick visual inspection.

    Processes only the first n_frames and saves side-by-side comparison PNGs.

    Args:
        video_path: input video path
        pipeline_name: key into config.PREPROCESS_PIPELINES
        n_frames: number of frames to sample
        output_dir: where to save PNGs (default: next to video)
    """
    if pipeline_name not in config.PREPROCESS_PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    pipeline_steps = config.PREPROCESS_PIPELINES[pipeline_name]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), f"inspect_{pipeline_name}")
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_interval = max(1, total_frames // n_frames)

    while saved < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            processed = apply_pipeline(frame, pipeline_steps)

            # Side-by-side
            label_h = 30
            before_label = frame.copy()
            after_label = processed.copy()
            cv2.putText(before_label, "BEFORE", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(after_label, f"AFTER ({pipeline_name})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            comparison = np.hstack([before_label, after_label])

            out_path = os.path.join(output_dir, f"compare_{frame_idx:05d}.png")
            cv2.imwrite(out_path, comparison)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} comparison frames to {output_dir}")


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

    # Helper: create a dark frame
    def dark_frame(mean_val=30):
        frame = np.full((100, 100, 3), mean_val, dtype=np.uint8)
        # Add some spatial variation for CLAHE to work with
        frame[20:80, 20:80] = min(mean_val + 20, 255)
        return frame

    def frame_luminance(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0].mean()

    def frame_l_std(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0].std()

    # ── Test 1: CLAHE changes local contrast on dark frame ───────
    print("\nTest 1: CLAHE local contrast enhancement")
    dark = dark_frame(30)
    clahe_out = apply_clahe(dark)

    std_before = frame_l_std(dark)
    std_after = frame_l_std(clahe_out)
    check("CLAHE changes luminance distribution", std_after != std_before)
    check("CLAHE output has valid range",
          clahe_out.min() >= 0 and clahe_out.max() <= 255)
    check("CLAHE preserves frame shape", clahe_out.shape == dark.shape)

    # ── Test 2: Gamma correction ─────────────────────────────────
    print("\nTest 2: Gamma correction")
    dark = dark_frame(50)
    lum_before = frame_luminance(dark)

    bright = apply_gamma(dark, gamma=0.5)  # should brighten
    lum_bright = frame_luminance(bright)
    check("Gamma 0.5 brightens (increases luminance)", lum_bright > lum_before)

    darker = apply_gamma(dark, gamma=2.0)  # should darken
    lum_darker = frame_luminance(darker)
    check("Gamma 2.0 darkens (decreases luminance)", lum_darker < lum_before)

    identity = apply_gamma(dark, gamma=1.0)
    check("Gamma 1.0 is near-identity",
          np.allclose(dark, identity, atol=1))

    # ── Test 3: Brightness normalization ─────────────────────────
    print("\nTest 3: Brightness normalization")
    dark = dark_frame(40)
    lum_before = frame_luminance(dark)
    target = 128.0

    normed = apply_brightness_norm(dark, target_mean=target)
    lum_after = frame_luminance(normed)
    check(f"Moves brightness toward target ({lum_before:.0f} -> {lum_after:.0f})",
          abs(lum_after - target) < abs(lum_before - target))
    check("Does not exceed 255", normed.max() <= 255)

    # Safety: very dark frame should not blow out
    very_dark = dark_frame(5)
    normed_vd = apply_brightness_norm(very_dark, target_mean=200, max_scale=3.0)
    check("Max scale cap prevents extreme scaling",
          frame_luminance(normed_vd) < 200)

    # ── Test 4: Denoising ────────────────────────────────────────
    print("\nTest 4: Denoising")
    # Create noisy frame
    base = np.full((100, 100, 3), 80, dtype=np.uint8)
    noise = np.random.randint(0, 40, base.shape, dtype=np.uint8)
    noisy = cv2.add(base, noise)
    noise_std_before = noisy.astype(float).std()

    denoised = apply_denoise(noisy)
    noise_std_after = denoised.astype(float).std()
    check("Denoise reduces pixel variance",
          noise_std_after < noise_std_before)
    check("Denoise preserves frame shape", denoised.shape == noisy.shape)

    # ── Test 5: Pipeline order is respected ──────────────────────
    print("\nTest 5: Pipeline order")
    dark = dark_frame(30)

    # CLAHE then gamma vs gamma then CLAHE should give different results
    out_cg = apply_pipeline(dark, [("clahe", {}), ("gamma", {"gamma": 0.7})])
    out_gc = apply_pipeline(dark, [("gamma", {"gamma": 0.7}), ("clahe", {})])
    check("Different pipeline orders produce different results",
          not np.array_equal(out_cg, out_gc))

    # ── Test 6: Full pipeline smoke test ─────────────────────────
    print("\nTest 6: Full pipeline (denoise_clahe_gamma)")
    dark = dark_frame(25)
    steps = config.PREPROCESS_PIPELINES["denoise_clahe_gamma"]
    out = apply_pipeline(dark, steps)
    check("Full pipeline output valid shape", out.shape == dark.shape)
    check("Full pipeline output valid dtype", out.dtype == np.uint8)

    # ── Test 7: Video preprocessing preserves frame count ────────
    print("\nTest 7: Video frame count preservation")
    import tempfile

    # Create a tiny test video
    tmp_dir = tempfile.mkdtemp()
    in_path = os.path.join(tmp_dir, "test_in.avi")
    out_path = os.path.join(tmp_dir, "test_out.avi")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(in_path, fourcc, 30, (100, 100))
    n_test_frames = 15
    for i in range(n_test_frames):
        f = np.full((100, 100, 3), 40 + i, dtype=np.uint8)
        writer.write(f)
    writer.release()

    stats = preprocess_video(in_path, out_path, "clahe")
    check(f"Frame count preserved ({stats['frame_count']})", stats["frame_count"] == n_test_frames)
    check("FPS recorded", stats["fps"] > 0)
    check("Resolution recorded", stats["resolution"] == [100, 100])
    check("Pipeline name recorded", stats["pipeline"] == "clahe")

    # Verify output video has same frame count
    cap_out = cv2.VideoCapture(out_path)
    count = 0
    while True:
        ret, _ = cap_out.read()
        if not ret:
            break
        count += 1
    cap_out.release()
    check("Output video frame count matches", count == n_test_frames)

    # Cleanup
    os.unlink(in_path)
    os.unlink(out_path)
    os.rmdir(tmp_dir)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Low-light video preprocessing")
    parser.add_argument("--video", help="Input video path")
    parser.add_argument("--pipeline", default="clahe",
                        choices=list(config.PREPROCESS_PIPELINES.keys()),
                        help="Preprocessing pipeline name")
    parser.add_argument("--output", help="Output video path (default: auto-named)")
    parser.add_argument("--inspect", type=int, metavar="N",
                        help="Save N before/after comparison frames instead of full video")
    parser.add_argument("--inspect-dir", help="Output directory for inspect frames")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    if not args.video:
        parser.error("--video is required (or use --test)")

    if args.inspect:
        inspect_pipeline(args.video, args.pipeline,
                         n_frames=args.inspect, output_dir=args.inspect_dir)
        return

    # Default output path: same dir as input, with pipeline name
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.video)
        out_path = f"{base}_{args.pipeline}{ext}"

    preprocess_video(args.video, out_path, args.pipeline)


if __name__ == "__main__":
    main()
