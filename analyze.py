"""
One-command rPPG analysis: video in -> pulse signal, heart rate, signal-quality score out.

No training, no weights (besides MediaPipe's face landmarker). Classical POS pipeline:
    MediaPipe face mesh ROI -> mean RGB per frame -> POS (Wang et al. 2017)
    -> Butterworth bandpass (42-210 BPM) -> windowed FFT HR + SNR

Usage:
    python analyze.py path/to/video.mp4
    python analyze.py path/to/video.mp4 --region combined --json out.json --plot out.png

Signal-quality score (0-1) is a heuristic summarising whether a plausible cardiac
pulse is present. It is NOT calibrated as a liveness/deepfake classifier — treat it
as one feature to evaluate against labelled data.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from rppg.face_detect import extract_rgb_signals  # noqa: E402
from rppg.pos import pos_from_dataframe  # noqa: E402
from rppg.signal_processing import (  # noqa: E402
    bandpass_filter,
    estimate_hr_windowed,
    smooth_hr_temporal,
)

# Windows shorter than the default 10 s are allowed for short clips (e.g. KYC selfies).
MIN_WINDOW_SEC = 4.0


def _video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.FPS
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if not fps or fps != fps or fps > 240:
        fps = config.FPS
    return float(fps), n


def _quality_score(windows, snr_threshold):
    """Heuristic 0-1 score: is there a consistent, strong cardiac peak?

    Combines (a) fraction of windows whose spectral SNR clears the threshold,
    (b) median SNR mapped to 0-1, (c) HR stability across valid windows.
    """
    snr = windows["snr"].to_numpy(dtype=float)
    hr = windows["hr_bpm_estimated"].to_numpy(dtype=float)
    if len(snr) == 0 or np.all(np.isnan(snr)):
        return 0.0, {}

    frac_valid = float(np.nanmean(snr >= snr_threshold))
    med_snr = float(np.nanmedian(snr))
    snr_term = float(np.clip((med_snr - 1.0) / (2 * snr_threshold), 0, 1))
    hr_valid = hr[~np.isnan(hr)]
    if len(hr_valid) >= 2:
        stability = float(np.clip(1 - np.std(hr_valid) / 20.0, 0, 1))
    elif len(hr_valid) == 1:
        stability = 0.5
    else:
        stability = 0.0

    score = 0.4 * frac_valid + 0.4 * snr_term + 0.2 * stability
    return round(score, 3), {
        "fraction_windows_valid": round(frac_valid, 3),
        "median_snr": round(med_snr, 2),
        "hr_stability": round(stability, 3),
    }


def analyze(video_path, region="forehead", window_sec=None, stride_sec=None, smooth=True, fps=None):
    """fps: override the container frame rate (some webcam recordings report 30 but run slower)."""
    if fps is None:
        fps, _ = _video_fps(video_path)
    rgb_df = extract_rgb_signals(video_path, region=region)
    if rgb_df.empty:
        raise ValueError("No frames could be read from the video.")

    n_frames = len(rgb_df)
    duration = n_frames / fps
    detection_rate = float(rgb_df["valid_detection"].mean())

    if window_sec is None:
        window_sec = min(config.EVAL_WINDOW_SEC, max(MIN_WINDOW_SEC, duration))
    if stride_sec is None:
        stride_sec = max(1.0, window_sec / 2)
    if duration < MIN_WINDOW_SEC:
        raise ValueError(f"Video is {duration:.1f}s; need at least {MIN_WINDOW_SEC:.0f}s.")

    pulse = pos_from_dataframe(rgb_df, fps)
    filtered = bandpass_filter(pulse, fps)
    windows = estimate_hr_windowed(filtered, fps, window_sec=window_sec, stride_sec=stride_sec)
    if smooth and config.TEMPORAL_SMOOTH and len(windows) > 2:
        windows = smooth_hr_temporal(windows)

    hr_valid = windows["hr_bpm_estimated"].dropna()
    score, parts = _quality_score(windows, config.HR_MIN_SNR)

    summary = {
        "video": os.path.basename(video_path),
        "fps": round(fps, 2),
        "frames": n_frames,
        "duration_sec": round(duration, 2),
        "region": region,
        "face_detection_rate": round(detection_rate, 3),
        "hr_bpm_median": round(float(hr_valid.median()), 1) if len(hr_valid) else None,
        "n_windows": int(len(windows)),
        "n_valid_windows": int(len(hr_valid)),
        "window_sec": window_sec,
        "signal_quality_score": score,
        "signal_quality_components": parts,
    }
    return {
        "summary": summary,
        "windows": windows,
        "pulse": filtered,
        "fps": fps,
    }


def plot_result(result, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pulse, fps = result["pulse"], result["fps"]
    windows = result["windows"]
    s = result["summary"]
    t = np.arange(len(pulse)) / fps

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(t, pulse, lw=0.8, color="#c0392b")
    axes[0].set_title("Filtered rPPG pulse signal (POS)")
    axes[0].set_xlabel("time (s)")

    clean = np.nan_to_num(pulse)
    if len(clean) > 8:
        spec = np.abs(np.fft.rfft(clean * np.hanning(len(clean)))) ** 2
        freqs = np.fft.rfftfreq(len(clean), 1 / fps) * 60
        band = (freqs >= 40) & (freqs <= 210)
        axes[1].plot(freqs[band], spec[band], color="#2c3e50")
    axes[1].set_title("Power spectrum (whole clip)")
    axes[1].set_xlabel("BPM")

    mid = (windows["window_start_sec"] + windows["window_end_sec"]) / 2
    axes[2].plot(mid, windows["hr_bpm_estimated"], "o-", color="#c0392b", label="HR (bpm)")
    ax2 = axes[2].twinx()
    ax2.bar(mid, windows["snr"], width=1.0, alpha=0.25, color="#2c3e50", label="SNR")
    ax2.axhline(config.HR_MIN_SNR, ls="--", color="#7f8c8d", lw=0.8)
    axes[2].set_title(
        f"Per-window HR & SNR — median {s['hr_bpm_median']} bpm, "
        f"quality {s['signal_quality_score']}"
    )
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("bpm")
    ax2.set_ylabel("SNR")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--region", default="forehead", choices=["forehead", "cheeks", "combined"])
    ap.add_argument("--window-sec", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None, help="override container FPS if it is wrong")
    ap.add_argument("--json", help="write summary + per-window results to this JSON file")
    ap.add_argument("--plot", help="save diagnostic plot to this PNG")
    args = ap.parse_args()

    res = analyze(args.video, region=args.region, window_sec=args.window_sec, fps=args.fps)
    print(json.dumps(res["summary"], indent=2))
    if args.json:
        out = dict(res["summary"])
        out["windows"] = json.loads(res["windows"].to_json(orient="records"))
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    if args.plot:
        plot_result(res, args.plot)


if __name__ == "__main__":
    main()
