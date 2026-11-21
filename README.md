# Circadify rPPG Robustness

Evaluate how a camera-based heart rate estimation (rPPG) pipeline performs under low-light conditions, and improve it with targeted preprocessing.

Uses the POS algorithm (Wang et al., 2017) to estimate heart rate from facial video, with Apple Watch as ground truth.

> **No trained model.** This is a classical signal-processing pipeline (POS, implemented from scratch in NumPy + MediaPipe face mesh). The only weights are MediaPipe's off-the-shelf face landmarker. That means it runs on any face video with no training data.

**Report (PDF):** https://pedroldrh.xyz/assets/circadify-rppg-results.pdf

## Analyze any video

```bash
pip install -r requirements.txt
python analyze.py face.mp4 --json out.json --plot out.png
# --region forehead|cheeks|combined   --fps 23.1 (override if the container FPS is wrong)
```

Outputs the filtered pulse signal, per-window heart rate + spectral SNR, and a **signal-quality score (0–1)**: a heuristic of how strong and consistent the cardiac peak is (fraction of windows above SNR threshold, median SNR, HR stability). It is *not* a calibrated liveness / deepfake classifier — it's a feature to evaluate against labelled data. `python app.py` launches the Gradio demo locally.

## Results & limitations

- 12 sessions, **one subject**, laptop webcam (~23 fps), Apple Watch HR as approximate ground truth.
- Baseline MAE per session ranges ~2–9.5 bpm (normal light) and ~3.3–7.4 bpm (low light); see `data/results/`.
- Low-light preprocessing (CLAHE / gamma / denoise) helped some sessions and hurt others — no single pipeline wins consistently at this sample size.
- Raw videos and health data are not included in the repo (privacy).


## Quick Start

```bash
# Install
cd circadify-rppg
pip install -r requirements.txt

# Record sessions (needs webcam)
python collect/record_video.py --duration 60 --condition normal --subject your_name
python collect/record_video.py --duration 60 --condition low_light --subject your_name

# Export Apple Watch HR from iPhone, place export.xml in data/

# Parse + align all sessions
python collect/batch_process.py --export data/export.xml --data-dir data/raw

# Evaluate
python evaluate/run_baseline.py --data-dir data/raw
python enhance/run_enhanced.py --data-dir data/raw
python enhance/plots.py --results-dir data/results
```

## How It Works

1. **Record** 60-second face videos on a laptop webcam (normal light + low light)
2. **Wear an Apple Watch** during recording to capture real heart rate
3. **Run the pipeline** — it extracts heart rate from the video by detecting tiny skin color changes
4. **Compare** the camera's estimate vs the watch's real reading
5. **Apply preprocessing** (CLAHE, gamma correction, denoising) to low-light videos and re-evaluate
6. **See the results** — charts showing baseline vs enhanced performance

## Project Structure

```
config.py                  # All tunable constants
collect/
    record_video.py        # Webcam recording with sync countdown
    parse_apple_health.py  # Apple Health XML/CSV parsing
    align.py               # Align HR labels to video windows
    batch_process.py       # One-command parse + align for all sessions
rppg/
    face_detect.py         # MediaPipe face mesh ROI extraction
    pos.py                 # POS algorithm (heart rate from RGB signals)
    signal_processing.py   # Bandpass filter + FFT heart rate estimation
evaluate/
    metrics.py             # MAE, RMSE, Median AE, Pearson r
    run_baseline.py        # Evaluate all sessions
enhance/
    preprocess.py          # CLAHE, gamma, denoising operations
    run_enhanced.py        # Evaluate with preprocessing pipelines
    plots.py               # Before/after comparison figures
```

## Requirements

- Python 3.10+
- Laptop with webcam (for recording only)
- Apple Watch + iPhone (for ground truth HR)
- No GPU needed

## Dependencies

```
opencv-python>=4.8
numpy>=1.24
scipy>=1.10
pandas>=2.0
mediapipe>=0.10
matplotlib>=3.7
```

## Tests

Every module has built-in tests:

```bash
python collect/parse_apple_health.py --test
python collect/align.py --test
python collect/batch_process.py --test
python rppg/face_detect.py --test
python rppg/pos.py --test
python rppg/signal_processing.py --test
python evaluate/metrics.py --test
python enhance/preprocess.py --test
python enhance/run_enhanced.py --test
python enhance/plots.py --test
```

212 tests total, all passing.

## Preprocessing Pipelines

| Pipeline | Steps | Purpose |
|----------|-------|---------|
| `clahe` | CLAHE on L channel | Improve local contrast |
| `gamma_07` | Gamma correction (0.7) | Brighten shadows |
| `clahe_gamma` | CLAHE + Gamma 0.7 | Contrast + brightness |
| `denoise_clahe_gamma` | Denoise + CLAHE + Gamma 0.7 | Full low-light enhancement |

## Useful Commands

```bash
# Preview what sessions exist and their status
python evaluate/run_baseline.py --data-dir data/raw --list-sessions

# Dry run batch processing (see what would happen)
python collect/batch_process.py --export data/export.xml --data-dir data/raw --dry-run

# Debug face detection on a video
python rppg/face_detect.py --video data/raw/session_.../video.avi --save-debug-frames debug/

# Preview preprocessing effect without full video
python enhance/preprocess.py --video data/raw/session_.../video.avi --pipeline clahe --inspect 5

# Run enhanced eval on specific pipelines only
python enhance/run_enhanced.py --data-dir data/raw --pipelines clahe denoise_clahe_gamma
```

## Documentation

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the full technical design, module details, and experiment methodology.
