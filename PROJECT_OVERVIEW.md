# Circadify rPPG Robustness Project

## Context

This project was built for Circadify, a startup working on camera-based vital sign estimation from facial video for telehealth and remote monitoring.

The goal is not to rebuild their product. The goal is to build something technically relevant and genuinely useful that shows strong project ownership and good research instincts — a small but real research prototype.

## Core Idea

We evaluate how a classical rPPG (remote photoplethysmography) pipeline performs under a real-world failure condition — **low light** — using face videos recorded on a laptop webcam plus Apple Watch heart-rate labels as approximate ground truth. Then we improve low-light performance with targeted preprocessing and show before/after results.

This is a **robustness project**, not a model-training project. We are trying to show where the pipeline breaks in realistic home conditions, and whether targeted preprocessing can recover performance.

## Tech Stack

- Python 3.13 (Windows)
- OpenCV, NumPy, SciPy, pandas, matplotlib
- MediaPipe Face Mesh (face detection + landmark ROI extraction)
- No deep learning, no GPU required, no heavy frameworks

## Algorithm: POS (Plane-Orthogonal-to-Skin)

We use the POS algorithm from Wang et al. (2017), implemented from scratch in ~30 lines of NumPy. POS is a standard classical rPPG method that:

1. Takes windowed mean RGB signals from a face ROI
2. Temporally normalizes each window
3. Projects onto the plane orthogonal to skin-tone variation (G-B and G+B-2R)
4. Combines projections using a standard deviation ratio
5. Overlap-adds windows to produce a continuous pulse signal

We chose POS over Green-channel or CHROM because it outperforms both in most benchmarks while remaining simple to implement. We avoided heavy frameworks like pyVHR or rPPG-Toolbox to keep setup time minimal.

## Project Structure

```
circadify-rppg/
    config.py                      # All tunable constants in one place
    requirements.txt               # opencv-python, numpy, scipy, pandas, mediapipe, matplotlib

    collect/
        record_video.py            # Webcam recording with preview, countdown, frame timestamps
        parse_apple_health.py      # Apple Health XML/CSV heart rate parsing
        align.py                   # Window-level HR alignment (primary) + frame-level (secondary)
        batch_process.py           # Single-command parse + align for all sessions

    rppg/
        face_detect.py             # MediaPipe face mesh ROI extraction (forehead/cheeks/combined)
        pos.py                     # POS algorithm implementation
        signal_processing.py       # Bandpass filter, Hann-windowed FFT HR estimation

    evaluate/
        metrics.py                 # MAE, RMSE, Median AE, Pearson r
        run_baseline.py            # End-to-end baseline evaluation orchestrator

    enhance/
        preprocess.py              # CLAHE, gamma correction, brightness normalization, denoising
        run_enhanced.py            # Enhanced evaluation orchestrator with pipeline sweep
        plots.py                   # Before/after comparison figures

    data/
        raw/                       # Session folders with videos, timestamps, HR labels
        processed/                 # Intermediate outputs (RGB signals, pulse, preprocessed videos)
        results/                   # JSON reports, CSV summaries, figures
```

## Pipeline Flow

```
Webcam video
    → face_detect (MediaPipe face mesh → forehead/cheeks ROI → mean RGB per frame)
    → pos (POS algorithm → raw pulse signal)
    → signal_processing (Butterworth bandpass 0.7-3.5 Hz → Hann-windowed FFT → HR per 10-sec window)
    → metrics (compare estimated HR windows vs Apple Watch ground truth)
```

For enhanced evaluation, the only change is inserting a preprocessing step before face detection:

```
Raw video → preprocess (CLAHE / gamma / denoise) → preprocessed video → same pipeline as above
```

## Data Collection Design

- **Target**: 8-12 sessions, roughly half normal light and half low light
- **Duration**: 30-60 seconds per clip
- **Resolution**: 640x480 at 30 FPS (MJPG codec)
- **Ground truth**: Apple Watch heart rate via Apple Health XML export (sampled every 3-6 seconds during workout mode)
- **Sync**: On-screen countdown during recording serves as the temporal anchor; video start timestamp (UTC) aligns to Apple Watch timestamps

Each session is stored as:
```
data/raw/session_YYYYMMDD_HHMMSS/
    video.avi
    frame_timestamps.csv       # frame_idx, elapsed_ms, system_time_ms
    metadata.json              # start/end UTC, condition, subject_id, fps, etc.
    hr_labels.csv              # parsed Apple Watch HR
    windows.csv                # aligned evaluation windows
```

## Evaluation Approach

- **Primary evaluation**: Window-level comparison (10-second windows, 5-second stride)
  - For each window: rPPG estimates HR via FFT peak, Apple Watch provides median HR from samples in that window
  - Windows with fewer than 2 Apple Watch samples are marked invalid and excluded
- **Metrics**: MAE, RMSE, Median Absolute Error, Pearson correlation — reported separately for normal vs low light
- **Success criteria**: ~20-30% reduction in low-light MAE after preprocessing, or a clear improvement on most low-light sessions

## Face ROI Design

- **Forehead** (default): MediaPipe Face Mesh landmarks for the upper face — good pulse signal, minimal motion artifact from mouth/eyes
- **Cheeks**: Left and right cheek regions as backup/supplement
- **Combined**: All three regions merged — switchable via `--region combined`
- Modular design allows ROI experimentation without changing the pipeline

## Preprocessing Pipelines

Four predefined pipelines, ordered from simplest to most aggressive:

| Pipeline | Steps | Rationale |
|----------|-------|-----------|
| `clahe` | CLAHE on L channel | Improve local contrast in dark frames |
| `gamma_07` | Gamma correction (0.7) | Brighten shadows directly |
| `clahe_gamma` | CLAHE → Gamma 0.7 | Contrast + brightness |
| `denoise_clahe_gamma` | Denoise → CLAHE → Gamma 0.7 | Reduce noise before amplifying contrast |

Order matters: denoising before contrast enhancement prevents amplifying photon noise.

## Key Design Decisions

1. **Window-level evaluation, not frame-level**: Apple Watch HR is sparse and irregular (every 3-6 seconds). Per-frame interpolation exists for visualization but is not used for metrics.

2. **NaN handling throughout**: Every module handles missing data explicitly. Face detection failures, POS window skips, insufficient HR samples — all produce NaN rather than zeros, and downstream modules propagate NaN correctly.

3. **Explicit validity tracking**: Every window carries a `valid` flag. Metrics are computed only on mutually valid windows (both ground truth and estimate valid).

4. **SNR-based quality gating**: HR estimates from windows with weak spectral peaks (low SNR) are marked invalid rather than reported as noise.

5. **Preprocessing is applied to the video, not the pipeline**: The rPPG pipeline (face detection, POS, signal processing) stays unchanged. Only the input video is modified. This keeps the experiment clean.

6. **Cached preprocessing**: Preprocessed videos are saved and reused across runs. `--force-preprocess` regenerates them.

7. **Baseline comparison built in**: The enhanced evaluation automatically loads baseline results and computes improvement columns (MAE delta, percent improvement).

## Module-by-Module Summary

### collect/record_video.py
- Opens webcam with live preview (press SPACE to start)
- 3-2-1 countdown with Apple Watch sync cue
- Records to MJPG .avi with per-frame timestamps (elapsed_ms + system_time_ms)
- Rich metadata.json (start/end UTC, actual vs requested FPS, condition, subject)

### collect/parse_apple_health.py
- Streaming XML parser (memory-efficient for large Apple Health exports)
- CSV fallback with flexible column matching
- Time window filtering during parse
- Output schema: timestamp_utc, hr_bpm, source_type, source_file

### collect/align.py
- Window-level alignment (primary): 10-sec windows, 5-sec stride
- Uses metadata.json start_time_utc + elapsed time as source of truth
- NaN for invalid windows (< 2 HR samples), not 0
- Frame-level interpolation (secondary, visualization only)

### rppg/face_detect.py
- MediaPipe Face Mesh with modular ROI (forehead/cheeks/combined)
- DataFrame output: frame_idx, rgb_r, rgb_g, rgb_b, valid_detection, interpolated
- Short gaps (≤5 frames) interpolated, long gaps stay NaN
- Debug visualization: `--save-debug-frames` overlays ROI on sampled frames

### rppg/pos.py
- Standard POS (Wang et al., 2017) implementation
- Configurable max NaN fraction per window (skips windows with too many missing frames)
- Numerical stability guards for near-zero variance
- Optional diagnostics dict (processed/skipped window counts)

### rppg/signal_processing.py
- Butterworth bandpass filter (0.7-3.5 Hz, order 4)
- NaN-safe: filters only contiguous valid segments independently
- Hann window before FFT to reduce spectral leakage
- SNR: peak power / mean non-peak power in cardiac band
- Validity gating: NaN fraction, SNR threshold, clear spectral peak required

### collect/batch_process.py
- Single-pass XML parsing — reads export.xml once, routes HR records to sessions by time window
- Produces hr_labels.csv, windows.csv, and frame_hr.csv per session
- `--dry-run` mode to preview what would be processed
- `--force` to reprocess already-completed sessions
- Reports: HR records per session, valid windows, clear failure reasons
- Replaces manual per-session parsing and alignment with one command

### evaluate/metrics.py
- MAE, RMSE, Median Absolute Error, Pearson r (with p-value)
- Window grid validation (raises error if estimate and ground truth grids don't match)
- Pearson edge cases: constant series → NaN with warning, < 2 points → NaN
- Optional merged per-window DataFrame output for debugging

### evaluate/run_baseline.py
- Thin orchestrator — calls lower-level modules, no reimplemented logic
- Auto-discovers sessions, validates required files
- `--list-sessions` dry-run mode
- JSON report + flat CSV summary (per-session metrics, detection rate, POS stats)
- Aggregation by condition: mean and median of session MAEs/RMSEs

### enhance/preprocess.py
- Four operations: CLAHE, gamma, brightness normalization, denoising
- All fully parameterized with config defaults
- Pipeline definitions centralized in config.py
- `--inspect N` mode: saves N before/after comparison frames without processing full video
- Reports input/output brightness stats per video

### enhance/run_enhanced.py
- Thin orchestrator for preprocessing + evaluation sweep
- Cached preprocessing (reuses existing preprocessed videos)
- Comparison CSV with baseline_mae, enhanced_mae, mae_delta, mae_pct_improvement
- Per-pipeline aggregation (mean/median MAE, success/fail counts)
- Session filtering by condition, failure tracking with reasons

### enhance/plots.py
- 5 plot types: MAE by pipeline (bar), MAE distribution by pipeline (box+scatter), best vs baseline per session, condition comparison, Bland-Altman (optional)
- Reads CSV summaries as primary source (not JSON)
- Best pipeline = lowest MAE per session, ties broken alphabetically
- Graceful skip for missing data — every plot checks inputs
- Saves plots_summary.txt documenting what was generated

## Test Coverage

Every module has built-in tests runnable via `--test`:

```
py collect/parse_apple_health.py --test     # 12 tests
py collect/align.py --test                   # 25 tests
py collect/batch_process.py --test           # 13 tests
py rppg/face_detect.py --test               # 22 tests
py rppg/pos.py --test                        # 18 tests
py rppg/signal_processing.py --test          # 17 tests
py evaluate/metrics.py --test                # 26 tests
py evaluate/run_baseline.py --test           # 17 tests
py enhance/preprocess.py --test              # 19 tests
py enhance/run_enhanced.py --test            # 26 tests
py enhance/plots.py --test                   # 17 tests
```

Total: 212 tests, all passing.

Tests cover: synthetic signal validation (1.2 Hz sinusoid detected by POS, 75 BPM estimated within 3 BPM), NaN handling, edge cases (constant series, near-zero variance, sparse sampling), schema validation, output shape verification, and pipeline integration.

## How to Run (End-to-End)

```bash
# 1. Install dependencies
cd circadify-rppg
py -m pip install opencv-python numpy scipy pandas mediapipe matplotlib

# 2. Record sessions (on a laptop with webcam)
py collect/record_video.py --duration 60 --condition normal --subject pedro
py collect/record_video.py --duration 60 --condition low_light --subject pedro
# Repeat for 8-12 total sessions across normal/low_light conditions

# 3. Export Apple Watch HR from iPhone, place export.xml in data/
#    (iPhone → Health app → Profile → Export All Health Data)

# 4. Batch parse + align all sessions (one command)
py collect/batch_process.py --export data/export.xml --data-dir data/raw

# 5. Run baseline evaluation
py evaluate/run_baseline.py --data-dir data/raw

# 6. Run enhanced evaluation
py enhance/run_enhanced.py --data-dir data/raw

# 7. Generate comparison plots
py enhance/plots.py --results-dir data/results
```

## What This Project Demonstrates

1. **Understanding of the problem domain**: rPPG signal extraction from facial video, cardiac frequency estimation, real-world failure modes
2. **Research methodology**: controlled experiment with normal/low-light conditions, proper ground truth alignment, window-level evaluation
3. **Engineering quality**: modular design, explicit NaN handling, validity tracking, configurable parameters, built-in tests
4. **Practical judgment**: chose a simple algorithm that works (POS), avoided overengineering, prioritized working code over fancy architecture
5. **Robustness analysis**: systematic evaluation of where the pipeline breaks and targeted preprocessing to improve it

## Known Limitations

- Apple Watch HR is approximate ground truth (not medical-grade)
- POS is a classical method — deep learning approaches (e.g., PhysNet, EfficientPhys) would likely perform better but require training data and GPU
- Small subject pool (2-3 people) limits generalizability
- Webcam quality varies significantly between laptops
- Preprocessing is frame-level only — temporal smoothing or adaptive enhancement could help further
