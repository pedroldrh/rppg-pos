"""
Circadify rPPG — Central configuration.
All tunable constants live here so nothing is scattered across modules.
"""

# ── Video recording ──────────────────────────────────────────────────
FPS = 30
RESOLUTION = (640, 480)          # width, height
CODEC = "MJPG"                   # reliable on Windows OpenCV
DURATION_SEC = 60                # default clip length

# ── rPPG signal processing ───────────────────────────────────────────
POS_WINDOW_SEC = 1.6             # POS sliding window (48 frames @ 30 fps)
POS_MAX_NAN_FRACTION = 0.1      # skip window if > 10% of frames are NaN
POS_MIN_STD = 1e-6              # numerical guard for near-zero variance
HR_BAND_HZ = (0.7, 3.5)         # 42–210 BPM
BANDPASS_ORDER = 4               # Butterworth filter order
HR_MAX_NAN_FRACTION = 0.5       # skip eval window if > 50% pulse samples NaN
HR_MIN_SNR = 2.0                # skip window if SNR below this threshold
HARMONIC_CHECK = True           # enable subharmonic preference in FFT peak selection
HARMONIC_POWER_RATIO = 0.3     # prefer subharmonic if its power >= this fraction of dominant peak
TEMPORAL_SMOOTH = True          # enable temporal outlier smoothing across windows
TEMPORAL_MAX_JUMP_BPM = 25.0   # flag window as outlier if it deviates > this from neighbors

# ── Evaluation windows ───────────────────────────────────────────────
EVAL_WINDOW_SEC = 10             # primary comparison window
EVAL_STRIDE_SEC = 5              # overlap stride
MIN_HR_SAMPLES_PER_WINDOW = 2   # skip window if fewer Apple Watch samples

# ── Low-light enhancement ────────────────────────────────────────────
CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)
DENOISE_H = 10                   # fastNlMeansDenoisingColored luminance strength
DENOISE_H_COLOR = 10             # fastNlMeansDenoisingColored color strength
DENOISE_TEMPLATE_WINDOW = 7      # template patch size
DENOISE_SEARCH_WINDOW = 21       # search area size
BRIGHTNESS_NORM_TARGET = 128.0   # target mean luminance for brightness normalization
BRIGHTNESS_NORM_MAX_SCALE = 3.0  # safety cap to prevent blowout
ADAPTIVE_BRIGHTNESS_THRESHOLD = 140  # L-channel mean below this triggers enhancement
ADAPTIVE_GAMMA = 0.7             # gamma value for adaptive enhancement

# Preprocessing pipelines: ordered list of (operation, params) tuples.
# Operations: "clahe", "gamma", "brightness_norm", "denoise"
PREPROCESS_PIPELINES = {
    "clahe": [
        ("clahe", {}),
    ],
    "gamma_07": [
        ("gamma", {"gamma": 0.7}),
    ],
    "clahe_gamma": [
        ("clahe", {}),
        ("gamma", {"gamma": 0.7}),
    ],
    "denoise_clahe_gamma": [
        ("denoise", {}),
        ("clahe", {}),
        ("gamma", {"gamma": 0.7}),
    ],
    "adaptive": [
        ("adaptive_gamma", {}),
    ],
}

# ── Paths (relative to project root) ────────────────────────────────
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
DATA_RESULTS = "data/results"
