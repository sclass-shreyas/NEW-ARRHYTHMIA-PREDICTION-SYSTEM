"""
PERSON 3 — Physiological Model
Day 2: HRV Feature Engine

File: physiological_model/hrv_features.py

What this module does:
    Given a LOCAL window of RR intervals around a beat,
    computes all time-domain, frequency-domain, and nonlinear
    HRV features needed by the physiological rule engine.

All metrics follow the Task Force Standards:
    "Heart rate variability: standards of measurement, physiological
    interpretation and clinical use." European Heart Journal, 1996.

Usage:
    from physiological_model.hrv_features import compute_hrv_features
    features = compute_hrv_features(rr_window)
"""

import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────
# CONSTANTS — Clinical reference ranges (Task Force 1996 + AHA)
# ─────────────────────────────────────────────────────────────

NORMAL_HR_MIN     = 60.0     # bpm
NORMAL_HR_MAX     = 100.0    # bpm
NORMAL_SDNN_MIN   = 20.0     # ms  (below this = severely depressed HRV)
NORMAL_SDNN_MAX   = 150.0    # ms
NORMAL_RMSSD_MAX  = 80.0     # ms  (above this = excessive vagal or ectopic)
NORMAL_PNN50_MAX  = 50.0     # %   (above this = highly irregular rhythm)
NORMAL_SD1SD2_MIN = 0.15     # ratio (below = rigid rhythm)
NORMAL_SD1SD2_MAX = 1.0      # ratio (above = chaotic rhythm)
NORMAL_LFHF_MAX   = 2.5      # ratio (above = sympathetic dominance)

MIN_BEATS_TIME    = 4        # minimum beats for time-domain features
MIN_BEATS_FREQ    = 20       # minimum beats for frequency-domain features
MIN_BEATS_NONLIN  = 10       # minimum beats for nonlinear features

FS_RESAMPLE       = 4.0      # Hz — standard RR resampling rate for freq domain


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def compute_hrv_features(rr_window: np.ndarray) -> dict | None:
    """
    Computes all HRV features for a local RR interval window.

    Parameters
    ----------
    rr_window : np.ndarray
        Array of RR intervals in SECONDS. Shape (k,) where k is
        typically 10–20 beats (the local context around one beat).
        Must contain values in physiologically valid range (0.25s–2.0s).

    Returns
    -------
    dict with all computed features, or None if window too short.

    Feature groups returned:
        Time-domain   : mean_rr_ms, mean_hr_bpm, sdnn_ms, rmssd_ms, pnn50_pct
        Poincaré      : sd1, sd2, sd1_sd2_ratio
        Frequency     : lf_power, hf_power, lf_hf_ratio  (None if < 20 beats)
        Nonlinear     : sample_entropy  (None if < 10 beats)
        Meta          : n_beats, has_freq_features, has_nonlinear_features
    """

    # ── Input validation ────────────────────────────────────────────────
    rr_window = np.asarray(rr_window, dtype=np.float64)

    # Remove physiologically impossible values before computing
    rr_clean = rr_window[(rr_window >= 0.25) & (rr_window <= 2.0)]

    if len(rr_clean) < MIN_BEATS_TIME:
        return None   # not enough valid beats for any meaningful analysis

    # ── Time-domain features ─────────────────────────────────────────────
    rr_ms     = rr_clean * 1000.0        # convert to milliseconds
    diff_rr   = np.diff(rr_ms)           # successive differences

    mean_rr   = float(np.mean(rr_ms))
    mean_hr   = 60000.0 / mean_rr        # beats per minute
    sdnn      = float(np.std(rr_ms, ddof=1))
    rmssd     = float(np.sqrt(np.mean(diff_rr ** 2)))
    pnn50     = float(np.sum(np.abs(diff_rr) > 50.0) / len(diff_rr) * 100.0)

    # ── Poincaré features ────────────────────────────────────────────────
    # SD1: short-term variability (beat-to-beat)
    # SD2: long-term variability (overall trend)
    sd1 = float(np.std(diff_rr / np.sqrt(2.0), ddof=1))
    sd2 = float(np.std((rr_ms[:-1] + rr_ms[1:]) / np.sqrt(2.0), ddof=1))
    sd1_sd2_ratio = float(sd1 / sd2) if sd2 > 1e-6 else 0.0

    # ── Frequency-domain features ────────────────────────────────────────
    lf_power    = None
    hf_power    = None
    lf_hf_ratio = None
    has_freq    = False

    if len(rr_clean) >= MIN_BEATS_FREQ:
        try:
            lf_power, hf_power, lf_hf_ratio = _compute_frequency_features(
                rr_clean, fs=FS_RESAMPLE
            )
            has_freq = True
        except Exception:
            pass   # frequency features optional — don't crash if interpolation fails

    # ── Nonlinear features ───────────────────────────────────────────────
    samp_en     = None
    has_nonlin  = False

    if len(rr_clean) >= MIN_BEATS_NONLIN:
        try:
            samp_en   = _sample_entropy(rr_ms, m=2, r_factor=0.2)
            has_nonlin = True
        except Exception:
            pass

    # ── Assemble and return ──────────────────────────────────────────────
    return {
        # Time-domain
        'mean_rr_ms'          : round(mean_rr, 4),
        'mean_hr_bpm'         : round(mean_hr, 4),
        'sdnn_ms'             : round(sdnn, 4),
        'rmssd_ms'            : round(rmssd, 4),
        'pnn50_pct'           : round(pnn50, 4),

        # Poincaré
        'sd1'                 : round(sd1, 4),
        'sd2'                 : round(sd2, 4),
        'sd1_sd2_ratio'       : round(sd1_sd2_ratio, 4),

        # Frequency-domain (None if insufficient beats)
        'lf_power'            : round(lf_power, 6)    if lf_power    is not None else None,
        'hf_power'            : round(hf_power, 6)    if hf_power    is not None else None,
        'lf_hf_ratio'         : round(lf_hf_ratio, 4) if lf_hf_ratio is not None else None,

        # Nonlinear (None if insufficient beats)
        'sample_entropy'      : round(samp_en, 4)     if samp_en     is not None else None,

        # Meta
        'n_beats'             : len(rr_clean),
        'has_freq_features'   : has_freq,
        'has_nonlinear_features': has_nonlin,
    }


# ─────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────

def _compute_frequency_features(rr_seconds: np.ndarray,
                                 fs: float = 4.0) -> tuple[float, float, float]:
    """
    Computes LF power, HF power, and LF/HF ratio via Welch's PSD
    on a uniformly resampled RR signal.

    LF band : 0.04 – 0.15 Hz  (sympathetic + parasympathetic)
    HF band : 0.15 – 0.40 Hz  (parasympathetic / vagal)

    Parameters
    ----------
    rr_seconds : RR intervals in seconds
    fs         : resampling frequency (standard = 4 Hz)

    Returns
    -------
    (lf_power, hf_power, lf_hf_ratio)
    """
    # Build time axis from cumulative RR intervals
    t_rr = np.cumsum(rr_seconds)
    t_rr = t_rr - t_rr[0]                      # start at t=0

    # Uniform time grid at fs Hz
    t_uniform = np.arange(0, t_rr[-1], 1.0 / fs)

    if len(t_uniform) < 8:
        raise ValueError("Too short for frequency analysis after resampling")

    # Cubic interpolation onto uniform grid
    interp_fn     = interp1d(t_rr, rr_seconds, kind='cubic',
                              bounds_error=False, fill_value='extrapolate')
    rr_resampled  = interp_fn(t_uniform)

    # Remove mean (detrend)
    rr_resampled  = rr_resampled - np.mean(rr_resampled)

    # Welch PSD
    nperseg = min(len(rr_resampled), 256)
    freqs, psd = welch(rr_resampled, fs=fs, nperseg=nperseg)

    # Band power via trapezoidal integration
    lf_mask  = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask  = (freqs >= 0.15) & (freqs < 0.40)

    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask]))
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask]))

    lf_hf    = lf_power / hf_power if hf_power > 1e-10 else 0.0

    return lf_power, hf_power, lf_hf


def _sample_entropy(rr_ms: np.ndarray, m: int = 2,
                    r_factor: float = 0.2) -> float:
    """
    Computes Sample Entropy (SampEn) of an RR interval sequence.

    SampEn measures the irregularity/complexity of a time series.
    Higher SampEn = more irregular = more likely arrhythmic.

    Parameters
    ----------
    rr_ms    : RR intervals in milliseconds
    m        : template length (standard = 2)
    r_factor : tolerance as fraction of std (standard = 0.2)

    Returns
    -------
    float : sample entropy value
    """
    N = len(rr_ms)
    r = r_factor * np.std(rr_ms, ddof=1)

    if r < 1e-10:
        return 0.0

    def _count_matches(length):
        count = 0
        for i in range(N - length):
            template = rr_ms[i:i + length]
            for j in range(i + 1, N - length + 1):
                if np.max(np.abs(rr_ms[j:j + length] - template)) < r:
                    count += 1
        return count

    A = _count_matches(m + 1)   # matches of length m+1
    B = _count_matches(m)       # matches of length m

    if B == 0 or A == 0:
        return 0.0

    return float(-np.log(A / B))


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST — run this file directly to verify
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    print("\n" + "="*55)
    print("  HRV FEATURES — Unit Test")
    print("="*55)

    # ── Test 1: Normal sinus rhythm (simulated) ──────────────────────────
    print("\n  Test 1: Simulated Normal Sinus Rhythm (75 bpm, low noise)")
    rr_normal = np.random.normal(loc=0.80, scale=0.03, size=30)
    rr_normal = np.clip(rr_normal, 0.5, 1.2)
    f = compute_hrv_features(rr_normal)
    print(f"    Mean HR     : {f['mean_hr_bpm']:.1f} bpm  (expect ~75)")
    print(f"    SDNN        : {f['sdnn_ms']:.1f} ms   (expect 20–60 for 30-beat window)")
    print(f"    RMSSD       : {f['rmssd_ms']:.1f} ms")
    print(f"    pNN50       : {f['pnn50_pct']:.1f} %")
    print(f"    SD1/SD2     : {f['sd1_sd2_ratio']:.3f}  (expect 0.15–1.0 for normal)")
    print(f"    SampEn      : {f['sample_entropy']}")
    print(f"    LF/HF ratio : {f['lf_hf_ratio']}")

    # ── Test 2: Atrial Fibrillation (highly irregular) ───────────────────
    print("\n  Test 2: Simulated Atrial Fibrillation (very irregular)")
    rr_af = np.random.uniform(low=0.45, high=1.1, size=30)
    f2 = compute_hrv_features(rr_af)
    print(f"    Mean HR     : {f2['mean_hr_bpm']:.1f} bpm")
    print(f"    SDNN        : {f2['sdnn_ms']:.1f} ms   (expect HIGH for AF)")
    print(f"    RMSSD       : {f2['rmssd_ms']:.1f} ms   (expect HIGH for AF)")
    print(f"    pNN50       : {f2['pnn50_pct']:.1f} %   (expect HIGH for AF)")
    print(f"    SD1/SD2     : {f2['sd1_sd2_ratio']:.3f}  (expect > 0.5 for AF)")
    print(f"    SampEn      : {f2['sample_entropy']}")

    # ── Test 3: Too few beats ─────────────────────────────────────────────
    print("\n  Test 3: Too few beats (should return None)")
    result = compute_hrv_features(np.array([0.8, 0.79, 0.81]))
    print(f"    Result      : {result}  (expect None)")

    # ── Test 4: Real data from record 100 ────────────────────────────────
    npy_path = "physiological_model/data/100_verified.npy"
    if os.path.exists(npy_path):
        print(f"\n  Test 4: Real MIT-BIH Record 100 (beats 1–20)")
        data = np.load(npy_path, allow_pickle=True).item()
        rr   = data['rr_intervals'][1:21]       # skip beat 0 (+marker), take 20
        f4   = compute_hrv_features(rr)
        print(f"    Mean HR     : {f4['mean_hr_bpm']:.1f} bpm  (expect ~75)")
        print(f"    SDNN        : {f4['sdnn_ms']:.1f} ms")
        print(f"    RMSSD       : {f4['rmssd_ms']:.1f} ms")
        print(f"    pNN50       : {f4['pnn50_pct']:.1f} %")
        print(f"    SD1/SD2     : {f4['sd1_sd2_ratio']:.3f}")
        print(f"    Has freq    : {f4['has_freq_features']}")
        print(f"    LF/HF       : {f4['lf_hf_ratio']}")
        print(f"    SampEn      : {f4['sample_entropy']}")
    else:
        print(f"\n  Test 4: Skipped (run day1_load_and_verify.py first)")

    print("\n  ✓ All tests complete.\n")