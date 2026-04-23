import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import wfdb
import joblib
from tensorflow import keras

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = r"C:\Users\shrey\OneDrive\Documents\ECG_data"
DB_DIR      = os.path.join(BASE_DIR, "ECG_data", "mit-bih-arrhythmia-database-1.0.0")
LSTM_PATH   = os.path.join(BASE_DIR, "models", "lstm_model.keras")
RF_PATH     = os.path.join(BASE_DIR, "models", "rf_model_v2.pkl")
WINDOW      = 360       # samples either side handled below (360 total)
HALF        = 180
FS          = 360       # Hz
THRESHOLD   = 0.3       # recommended operating point

# ── Load models once at import time ──────────────────────────────────────────
_lstm_model = None
_rf_model   = None

def _get_lstm():
    global _lstm_model
    if _lstm_model is None:
        _lstm_model = keras.models.load_model(LSTM_PATH)
    return _lstm_model

def _get_rf():
    global _rf_model
    if _rf_model is None:
        _rf_model = joblib.load(RF_PATH)
    return _rf_model

# ── Feature extraction for RF (20 features, same as training) ────────────────
def _extract_features(windows: np.ndarray) -> np.ndarray:
    """windows: (N, 360) float32 → returns (N, 20) float64"""
    w = windows.astype(np.float64)
    d1 = np.diff(w, axis=1)
    d2 = np.diff(d1, axis=1)
    center = w[:, 140:220]
    left   = w[:, :180]
    right  = w[:, 180:]

    feats = np.column_stack([
        w.mean(1), w.std(1), w.max(1), w.min(1),
        w.max(1) - w.min(1),
        np.percentile(w, 25, axis=1), np.percentile(w, 75, axis=1),
        np.abs(w).mean(1),
        d1.std(1), d1.max(1), d1.min(1), np.abs(d1).mean(1),
        d2.std(1), np.abs(d2).mean(1),
        center.max(1), center.min(1), center.std(1),
        center.max(1) - center.min(1),
        left.mean(1) - right.mean(1),
        left.std(1)  - right.std(1),
    ])
    return feats

# ── R-peak detection (load from pre-saved CSV, fall back to wfdb annotations) ─
def _get_rpeaks(record_id: str) -> np.ndarray:
    csv_path = os.path.join(BASE_DIR, f"{record_id}_rpeaks.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        return pd.read_csv(csv_path)["R_peak_index"].values.astype(int)
    # fallback: read from wfdb annotation
    ann = wfdb.rdann(os.path.join(DB_DIR, record_id), "atr")
    return np.array(ann.sample)

# ── Main prediction function ──────────────────────────────────────────────────
def predict_record(
    record_id: str,
    threshold: float = THRESHOLD,
    use_lstm: bool = True,
) -> dict:
    """
    Predict PVC probability for every beat in a MIT-BIH record.

    Parameters
    ----------
    record_id : str   e.g. "100", "200"
    threshold : float operating threshold (default 0.3)
    use_lstm  : bool  True → LSTM probabilities, False → RF probabilities

    Returns
    -------
    dict with keys:
        beat_probs   : np.ndarray (N,)  float  — PVC probability per beat
        beat_labels  : np.ndarray (N,)  int    — 1=PVC, 0=Normal at threshold
        beat_samples : np.ndarray (N,)  int    — R-peak sample index per beat
        risk_score   : float            — fraction of beats flagged as PVC
        summary      : dict             — counts and metadata
    """
    # 1. Load signal
    record_path = os.path.join(DB_DIR, record_id)
    record  = wfdb.rdrecord(record_path)
    signal  = record.p_signal[:, 0].astype(np.float32)

    # 2. Get R-peaks
    rpeaks = _get_rpeaks(record_id)

    # 3. Extract windows
    windows, valid_peaks = [], []
    for rp in rpeaks:
        start, end = rp - HALF, rp + HALF
        if start >= 0 and end <= len(signal):
            windows.append(signal[start:end])
            valid_peaks.append(rp)

    if len(windows) == 0:
        raise ValueError(f"No valid windows extracted for record {record_id}")

    windows      = np.array(windows, dtype=np.float32)   # (N, 360)
    valid_peaks  = np.array(valid_peaks, dtype=int)

    # 4. Normalise (per-beat z-score, same as training)
    X_norm = (windows - windows.mean(axis=1, keepdims=True)) / \
             (windows.std(axis=1, keepdims=True) + 1e-8)

    # 5. Get probabilities
    if use_lstm:
        X_lstm = X_norm[:, :, np.newaxis]          # (N, 360, 1)
        probs  = _get_lstm().predict(X_lstm, batch_size=256, verbose=0).flatten()
    else:
        feats = _extract_features(X_norm)
        probs = _get_rf().predict_proba(feats)[:, 1]

    # 6. Apply threshold
    labels = (probs >= threshold).astype(int)

    # 7. Build summary
    n_beats  = len(labels)
    n_pvc    = int(labels.sum())
    risk     = round(float(n_pvc / n_beats), 4) if n_beats > 0 else 0.0

    summary = {
        "record_id"    : record_id,
        "total_beats"  : n_beats,
        "pvc_count"    : n_pvc,
        "normal_count" : n_beats - n_pvc,
        "risk_score"   : risk,
        "threshold"    : threshold,
        "model"        : "lstm" if use_lstm else "rf_v2",
        "fs_hz"        : FS,
    }

    return {
        "beat_probs"   : probs,
        "beat_labels"  : labels,
        "beat_samples" : valid_peaks,
        "risk_score"   : risk,
        "summary"      : summary,
    }


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = predict_record("200")
    s = result["summary"]
    print(f"Record      : {s['record_id']}")
    print(f"Total beats : {s['total_beats']}")
    print(f"PVC flagged : {s['pvc_count']}")
    print(f"Risk score  : {s['risk_score']:.4f}  ({s['risk_score']*100:.1f}% of beats)")
    print(f"Threshold   : {s['threshold']}")
    print(f"Model       : {s['model']}")