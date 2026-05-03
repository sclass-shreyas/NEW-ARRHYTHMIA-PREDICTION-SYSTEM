import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import wfdb
from scipy.signal import resample_poly
from tensorflow import keras
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(_HERE)  # parent of incart/ folder
INCART_DIR  = _HERE
LSTM_PATH   = os.path.join(BASE_DIR, "models", "lstm_model.keras")
RECORDS     = ["I01", "I02"]

INCART_FS   = 257   # INCART sample rate
TARGET_FS   = 360   # what our model expects
HALF        = 180   # half window size at 360 Hz
THRESHOLD   = 0.3

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading LSTM model...")
model = keras.models.load_model(LSTM_PATH)

all_probs  = []
all_labels = []

for record_id in RECORDS:
    print(f"\nProcessing {record_id}...")
    
    # 1. Load signal and annotations
    record = wfdb.rdrecord(os.path.join(INCART_DIR, record_id))
    ann    = wfdb.rdann(os.path.join(INCART_DIR, record_id), "atr")
    
    # 2. Take lead I (index 0), resample from 257 Hz → 360 Hz
    signal_raw = record.p_signal[:, 0].astype(np.float32)
    signal     = resample_poly(signal_raw, TARGET_FS, INCART_FS).astype(np.float32)
    
    # 3. Scale annotation sample indices to new sample rate
    scale        = TARGET_FS / INCART_FS
    rpeaks_orig  = np.array(ann.sample)
    rpeaks       = (rpeaks_orig * scale).astype(int)
    symbols      = np.array(ann.symbol)
    
    # 4. Binary labels: V → 1, N → 0
    beat_labels = (symbols == 'V').astype(int)
    
    # 5. Extract windows
    windows, valid_labels = [], []
    for rp, lbl in zip(rpeaks, beat_labels):
        start, end = rp - HALF, rp + HALF
        if start >= 0 and end <= len(signal):
            windows.append(signal[start:end])
            valid_labels.append(lbl)
    
    windows      = np.array(windows, dtype=np.float32)
    valid_labels = np.array(valid_labels, dtype=int)
    
    # 6. Normalise per-beat
    X_norm = (windows - windows.mean(axis=1, keepdims=True)) / \
             (windows.std(axis=1, keepdims=True) + 1e-8)
    X_lstm = X_norm[:, :, np.newaxis]
    
    # 7. Predict
    probs = model.predict(X_lstm, batch_size=256, verbose=0).flatten()
    
    # 8. Record-level summary
    pvc_true  = int(beat_labels.sum())
    pvc_pred  = int((probs >= THRESHOLD).sum())
    print(f"  Beats: {len(valid_labels)} | True PVCs: {pvc_true} | Predicted PVCs: {pvc_pred}")
    
    all_probs.append(probs)
    all_labels.append(valid_labels)

# ── Combined metrics ──────────────────────────────────────────────────────────
y_prob = np.concatenate(all_probs)
y_true = np.concatenate(all_labels)
y_pred = (y_prob >= THRESHOLD).astype(int)

print(f"\n{'='*50}")
print(f"INCART GENERALIZATION TEST — threshold={THRESHOLD}")
print(f"{'='*50}")
print(f"Total beats : {len(y_true)}")
print(f"True PVCs   : {int(y_true.sum())}")
print(f"Recall      : {recall_score(y_true, y_pred):.3f}")
print(f"Precision   : {precision_score(y_true, y_pred):.3f}")
print(f"F1          : {f1_score(y_true, y_pred):.3f}")
cm = confusion_matrix(y_true, y_pred)
print(f"Confusion matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")