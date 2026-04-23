# =========================================
# ECG DATA PROCESSING PIPELINE (FINAL SUBMISSION)
# =========================================

import os
import zipfile
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch

# =========================================
# 1. EXTRACT DATASET
# =========================================

zip_path = "mit-bih-arrhythmia-database-1.0.0.zip"
extract_path = "ecg_data"

if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print("Dataset ready.\n")

# =========================================
# 2. CORRECT DATA PATH (IMPORTANT)
# =========================================

data_path = os.path.join(extract_path, "mit-bih-arrhythmia-database-1.0.0")

# =========================================
# 3. ECG CLEANING FUNCTION
# =========================================

def clean_ecg(signal, fs):
    # Bandpass filter (0.5–40 Hz)
    b, a = butter(2, [0.5/(fs/2), 40/(fs/2)], btype='band')
    signal = filtfilt(b, a, signal)

    # Notch filter (50 Hz)
    b_notch, a_notch = iirnotch(50/(fs/2), Q=30)
    signal = filtfilt(b_notch, a_notch, signal)

    return signal

# =========================================
# 4. FEATURE EXTRACTION
# =========================================

def extract_features(r_peaks, fs):
    rr = np.diff(r_peaks) / fs

    return {
        "mean_rr": np.mean(rr),
        "sdnn": np.std(rr),
        "rmssd": np.sqrt(np.mean(np.square(np.diff(rr)))),
        "heart_rate": 60 / np.mean(rr)
    }

# =========================================
# 5. LOAD RECORDS
# =========================================

records = list(set([
    f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.hea')
]))

print(f"Total records found: {len(records)}\n")

# =========================================
# 6. MAIN PIPELINE
# =========================================

all_features = []
all_clean_signals = {}

# OPTIONAL: store R-peaks for proof
save_rpeaks = True

for rec in records:
    try:
        print(f"Processing record: {rec}")

        record_path = os.path.join(data_path, rec)

        # Load ECG signal + annotations
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')

        signal = record.p_signal[:, 0]   # MLII lead
        fs = record.fs
        r_peaks = annotation.sample      # ✅ Annotated R-peaks (GROUND TRUTH)

        # Clean signal
        clean_signal = clean_ecg(signal, fs)

        # Extract HRV features
        features = extract_features(r_peaks, fs)
        features["record"] = rec

        # Store
        all_features.append(features)
        all_clean_signals[rec] = clean_signal

        # OPTIONAL: Save R-peaks (for demonstration)
        if save_rpeaks:
            pd.DataFrame(r_peaks, columns=["R_peak_index"]).to_csv(
                f"{rec}_rpeaks.csv", index=False
            )

    except Exception as e:
        print(f"Skipping {rec}: {e}")

print("\nProcessing complete.\n")

# =========================================
# 7. SAVE FINAL OUTPUTS
# =========================================

# Clean ECG dataset
np.save("clean_ecg_dataset.npy", all_clean_signals)
print("Saved: clean_ecg_dataset.npy")

# Feature dataset
df = pd.DataFrame(all_features)
df.to_csv("ecg_features.csv", index=False)
print("Saved: ecg_features.csv")

print("\n✅ FINAL: Data ready for ML training.")