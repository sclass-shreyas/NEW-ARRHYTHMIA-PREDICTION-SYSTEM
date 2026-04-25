"""
PERSON 3 — Physiological Model
Day 1: Data Loading & Verification Script

What this script does:
- Loads R-peaks from Person 1's CSV (R_peak_index column)
- Loads beat labels from the .atr annotation file using wfdb
- Computes RR intervals in seconds
- Prints a full sanity check summary
- Saves a verified data dict as .npy for use in next steps

Run this from the ROOT of your cloned repo:
    python day1_load_and_verify.py
"""

import numpy as np
import pandas as pd
import wfdb
import os

# ─────────────────────────────────────────────
# CONFIGURATION — adjust paths if needed
# ─────────────────────────────────────────────

RECORD_ID      = "100"                          # change to test other records
FS             = 360.0                          # sampling frequency (Hz), confirmed

# Path to the raw MIT-BIH record files (.dat, .hea, .atr)
RECORD_PATH    = f"ecg_data/mit-bih-arrhythmia-database-1.0.0/{RECORD_ID}"

# Path to Person 1's R-peaks CSV
RPEAKS_CSV     = f"{RECORD_ID}_rpeaks.csv"

# Output path for verified data dict
OUTPUT_PATH    = f"physiological_model/data/{RECORD_ID}_verified.npy"

# ─────────────────────────────────────────────
# AAMI LABEL MAP
# MIT-BIH annotation symbols → AAMI integer classes
# ─────────────────────────────────────────────
AAMI_MAP = {
    # Class 0 — Normal
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    # Class 1 — Supraventricular ectopic (S)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    # Class 2 — Ventricular ectopic / PVC (V)
    'V': 2, 'E': 2,
    # Class 3 — Fusion beat (F)
    'F': 3,
    # Class 4 — Unknown / paced (Q)
    'f': 4, 'Q': 4, '/': 4, '~': 4
}


def load_record(record_path, rpeaks_csv, fs):
    """
    Loads and aligns all data needed by the physiological model.

    Returns a dict with:
        signal        : raw ECG signal array (Lead MLII)
        fs            : sampling frequency
        rpeaks        : R-peak sample indices from Person 1's CSV
        rr_intervals  : RR intervals in seconds, shape (N-1,)
        beat_symbols  : raw annotation chars per beat e.g. 'N', 'V'
        beat_labels   : AAMI integer class per beat (0–4)
        n_beats       : total number of beats
    """

    # ── 1. Load raw ECG signal ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Loading record: {record_path}")
    print(f"{'='*55}")

    record     = wfdb.rdrecord(record_path)
    signal     = record.p_signal[:, 0]          # Lead MLII (channel 0)
    print(f"  ✓ ECG signal loaded     | shape: {signal.shape} | fs: {record.fs} Hz")

    # ── 2. Load annotations (.atr) ──────────────────────────────────────
    annotation   = wfdb.rdann(record_path, 'atr')
    ann_samples  = annotation.sample             # sample indices of all annotations
    ann_symbols  = annotation.symbol             # annotation chars

    print(f"  ✓ Annotations loaded    | {len(ann_samples)} total annotation events")

    # ── 3. Load Person 1's R-peaks CSV ──────────────────────────────────
    rpeaks_df   = pd.read_csv(rpeaks_csv)
    rpeaks_arr  = rpeaks_df['R_peak_index'].values.astype(int)

    print(f"  ✓ R-peaks CSV loaded    | {len(rpeaks_arr)} R-peaks from Person 1")

    # ── 4. Align beat labels to Person 1's R-peaks ──────────────────────
    # Strategy: for each of Person 1's R-peaks, find the closest annotation
    # sample and take its symbol. Tolerance = ±10 samples (~28ms at 360Hz)
    TOLERANCE = 10
    beat_symbols = []
    beat_labels  = []
    unmatched    = 0

    for rpeak in rpeaks_arr:
        diffs = np.abs(ann_samples - rpeak)
        closest_idx = np.argmin(diffs)

        if diffs[closest_idx] <= TOLERANCE:
            sym = ann_symbols[closest_idx]
            beat_symbols.append(sym)
            beat_labels.append(AAMI_MAP.get(sym, 4))   # default to Q (4) if unknown
        else:
            # No annotation close enough — mark as unknown
            beat_symbols.append('?')
            beat_labels.append(4)
            unmatched += 1

    beat_symbols = np.array(beat_symbols)
    beat_labels  = np.array(beat_labels, dtype=np.int32)

    print(f"  ✓ Beat labels aligned   | unmatched: {unmatched}")

    # ── 5. Compute RR intervals ──────────────────────────────────────────
    rr_intervals = np.diff(rpeaks_arr) / fs       # seconds, shape (N-1,)

    print(f"  ✓ RR intervals computed | shape: {rr_intervals.shape}")

    return {
        'record_id'   : RECORD_ID,
        'signal'      : signal,
        'fs'          : fs,
        'rpeaks'      : rpeaks_arr,
        'rr_intervals': rr_intervals,
        'beat_symbols': beat_symbols,
        'beat_labels' : beat_labels,
        'n_beats'     : len(rpeaks_arr),
    }


def sanity_check(data):
    """
    Prints a full sanity check report.
    Run this every time you load a new record.
    """
    rr  = data['rr_intervals']
    lbl = data['beat_labels']
    sym = data['beat_symbols']

    print(f"\n{'='*55}")
    print(f"  SANITY CHECK — Record {data['record_id']}")
    print(f"{'='*55}")

    # ── RR interval stats ────────────────────────────────────────────────
    mean_hr = 60.0 / np.mean(rr)
    print(f"\n  RR Intervals (seconds):")
    print(f"    Count   : {len(rr)}")
    print(f"    Min     : {np.min(rr):.4f} s  ({60/np.max(rr):.1f} bpm)")
    print(f"    Max     : {np.max(rr):.4f} s  ({60/np.min(rr):.1f} bpm)")
    print(f"    Mean    : {np.mean(rr):.4f} s  ({mean_hr:.1f} bpm)")
    print(f"    Std     : {np.std(rr):.4f} s")

    # ── Physiological sanity flags ───────────────────────────────────────
    print(f"\n  Physiological Flags:")
    if np.any(rr < 0.25):
        print(f"    ⚠ WARNING: {np.sum(rr < 0.25)} RR intervals < 0.25s (>240 bpm) — likely noise or mis-detection")
    else:
        print(f"    ✓ No extreme short RR intervals detected")

    if np.any(rr > 2.0):
        print(f"    ⚠ WARNING: {np.sum(rr > 2.0)} RR intervals > 2.0s (<30 bpm) — possible pause or missed beat")
    else:
        print(f"    ✓ No extreme long RR intervals detected")

    # ── Beat label distribution ──────────────────────────────────────────
    label_names = {0: 'Normal (N)', 1: 'Supraventricular (S)',
                   2: 'Ventricular/PVC (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'}

    print(f"\n  Beat Label Distribution:")
    unique, counts = np.unique(lbl, return_counts=True)
    for u, c in zip(unique, counts):
        pct = 100 * c / len(lbl)
        print(f"    Class {u} — {label_names[u]:<28} : {c:>5} beats  ({pct:.1f}%)")

    # ── First 10 beats preview ───────────────────────────────────────────
    print(f"\n  First 10 Beats Preview:")
    print(f"    {'Beat':<6} {'R-peak':<10} {'RR (s)':<10} {'Symbol':<8} {'AAMI Label'}")
    print(f"    {'-'*50}")
    for i in range(min(10, data['n_beats'] - 1)):
        print(f"    {i:<6} {data['rpeaks'][i]:<10} {rr[i]:<10.4f} {sym[i]:<8} {lbl[i]}")

    print(f"\n  ✓ Sanity check complete. Record looks {'CLEAN' if np.sum(rr < 0.25) + np.sum(rr > 2.0) == 0 else 'NEEDS REVIEW'}.")
    print(f"{'='*55}\n")


def save_verified_data(data, output_path):
    """Saves the verified data dict as .npy for use in subsequent scripts."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, data)
    print(f"  ✓ Verified data saved → {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Check that required files exist
    missing = []
    for f in [RPEAKS_CSV, RECORD_PATH + '.hea', RECORD_PATH + '.dat', RECORD_PATH + '.atr']:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        print("\n  ✗ Missing files — check your paths:")
        for f in missing:
            print(f"      {f}")
        print("\n  Make sure you run this from the ROOT of the repo.\n")
        exit(1)

    # Load
    data = load_record(RECORD_PATH, RPEAKS_CSV, FS)

    # Verify
    sanity_check(data)

    # Save for next steps
    save_verified_data(data, OUTPUT_PATH)
