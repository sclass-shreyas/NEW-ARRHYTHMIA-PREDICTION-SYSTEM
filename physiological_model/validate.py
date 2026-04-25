"""
PERSON 3 — Physiological Model
Day 5: Validation Script (All Records)

File: physiological_model/validate.py

What this script does:
    1. Auto-discovers all available records from Person 1's rpeaks CSVs
    2. Runs physiological_analysis on every beat of every record
    3. Computes precision, recall, F1, accuracy across all records
    4. Saves a master per-beat CSV — handoff to Person 4 and Person 5
    5. Prints a full research-level validation report

Run from repo ROOT:
    python physiological_model/validate.py

Output files:
    physiological_model/results/master_results.csv
    physiological_model/results/per_record_metrics.csv
    physiological_model/results/validation_report.txt
"""

import numpy as np
import pandas as pd
import wfdb
import os
import sys
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
PHYS_DIR   = ROOT_DIR / "physiological_model"
DATA_DIR   = ROOT_DIR / "ecg_data" / "mit-bih-arrhythmia-database-1.0.0"
RESULT_DIR = PHYS_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PHYS_DIR))
from hrv_features import compute_hrv_features
from arrhythmia_classifier import classify_beat
from interface import analyze_full_record

# ── AAMI label map ───────────────────────────────────────────
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'f': 4, 'Q': 4, '/': 4, '~': 4
}

LABEL_NAMES = {
    0: 'Normal (N)',
    1: 'Supraventricular (S)',
    2: 'Ventricular/PVC (V)',
    3: 'Fusion (F)',
    4: 'Unknown (Q)'
}

CLASSIFICATION_THRESHOLD = 0.3   # phys_risk >= this → 'abnormal'
FS = 360.0
TOLERANCE = 10                    # samples, for annotation alignment


# ─────────────────────────────────────────────────────────────
# STEP 1: Auto-discover all available records
# ─────────────────────────────────────────────────────────────

def discover_records() -> list[str]:
    """
    Finds all record IDs that have BOTH:
    - A rpeaks CSV at repo root (Person 1's output)
    - The raw .atr file in the MIT-BIH folder (for labels)
    """
    csv_files = sorted(ROOT_DIR.glob("*_rpeaks.csv"))
    valid_records = []

    for csv_path in csv_files:
        record_id = csv_path.stem.replace("_rpeaks", "")
        atr_path  = DATA_DIR / f"{record_id}.atr"
        hea_path  = DATA_DIR / f"{record_id}.hea"

        if atr_path.exists() and hea_path.exists():
            valid_records.append(record_id)
        else:
            print(f"  ⚠ Skipping {record_id} — missing .atr or .hea file")

    return valid_records


# ─────────────────────────────────────────────────────────────
# STEP 2: Load a single record
# ─────────────────────────────────────────────────────────────

def load_record(record_id: str) -> dict | None:
    """
    Loads rpeaks CSV + .atr annotations for one record.
    Returns aligned dict or None if loading fails.
    """
    try:
        csv_path    = ROOT_DIR / f"{record_id}_rpeaks.csv"
        record_path = str(DATA_DIR / record_id)

        # Load R-peaks from Person 1's CSV
        rpeaks_df  = pd.read_csv(csv_path)
        rpeaks_arr = rpeaks_df['R_peak_index'].values.astype(int)

        # Load annotations from .atr
        annotation   = wfdb.rdann(record_path, 'atr')
        ann_samples  = annotation.sample
        ann_symbols  = annotation.symbol

        # Align labels to Person 1's R-peaks
        beat_symbols = []
        beat_labels  = []

        for rpeak in rpeaks_arr:
            diffs       = np.abs(ann_samples - rpeak)
            closest_idx = np.argmin(diffs)
            if diffs[closest_idx] <= TOLERANCE:
                sym = ann_symbols[closest_idx]
                beat_symbols.append(sym)
                beat_labels.append(AAMI_MAP.get(sym, 4))
            else:
                beat_symbols.append('?')
                beat_labels.append(4)

        beat_symbols = np.array(beat_symbols)
        beat_labels  = np.array(beat_labels, dtype=np.int32)

        # Remove + marker at beat 0 if present
        if len(beat_symbols) > 0 and beat_symbols[0] == '+':
            rpeaks_arr   = rpeaks_arr[1:]
            beat_symbols = beat_symbols[1:]
            beat_labels  = beat_labels[1:]

        rr_intervals = np.diff(rpeaks_arr) / FS

        return {
            'record_id'   : record_id,
            'rpeaks'      : rpeaks_arr,
            'rr_intervals': rr_intervals,
            'beat_labels' : beat_labels,
            'beat_symbols': beat_symbols,
            'n_beats'     : len(rpeaks_arr),
        }

    except Exception as e:
        print(f"  ✗ Failed to load record {record_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# STEP 3: Process one record → per-beat rows
# ─────────────────────────────────────────────────────────────

def process_record(data: dict) -> list[dict]:
    """
    Runs physiological analysis on all beats of one record.
    Returns list of row dicts for the master CSV.
    """
    record_id    = data['record_id']
    rr_intervals = data['rr_intervals']
    n_beats      = data['n_beats']
    rpeaks       = data['rpeaks']
    beat_labels  = data['beat_labels']
    beat_symbols = data['beat_symbols']

    risk_scores, results = analyze_full_record(rr_intervals, n_beats)

    rows = []
    for i, result in enumerate(results):
        predicted_abnormal = 1 if risk_scores[i] >= CLASSIFICATION_THRESHOLD else 0
        true_abnormal      = 1 if beat_labels[i] >= 1 else 0  # label 0 = normal, 1–4 = abnormal

        rows.append({
            'record_id'        : record_id,
            'beat_index'       : i,
            'rpeak_sample'     : int(rpeaks[i]),
            'true_label'       : int(beat_labels[i]),
            'true_symbol'      : beat_symbols[i],
            'true_abnormal'    : true_abnormal,
            'phys_risk_score'  : round(float(risk_scores[i]), 4),
            'predicted_abnormal': predicted_abnormal,
            'classification'   : result['classification'],
            'arrhythmia_type'  : result['arrhythmia_type'],
            'explanation'      : result['explanation'],
        })

    return rows


# ─────────────────────────────────────────────────────────────
# STEP 4: Compute metrics for a set of rows
# ─────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    """
    Computes binary classification metrics.
    Positive class = abnormal (true_label >= 1).
    """
    y_true = np.array([r['true_abnormal']      for r in rows])
    y_pred = np.array([r['predicted_abnormal'] for r in rows])

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    scores = np.array([r['phys_risk_score'] for r in rows])
    labels = np.array([r['true_label']      for r in rows])

    mean_by_class = {}
    for cls in range(5):
        mask = labels == cls
        if np.any(mask):
            mean_by_class[cls] = float(np.mean(scores[mask]))

    return {
        'n_beats'    : len(rows),
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'precision'  : round(precision,   4),
        'recall'     : round(recall,      4),
        'f1_score'   : round(f1,          4),
        'accuracy'   : round(accuracy,    4),
        'specificity': round(specificity, 4),
        'mean_risk_by_class': mean_by_class,
    }


# ─────────────────────────────────────────────────────────────
# STEP 5: Print and save validation report
# ─────────────────────────────────────────────────────────────

def print_report(all_metrics: dict, per_record_metrics: list[dict],
                 record_ids: list[str]) -> str:
    """Builds and prints the full validation report string."""

    lines = []
    lines.append("=" * 65)
    lines.append("  PHYSIOLOGICAL MODEL — VALIDATION REPORT")
    lines.append("  Person 3 | MIT-BIH Arrhythmia Database")
    lines.append("=" * 65)

    lines.append(f"\n  Records processed : {len(record_ids)}")
    lines.append(f"  Total beats       : {all_metrics['n_beats']}")
    lines.append(f"  Classification threshold : {CLASSIFICATION_THRESHOLD}")

    lines.append("\n  --- Overall Binary Classification Metrics ---")
    lines.append(f"  (Positive = abnormal, label >= 1)")
    lines.append(f"  Accuracy    : {all_metrics['accuracy']*100:.2f}%")
    lines.append(f"  Precision   : {all_metrics['precision']*100:.2f}%")
    lines.append(f"  Recall      : {all_metrics['recall']*100:.2f}%")
    lines.append(f"  F1-Score    : {all_metrics['f1_score']*100:.2f}%")
    lines.append(f"  Specificity : {all_metrics['specificity']*100:.2f}%")
    lines.append(f"  TP={all_metrics['TP']}  FP={all_metrics['FP']}  "
                 f"TN={all_metrics['TN']}  FN={all_metrics['FN']}")

    lines.append("\n  --- Mean Physiological Risk Score by AAMI Class ---")
    for cls, mean_risk in sorted(all_metrics['mean_risk_by_class'].items()):
        bar  = "█" * int(mean_risk * 40)
        lines.append(f"  Class {cls} {LABEL_NAMES.get(cls,''):<28} "
                     f"mean={mean_risk:.4f}  {bar}")

    lines.append("\n  --- Per-Record Summary ---")
    lines.append(f"  {'Record':<10} {'Beats':>7} {'Abnormal':>9} "
                 f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
    lines.append("  " + "-"*55)

    for rm in per_record_metrics:
        n_abn = rm['TP'] + rm['FN']
        lines.append(f"  {rm['record_id']:<10} {rm['n_beats']:>7} {n_abn:>9} "
                     f"{rm['precision']:>10.4f} {rm['recall']:>8.4f} "
                     f"{rm['f1_score']:>8.4f}")

    lines.append("\n" + "=" * 65)

    report = "\n".join(lines)
    print(report)
    return report


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*65)
    print("  PHYSIOLOGICAL MODEL — FULL VALIDATION RUN")
    print("="*65)

    # ── Discover records ─────────────────────────────────────────────────
    record_ids = discover_records()
    print(f"\n  Found {len(record_ids)} valid records: {record_ids}\n")

    if not record_ids:
        print("  ✗ No valid records found. Check that rpeaks CSVs and")
        print("    MIT-BIH .atr files are both present.\n")
        sys.exit(1)

    # ── Process all records ──────────────────────────────────────────────
    all_rows           = []
    per_record_metrics = []

    for i, record_id in enumerate(record_ids):
        print(f"  [{i+1:>2}/{len(record_ids)}] Processing record {record_id}...",
              end=" ", flush=True)

        data = load_record(record_id)
        if data is None:
            print("SKIPPED")
            continue

        rows    = process_record(data)
        metrics = compute_metrics(rows)

        per_record_metrics.append({'record_id': record_id, **metrics})
        all_rows.extend(rows)

        print(f"  {data['n_beats']} beats | "
              f"F1={metrics['f1_score']:.3f} | "
              f"Recall={metrics['recall']:.3f}")

    # ── Overall metrics ──────────────────────────────────────────────────
    print(f"\n  Total beats processed: {len(all_rows)}")
    all_metrics = compute_metrics(all_rows)

    # ── Save master CSV ──────────────────────────────────────────────────
    master_csv_path = RESULT_DIR / "master_results.csv"
    master_df       = pd.DataFrame(all_rows)
    master_df.to_csv(master_csv_path, index=False)
    print(f"\n  ✓ Master CSV saved → {master_csv_path}")
    print(f"    Columns: {list(master_df.columns)}")
    print(f"    Shape  : {master_df.shape}")

    # ── Save per-record metrics CSV ──────────────────────────────────────
    per_record_csv_path = RESULT_DIR / "per_record_metrics.csv"
    pr_df = pd.DataFrame([
        {k: v for k, v in rm.items() if k != 'mean_risk_by_class'}
        for rm in per_record_metrics
    ])
    pr_df.to_csv(per_record_csv_path, index=False)
    print(f"  ✓ Per-record metrics saved → {per_record_csv_path}")

    # ── Print and save report ────────────────────────────────────────────
    report = print_report(all_metrics, per_record_metrics, record_ids)

    report_path = RESULT_DIR / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✓ Validation report saved → {report_path}")
    print("\n  ✓ All done. Hand master_results.csv to Person 4 and Person 5.\n")
