"""
evaluate.py — Full Evaluation Across All 48 Records (Person 4)
==============================================================
Two-pass strategy:
  Pass 1: Collect ALL phys+lstm scores → train global LogisticRegression
  Pass 2: Evaluate all 4 methods on every record using the trained model

Outputs:
  - Console: comparison table (Precision / Recall / F1)
  - results/evaluate_results.csv  — per-record breakdown
  - results/hybrid_results.csv    — FULL output for Person 5 (Streamlit UI)

Usage:
    python evaluate.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from predict import predict_record
from fusion import (
    align_scores,
    weighted_fusion,
    max_fusion,
    apply_learned_fusion,
    get_alpha,
    compute_metrics,
    MASTER_CSV,
)

# ── All 48 MIT-BIH records ────────────────────────────────────────────────────
ALL_RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

THRESHOLD = 0.5    # final decision boundary for all methods


def _banner(msg: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")


def _safe_predict(record_id: str) -> dict | None:
    """Run predict_record and suppress any per-record errors."""
    try:
        return predict_record(record_id, threshold=0.3)
    except Exception as e:
        print(f"  [WARN] predict_record('{record_id}') failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
#  PASS 1 — Collect all aligned data for training the global fusion model
# ════════════════════════════════════════════════════════════════════════════════

def collect_all_data(phys_df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate all 48 records, align phys+LSTM scores, return a single DataFrame.
    Used to train the global LogisticRegression fusion model.
    """
    _banner("PASS 1 — Collecting data for all 48 records")
    all_frames = []
    failed = []

    for i, rid in enumerate(ALL_RECORDS, 1):
        print(f"  [{i:02d}/48] Record {rid} ...", end=" ", flush=True)
        lstm_result = _safe_predict(rid)
        if lstm_result is None:
            failed.append(rid)
            print("SKIP")
            continue

        merged = align_scores(rid, phys_df, lstm_result)
        if merged.empty:
            failed.append(rid)
            print("EMPTY")
            continue

        all_frames.append(merged)
        n_abn = merged['true_abnormal'].sum()
        print(f"OK  ({len(merged)} beats, {n_abn} abnormal)")

    if failed:
        print(f"\n  FAILED records: {failed}")

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n  Total aligned beats: {len(combined):,}")
    print(f"  Abnormal beats     : {combined['true_abnormal'].sum():,} "
          f"({combined['true_abnormal'].mean()*100:.1f}%)")
    return combined


# ════════════════════════════════════════════════════════════════════════════════
#  TRAIN GLOBAL LOGISTIC REGRESSION FUSION MODEL
# ════════════════════════════════════════════════════════════════════════════════

def train_global_lr(combined: pd.DataFrame) -> LogisticRegression:
    """Train LR on all records to show global coefficients for the paper."""
    _banner("Training Global Logistic Regression Fusion Model")

    X = np.column_stack([
        combined['phys_score'].values,
        combined['lstm_score'].values,
    ])
    y = combined['true_abnormal'].values.astype(int)

    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X, y)

    print(f"  Training samples   : {len(y):,}")
    print(f"  Positive rate      : {y.mean()*100:.2f}%")
    print(f"\n  -- Learned Fusion Coefficients (KEY RESULT) ----------")
    print(f"  > Physio weight: {model.coef_[0][0]:.3f}")
    print(f"  > LSTM weight  : {model.coef_[0][1]:.3f}")
    print(f"  ------------------------------------------------------------")
    return model


# ════════════════════════════════════════════════════════════════════════════════
#  PASS 2 — Evaluate all methods on every record
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_all(phys_df: pd.DataFrame,
                 combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Second pass: compute per-record metrics using Leave-One-Patient-Out CV.
    Also builds the full hybrid_results.csv DataFrame.

    Returns
    -------
    per_record_df : DataFrame with per-record metrics
    all_results   : DataFrame with beat-level results for hybrid_results.csv
    """
    _banner("PASS 2 — Per-record Evaluation")

    per_record_rows = []
    all_result_frames = []

    for i, rid in enumerate(ALL_RECORDS, 1):
        print(f"  [{i:02d}/48] Record {rid} ...", end=" ", flush=True)
        lstm_result = _safe_predict(rid)
        if lstm_result is None:
            print("SKIP")
            continue

        merged = align_scores(rid, phys_df, lstm_result)
        if merged.empty:
            print("EMPTY")
            continue

        phys_scores = merged['phys_score'].values.astype(np.float32)
        lstm_scores = merged['lstm_score'].values.astype(np.float32)
        true_binary = merged['true_abnormal'].values.astype(int)
        alpha       = get_alpha(rid)

        # ── Leave-One-Patient-Out Cross-Validation ─────────────────────────
        train_data = combined[combined['record_id'] != int(rid)]
        X_train = np.column_stack([train_data['phys_score'], train_data['lstm_score']])
        y_train = train_data['true_abnormal'].values.astype(int)
        
        lr_model = LogisticRegression(class_weight='balanced', random_state=42)
        lr_model.fit(X_train, y_train)

        # ── Compute all fusion scores ──────────────────────────────────────
        w_scores  = weighted_fusion(phys_scores, lstm_scores, alpha=alpha)
        mx_scores = max_fusion(phys_scores, lstm_scores, phys_thresh=THRESHOLD, lstm_thresh=0.3)
        lr_scores = apply_learned_fusion(lr_model, phys_scores, lstm_scores)

        # ── Per-record metrics ─────────────────────────────────────────────
        m_phys = compute_metrics(true_binary, phys_scores,  THRESHOLD)
        m_lstm = compute_metrics(true_binary, lstm_scores,  0.3)
        m_wgt  = compute_metrics(true_binary, w_scores,     THRESHOLD)
        m_max  = compute_metrics(true_binary, mx_scores,    THRESHOLD)
        m_lr   = compute_metrics(true_binary, lr_scores,    THRESHOLD)

        best_f1 = max(m_phys['f1'], m_lstm['f1'], m_wgt['f1'],
                      m_max['f1'], m_lr['f1'])

        per_record_rows.append({
            'record_id'        : rid,
            'n_beats'          : len(true_binary),
            'n_abnormal'       : int(true_binary.sum()),
            'pct_abnormal'     : round(true_binary.mean() * 100, 2),
            'is_paced'         : rid in {'102','107','109','111','212'},
            'alpha'            : alpha,
            # Physio only
            'phys_precision'   : round(m_phys['precision'], 4),
            'phys_recall'      : round(m_phys['recall'], 4),
            'phys_f1'          : round(m_phys['f1'], 4),
            # LSTM only
            'lstm_precision'   : round(m_lstm['precision'], 4),
            'lstm_recall'      : round(m_lstm['recall'], 4),
            'lstm_f1'          : round(m_lstm['f1'], 4),
            # Weighted fusion
            'wgt_precision'    : round(m_wgt['precision'], 4),
            'wgt_recall'       : round(m_wgt['recall'], 4),
            'wgt_f1'           : round(m_wgt['f1'], 4),
            # Max fusion
            'max_precision'    : round(m_max['precision'], 4),
            'max_recall'       : round(m_max['recall'], 4),
            'max_f1'           : round(m_max['f1'], 4),
            # Learned LR fusion
            'lr_precision'     : round(m_lr['precision'], 4),
            'lr_recall'        : round(m_lr['recall'], 4),
            'lr_f1'            : round(m_lr['f1'], 4),
            'best_f1'          : round(best_f1, 4),
        })

        # ── Build beat-level output for hybrid_results.csv ─────────────────
        final_label = np.where(lr_scores >= THRESHOLD, 'abnormal', 'normal')
        rec_df = pd.DataFrame({
            'record_id'      : rid,
            'beat_index'     : merged['beat_index'].values,
            'rpeak_sample'   : merged['rpeak_sample'].values,
            'phys_score'     : np.round(phys_scores, 6),
            'lstm_score'     : np.round(lstm_scores, 6),
            'hybrid_score'   : np.round(lr_scores, 6),
            'final_label'    : final_label,
            'true_label'     : merged['true_label'].values,
            'arrhythmia_type': merged['arrhythmia_type'].values,
            'explanation'    : merged['explanation'].values,
        })
        all_result_frames.append(rec_df)

        best_tag = "*" if np.isclose(m_lr['f1'], best_f1) else " "
        print(f"OK  | RF F1={m_lr['f1']:.3f}{best_tag} | "
              f"LSTM F1={m_lstm['f1']:.3f} | "
              f"Phys F1={m_phys['f1']:.3f}")

    per_record_df = pd.DataFrame(per_record_rows)
    all_results   = pd.concat(all_result_frames, ignore_index=True)
    return per_record_df, all_results


# ════════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════════

def print_summary_table(per_record_df: pd.DataFrame, all_results: pd.DataFrame) -> None:
    """Print the globally-averaged comparison table for the IEEE paper."""
    _banner("SUMMARY TABLE — Global Metrics (Excluding Pacemakers)")
    
    paced_records = ['102', '107', '109', '111', '212']
    valid_results = all_results[~all_results['record_id'].astype(str).isin(paced_records)]

    y_true = (valid_results['true_label'] > 0).astype(int)
    
    # Calculate global w_scores and mx_scores for the table
    lstm_global = valid_results['lstm_score'].values
    scaled_lstm = np.where(lstm_global < 0.3,
                           lstm_global * (0.5 / 0.3),
                           0.5 + (lstm_global - 0.3) * (0.5 / 0.7))
    alphas = valid_results['record_id'].astype(str).map(get_alpha).values
    w_scores = alphas * scaled_lstm + (1 - alphas) * valid_results['phys_score']
    mx_scores = max_fusion(valid_results['phys_score'].values, valid_results['lstm_score'].values, phys_thresh=THRESHOLD, lstm_thresh=0.3)
    
    def _get_global_metrics(scores, th):
        y_pred = (scores >= th).astype(int)
        return (
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0)
        )

    results = {
        'LSTM only':      _get_global_metrics(valid_results['lstm_score'], 0.3),
        'Physio only':    _get_global_metrics(valid_results['phys_score'], THRESHOLD),
        'Hybrid weighted': _get_global_metrics(w_scores, THRESHOLD),
        'Hybrid max':     _get_global_metrics(mx_scores, THRESHOLD),
        'Hybrid learned': _get_global_metrics(valid_results['hybrid_score'], THRESHOLD),
    }

    best_f1 = max(v[2] for v in results.values())
    phys_f1 = results['Physio only'][2]
    lstm_f1 = results['LSTM only'][2]

    print(f"\n  {'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 54)

    for method, (p, r, f) in results.items():
        marker = " << BEST" if np.isclose(f, best_f1) else ""
        print(f"  {method:<20} {p:>10.3f} {r:>10.3f} {f:>10.3f}{marker}")

    print(f"\n  -- Improvement over baselines (global F1) ---------------")
    for method, (p, r, f) in results.items():
        if method in ('Physio only', 'LSTM only'):
            continue
        delta_phys = (f - phys_f1) * 100
        delta_lstm = (f - lstm_f1) * 100
        print(f"  {method:<20} vs Physio: {delta_phys:+.1f}pp  "
              f"vs LSTM: {delta_lstm:+.1f}pp")

    print(f"\n  Known baselines:  Physio standalone F1=35.28%")
    print(f"                    LSTM standalone F1=57.1% (@ threshold=0.3)")


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    _banner("EVALUATE.PY — Full 48-Record Evaluation")
    print(f"  Results will be saved to: {RESULTS_DIR}")

    # Load master CSV once
    print("\n  Loading master_results.csv ...")
    phys_df = pd.read_csv(MASTER_CSV)
    print(f"  Loaded {len(phys_df):,} rows.")

    # Pass 1: collect all data
    combined = collect_all_data(phys_df)

    # Train global model (for feature importance printing only)
    train_global_lr(combined)

    # Pass 2: evaluate with Leave-One-Patient-Out Cross-Validation
    per_record_df, all_results = evaluate_all(phys_df, combined)

    # Print summary table
    print_summary_table(per_record_df, all_results)

    # ── Save results ───────────────────────────────────────────────────────────
    eval_path   = os.path.join(RESULTS_DIR, "evaluate_results.csv")
    hybrid_path = os.path.join(RESULTS_DIR, "hybrid_results.csv")

    per_record_df.to_csv(eval_path, index=False)
    all_results.to_csv(hybrid_path, index=False)

    _banner("FILES SAVED")
    print(f"  Per-record metrics : {eval_path}")
    print(f"  Hybrid results CSV : {hybrid_path}")
    print(f"  Rows in hybrid CSV : {len(all_results):,}")
    print(f"\n  Tell Person 5:")
    print(f"  > CSV at results/hybrid_results.csv")
    print(f"  > Filter by record_id to load one record at a time.")
    print(f"  > 'explanation' column has human-readable text for Streamlit.")
    print(f"  > 'final_label' is 'normal' or 'abnormal' - use for colour coding.")
    print(f"  > 'hybrid_score' is 0-1 float - use for risk gauge.")
    print(f"\n{'='*65}\n  DONE\n{'='*65}\n")


if __name__ == "__main__":
    main()
