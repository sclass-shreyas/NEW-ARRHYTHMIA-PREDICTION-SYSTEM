"""
fusion.py — Hybrid Fusion Layer (Person 4)
==========================================
Combines physiological risk scores (Person 3) and LSTM probabilities
(Person 2) into a single hybrid arrhythmia risk score.

Three fusion methods:
  1. Weighted Average Fusion  — simple, interpretable
  2. Max Fusion               — maximises recall (clinical safety)
  3. Learned Logistic Fusion  — data-driven, research-level (MAIN RESULT)

Usage:
    python fusion.py                  # runs demo on record "100"
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV  = os.path.join(REPO_ROOT, "physiological_model", "results", "master_results.csv")

# Paced-rhythm records: HRV is artificially suppressed → trust LSTM more
PACED_RECORDS = {'102', '107', '109', '111', '212'}


def get_alpha(record_id: str) -> float:
    """
    Return the LSTM weight (alpha) for weighted fusion.
    alpha=1.0 → trust LSTM only, alpha=0.0 → trust physiology only.
    Paced records get alpha=0.85 (LSTM-heavy) because HRV-based physio
    scores are near-zero for all beats in those records.
    """
    return 0.85 if str(record_id) in PACED_RECORDS else 0.5


# ─── FUSION METHOD 1: Weighted Average ────────────────────────────────────────

def weighted_fusion(phys_scores: np.ndarray,
                    lstm_scores: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Weighted average of physiological and LSTM risk scores.

    Parameters
    ----------
    phys_scores : np.ndarray, shape (N,), values 0-1
        Person 3's physiological risk scores.
    lstm_scores : np.ndarray, shape (N,), values 0-1
        Person 2's LSTM PVC probabilities.
    alpha : float
        Weight given to LSTM output.
        alpha=0.0 → trust physiology only
        alpha=0.5 → equal weight (default)
    Combine scores using a weighted average.
    Scales LSTM scores from a 0.3 threshold to a 0.5 threshold to align with phys_scores.
    """
    scaled_lstm = np.where(lstm_scores < 0.3,
                           lstm_scores * (0.5 / 0.3),
                           0.5 + (lstm_scores - 0.3) * (0.5 / 0.7))
    return (alpha * scaled_lstm + (1 - alpha) * phys_scores).astype(np.float32)


# ─── FUSION METHOD 2: Max Fusion ──────────────────────────────────────────────

def max_fusion(phys_scores: np.ndarray,
               lstm_scores: np.ndarray,
               phys_thresh: float = 0.5,
               lstm_thresh: float = 0.3) -> np.ndarray:
    """
    Flag a beat as abnormal if EITHER model is confident.
    Maximises recall — missing an arrhythmia is worse than a false alarm.

    Scores are scaled before fusion to prevent uncalibrated suppression.
    """
    assert phys_scores.shape == lstm_scores.shape, (
        f"Shape mismatch: phys={phys_scores.shape}, lstm={lstm_scores.shape}"
    )
    scaled_lstm = lstm_scores * (phys_thresh / lstm_thresh)
    return np.clip(np.maximum(phys_scores, scaled_lstm), 0.0, 1.0).astype(np.float32)


# ─── FUSION METHOD 3: Learned Logistic Regression ─────────────────────────────

def train_learned_fusion(phys_scores: np.ndarray,
                         lstm_scores: np.ndarray,
                         true_labels_binary: np.ndarray):
    """
    Train a Logistic Regression model that learns the optimal combination
    of physiological and LSTM scores from labelled data.

    Parameters
    ----------
    phys_scores        : np.ndarray (N,) — physiological risk scores
    lstm_scores        : np.ndarray (N,) — LSTM probabilities
    true_labels_binary : np.ndarray (N,) — 1=abnormal (AAMI 1-4), 0=normal

    Returns
    -------
    model         : trained sklearn LogisticRegression
    hybrid_probs  : np.ndarray (N_test,) — predicted probabilities on test set
    y_test        : np.ndarray (N_test,) — ground-truth labels for test set
    """
    X = np.column_stack([phys_scores, lstm_scores])
    y = true_labels_binary.astype(int)

    # NOTE: train_test_split on beats from a single record is purely for this
    # 1-record smoke test demo. In a real pipeline (evaluate.py), we use 
    # Leave-One-Patient-Out Cross Validation to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    print("\n  ── Learned Fusion Coefficients (KEY RESULT FOR PAPER) ──")
    print(f"  > Physio weight : {model.coef_[0][0]:.3f}")
    print(f"  > LSTM weight   : {model.coef_[0][1]:.3f}")
    print("  --------------------------------------------------------------")

    hybrid_probs = model.predict_proba(X_test)[:, 1]
    return model, hybrid_probs, y_test


def apply_learned_fusion(model,
                         phys_scores: np.ndarray,
                         lstm_scores: np.ndarray) -> np.ndarray:
    """
    Apply a trained logistic regression fusion model to new data.

    Parameters
    ----------
    model       : trained LogisticRegression from train_learned_fusion()
    phys_scores : np.ndarray (N,)
    lstm_scores : np.ndarray (N,)

    Returns
    -------
    np.ndarray (N,) — hybrid probability 0-1
    """
    X = np.column_stack([phys_scores, lstm_scores])
    return model.predict_proba(X)[:, 1].astype(np.float32)


# ─── ALIGNMENT HELPER ─────────────────────────────────────────────────────────

def align_scores(record_id: str,
                 phys_df: pd.DataFrame,
                 lstm_result: dict) -> pd.DataFrame:
    """
    Align Person 3's physiological scores with Person 2's LSTM predictions
    by matching on R-peak sample position (the only reliable shared key).

    Also skips beat 0 (always a record-start '+' annotation, not a real beat).

    Parameters
    ----------
    record_id   : str  e.g. "100"
    phys_df     : full master_results DataFrame (all records)
    lstm_result : dict returned by predict_record(record_id)

    Returns
    -------
    pd.DataFrame with columns:
        record_id, beat_index, rpeak_sample,
        phys_score, lstm_score,
        true_label, true_abnormal, arrhythmia_type, explanation
    """
    # ── Person 3 data for this record ──────────────────────────────────────
    rec_phys = phys_df[phys_df['record_id'] == int(record_id)].copy()

    # ── Person 2 data ───────────────────────────────────────────────────────
    lstm_df = pd.DataFrame({
        'rpeak_sample': lstm_result['beat_samples'],
        'lstm_score'  : lstm_result['beat_probs'].astype(np.float32),
        'lstm_label'  : lstm_result['beat_labels'],
    })

    # ── Merge on R-peak position (safe alignment) ───────────────────────────
    merged = rec_phys.merge(lstm_df, on='rpeak_sample', how='inner')

    # ── Skip beat 0 (record-start marker, always phys_score=0.0) ───────────
    merged = merged[merged['beat_index'] > 0].reset_index(drop=True)

    # ── Rename for clarity ──────────────────────────────────────────────────
    merged = merged.rename(columns={'phys_risk_score': 'phys_score'})

    return merged


# ─── METRICS HELPER ───────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    scores: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute precision, recall, F1 at a given threshold."""
    y_pred = (scores >= threshold).astype(int)
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall'   : recall_score(y_true, y_pred, zero_division=0),
        'f1'       : f1_score(y_true, y_pred, zero_division=0),
    }


# ─── DEMO / SMOKE TEST ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Ensure predict.py can be imported from repo root
    sys.path.insert(0, REPO_ROOT)
    from predict import predict_record

    DEMO_RECORD = "100"
    THRESHOLD   = 0.5

    print("=" * 60)
    print("  FUSION.PY — Hybrid Fusion Demo")
    print(f"  Record: {DEMO_RECORD}")
    print("=" * 60)

    # ── Step 1: Load physiological scores ──────────────────────────────────
    print(f"\n[1/5] Loading master_results.csv ...")
    phys_df = pd.read_csv(MASTER_CSV)
    print(f"      Loaded {len(phys_df):,} rows across {phys_df['record_id'].nunique()} records.")

    # ── Step 2: LSTM predictions ────────────────────────────────────────────
    print(f"\n[2/5] Running LSTM predict_record('{DEMO_RECORD}') ...")
    lstm_result = predict_record(DEMO_RECORD, threshold=0.3)
    print(f"      LSTM returned {len(lstm_result['beat_probs'])} beats.")

    # ── Step 3: Align ───────────────────────────────────────────────────────
    print(f"\n[3/5] Aligning by rpeak_sample, skipping beat 0 ...")
    merged = align_scores(DEMO_RECORD, phys_df, lstm_result)
    print(f"      Aligned beats: {len(merged)}")

    phys_scores  = merged['phys_score'].values.astype(np.float32)
    lstm_scores  = merged['lstm_score'].values.astype(np.float32)
    true_binary  = merged['true_abnormal'].values.astype(int)

    print(f"      Abnormal beats: {true_binary.sum()} / {len(true_binary)} "
          f"({true_binary.mean()*100:.1f}%)")

    # ── Step 4: Run all fusion methods ──────────────────────────────────────
    print(f"\n[4/5] Computing fusion scores ...")
    alpha = get_alpha(DEMO_RECORD)
    w_scores  = weighted_fusion(phys_scores, lstm_scores, alpha=alpha)
    mx_scores = max_fusion(phys_scores, lstm_scores)

    # Learned fusion needs more data — use record 100 only for demo
    lr_model, lr_probs_test, y_test = train_learned_fusion(
        phys_scores, lstm_scores, true_binary
    )
    # Full-record learned scores for comparison table
    lr_scores_full = apply_learned_fusion(lr_model, phys_scores, lstm_scores)

    # ── Step 5: Print comparison table ─────────────────────────────────────
    print(f"\n[5/5] Comparison Table (threshold={THRESHOLD})")
    print(f"\n  {'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 56)

    methods = {
        "Physio only"      : phys_scores,
        "LSTM only"        : lstm_scores,
        f"Weighted (α={alpha})" : w_scores,
        "Max fusion"       : mx_scores,
        "Learned (LR)"     : lr_scores_full,
    }

    for name, scores in methods.items():
        m = compute_metrics(true_binary, scores, threshold=THRESHOLD)
        marker = " <- BEST" if np.isclose(m['f1'], max(
            compute_metrics(true_binary, s, threshold=THRESHOLD)['f1']
            for s in methods.values()
        )) else ""
        print("  {:<22} {:>10.3f} {:>10.3f} {:>10.3f}{}".format(
            name, m['precision'], m['recall'], m['f1'], marker))

    print(f"\n  Physio baseline F1 (known): 0.353")
    print(f"  LSTM baseline F1 (known) : 0.571 @ threshold=0.3")
    print("\n" + "=" * 60)
    print("  Done. Import fusion.py in pipeline.py for full evaluation.")
    print("=" * 60)
