"""
pipeline.py — End-to-End Hybrid Pipeline (Person 4)
=====================================================
Given a record_id, runs the full hybrid fusion and returns a structured
DataFrame ready for Person 5's Streamlit UI.

Usage (standalone):
    python pipeline.py              # runs on record "100" as demo

Usage (import):
    from pipeline import run_pipeline
    df = run_pipeline("100")
    # df has all columns Person 5 needs
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

# ── Make imports work from any working directory ───────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from predict import predict_record
from fusion import (
    align_scores,
    weighted_fusion,
    max_fusion,
    apply_learned_fusion,
    get_alpha,
    MASTER_CSV,
    PACED_RECORDS,
)

# ── Lazy-load the master CSV once and cache it ─────────────────────────────────
_phys_df_cache: pd.DataFrame | None = None

def _get_phys_df() -> pd.DataFrame:
    global _phys_df_cache
    if _phys_df_cache is None:
        _phys_df_cache = pd.read_csv(MASTER_CSV)
    return _phys_df_cache


# ── Main pipeline function ─────────────────────────────────────────────────────

def run_pipeline(
    record_id: str,
    fusion_model=None,
    alpha: Optional[float] = None,
    threshold: float = 0.5,
    lstm_threshold: float = 0.3,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Full end-to-end hybrid arrhythmia prediction for one MIT-BIH record.

    Parameters
    ----------
    record_id     : str   — e.g. "100"
    fusion_model  : trained LogisticRegression or None
                    If None → uses weighted fusion (or max for paced records)
                    If provided → uses learned logistic fusion
    alpha         : float or None
                    LSTM weight for weighted fusion.
                    If None → auto-selected by get_alpha() (0.85 paced, 0.5 normal)
    threshold     : float — decision boundary for hybrid_score → final_label (default 0.5)
    lstm_threshold: float — threshold passed to predict_record (default 0.3)
    verbose       : bool  — print progress messages

    Returns
    -------
    pd.DataFrame with columns:
        record_id, beat_index, rpeak_sample,
        phys_score, lstm_score, hybrid_score,
        final_label,          ← 'normal' or 'abnormal'
        true_label,           ← AAMI 0-4
        true_abnormal,        ← int 0/1
        arrhythmia_type,      ← string e.g. 'PVC', 'Normal'
        explanation           ← human-readable string for Streamlit
    """
    record_id = str(record_id)

    if verbose:
        print(f"  [{record_id}] Loading physiological scores ...")

    phys_df = _get_phys_df()

    # ── LSTM inference ─────────────────────────────────────────────────────────
    if verbose:
        print(f"  [{record_id}] Running LSTM ...")
    try:
        lstm_result = predict_record(record_id, threshold=lstm_threshold)
    except Exception as e:
        print(f"  [{record_id}] ERROR in predict_record: {e}")
        return pd.DataFrame()

    # ── Align physio + LSTM on R-peak sample (skips beat 0 automatically) ─────
    if verbose:
        print(f"  [{record_id}] Aligning scores ...")
    merged = align_scores(record_id, phys_df, lstm_result)

    if merged.empty:
        print(f"  [{record_id}] WARNING: alignment produced empty DataFrame — skipping.")
        return pd.DataFrame()

    phys_scores = merged['phys_score'].values.astype(np.float32)
    lstm_scores = merged['lstm_score'].values.astype(np.float32)

    # ── Determine alpha (paced records get LSTM-heavy weight) ─────────────────
    if alpha is None:
        alpha = get_alpha(record_id)

    # ── Apply fusion ───────────────────────────────────────────────────────────
    if fusion_model is not None:
        hybrid_scores = apply_learned_fusion(fusion_model, phys_scores, lstm_scores)
        fusion_method = "learned_lr"
    else:
        hybrid_scores = weighted_fusion(phys_scores, lstm_scores, alpha=alpha)
        fusion_method = f"weighted_alpha{alpha}"

    # ── Binary label ───────────────────────────────────────────────────────────
    final_label = np.where(hybrid_scores >= threshold, 'abnormal', 'normal')

    # ── Build output DataFrame ─────────────────────────────────────────────────
    out = pd.DataFrame({
        'record_id'      : record_id,
        'beat_index'     : merged['beat_index'].values,
        'rpeak_sample'   : merged['rpeak_sample'].values,
        'phys_score'     : np.round(phys_scores, 6),
        'lstm_score'     : np.round(lstm_scores, 6),
        'hybrid_score'   : np.round(hybrid_scores, 6),
        'final_label'    : final_label,
        'true_label'     : merged['true_label'].values,
        'true_abnormal'  : merged['true_abnormal'].values,
        'arrhythmia_type': merged['arrhythmia_type'].values,
        'explanation'    : merged['explanation'].values,
        'fusion_method'  : fusion_method,
        'alpha'          : alpha,
        'threshold'      : threshold,
    })

    if verbose:
        n_abn = (out['final_label'] == 'abnormal').sum()
        print(f"  [{record_id}] Done: {len(out)} beats, {n_abn} flagged abnormal "
              f"({n_abn/len(out)*100:.1f}%), fusion={fusion_method}")

    return out


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEMO_RECORD = "100"
    print("=" * 60)
    print("  PIPELINE.PY — Standalone Demo")
    print(f"  Record: {DEMO_RECORD}")
    print("=" * 60)

    df = run_pipeline(DEMO_RECORD, verbose=True)

    if df.empty:
        print("Pipeline returned empty DataFrame — check paths and data.")
        sys.exit(1)

    print(f"\n  Output shape : {df.shape}")
    print(f"  Columns      : {df.columns.tolist()}")
    print(f"\n  Sample rows:")
    print(df[['beat_index', 'rpeak_sample', 'phys_score', 'lstm_score',
               'hybrid_score', 'final_label', 'true_label',
               'arrhythmia_type']].head(10).to_string(index=False))

    print(f"\n  Label distribution:")
    print(df['final_label'].value_counts().to_string())

    print(f"\n  True AAMI distribution:")
    print(df['true_label'].value_counts().sort_index().to_string())

    # Quick accuracy check
    from sklearn.metrics import classification_report
    y_true = df['true_abnormal'].values
    y_pred = (df['final_label'] == 'abnormal').astype(int).values
    print(f"\n  Classification Report (hybrid vs ground truth):")
    print(classification_report(y_true, y_pred,
                                target_names=['Normal', 'Abnormal'],
                                digits=3))

    print("=" * 60)
    print("  Pipeline OK. Run evaluate.py for all 48 records.")
    print("=" * 60)
