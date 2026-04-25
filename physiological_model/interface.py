"""
PERSON 3 — Physiological Model
Day 4: Public Interface

File: physiological_model/interface.py

THIS IS THE FILE PERSON 4 IMPORTS.

Two public functions:

    1. physiological_analysis(rr_intervals, beat_index, window=10)
       → Analyzes a single beat. Returns full result dict.

    2. analyze_full_record(rr_intervals, n_beats)
       → Analyzes every beat in a record.
       → Returns list of dicts aligned with Person 2's (n_beats,) output.
       → Also returns np.ndarray of risk scores for direct fusion.

Integration contract with Person 4:
    - Input : rr_intervals as np.ndarray (seconds), n_beats (int)
    - Output: risk_scores array shape (n_beats,) float32
              results_list — full metadata per beat
    - Beat 0 is always skipped (+ marker), returns risk=0.0
    - All arrays are zero-indexed and aligned to R-peak indices

Usage (Person 4):
    from physiological_model.interface import analyze_full_record
    risk_scores, results = analyze_full_record(rr_intervals, n_beats)
    # risk_scores shape: (n_beats,) — fuse with LSTM output directly
"""

import numpy as np
import os
import sys

# Allow running as standalone script from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hrv_features import compute_hrv_features
from arrhythmia_classifier import classify_beat


# ─────────────────────────────────────────────────────────────
# DEFAULT RESULT — returned for beats that can't be analyzed
# (beat 0 / + marker, or windows with too few valid RR values)
# ─────────────────────────────────────────────────────────────

def _default_result(beat_index: int, reason: str = "insufficient data") -> dict:
    return {
        'beat_index'      : beat_index,
        'phys_risk_score' : 0.0,
        'classification'  : 'unknown',
        'arrhythmia_type' : 'Insufficient Data',
        'flags'           : {},
        'explanation'     : f'Physiological analysis skipped: {reason}',
        'risk_breakdown'  : {},
        'features'        : None,
        'window_size'     : 0,
    }


# ─────────────────────────────────────────────────────────────
# FUNCTION 1 — Single beat analysis
# ─────────────────────────────────────────────────────────────

def physiological_analysis(rr_intervals: np.ndarray,
                            beat_index: int,
                            window: int = 10) -> dict:
    """
    Analyzes the physiological state around a single beat.

    Parameters
    ----------
    rr_intervals : np.ndarray
        Full RR interval array for the record, in seconds.
        Shape (N-1,) where N = number of R-peaks.
        From Person 1's CSV: np.diff(rpeaks) / 360.0

    beat_index : int
        Index of the beat to analyze (0-indexed, matches R-peak index).
        Beat 0 is always skipped (record-start + marker).

    window : int
        Number of surrounding RR intervals to include on each side.
        Default 10 → uses up to 20 RR intervals total.
        Minimum 2 → must allow at least 4 valid beats for time-domain.

    Returns
    -------
    dict with keys:
        beat_index       : int
        phys_risk_score  : float (0.0 – 1.0)
        classification   : 'normal' | 'abnormal' | 'unknown'
        arrhythmia_type  : str
        flags            : dict of bool
        explanation      : str
        risk_breakdown   : dict
        features         : dict (raw HRV features, for Person 5 display)
        window_size      : int (actual number of RR intervals used)
    """

    # ── Guard: always skip beat 0 (record-start marker) ─────────────────
    if beat_index == 0:
        return _default_result(beat_index, reason="record-start marker (+), not a real beat")

    # ── Guard: beat_index must be within valid range ─────────────────────
    if beat_index < 0 or beat_index >= len(rr_intervals) + 1:
        return _default_result(beat_index, reason=f"beat_index {beat_index} out of range")

    # ── Slice local RR window around beat ───────────────────────────────
    # rr_intervals[i] = interval BETWEEN beat i and beat i+1
    # For beat at index i, we want RR intervals surrounding it:
    # from max(0, i-window) to min(len(rr), i+window)
    start      = max(0, beat_index - window)
    end        = min(len(rr_intervals), beat_index + window)
    rr_window  = rr_intervals[start:end]

    # ── Compute HRV features ─────────────────────────────────────────────
    features = compute_hrv_features(rr_window)

    if features is None:
        return _default_result(beat_index, reason="window too short for HRV analysis")

    # ── Classify ─────────────────────────────────────────────────────────
    result = classify_beat(features)

    # ── Attach metadata and return ───────────────────────────────────────
    result['beat_index']  = beat_index
    result['features']    = features
    result['window_size'] = len(rr_window)

    return result


# ─────────────────────────────────────────────────────────────
# FUNCTION 2 — Full record analysis (what Person 4 uses)
# ─────────────────────────────────────────────────────────────

def analyze_full_record(rr_intervals: np.ndarray,
                        n_beats: int,
                        window: int = 10) -> tuple[np.ndarray, list]:
    """
    Runs physiological analysis on every beat in a record.

    Parameters
    ----------
    rr_intervals : np.ndarray
        Full RR interval array in seconds. Shape (n_beats - 1,).

    n_beats : int
        Total number of beats (= number of R-peaks).

    window : int
        Local RR window size per beat. Default 10.

    Returns
    -------
    risk_scores : np.ndarray
        Float32 array of physiological risk scores, shape (n_beats,).
        Beat 0 = 0.0 (skipped). Directly fusable with Person 2's output.

    results_list : list[dict]
        Full result dict per beat. Length = n_beats.
        Each dict contains all classification metadata.

    Example (Person 4 usage):
        risk_scores, results = analyze_full_record(rr_intervals, n_beats)
        # fuse:
        hybrid_score = 0.5 * risk_scores + 0.5 * lstm_scores
    """

    risk_scores  = np.zeros(n_beats, dtype=np.float32)
    results_list = []

    for beat_idx in range(n_beats):
        result = physiological_analysis(rr_intervals, beat_idx, window=window)
        risk_scores[beat_idx] = result['phys_risk_score']
        results_list.append(result)

    return risk_scores, results_list


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  INTERFACE — Standalone Test")
    print("="*60)

    # ── Load real record 100 ─────────────────────────────────────────────
    npy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "100_verified.npy"
    )

    if not os.path.exists(npy_path):
        print("\n  ✗ 100_verified.npy not found.")
        print("    Run day1_load_and_verify.py first.\n")
        sys.exit(1)

    data         = np.load(npy_path, allow_pickle=True).item()
    rr_intervals = data['rr_intervals']
    beat_labels  = data['beat_labels']
    n_beats      = data['n_beats']

    print(f"\n  Record 100 loaded: {n_beats} beats, {len(rr_intervals)} RR intervals")

    # ── Test 1: Single beat analysis ────────────────────────────────────
    print("\n  --- Test 1: Single Beat Analysis ---")
    for test_beat in [0, 1, 8, 50, 100]:
        r = physiological_analysis(rr_intervals, test_beat)
        print(f"  Beat {test_beat:>4} | label={beat_labels[test_beat]} "
              f"| risk={r['phys_risk_score']:.4f} "
              f"| class={r['classification']:<8} "
              f"| type={r['arrhythmia_type']}")

    # ── Test 2: Full record analysis ────────────────────────────────────
    print("\n  --- Test 2: Full Record Analysis ---")
    risk_scores, results = analyze_full_record(rr_intervals, n_beats)

    print(f"  risk_scores shape : {risk_scores.shape}  (must be ({n_beats},))")
    print(f"  results_list len  : {len(results)}")
    print(f"  risk_scores dtype : {risk_scores.dtype}")
    print(f"  Beat 0 risk       : {risk_scores[0]:.4f}  (must be 0.0 — skipped marker)")
    print(f"  Min risk          : {np.min(risk_scores):.4f}")
    print(f"  Max risk          : {np.max(risk_scores):.4f}")
    print(f"  Mean risk         : {np.mean(risk_scores):.4f}")

    # ── Test 3: Risk score separation by AAMI class ──────────────────────
    print("\n  --- Test 3: Risk Score by AAMI Class ---")
    label_names = {0: 'Normal (N)', 1: 'Supraventricular (S)',
                   2: 'Ventricular/PVC (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'}

    for cls in sorted(np.unique(beat_labels)):
        mask  = beat_labels == cls
        mean  = np.mean(risk_scores[mask])
        std   = np.std(risk_scores[mask])
        count = np.sum(mask)
        bar   = "█" * int(mean * 30)
        print(f"  Class {cls} {label_names[cls]:<28} n={count:>5} "
              f"| mean={mean:.4f} ± {std:.4f}  {bar}")

    # ── Test 4: Output format check for Person 4 ────────────────────────
    print("\n  --- Test 4: Person 4 Integration Format Check ---")
    sample = results[8]   # beat 8 is an 'A' (Supraventricular) beat
    print(f"  Sample result dict keys : {list(sample.keys())}")
    print(f"  phys_risk_score  : {sample['phys_risk_score']}")
    print(f"  classification   : {sample['classification']}")
    print(f"  arrhythmia_type  : {sample['arrhythmia_type']}")
    print(f"  explanation      : {sample['explanation'][:80]}...")
    print(f"  features present : {sample['features'] is not None}")
    print(f"  window_size      : {sample['window_size']}")

    # ── Test 5: Simulate Person 4 fusion ────────────────────────────────
    print("\n  --- Test 5: Simulated Hybrid Fusion (Person 4 preview) ---")
    # Simulate what Person 2's LSTM might output
    np.random.seed(42)
    lstm_scores = np.random.beta(0.5, 5.0, size=n_beats).astype(np.float32)
    lstm_scores[beat_labels == 2] = 0.85   # PVC beats get high LSTM score

    # Simple equal-weight fusion (Person 4 will refine this)
    hybrid_scores = 0.5 * risk_scores + 0.5 * lstm_scores

    print(f"  Fusion method    : equal weight (0.5 phys + 0.5 lstm)")
    print(f"  Hybrid shape     : {hybrid_scores.shape}")
    for cls in sorted(np.unique(beat_labels)):
        mask = beat_labels == cls
        print(f"  Class {cls} {label_names[cls]:<28} "
              f"hybrid mean = {np.mean(hybrid_scores[mask]):.4f}")

    print("\n" + "="*60)
    print("  ✓ Interface tests complete. Ready for Person 4.")
    print("="*60 + "\n")
