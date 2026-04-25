# Person 4 — Integration & Fusion Layer
## Complete Handoff Document from Person 3

> This document contains everything you need to take over and build the hybrid fusion layer.
> Read it fully before writing a single line of code.

---

## 1. What You Are Building

Your job is to **combine two scores** into one final hybrid prediction per beat:

- **Person 3's physiological risk score** — based on heart rhythm timing rules
- **Person 2's LSTM probability score** — based on ECG waveform shape

Neither score alone is sufficient. The physiological model is high-recall but low-precision (catches most arrhythmias but raises false alarms). The LSTM is morphology-aware but a black box. Together they are stronger than either alone.

**Your fusion layer is the core innovation of this project.**

---

## 2. Project Context (Quick Summary)

| Person | Role | Status |
|--------|------|--------|
| Person 1 | Data & Signal Processing | ✅ Complete |
| Person 2 | LSTM Model | ✅ Complete |
| Person 3 | Physiological Model | ✅ Complete |
| **Person 4** | **Hybrid Fusion (you)** | 🔄 Your turn |
| Person 5 | Streamlit UI | Waiting on you |

**Dataset:** MIT-BIH Arrhythmia Database, 48 records, 112,600 beats total  
**Sampling frequency:** 360 Hz  
**AAMI Labels:** 0 = Normal, 1 = Supraventricular, 2 = Ventricular/PVC, 3 = Fusion, 4 = Unknown

---

## 3. Files Person 3 Hands You

### Module files (for live inference)
```
physiological_model/
├── interface.py             ← THE file you import and call
├── hrv_features.py          ← imported automatically by interface.py
├── arrhythmia_classifier.py ← imported automatically by interface.py
└── data/
    └── 100_verified.npy     ← verified data dict for record 100
```

### Result files (pre-computed, ready to use)
```
physiological_model/results/
├── master_results.csv       ← 112,600 rows, all 48 records — USE THIS
├── per_record_metrics.csv   ← per-record F1, precision, recall
└── validation_report.txt    ← full validation summary
```

---

## 4. How to Call Person 3's Code

### Option A — Live inference (recommended for real-time fusion)

```python
import numpy as np
from physiological_model.interface import analyze_full_record

# Load Person 1's verified data
record       = np.load("physiological_model/data/100_verified.npy", allow_pickle=True).item()
rr_intervals = record['rr_intervals']   # shape (N-1,) float64, in seconds
n_beats      = record['n_beats']        # int

# Call Person 3's model
phys_scores, phys_results = analyze_full_record(rr_intervals, n_beats)

# phys_scores  → np.ndarray float32, shape (n_beats,)
# phys_results → list of dicts, one per beat, with full metadata

print(phys_scores.shape)   # e.g. (2274,)
print(phys_scores[1:5])    # sample values
```

### Option B — Load from CSV (fastest, no recomputation)

```python
import pandas as pd

df = pd.read_csv("physiological_model/results/master_results.csv")

# Filter to one record
rec100 = df[df['record_id'] == '100']

# The column you need for fusion
phys_scores = rec100['phys_risk_score'].values   # shape (n_beats,)

# Other useful columns available:
# rec100['arrhythmia_type']     → string label per beat
# rec100['explanation']         → human-readable reason (pass to Person 5)
# rec100['true_label']          → AAMI ground truth 0–4
# rec100['predicted_abnormal']  → Person 3's binary prediction (0 or 1)
# rec100['rpeak_sample']        → R-peak location in original signal
```

---

## 5. What Person 2 Gives You

```python
# Person 2's LSTM output format
lstm_scores   # np.ndarray float32, shape (n_beats,)
              # sigmoid output 0.0–1.0
              # higher = more likely PVC / arrhythmia
              # per-beat, window = 360 samples centred on R-peak
```

---

## 6. ⚠️ Critical Warnings — Read Before Coding

### Warning 1 — Array shape must match before fusion
```python
# ALWAYS check this before any fusion operation
assert lstm_scores.shape == phys_scores.shape, \
    f"Shape mismatch: lstm={lstm_scores.shape}, phys={phys_scores.shape}"
```

### Warning 2 — Beat 0 is ALWAYS 0.0, skip it
Beat index 0 in every record is a record-start annotation marker (`+`), not a real heartbeat.
Person 3's model always returns `phys_risk_score = 0.0` for beat 0.
If you don't account for this, your array alignment will be off by one.

```python
# When computing metrics, always skip beat 0
phys_valid = phys_scores[1:]
lstm_valid = lstm_scores[1:]
labels_valid = true_labels[1:]
```

### Warning 3 — Paced rhythm records have near-zero physiological scores
These records have mechanically paced rhythms — HRV is artificially suppressed.
Person 3's model scores will be near zero for ALL beats in these records.
**Rely more heavily on the LSTM score for these records.**

| Record | Issue |
|--------|-------|
| 102 | Paced rhythm |
| 107 | Paced rhythm |
| 109 | Paced rhythm |
| 111 | Paced rhythm |
| 212 | Paced rhythm |

```python
# Option: use record-adaptive weighting
PACED_RECORDS = ['102', '107', '109', '111', '212']

def get_alpha(record_id):
    # alpha = weight given to LSTM (0 = phys only, 1 = lstm only)
    return 0.85 if record_id in PACED_RECORDS else 0.5
```

### Warning 4 — Do NOT use beat 0's score in training the fusion model
If you train a logistic regression fusion model, exclude beat 0 from training data.
It will artificially inflate your accuracy (it is always normal and always scored 0.0).

---

## 7. Key Numbers from Person 3's Validation

| Metric | Value |
|--------|-------|
| Records processed | 48 |
| Total beats | 112,600 |
| Standalone accuracy | 62.28% |
| Standalone precision | 26.52% |
| Standalone recall | 52.70% |
| Standalone F1 | **35.28%** ← your hybrid must beat this |
| Best single record F1 | 84.5% (record 232) |

### Mean risk score by AAMI class (key result)
```
Class 0 — Normal (N)           : 0.2501
Class 1 — Supraventricular (S) : 0.4205   ← correctly higher than normal
Class 2 — Ventricular/PVC (V)  : 0.5214   ← correctly highest
Class 3 — Fusion (F)           : 0.3744
Class 4 — Unknown (Q)          : 0.1793
```
Risk increases monotonically Normal → Supraventricular → Ventricular.
This confirms the physiological model ranks beat danger correctly without training.

---

## 8. Three Fusion Approaches — Implement All Three and Compare

### Approach 1 — Weighted average (start here)
Simple, explainable, works well. Tune `alpha` to control trust balance.

```python
def weighted_fusion(phys_scores, lstm_scores, alpha=0.5):
    """
    alpha = 0.0 → trust physiology only
    alpha = 0.5 → equal weight (default)
    alpha = 1.0 → trust LSTM only
    """
    return alpha * lstm_scores + (1 - alpha) * phys_scores
```

### Approach 2 — Max fusion (conservative, maximizes recall)
Flag a beat if EITHER model is confident. Catches more arrhythmias, more false positives.

```python
def max_fusion(phys_scores, lstm_scores):
    """
    Any beat either model flags above threshold → flagged.
    Best when missing an arrhythmia is worse than a false alarm.
    """
    return np.maximum(phys_scores, lstm_scores)
```

### Approach 3 — Learned logistic regression (research-level, best for marks)
Trains a simple model to find the optimal weight from data.
Print the coefficients — they become a key result in the report.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

def learned_fusion(phys_scores, lstm_scores, true_labels_binary):
    """
    Trains logistic regression on [phys_score, lstm_score] → abnormal probability.
    Returns trained model + predicted probabilities.
    """
    # Stack both scores as 2-feature input
    X = np.column_stack([phys_scores, lstm_scores])  # shape (n_beats, 2)
    y = true_labels_binary                            # 0=normal, 1=abnormal

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Print learned weights — this is a key result
    print(f"Physiology weight : {model.coef_[0][0]:.4f}")
    print(f"LSTM weight       : {model.coef_[0][1]:.4f}")

    hybrid_probs = model.predict_proba(X_test)[:, 1]
    return model, hybrid_probs, y_test
```

---

## 9. Your Deliverables — In Order

### 1. `fusion.py`
All three fusion functions in one file. Takes phys_scores + lstm_scores, returns hybrid_scores.
Include a standalone test that runs all three methods and prints F1 for each.

### 2. `pipeline.py`
The master end-to-end script. For a given record:
1. Load Person 1's rpeaks CSV
2. Compute RR intervals
3. Call Person 3's `analyze_full_record()` → phys_scores
4. Load Person 2's LSTM predictions → lstm_scores
5. Run fusion → hybrid_scores
6. Output final predictions

### 3. `evaluate.py`
Comparison table across all 48 records:
```
Method          Precision   Recall   F1
LSTM only         X.XX       X.XX    X.XX
Physio only       26.52      52.70   35.28
Hybrid (weighted) X.XX       X.XX    X.XX  ← must be highest
Hybrid (learned)  X.XX       X.XX    X.XX
```
This table is your main result for the report and PPT.

### 4. Final output CSV for Person 5
```python
# Required columns
['record_id', 'beat_index', 'rpeak_sample',
 'phys_score', 'lstm_score', 'hybrid_score',
 'final_label',       # 'normal' or 'abnormal'
 'true_label',        # AAMI ground truth 0–4
 'arrhythmia_type',   # string from Person 3
 'explanation']       # string from Person 3 — Person 5 displays this
```

---

## 10. Claude Continuation Prompt

Paste this into a new Claude session to get started immediately:

```
I am Person 4 in a 5-person team building a hybrid cardiac digital twin for
arrhythmia prediction (140 marks semester project). My job is to build the
fusion layer combining Person 2's LSTM output with Person 3's physiological
model output.

PERSON 3 OUTPUT (fully complete):
- Function: from physiological_model.interface import analyze_full_record
- Call: phys_scores, results = analyze_full_record(rr_intervals, n_beats)
- Returns: float32 array shape (n_beats,), values 0–1. Beat 0 = 0.0 always (skip).
- Pre-computed CSV: physiological_model/results/master_results.csv
  Columns: record_id, beat_index, rpeak_sample, true_label, true_symbol,
  true_abnormal, phys_risk_score, predicted_abnormal, classification,
  arrhythmia_type, explanation
- Validated: 48 records, 112,600 beats
- Standalone metrics: Accuracy=62.28%, Precision=26.52%, Recall=52.70%, F1=35.28%
- Risk ordering confirmed: Normal=0.250, Supra=0.421, PVC=0.521

PERSON 2 OUTPUT (LSTM):
- Returns: float32 array shape (n_beats,), sigmoid 0–1, per-beat PVC probability
- Window: 360 samples (1 second) centred on each R-peak, per-beat granularity

PERSON 1 DATA FORMAT:
- CSV: {record_id}_rpeaks.csv, column R_peak_index (integer sample indices)
- Raw files: ecg_data/mit-bih-arrhythmia-database-1.0.0/{id}.dat/.hea/.atr
- Verified npy: physiological_model/data/100_verified.npy
  Keys: signal, fs(360Hz), rpeaks, rr_intervals(seconds,N-1),
  beat_labels(AAMI 0-4), beat_symbols, n_beats

AAMI LABELS: 0=Normal, 1=Supraventricular, 2=Ventricular/PVC, 3=Fusion, 4=Unknown

CRITICAL WARNINGS:
1. Always confirm lstm_scores.shape == phys_scores.shape before fusion
2. Beat 0 is always 0.0 (record-start marker, not a real beat) — skip it
3. Paced records 102,107,109,111,212 have near-zero phys scores — weight LSTM more
4. Exclude beat 0 when training the learned fusion model

MY DELIVERABLES:
1. fusion.py — weighted/max/learned fusion, compare all three F1 scores
2. pipeline.py — end-to-end record processing
3. evaluate.py — LSTM-only vs Phys-only vs Hybrid comparison table
4. Final CSV for Person 5: record_id, beat_index, rpeak_sample, phys_score,
   lstm_score, hybrid_score, final_label, true_label, arrhythmia_type, explanation

TARGET: Hybrid F1 must exceed standalone physio F1 of 35.28%.

Please help me build fusion.py first — production-ready, research-level,
with all three fusion methods and a comparison test.
```

---

## 11. What to Tell Person 5

Once your fusion CSV is ready, tell Person 5:

> "Final results CSV is at `results/hybrid_results.csv`.
> Columns for display: `hybrid_score`, `arrhythmia_type`, `explanation`.
> The explanation column has human-readable text ready to show directly in Streamlit.
> Filter by `record_id` to load one record at a time."

---

*Person 3 handoff complete — all files committed to repo — master_results.csv ready*
