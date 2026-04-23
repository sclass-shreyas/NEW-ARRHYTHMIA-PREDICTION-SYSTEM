# ML Model Handoff — Person 2 → Person 4

## What you're getting
A single Python file: `predict.py`
One function: `predict_record(record_id)` — call this for any MIT-BIH record.

## How to import and call it

```python
from predict import predict_record

result = predict_record("200")          # pass record ID as string
```

## Output — what comes back

`result` is a dict with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `beat_probs` | np.ndarray (N,) float | Raw LSTM PVC probability per beat, range 0–1 |
| `beat_labels` | np.ndarray (N,) int | 1=PVC, 0=Normal, applied at threshold=0.3 |
| `beat_samples` | np.ndarray (N,) int | R-peak sample index for each beat (use for alignment with Person 3) |
| `risk_score` | float | Fraction of beats flagged as PVC, range 0–1 |
| `summary` | dict | Metadata — see below |

### summary dict keys
```python
{
    "record_id"    : "200",
    "total_beats"  : 2790,
    "pvc_count"    : 1000,
    "normal_count" : 1790,
    "risk_score"   : 0.3584,
    "threshold"    : 0.3,
    "model"        : "lstm",
    "fs_hz"        : 360
}
```

## Threshold
Default is **0.3** (tuned for high recall — missing a PVC is worse than a false alarm).
You can override it: `predict_record("200", threshold=0.5)`

## Model performance (test set, 8 records)
| Threshold | Recall | Precision | F1 | Missed PVCs |
|-----------|--------|-----------|----|-------------|
| 0.5 | 0.829 | 0.491 | 0.616 | 473 |
| 0.3 | 0.857 | 0.428 | 0.571 | 394 |

Recommended: **0.3** for the hybrid system.

## What to feed into your hybrid pipeline

```python
result = predict_record(record_id)

ml_probs      = result["beat_probs"]      # per-beat ML probability → combine with Person 3
ml_labels     = result["beat_labels"]     # per-beat binary flag
beat_samples  = result["beat_samples"]    # use to align with Person 3's beat-level output
ml_risk_score = result["risk_score"]      # record-level risk → combine with Person 3's score
```

## Dependencies
- tensorflow >= 2.11
- numpy, pandas, wfdb, scikit-learn, joblib
- models/lstm_model.keras must be present
- models/rf_model_v2.pkl must be present (only used if use_lstm=False)
- {record_id}_rpeaks.csv files must be present in BASE_DIR

## Files Person 4 needs
```
predict.py
models/lstm_model.keras
models/rf_model_v2.pkl
{record_id}_rpeaks.csv  (one per record)
```