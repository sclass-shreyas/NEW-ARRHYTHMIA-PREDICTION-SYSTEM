import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from sklearn.metrics import recall_score, precision_score, f1_score

BASE_DIR = r"C:\Users\shrey\OneDrive\Documents\ECG_data"
TEST_RECORDS = {"200","201","202","203","205","207","208","209"}

X = np.load(os.path.join(BASE_DIR, "X_windows.npy"))
y = np.load(os.path.join(BASE_DIR, "y_labels.npy"))
meta = np.load(os.path.join(BASE_DIR, "meta_records.npy"))

X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
X_norm = X_norm[:, :, np.newaxis]

test_mask = np.isin(meta, list(TEST_RECORDS))
X_test, y_test = X_norm[test_mask], y[test_mask]

model = keras.models.load_model(os.path.join(BASE_DIR, "models", "lstm_model.keras"))
y_prob = model.predict(X_test, batch_size=256, verbose=0).flatten()

print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>6} {'Missed':>8} {'FalseAlarm':>12}")
for t in [0.5, 0.4, 0.3, 0.2, 0.15]:
    y_pred = (y_prob >= t).astype(int)
    r = recall_score(y_test, y_pred)
    p = precision_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    missed = int((y_test==1).sum()) - int(((y_pred==1)&(y_test==1)).sum())
    fa = int(((y_pred==1)&(y_test==0)).sum())
    print(f"{t:>10.2f} {r:>8.3f} {p:>10.3f} {f:>6.3f} {missed:>8} {fa:>12}")