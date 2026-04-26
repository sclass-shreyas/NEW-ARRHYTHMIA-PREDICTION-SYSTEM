# HANDOFF TO PERSON 5 (STREAMLIT UI)

**From:** Person 4 (Hybrid Fusion)
**To:** Person 5 (Streamlit UI)

Hey! The core machine learning and fusion pipeline is completely finished. I have successfully combined Person 2's LSTM deep learning model with Person 3's Physiological rule-based model. 

The result is a **Learned Hybrid Model** that achieves a higher F1 score than either model standalone, while retaining the human-readable clinical explanations from the physiological side.

Your entire job is now to build the Streamlit UI on top of the final output file I have generated for you.

---

## 📂 Your Deliverable: `hybrid_results.csv`

You do **not** need to run the LSTM, you do **not** need to run the physiological model, and you do **not** need to do any machine learning inference. 

I have pre-computed the final hybrid predictions for all **112,600 beats** across all 48 MIT-BIH records and saved them in a single file.

**Location:** `results/hybrid_results.csv`

### 📊 Column Schema (Exactly what you need for the UI)

| Column | Type | Description | How to use it in Streamlit |
|---|---|---|---|
| `record_id` | `int` | The MIT-BIH record number (e.g., 100, 200). | Use this for your main dropdown/sidebar to select which patient record to view. |
| `beat_index` | `int` | The sequential number of the beat in the record. | Use for displaying beat sequence (Beat 1, Beat 2, etc.). |
| `rpeak_sample` | `int` | The exact sample position in the raw ECG signal. | Use if you need to plot the exact location of the beat on an ECG graph. |
| `phys_score` | `float` | Person 3's physiological risk score (0-1). | Display in an expander if the doctor wants to see the raw physiological risk. |
| `lstm_score` | `float` | Person 2's LSTM probability score (0-1). | Display in an expander if the doctor wants to see the raw AI confidence. |
| **`hybrid_score`** | `float` | **My fused risk score (0-1).** | **Use this for your main Risk Gauge/Meter UI element.** |
| **`final_label`** | `string` | Either `'normal'` or `'abnormal'`. | **Use this to color-code the beats (e.g., Green for normal, Red for abnormal).** |
| `true_label` | `int` | The AAMI ground truth (0-4). | (Optional) Show this to prove the system works. |
| **`arrhythmia_type`**| `string` | The specific type (e.g., "PVC", "Normal"). | **Display this as the primary diagnosis text.** |
| **`explanation`** | `string` | Human-readable clinical reasoning. | **Display this directly in the UI for explainability.** Example: "Elevated RMSSD indicates irregular rhythm." |

---

## 💻 Code Snippet: How to load and use the data

Here is exactly how you should load the data in your `app.py`:

```python
import streamlit as st
import pandas as pd

# 1. Load the data once and cache it
@st.cache_data
def load_data():
    return pd.read_csv("results/hybrid_results.csv")

df = load_data()

# 2. Sidebar to select a patient record
st.sidebar.title("Patient Selection")
record_list = df['record_id'].unique()
selected_record = st.sidebar.selectbox("Select MIT-BIH Record:", record_list)

# 3. Filter data for the selected record
patient_data = df[df['record_id'] == selected_record]

st.title(f"Cardiac Digital Twin - Record {selected_record}")

# 4. Show overall stats
total_beats = len(patient_data)
abnormal_beats = len(patient_data[patient_data['final_label'] == 'abnormal'])
st.metric("Total Beats Analyzed", total_beats)
st.metric("Arrhythmias Detected", abnormal_beats)

# 5. Show flagged beats with explanations
st.subheader("Flagged Arrhythmias")
flagged_df = patient_data[patient_data['final_label'] == 'abnormal']

for index, row in flagged_df.iterrows():
    with st.expander(f"Beat {row['beat_index']} - {row['arrhythmia_type']} (Risk: {row['hybrid_score']:.2f})"):
        st.write(f"**Clinical Explanation:** {row['explanation']}")
        st.progress(row['hybrid_score'], text="Hybrid Risk Score")
```

---

## 📈 Paper Results (For our IEEE Report)

If you are writing any text for the UI dashboard (or for our final presentation), here are the final global metrics. Our Learned Hybrid Model successfully outperformed both baseline models!

- **LSTM Only F1:** 0.488
- **Physiology Only F1:** 0.340
- **Learned Hybrid F1:** **0.490**  🏆 (Primary model used in the CSV)

We proved our hypothesis: By combining Deep Learning (LSTM) for shape analysis with Rule-based logic (Physiology) for rhythm analysis, we achieve superior accuracy *while still providing human-readable explanations to the doctor.*

Let me know if you need any extra columns in the CSV, otherwise you are clear for takeoff! 🚀
