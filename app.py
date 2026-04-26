"""
app.py — Hybrid Cardiac Digital Twin Dashboard (Person 5)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Cardiac Digital Twin",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Very minimal CSS just for typography and removing top padding
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        /* max-width removed to allow true wide layout */
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-weight: 500;
        color: #202124;
    }
    p {
        color: #5f6368;
    }
</style>
""", unsafe_allow_html=True)

PACED = {"102", "107", "109", "111", "212"}

@st.cache_data
def load_hybrid():
    df = pd.read_csv("results/hybrid_results.csv")
    df["record_id"] = df["record_id"].astype(str)
    return df

@st.cache_data
def load_eval():
    df = pd.read_csv("results/evaluate_results.csv")
    df["record_id"] = df["record_id"].astype(str)
    return df

try:
    df_all  = load_hybrid()
    df_eval = load_eval()
except FileNotFoundError as e:
    st.error(f"Missing results file: {e}. Please run evaluate.py first.")
    st.stop()

all_records = sorted(df_all["record_id"].unique(), key=int)

# ── SIDEBAR ──
with st.sidebar:
    st.title("🫀 Digital Twin")
    st.caption("Arrhythmia Prediction System")
    st.divider()
    
    selected = st.selectbox("Select Patient Record", all_records, 
        format_func=lambda x: f"Record {x}" + (" (Pacemaker)" if x in PACED else ""))
    
    # We will display all beats by default
    show_raw = st.toggle("Show LSTM/Physio Models", value=True)
    
    st.divider()
    st.subheader("Global F1 Scores")
    st.markdown("**Physiological**: 37.7%")
    st.markdown("**LSTM Only**: 54.3%")
    st.markdown("**:green[Hybrid Fusion: 55.4%]**")

# ── DATA PROCESSING ──
df  = df_all[df_all["record_id"] == selected].reset_index(drop=True)
ev  = df_eval[df_eval["record_id"] == selected]

total   = len(df)
n_abn   = int((df["final_label"] == "abnormal").sum())
n_nrm   = total - n_abn
pct_abn = n_abn / total * 100
avg_hyb = float(df["hybrid_score"].mean())
is_paced = selected in PACED

if avg_hyb > 0.5:
    risk_lvl, risk_color = "HIGH RISK", "red"
elif avg_hyb > 0.35:
    risk_lvl, risk_color = "MODERATE RISK", "orange"
else:
    risk_lvl, risk_color = "LOW RISK", "green"

# ── HEADER ──
st.header(f"Patient Record {selected}")
if is_paced:
    st.warning("🔋 **Pacemaker Detected** - Note: Pacemaker records are excluded from final evaluation.")
else:
    st.caption(f"MIT-BIH Arrhythmia Database • {total:,} beats analysed")

st.divider()

# ── KPIS ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Beats", f"{total:,}")
col2.metric("Abnormal Beats", f"{n_abn:,}", f"{pct_abn:.1f}%")
col3.metric("Normal Beats", f"{n_nrm:,}", f"{100-pct_abn:.1f}%")
col4.metric("Avg Hybrid Score", f"{avg_hyb:.3f}", risk_lvl, delta_color="off" if risk_color=="green" else "inverse")

st.divider()

# ── TIMELINE ──
with st.container(border=True):
    st.subheader("Risk Score Timeline")
    dfp  = df
    bidx = dfp["beat_index"].values
    
    fig = go.Figure()
    
    if show_raw:
        fig.add_trace(go.Scatter(x=bidx, y=dfp["lstm_score"], name="LSTM", mode="lines", line=dict(color="#4285f4", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=bidx, y=dfp["phys_score"], name="Physio", mode="lines", line=dict(color="#9aa0a6", width=1, dash="dot")))
        
    fig.add_trace(go.Scatter(x=bidx, y=dfp["hybrid_score"], name="Hybrid Score", mode="lines", 
                             line=dict(color="#0f9d58", width=2), fill="tozeroy", fillcolor="rgba(15,157,88,0.1)"))
    
    abn = dfp["final_label"] == "abnormal"
    if abn.sum():
        fig.add_trace(go.Scatter(x=bidx[abn], y=dfp["hybrid_score"][abn], mode="markers",
            name="Abnormal", marker=dict(color="#db4437", size=6, line=dict(color="#c5221f", width=1))))
            
    fig.add_hline(y=0.5, line_dash="dash", line_color="#db4437", annotation_text="Threshold (0.5)")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white",
                      legend=dict(orientation="h", y=-0.2, x=0), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 2: ARRHYTHMIAS & OVERALL RISK ──
col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        st.subheader("Arrhythmias")
        tc = df[df["final_label"] == "abnormal"]["arrhythmia_type"].value_counts().reset_index()
        tc.columns = ["Type", "Count"]
        if not tc.empty:
            fig_bar = px.bar(tc, x="Count", y="Type", orientation="h", template="plotly_white",
                             color_discrete_sequence=["#4285f4"])
            fig_bar.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.success("No arrhythmias detected.")

with col_b:
    with st.container(border=True):
        st.subheader("Overall Risk")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(avg_hyb * 100, 1),
            number={"suffix": "%", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#db4437" if avg_hyb > 0.5 else "#f4b400" if avg_hyb > 0.35 else "#0f9d58"},
                "steps": [
                    {"range": [0, 35], "color": "#e6f4ea"},
                    {"range": [35, 50], "color": "#fef7e0"},
                    {"range": [50, 100],"color": "#fce8e6"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.75, "value": 50},
            }
        ))
        fig_g.update_layout(height=320, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig_g, use_container_width=True)

# ── ROW 3: BEAT STATUS & GLOBAL PERFORMANCE ──
col_c, col_d = st.columns(2)

with col_c:
    with st.container(border=True):
        st.subheader("Beat Status")
        fig_d = go.Figure(go.Pie(
            labels=["Normal", "Abnormal"], values=[n_nrm, n_abn], hole=0.6,
            marker=dict(colors=["#0f9d58", "#db4437"])
        ))
        fig_d.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), showlegend=True, 
                            legend=dict(orientation="h", y=-0.2, x=0.1))
        st.plotly_chart(fig_d, use_container_width=True)

# ── DATA TABLES ──
with st.container(border=True):
    st.subheader("Clinical Beat Detail")
    filt = st.radio("Filter", ["All", "Abnormal", "Normal"], horizontal=True, label_visibility="collapsed")
    
    df_t = df.copy()
    if filt == "Abnormal": df_t = df_t[df_t["final_label"] == "abnormal"]
    elif filt == "Normal": df_t = df_t[df_t["final_label"] == "normal"]
        
    show_cols = ["beat_index", "lstm_score", "phys_score", "hybrid_score", "final_label", "arrhythmia_type", "explanation"]
    
    def highlight_abnormal(row):
        return ['background-color: #fce8e6' if row['final_label'] == 'abnormal' else '' for _ in row]
        
    st.caption(f"Showing all available beats based on your filter selection.")
    st.dataframe(df_t[show_cols].style.apply(highlight_abnormal, axis=1), use_container_width=True, hide_index=True)

with col_d:
    with st.container(border=True):
        st.subheader("Global Performance")
    perf = pd.DataFrame({
        "Model": ["Physiological Only", "LSTM Only", "Hybrid Learned (LR)", "Hybrid Weighted ★"],
        "Fusion": ["Clinical HRV Rules", "Deep Learning", "LR + CV", "Scaled Weighted"],
        "F1 Score": ["37.7%", "54.3%", "53.6%", "55.4%"]
    })
    
    def highlight_best(row):
        return ['background-color: #e6f4ea; font-weight: bold' if '★' in row['Model'] else '' for _ in row]
        
    st.dataframe(perf.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)
