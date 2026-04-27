import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Hybrid Cardiac Digital Twin",
    page_icon="❤️",
    layout="wide"
)

# Google-Material Style CSS
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-family: 'Segoe UI', Roboto, sans-serif; font-weight: 500; color: #202124; }
    p { color: #5f6368; }
    .stMetric { border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # We added compression='zip' so it can read your uploaded zip file
    df = pd.read_csv('results/hybrid_results.zip', compression='zip')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'results/hybrid_results.csv' not found. Please ensure the data folder exists.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("ARRHYTHMIA DETECTION SYSTEM")
    st.subheader("Hybrid Cardiac Digital Twin")
    st.divider()
    
    # Record ID Selection with Pacemaker Flags
    pacemaker_ids = [102, 107, 109, 111, 212]
    unique_ids = sorted(df['record_id'].unique())
    
    def format_id(rid):
        return f"Patient {rid} (Pacemaker)" if rid in pacemaker_ids else f"Patient {rid}"
    
    selected_id = st.selectbox("Select Patient Record ID", unique_ids, format_func=format_id)
    
    # Filter data for selected patient
    p_df = df[df['record_id'] == selected_id].reset_index(drop=True)
    
    # Toggle switch
    show_raw = st.toggle("Show Raw LSTM/Physio Models", value=True)
    
    st.divider()
    st.markdown("### Final Global F1 Scores")
    st.write("• **Physio:** 37.7%")
    st.write("• **LSTM:** 54.3%")
    st.write("• **Hybrid:** 55.4%")

# --- TOP METRICS ROW ---
total_beats = len(p_df)
abnormal_count = len(p_df[p_df['final_label'] == 'abnormal'])
normal_count = total_beats - abnormal_count
avg_risk = p_df['hybrid_score'].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Beats", total_beats)
m2.metric("Abnormal Beats", f"{abnormal_count} ({abnormal_count/total_beats:.1%})")
m3.metric("Normal Beats", f"{normal_count} ({normal_count/total_beats:.1%})")
m4.metric("Avg Hybrid Risk", f"{avg_risk:.2f}", 
          delta_color="normal" if avg_risk <= 0.5 else "inverse")

# --- MAIN TIMELINE ---
with st.container(border=True):
    st.subheader("Hybrid Risk Score Timeline")
    fig_main = go.Figure()
    
    # Primary Hybrid Score
    fig_main.add_trace(go.Scatter(
        x=p_df['beat_index'], y=p_df['hybrid_score'],
        mode='lines', name='Hybrid Score',
        fill='tozeroy', line=dict(color='#0f9d58', width=2)
    ))
    
    # Optional Raw Models
    if show_raw:
        fig_main.add_trace(go.Scatter(
            x=p_df['beat_index'], y=p_df['lstm_score'],
            name='LSTM Score', line=dict(color='#4285f4', dash='dot')
        ))
        fig_main.add_trace(go.Scatter(
            x=p_df['beat_index'], y=p_df['phys_score'],
            name='Physio Score', line=dict(color='#f4b400', dash='dot')
        ))
    
    # Threshold Line
    fig_main.add_hline(y=0.5, line_dash="dash", line_color="#db4437", annotation_text="Risk Threshold")
    
    fig_main.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_main, use_container_width=True)

# --- ANALYTICS GRID (2x2) ---
row1_col1, row1_col2 = st.columns(2)

# Row 1, Left: Arrhythmia Breakdown
with row1_col1:
    with st.container(border=True):
        st.write("**Arrhythmia Breakdown**")
        breakdown = p_df['arrhythmia_type'].value_counts().reset_index()
        fig_bar = px.bar(breakdown, x='count', y='arrhythmia_type', orientation='h',
                         color_discrete_sequence=['#4285f4'])
        fig_bar.update_layout(height=320, margin=dict(l=0, r=0, t=0, b=0), template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

# Row 1, Right: Risk Gauge
with row1_col2:
    with st.container(border=True):
        st.write("**Instantaneous Risk Gauge**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_risk * 100,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#202124"},
                'steps': [
                    {'range': [0, 35], 'color': "#e6f4ea"},
                    {'range': [35, 50], 'color': "#fef7e0"},
                    {'range': [50, 100], 'color': "#fce8e6"}
                ]
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

# Row 2, Left: Beat Status
with row2_col1:
    with st.container(border=True):
        st.write("**Beat Classification Ratio**")
        fig_pie = px.pie(p_df, names='final_label', hole=0.6,
                         color='final_label',
                         color_discrete_map={'normal': '#0f9d58', 'abnormal': '#db4437'})
        fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

# Row 2, Right: Global Performance Table
with row2_col2:
    with st.container(border=True):
        st.write("**Model Performance Comparison**")
        perf_data = {
            "Model": ["Physio Baseline", "LSTM RNN", "Hybrid LR", "Hybrid Weighted"],
            "F1 Score": ["37.7%", "54.3%", "54.9%", "55.4%"]
        }
        perf_df = pd.DataFrame(perf_data)
        
        def highlight_hybrid(s):
            return ['background-color: #e6f4ea' if v == "55.4%" else '' for v in s]
        
        st.table(perf_df.style.apply(highlight_hybrid, subset=['F1 Score']))

# --- DATA TABLE ---
with st.container(border=True):
    st.subheader("Beat-by-Beat Analysis")
    filter_choice = st.radio("Filter Status:", ["All", "Abnormal", "Normal"], horizontal=True)
    
    if filter_choice == "Abnormal":
        display_df = p_df[p_df['final_label'] == 'abnormal']
    elif filter_choice == "Normal":
        display_df = p_df[p_df['final_label'] == 'normal']
    else:
        display_df = p_df

    def highlight_abnormal(row):
        return ['background-color: #fce8e6' if row.final_label == 'abnormal' else '' for _ in row]

    st.dataframe(
        display_df.style.apply(highlight_abnormal, axis=1),
        use_container_width=True,
        height=300
    )
