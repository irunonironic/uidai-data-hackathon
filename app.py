import streamlit as st
import pandas as pd
from PIL import Image
import os
import subprocess

st.set_page_config(layout="wide", page_title="UIDAI Data Intelligence Dashboard")


CHART_DIR = "outputs/charts/heatmaps"
TABLE_DIR = "outputs/tables"

STATE_HEATMAP = f"{CHART_DIR}/state_heatmap.png"
DISTRICT_HEATMAP = f"{CHART_DIR}/district_heatmap.png"

ANOMALY_TABLE = f"{TABLE_DIR}/anomalies.csv"
CORRELATION_TABLE = f"{TABLE_DIR}/state_update_enrolment_correlation.csv"
URBAN_TABLE = f"{TABLE_DIR}/urban_concentration.csv"
SPIKE_TABLE = f"{TABLE_DIR}/spike_strength.csv"


st.title(" UIDAI Demographic Intelligence Platform")
st.caption("Automated anomaly detection, hotspot analysis, and spatial heatmaps")

if st.button(" Run Data Pipeline"):
    with st.spinner("Running analysis..."):
        subprocess.run(["python3", "main.py"])
    st.success("Pipeline executed successfully!")

st.divider()

st.header(" Heatmaps")

col1, col2 = st.columns(2)

with col1:
    st.subheader("State Heatmap")
    if os.path.exists(STATE_HEATMAP):
        st.image(Image.open(STATE_HEATMAP), use_container_width=True)
    else:
        st.warning("State heatmap not found. Run pipeline.")

with col2:
    st.subheader("District Heatmap")
    if os.path.exists(DISTRICT_HEATMAP):
        st.image(Image.open(DISTRICT_HEATMAP), use_container_width=True)
    else:
        st.warning("District heatmap not found. Run pipeline.")

st.divider()

st.header("📊 Analytical Tables")

def show_table(title, path):
    st.subheader(title)
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("File not found.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Anomalies",
    "📈 Correlation",
    "🏙 Urban Concentration",
    "⚡ Spike Strength"
])

with tab1:
    show_table("Detected Anomalies", ANOMALY_TABLE)

with tab2:
    show_table("State Correlation", CORRELATION_TABLE)

with tab3:
    show_table("Urban Concentration", URBAN_TABLE)

with tab4:
    show_table("Spike Strength", SPIKE_TABLE)

st.divider()

st.caption("Built for UIDAI Hackathon • Python + Streamlit")
