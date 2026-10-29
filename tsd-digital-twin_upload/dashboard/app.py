"""
app.py - entry point of the TSD digital twin dashboard.

    streamlit run dashboard/app.py

Pages: Live batch, What-if, Batch history, Alerts.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="TSD Digital Twin", page_icon="🌀", layout="wide", initial_sidebar_state="expanded")

pages = [
    st.Page("pages/1_Live_batch.py", title="Live batch", icon="📡", default=True, url_path="live"),
    st.Page("pages/2_What_if.py", title="What-if", icon="🎛️", url_path="whatif"),
    st.Page("pages/3_Batch_history.py", title="Batch history", icon="📊", url_path="history"),
    st.Page("pages/4_Alerts.py", title="Alerts", icon="🚨", url_path="alerts"),
]
nav = st.navigation(pages)
with st.sidebar:
    st.markdown("### TSD Digital Twin")
    st.caption("Twin-shaft disperser - 3,000 L, 55 kW HSD, 11 kW anchor")
    import os
    src = os.environ.get("TSD_DATA_DIR", "data/raw")
    st.caption("Data source: SIMULATED (Step 3)" if src.endswith("raw") else f"Data source: {src} (plant historian export)")
nav.run()
