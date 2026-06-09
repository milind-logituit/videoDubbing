"""
Streamlit dashboard for VideoDubbing POC.
Run: uv run streamlit run code/dashboard.py
"""
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent.parent

st.set_page_config(page_title="VideoDubbing POC", layout="wide")
st.title("🎬 VideoDubbing POC")
st.info("Dashboard coming soon.")
