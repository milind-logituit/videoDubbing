"""
EDA for VideoDubbing POC.
Run: uv run python code/eda.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data/raw"
OUT  = ROOT / "eda_outputs"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

print("TODO: load data and generate EDA charts")
