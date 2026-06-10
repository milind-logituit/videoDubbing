"""
Data preparation for VideoDubbing POC.
Run: uv run python code/data_prep.py
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data/raw"
OUT  = ROOT / "data/prepared"
OUT.mkdir(parents=True, exist_ok=True)

print("TODO: load, clean, split, and save prepared data")
