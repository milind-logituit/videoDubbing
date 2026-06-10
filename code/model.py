"""
Models for VideoDubbing POC.
Run: uv run python code/model.py
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "model_outputs"
OUT.mkdir(exist_ok=True)

print("TODO: train models, evaluate, save metrics.json")
