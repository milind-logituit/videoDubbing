"""Regression gate for emotion quality — marks.slow, CI-only on emotion-touching PRs."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT       = Path(__file__).parent.parent
CLIP       = ROOT / "test_clips" / "tears_of_steel_2min.mp4"
METRICS    = ROOT / "model_outputs" / "metrics_tears_of_steel_2min.json"
PIPELINE   = ROOT / "code" / "pipeline_v2.py"

MIN_EMOTION_SOFT_SCORE   = 80.0
MIN_TTS_FIDELITY         = 60.0
MIN_MOS                  = 80.0


@pytest.mark.slow
def test_emotion_regression():
    if not CLIP.exists():
        pytest.skip(f"Test clip not found: {CLIP}")

    result = subprocess.run(
        [sys.executable, str(PIPELINE), "--input", str(CLIP), "--target-lang", "hi",
         "--no-diarize"],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0, f"Pipeline failed:\n{result.stderr[-2000:]}"

    assert METRICS.exists(), "metrics JSON not written"
    metrics = json.loads(METRICS.read_text())

    emotion = metrics.get("emotion", {})
    soft = emotion.get("avg_soft_score")
    fidelity = emotion.get("tts_fidelity", {}).get("avg_soft_score")
    mos = metrics.get("mos_rubric", {}).get("mos")

    assert soft     is not None, "emotion.avg_soft_score missing from metrics"
    assert soft     >= MIN_EMOTION_SOFT_SCORE,  f"Emotion soft score {soft}% < {MIN_EMOTION_SOFT_SCORE}%"
    assert fidelity is not None, "emotion.tts_fidelity.avg_soft_score missing from metrics"
    assert fidelity >= MIN_TTS_FIDELITY,        f"TTS fidelity {fidelity}% < {MIN_TTS_FIDELITY}%"
    assert mos      is not None, "mos_rubric.mos missing from metrics"
    assert mos      >= MIN_MOS,                 f"MOS {mos} < {MIN_MOS}"
