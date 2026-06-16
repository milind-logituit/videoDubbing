"""Regression gate for emotion quality — marks.slow, CI-only on emotion-touching PRs.

Thresholds are taken directly from docs/emotion_matching_spec.docx:

  Metric                   Gate (P1)   Success target
  avg_emotion_register     ≥ 3.8       ≥ 4.0
  tts_fidelity_soft_score  ≥ 65%       ≥ 75%
  MOS composite            (no gate)   qualitative

avg_emotion_register is the Claude-scored per-segment emotional delivery (1–5 rubric),
averaged over non-neutral segments in segment_quality. It is NOT the same as
emotion.avg_soft_score (text emotion consistency between source and dub).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT       = Path(__file__).parent.parent
CLIP       = ROOT / "test_clips" / "tears_of_steel_2min.mp4"
METRICS    = ROOT / "model_outputs" / "metrics_tears_of_steel_2min.json"
PIPELINE   = ROOT / "code" / "pipeline_v2.py"

# From emotion_matching_spec.docx — P1 regression gate thresholds
MIN_EMOTION_REGISTER = 3.8   # avg over non-neutral segments scored by Claude MOS rubric
MIN_TTS_FIDELITY     = 65.0  # emotion.tts_fidelity.avg_soft_score


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

    # Primary gate: avg_emotion_register (spec target ≥ 4.0; gate ≥ 3.8)
    scored = [s["emotion_register"] for s in metrics.get("segment_quality", [])
              if "emotion_register" in s]
    assert scored, "No emotion_register scores found in segment_quality"
    avg_er = sum(scored) / len(scored)
    assert avg_er >= MIN_EMOTION_REGISTER, (
        f"avg_emotion_register {avg_er:.2f} < {MIN_EMOTION_REGISTER} "
        f"(n={len(scored)}, values={scored})"
    )

    # Secondary gate: TTS fidelity (spec target ≥ 75%; gate ≥ 65%)
    fidelity = metrics.get("emotion", {}).get("tts_fidelity", {}).get("avg_soft_score")
    assert fidelity is not None, "emotion.tts_fidelity.avg_soft_score missing from metrics"
    assert fidelity >= MIN_TTS_FIDELITY, (
        f"TTS fidelity {fidelity}% < {MIN_TTS_FIDELITY}%"
    )
