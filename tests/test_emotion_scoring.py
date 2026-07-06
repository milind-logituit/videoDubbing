"""Tests for score_emotion_consistency and score_tts_emotion_fidelity in emotion.py."""
import importlib.util
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "emotion", Path(__file__).parent.parent / "code" / "emotion.py"
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["emotion"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

score_emotion_consistency   = _mod.score_emotion_consistency
score_tts_emotion_fidelity  = _mod.score_tts_emotion_fidelity
score_tts_prosody_transfer  = _mod.score_tts_prosody_transfer
_va_similarity              = _mod._va_similarity


def _write_glide_wav(path: Path, f0_start: float, f0_end: float,
                     dur_s: float = 1.2, sr: int = 16000) -> None:
    """Write a mono WAV whose pitch glides linearly from f0_start→f0_end Hz.

    A rising glide and a falling glide have anti-correlated F0 contours, which
    lets us assert on prosody-transfer direction deterministically.
    """
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    f0 = np.linspace(f0_start, f0_end, t.size)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = 0.6 * np.sin(phase)
    pcm = (sig * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_text_pipe(label: str, score: float = 0.9):
    """Return a pipeline callable that always predicts the given label."""
    return lambda text, **kwargs: [{"label": label, "score": score}]


def _seg(seg_id: int, en_text: str, hi_text: str, emotion: str = "neutral",
         start: float = 0.0, end: float = 1.0) -> dict:
    return {
        "id": seg_id,
        "en_text": en_text,
        "hi_text": hi_text,
        "emotion": emotion,
        "start": start,
        "end": end,
    }


# ── score_emotion_consistency ─────────────────────────────────────────────────

def test_consistency_empty_segments():
    with (
        patch.object(_mod, "_get_text_pipeline", return_value=_fake_text_pipe("joy")),
        patch.object(_mod, "_back_translate", side_effect=lambda t, **kw: t),
    ):
        result = score_emotion_consistency(Path("/dev/null"), Path("/dev/null"), [], "hi")

    assert result["n_segments"] == 0
    assert result["match_pct"] is None


def test_consistency_exact_match():
    with (
        patch.object(_mod, "_get_text_pipeline", return_value=_fake_text_pipe("joy", 0.9)),
        patch.object(_mod, "_back_translate", side_effect=lambda t, **kw: t),
    ):
        result = score_emotion_consistency(
            Path("/dev/null"),
            Path("/dev/null"),
            [_seg(0, "I am happy", "मैं खुश हूँ")],
            "hi",
        )

    assert result["n_segments"] == 1
    assert result["match_pct"] == 100.0
    assert result["avg_soft_score"] == 100.0


def test_consistency_mismatch():
    call_count = {"n": 0}

    def _alternating_pipe(text, **kwargs):
        call_count["n"] += 1
        if call_count["n"] % 2 == 1:
            return [{"label": "joy", "score": 0.9}]
        return [{"label": "sadness", "score": 0.9}]

    with (
        patch.object(_mod, "_get_text_pipeline", return_value=_alternating_pipe),
        patch.object(_mod, "_back_translate", side_effect=lambda t, **kw: t),
    ):
        result = score_emotion_consistency(
            Path("/dev/null"),
            Path("/dev/null"),
            [_seg(0, "I am happy", "मैं दुखी हूँ")],
            "hi",
        )

    assert result["match_pct"] == 0.0
    assert result["avg_soft_score"] < 50.0


def test_consistency_skips_empty_text():
    segments = [
        _seg(0, "",          "मैं खुश हूँ"),
        _seg(1, "I am happy", ""),
        _seg(2, "",          ""),
        _seg(3, "Valid text", "Valid hindi"),
    ]

    with (
        patch.object(_mod, "_get_text_pipeline", return_value=_fake_text_pipe("joy")),
        patch.object(_mod, "_back_translate", side_effect=lambda t, **kw: t),
    ):
        result = score_emotion_consistency(
            Path("/dev/null"), Path("/dev/null"), segments, "hi"
        )

    assert result["n_segments"] == 1


def test_consistency_soft_score_uses_va_similarity():
    """avg_soft_score ≈ (_va_similarity(happy,happy) + _va_similarity(happy,sad)) / 2 * 100."""
    call_count = {"n": 0}

    def _pipe(text, **kwargs):
        call_count["n"] += 1
        # Calls: seg0 src→joy, seg0 tgt→joy, seg1 src→joy, seg1 tgt→sadness
        if call_count["n"] in (1, 2, 3):
            return [{"label": "joy", "score": 0.9}]
        return [{"label": "sadness", "score": 0.9}]

    segments = [
        _seg(0, "I am happy", "मैं खुश हूँ"),
        _seg(1, "I am happy", "मैं दुखी हूँ"),
    ]

    with (
        patch.object(_mod, "_get_text_pipeline", return_value=_pipe),
        patch.object(_mod, "_back_translate", side_effect=lambda t, **kw: t),
    ):
        result = score_emotion_consistency(
            Path("/dev/null"), Path("/dev/null"), segments, "hi"
        )

    sim_match    = _va_similarity("happy", "happy")   # 1.0
    sim_mismatch = _va_similarity("happy", "sad")     # 0.07
    expected     = round((sim_match + sim_mismatch) / 2 * 100, 1)
    assert result["avg_soft_score"] == expected


# ── score_tts_emotion_fidelity ────────────────────────────────────────────────

def test_fidelity_missing_audio(tmp_path: Path):
    missing = tmp_path / "does_not_exist.mp3"
    result = score_tts_emotion_fidelity(missing, [])
    assert "note" in result


def test_fidelity_audio_model_load_failure(tmp_path: Path):
    audio_file = tmp_path / "dubbed.wav"
    audio_file.write_bytes(b"")

    with patch.object(_mod, "_get_audio_pipeline", side_effect=RuntimeError("no model")):
        result = score_tts_emotion_fidelity(audio_file, [])

    assert "note" in result


def test_fidelity_returns_avg_score(tmp_path: Path):
    audio_file = tmp_path / "dubbed.wav"
    audio_file.write_bytes(b"RIFF")

    fake_model = MagicMock()
    fake_processor = MagicMock()

    fake_audio = MagicMock()
    fake_audio.set_channels.return_value = fake_audio
    fake_audio.set_frame_rate.return_value = fake_audio

    # Return neutral-dominant distribution so perceived==intended→similarity==1.0
    neutral_dist = {"neutral": 0.9, "happy": 0.02, "angry": 0.02,
                    "sad": 0.02, "fearful": 0.02, "disgust": 0.01, "surprised": 0.01}

    segments = [{"id": 0, "start": 0.0, "end": 1.0, "emotion": "neutral"}]

    with (
        patch.object(_mod, "_get_audeering_model", return_value=(fake_model, fake_processor)),
        patch.object(_mod, "_classify_audio_segment_audeering", return_value=neutral_dist),
        patch("pydub.AudioSegment.from_file", return_value=fake_audio),
    ):
        result = score_tts_emotion_fidelity(audio_file, segments)

    assert "note" not in result
    assert result["avg_soft_score"] == 100.0
    assert result["n_segments"] == 1


def test_fidelity_skips_zero_duration(tmp_path: Path):
    audio_file = tmp_path / "dubbed.wav"
    audio_file.write_bytes(b"RIFF")

    fake_audio = MagicMock()
    fake_audio.set_channels.return_value = fake_audio
    fake_audio.set_frame_rate.return_value = fake_audio

    segments = [
        {"id": 0, "start": 1.0, "end": 1.0, "emotion": "neutral"},
        {"id": 1, "start": 2.0, "end": 1.5, "emotion": "neutral"},
    ]

    with (
        patch.object(_mod, "_get_audio_pipeline", return_value=MagicMock()),
        patch("pydub.AudioSegment.from_file", return_value=fake_audio),
    ):
        result = score_tts_emotion_fidelity(audio_file, segments)

    assert result["n_segments"] == 0


# ── score_tts_prosody_transfer ────────────────────────────────────────────────

def test_prosody_transfer_missing_source(tmp_path: Path):
    dub = tmp_path / "dub.wav"
    _write_glide_wav(dub, 120, 200)
    result = score_tts_prosody_transfer(tmp_path / "nope.wav", dub, [])
    assert "source audio not found" in result["note"]


def test_prosody_transfer_missing_dub(tmp_path: Path):
    src = tmp_path / "src.wav"
    _write_glide_wav(src, 120, 200)
    result = score_tts_prosody_transfer(src, tmp_path / "nope.wav", [])
    assert "dubbed audio not found" in result["note"]


def test_prosody_transfer_matching_contour_scores_high(tmp_path: Path):
    # Source and dub both glide up over the same window → contours correlate → high.
    src = tmp_path / "src.wav"
    dub = tmp_path / "dub.wav"
    _write_glide_wav(src, 120, 240)
    _write_glide_wav(dub, 130, 250)  # different absolute pitch, same shape

    segments = [{"id": 0, "start": 0.05, "end": 1.15, "emotion": "happy"}]
    result = score_tts_prosody_transfer(src, dub, segments)

    assert result["n_segments"] == 1
    assert result["avg_soft_score"] > 75.0
    assert result["segments"][0]["f0_shape_r"] > 0.5


def test_prosody_transfer_opposite_contour_scores_low(tmp_path: Path):
    # Source glides up, dub glides down → anti-correlated F0 → below matching case.
    src = tmp_path / "src.wav"
    dub = tmp_path / "dub.wav"
    _write_glide_wav(src, 120, 240)
    _write_glide_wav(dub, 240, 120)

    segments = [{"id": 0, "start": 0.05, "end": 1.15, "emotion": "happy"}]
    result = score_tts_prosody_transfer(src, dub, segments)

    assert result["n_segments"] == 1
    assert result["segments"][0]["f0_shape_r"] < 0.0


def test_prosody_transfer_skips_unvoiced_source(tmp_path: Path):
    # A silent source slice carries no prosody → segment is skipped, not scored.
    src = tmp_path / "src.wav"
    dub = tmp_path / "dub.wav"
    sr = 16000
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(int(sr * 1.2), dtype="<i2").tobytes())
    _write_glide_wav(dub, 130, 250)

    segments = [{"id": 0, "start": 0.05, "end": 1.15, "emotion": "happy"}]
    result = score_tts_prosody_transfer(src, dub, segments)

    assert result["n_segments"] == 0
    assert result["avg_soft_score"] is None
