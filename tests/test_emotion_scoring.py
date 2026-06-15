"""Tests for score_emotion_consistency and score_tts_emotion_fidelity in emotion.py."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_spec = importlib.util.spec_from_file_location(
    "emotion", Path(__file__).parent.parent / "code" / "emotion.py"
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["emotion"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

score_emotion_consistency  = _mod.score_emotion_consistency
score_tts_emotion_fidelity = _mod.score_tts_emotion_fidelity
_va_similarity             = _mod._va_similarity


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

    fake_pipe = MagicMock(return_value=[{"label": "neu", "score": 0.9}])

    fake_chunk = MagicMock()
    fake_chunk.__len__ = MagicMock(return_value=1000)
    fake_chunk.export = MagicMock()

    fake_audio = MagicMock()
    fake_audio.set_channels.return_value = fake_audio
    fake_audio.set_frame_rate.return_value = fake_audio
    fake_audio.__getitem__ = MagicMock(return_value=fake_chunk)

    segments = [{"id": 0, "start": 0.0, "end": 1.0, "emotion": "neutral"}]

    with (
        patch.object(_mod, "_get_audio_pipeline", return_value=fake_pipe),
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
