"""Unit tests for emotion.py — smooth_emotion_arc, load_prosody_config, _va_to_dist,
classify_segment_emotions (text-only and audio-fused paths)."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_spec = importlib.util.spec_from_file_location(
    "emotion", Path(__file__).parent.parent / "code" / "emotion.py"
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["emotion"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

smooth_emotion_arc         = _mod.smooth_emotion_arc
load_prosody_config        = _mod.load_prosody_config
SSML_PROSODY               = _mod.SSML_PROSODY
_va_to_dist                = _mod._va_to_dist
classify_segment_emotions  = _mod.classify_segment_emotions
_VALENCE_AROUSAL           = _mod._VALENCE_AROUSAL


# ── helpers ───────────────────────────────────────────────────────────────────

def _seg(seg_id: int, emotion: str, speaker: str = "SP0") -> dict:
    return {"id": seg_id, "start": float(seg_id), "end": float(seg_id + 1),
            "emotion": emotion, "speaker": speaker}


def _emotions(segs: list[dict]) -> list[str]:
    return [s["emotion"] for s in segs]


# ── smooth_emotion_arc ────────────────────────────────────────────────────────

def test_arc_flagged_initialised_on_all_segments():
    segs = [_seg(i, "neutral") for i in range(3)]
    result = smooth_emotion_arc(segs)
    assert all("arc_flagged" in s for s in result)
    assert all(s["arc_flagged"] is False for s in result)


def test_clear_outlier_flagged():
    # neutral neutral ANGRY neutral neutral → angry is the outlier
    segs = [_seg(i, "neutral") for i in range(5)]
    segs[2]["emotion"] = "angry"
    result = smooth_emotion_arc(segs)
    assert result[2]["arc_flagged"] is True
    assert all(not result[i]["arc_flagged"] for i in [0, 1, 3, 4])


def test_no_outlier_when_all_same():
    segs = [_seg(i, "happy") for i in range(7)]
    result = smooth_emotion_arc(segs)
    assert all(s["arc_flagged"] is False for s in result)


def test_emotions_not_overridden():
    segs = [_seg(i, "neutral") for i in range(5)]
    segs[2]["emotion"] = "angry"
    result = smooth_emotion_arc(segs)
    assert result[2]["emotion"] == "angry"


def test_short_speaker_group_skipped():
    # 3 segments < default window of 5 → nothing flagged
    segs = [_seg(i, "neutral") for i in range(3)]
    segs[1]["emotion"] = "angry"
    result = smooth_emotion_arc(segs)
    assert all(s["arc_flagged"] is False for s in result)


def test_edge_segments_not_flagged():
    # window=5 → first 2 and last 2 are always boundary, never flagged
    segs = [_seg(i, "neutral") for i in range(7)]
    segs[0]["emotion"] = "angry"
    segs[6]["emotion"] = "angry"
    result = smooth_emotion_arc(segs)
    assert result[0]["arc_flagged"] is False
    assert result[6]["arc_flagged"] is False


def test_multi_speaker_isolated():
    # SP0 has a clear outlier; SP1 has no outlier — must not cross-contaminate
    sp0 = [_seg(i, "neutral", "SP0") for i in range(5)]
    sp0[2]["emotion"] = "angry"
    sp1 = [_seg(i + 10, "happy", "SP1") for i in range(5)]
    result = smooth_emotion_arc(sp0 + sp1)
    by_id = {s["id"]: s for s in result}
    assert by_id[2]["arc_flagged"] is True         # SP0 outlier caught
    assert all(not by_id[i + 10]["arc_flagged"] for i in range(5))  # SP1 clean


def test_anonymous_speaker_grouped_together():
    # segments without speaker key → treated as one group
    segs = [{"id": i, "start": float(i), "end": float(i + 1), "emotion": "neutral"}
            for i in range(5)]
    segs[2]["emotion"] = "angry"
    result = smooth_emotion_arc(segs)
    assert result[2]["arc_flagged"] is True


def test_arc_flagged_false_not_nan():
    # Ensures no NaN ends up in the dict (would be truthy in pandas)
    segs = [_seg(i, "neutral") for i in range(5)]
    result = smooth_emotion_arc(segs)
    for s in result:
        assert s["arc_flagged"] is False, f"expected False, got {s['arc_flagged']!r}"


def test_custom_window():
    # window=3: centre of 3 is flagged if it differs from majority
    segs = [_seg(i, "neutral") for i in range(3)]
    segs[1]["emotion"] = "angry"
    result = smooth_emotion_arc(segs, window=3)
    assert result[1]["arc_flagged"] is True


# ── load_prosody_config ───────────────────────────────────────────────────────

def test_falls_back_to_ssml_prosody_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(_mod, "_PROSODY_CONFIG_PATH", tmp_path / "nonexistent.yaml")
    result = load_prosody_config("hi")
    assert result == SSML_PROSODY


def test_loads_default_block(tmp_path, monkeypatch):
    yaml_content = """
default:
  neutral:
    pitch: +0%
    volume: medium
    rate: +0%
  happy:
    pitch: +20%
    volume: loud
    rate: +10%
"""
    cfg = tmp_path / "emotion_prosody.yaml"
    cfg.write_text(yaml_content)
    monkeypatch.setattr(_mod, "_PROSODY_CONFIG_PATH", cfg)
    result = load_prosody_config("hi")
    assert result["neutral"]["pitch"] == "+0%"
    assert result["happy"]["pitch"] == "+20%"


def test_lang_override_merges_on_top(tmp_path, monkeypatch):
    yaml_content = """
default:
  happy:
    pitch: +15%
    volume: loud
    rate: +8%
hi:
  happy:
    pitch: +0%
    rate: +0%
"""
    cfg = tmp_path / "emotion_prosody.yaml"
    cfg.write_text(yaml_content)
    monkeypatch.setattr(_mod, "_PROSODY_CONFIG_PATH", cfg)
    result = load_prosody_config("hi")
    assert result["happy"]["pitch"] == "+0%"
    assert result["happy"]["rate"] == "+0%"
    assert result["happy"]["volume"] == "loud"   # inherited from default


def test_unknown_lang_returns_default_block(tmp_path, monkeypatch):
    yaml_content = """
default:
  neutral:
    pitch: +0%
    volume: medium
    rate: +0%
"""
    cfg = tmp_path / "emotion_prosody.yaml"
    cfg.write_text(yaml_content)
    monkeypatch.setattr(_mod, "_PROSODY_CONFIG_PATH", cfg)
    result = load_prosody_config("ta")   # no ta: block in file
    assert result["neutral"]["pitch"] == "+0%"


def test_falls_back_when_yaml_malformed(tmp_path, monkeypatch):
    cfg = tmp_path / "emotion_prosody.yaml"
    cfg.write_text("{{invalid: yaml: [}")
    monkeypatch.setattr(_mod, "_PROSODY_CONFIG_PATH", cfg)
    result = load_prosody_config("hi")
    assert result == SSML_PROSODY


def test_empty_yaml_falls_back(tmp_path, monkeypatch):
    cfg = tmp_path / "emotion_prosody.yaml"
    cfg.write_text("")
    monkeypatch.setattr(_mod, "_PROSODY_CONFIG_PATH", cfg)
    result = load_prosody_config("hi")
    assert result == SSML_PROSODY


# ── _va_to_dist ───────────────────────────────────────────────────────────────

def test_va_to_dist_sums_to_one():
    dist = _va_to_dist(0.0, 0.0)
    assert abs(sum(dist.values()) - 1.0) < 0.01


def test_va_to_dist_covers_all_emotions():
    dist = _va_to_dist(0.5, 0.5)
    assert set(dist.keys()) == set(_VALENCE_AROUSAL.keys())


def test_va_to_dist_exact_match_dominates():
    # neutral is at (0.0, 0.0) — query exactly there should give neutral most mass
    v, a = _VALENCE_AROUSAL["neutral"]
    dist = _va_to_dist(v, a)
    assert dist["neutral"] == max(dist.values())


def test_va_to_dist_happy_coords_dominate_happy():
    v, a = _VALENCE_AROUSAL["happy"]
    dist = _va_to_dist(v, a)
    assert dist["happy"] == max(dist.values())


def test_va_to_dist_no_negative_probs():
    dist = _va_to_dist(-1.0, -1.0)
    assert all(p >= 0.0 for p in dist.values())


# ── classify_segment_emotions — text-only path ────────────────────────────────

def _fake_text_pipe_classify(preds_list: list[dict]):
    """Return a pipeline callable yielding preds_list for any input."""
    return lambda text, **kwargs: preds_list


def _text_seg(seg_id: int, en_text: str) -> dict:
    return {"id": seg_id, "start": float(seg_id), "end": float(seg_id + 1),
            "en_text": en_text, "speaker": "SP0"}


def test_classify_text_only_empty_text(tmp_path):
    """Empty en_text produces neutral with score 1.0."""
    seg = {**_text_seg(0, ""), "en_text": ""}
    with patch.object(_mod, "_get_text_pipeline",
                      return_value=_fake_text_pipe_classify([{"label": "joy", "score": 0.9}])):
        result = classify_segment_emotions(tmp_path / "no_audio.wav", [seg],
                                           use_audio_ser=False)
    assert result[0]["emotion"] == "neutral"
    assert result[0]["emotion_score"] == 1.0


def test_classify_text_only_happy(tmp_path):
    segs = [_text_seg(0, "I am so happy today!")]
    preds = [{"label": "joy", "score": 0.85}, {"label": "neutral", "score": 0.10},
             {"label": "anger", "score": 0.05}]
    with patch.object(_mod, "_get_text_pipeline",
                      return_value=_fake_text_pipe_classify(preds)):
        result = classify_segment_emotions(tmp_path / "no_audio.wav", segs,
                                           use_audio_ser=False)
    assert result[0]["emotion"] == "happy"
    assert result[0]["emotion_score"] == pytest.approx(0.85, abs=0.01)
    assert "arousal" in result[0]
    assert "valence" in result[0]


def test_classify_text_only_propagates_existing_fields(tmp_path):
    segs = [{"id": 0, "start": 0.0, "end": 1.0, "en_text": "hello",
             "hi_text": "नमस्ते", "speaker": "SP0"}]
    preds = [{"label": "neutral", "score": 0.9}]
    with patch.object(_mod, "_get_text_pipeline",
                      return_value=_fake_text_pipe_classify(preds)):
        result = classify_segment_emotions(tmp_path / "no_audio.wav", segs,
                                           use_audio_ser=False)
    assert result[0]["hi_text"] == "नमस्ते"


# ── classify_segment_emotions — audio-fused path ─────────────────────────────

def test_classify_fuses_audio_when_available(tmp_path):
    """When audio SER returns a distribution, fused emotion may differ from text-only."""
    audio_file = tmp_path / "src.wav"
    audio_file.write_bytes(b"RIFF")

    segs = [_text_seg(0, "I am sad")]
    text_preds = [{"label": "sadness", "score": 0.8}, {"label": "neutral", "score": 0.2}]

    # audio SER returns strong angry signal — fused result should shift toward angry
    fake_audio_dist = {"angry": 0.9, "neutral": 0.1}

    fake_audio_obj = MagicMock()
    fake_audio_obj.set_channels.return_value = fake_audio_obj
    fake_audio_obj.set_frame_rate.return_value = fake_audio_obj

    with (
        patch.object(_mod, "_get_text_pipeline",
                     return_value=_fake_text_pipe_classify(text_preds)),
        patch("pydub.AudioSegment.from_file", return_value=fake_audio_obj),
        patch.object(_mod, "_get_audeering_model",
                     return_value=(MagicMock(), MagicMock())),
        patch.object(_mod, "_classify_audio_segment_audeering",
                     return_value=fake_audio_dist),
    ):
        result = classify_segment_emotions(audio_file, segs, use_audio_ser=True)

    # fused distribution: sad gets 0.65*0.8=0.52, angry gets 0.35*0.9=0.315 —
    # sad still wins, but angry mass is present
    fused_dist = result[0]["emotion_dist"]
    assert fused_dist.get("angry", 0.0) > 0.0


def test_classify_audio_ser_failure_falls_back_to_text(tmp_path):
    """If audio setup raises, text-only result is returned without crash."""
    audio_file = tmp_path / "src.wav"
    audio_file.write_bytes(b"RIFF")

    segs = [_text_seg(0, "I am so happy!")]
    preds = [{"label": "joy", "score": 0.9}]

    with (
        patch.object(_mod, "_get_text_pipeline",
                     return_value=_fake_text_pipe_classify(preds)),
        patch("pydub.AudioSegment.from_file", side_effect=RuntimeError("load fail")),
    ):
        result = classify_segment_emotions(audio_file, segs, use_audio_ser=True)

    assert result[0]["emotion"] == "happy"


def test_classify_audio_ser_empty_dist_uses_text(tmp_path):
    """If audeering returns {}, the text-only distribution is kept unchanged."""
    audio_file = tmp_path / "src.wav"
    audio_file.write_bytes(b"RIFF")

    segs = [_text_seg(0, "I am neutral")]
    preds = [{"label": "neutral", "score": 0.95}]

    fake_audio_obj = MagicMock()
    fake_audio_obj.set_channels.return_value = fake_audio_obj
    fake_audio_obj.set_frame_rate.return_value = fake_audio_obj

    with (
        patch.object(_mod, "_get_text_pipeline",
                     return_value=_fake_text_pipe_classify(preds)),
        patch("pydub.AudioSegment.from_file", return_value=fake_audio_obj),
        patch.object(_mod, "_get_audeering_model",
                     return_value=(MagicMock(), MagicMock())),
        patch.object(_mod, "_classify_audio_segment_audeering", return_value={}),
    ):
        result = classify_segment_emotions(audio_file, segs, use_audio_ser=True)

    assert result[0]["emotion"] == "neutral"
