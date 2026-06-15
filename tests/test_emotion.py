"""Unit tests for emotion.py — smooth_emotion_arc and load_prosody_config."""
import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "emotion", Path(__file__).parent.parent / "code" / "emotion.py"
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["emotion"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

smooth_emotion_arc  = _mod.smooth_emotion_arc
load_prosody_config = _mod.load_prosody_config
SSML_PROSODY        = _mod.SSML_PROSODY


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
