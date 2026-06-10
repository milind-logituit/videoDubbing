"""Unit tests for pipeline_v2 logic that doesn't require GPU or network."""
import importlib.util
import sys
from pathlib import Path

# Load pipeline module without executing __main__ block
_spec = importlib.util.spec_from_file_location(
    "pipeline_v2", Path(__file__).parent.parent / "code" / "pipeline_v2.py"
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["pipeline_v2"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

assign_speakers = _mod.assign_speakers
_seg_voice = _mod._seg_voice
_voice_tag = _mod._voice_tag
_GENDER_LABEL_MAP = _mod._GENDER_LABEL_MAP
TTS_VOICE_FEMALE_HI = _mod.TTS_VOICE_FEMALE_HI
TTS_VOICE_MALE_HI = _mod.TTS_VOICE_MALE_HI
MAX_RATE_PCT = _mod.MAX_RATE_PCT
_LLM_BATCH = _mod._LLM_BATCH


# ── assign_speakers ───────────────────────────────────────────────────────────

def _seg(start: float, end: float, seg_id: int = 0) -> dict:
    return {"id": seg_id, "start": start, "end": end}

def _turn(start: float, end: float, speaker: str) -> dict:
    return {"start": start, "end": end, "speaker": speaker}


def test_assign_speakers_exact_overlap():
    segs = [_seg(0.0, 2.0)]
    turns = [_turn(0.0, 2.0, "SPEAKER_00")]
    result = assign_speakers(segs, turns)
    assert result[0]["speaker"] == "SPEAKER_00"


def test_assign_speakers_picks_dominant():
    segs = [_seg(1.0, 4.0)]
    turns = [_turn(0.0, 2.0, "SPEAKER_00"), _turn(2.0, 5.0, "SPEAKER_01")]
    result = assign_speakers(segs, turns)
    # overlap with SPEAKER_00 = 1s, SPEAKER_01 = 2s → SPEAKER_01 wins
    assert result[0]["speaker"] == "SPEAKER_01"


def test_assign_speakers_gap_falls_back_to_nearest():
    # segment sits in a silence gap between two turns
    segs = [_seg(5.0, 6.0)]
    turns = [_turn(0.0, 4.0, "SPEAKER_00"), _turn(7.0, 10.0, "SPEAKER_01")]
    result = assign_speakers(segs, turns)
    # seg mid=5.5; dist to SPK_00 mid=2.0 → 3.5, dist to SPK_01 mid=8.5 → 3.0
    assert result[0]["speaker"] == "SPEAKER_01"
    assert "speaker" in result[0]


def test_assign_speakers_empty_turns_returns_default():
    segs = [_seg(0.0, 1.0)]
    result = assign_speakers(segs, [])
    assert result[0]["speaker"] == "SPEAKER_00"


def test_assign_speakers_preserves_all_fields():
    segs = [{"id": 7, "start": 0.0, "end": 1.0, "en_text": "hi", "hi_text": "हाय"}]
    turns = [_turn(0.0, 1.0, "SPEAKER_00")]
    result = assign_speakers(segs, turns)
    assert result[0]["en_text"] == "hi"
    assert result[0]["hi_text"] == "हाय"


def test_assign_speakers_multiple_segments():
    segs = [_seg(0.0, 2.0, 0), _seg(3.0, 5.0, 1)]
    turns = [_turn(0.0, 2.5, "A"), _turn(2.5, 6.0, "B")]
    result = assign_speakers(segs, turns)
    assert result[0]["speaker"] == "A"
    assert result[1]["speaker"] == "B"


# ── voice helpers ─────────────────────────────────────────────────────────────

def test_voice_tag_female():
    assert _voice_tag(TTS_VOICE_FEMALE_HI) == "f"


def test_voice_tag_male():
    assert _voice_tag(TTS_VOICE_MALE_HI) == "m"


def test_seg_voice_uses_speaker_map():
    voices = {"SPEAKER_00": TTS_VOICE_MALE_HI}
    seg = {"speaker": "SPEAKER_00", "hi_text": "test"}
    assert _seg_voice(seg, voices) == TTS_VOICE_MALE_HI


def test_seg_voice_defaults_to_female_when_missing():
    seg = {"hi_text": "test"}  # no speaker key
    assert _seg_voice(seg, {}) == TTS_VOICE_FEMALE_HI


def test_seg_voice_defaults_to_female_when_speaker_not_in_map():
    seg = {"speaker": "SPEAKER_99", "hi_text": "test"}
    assert _seg_voice(seg, {}) == TTS_VOICE_FEMALE_HI


# ── gender label map ──────────────────────────────────────────────────────────

def test_gender_label_map_coverage():
    assert _GENDER_LABEL_MAP[0] == "female"
    assert _GENDER_LABEL_MAP[1] == "male"
    assert _GENDER_LABEL_MAP[2] == "female"  # child → female voice


# ── global rate cap ───────────────────────────────────────────────────────────

def test_max_rate_pct_is_capped():
    # simulate the rate calculation
    total_tts_s, total_speech_s = 200.0, 100.0
    raw = int((total_tts_s / total_speech_s - 1) * 100)
    global_rate = max(0, min(raw, MAX_RATE_PCT))
    assert global_rate == MAX_RATE_PCT


def test_global_rate_zero_when_tts_fits():
    total_tts_s, total_speech_s = 90.0, 100.0
    raw = int((total_tts_s / total_speech_s - 1) * 100)
    global_rate = max(0, min(raw, MAX_RATE_PCT))
    assert global_rate == 0


# ── LLM batch sizing ─────────────────────────────────────────────────────────

def test_llm_batch_splits_correctly():
    n = 250
    n_batches = (n + _LLM_BATCH - 1) // _LLM_BATCH
    assert n_batches == 3
    batches = [list(range(n))[i * _LLM_BATCH:(i + 1) * _LLM_BATCH]
               for i in range(n_batches)]
    assert sum(len(b) for b in batches) == n
    assert len(batches[0]) == _LLM_BATCH
    assert len(batches[-1]) == n % _LLM_BATCH or len(batches[-1]) == _LLM_BATCH
