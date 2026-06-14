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
_assign_genders = _mod._assign_genders
_assign_voice_pool = _mod._assign_voice_pool
_seg_voice = _mod._seg_voice
_voice_tag = _mod._voice_tag
_ctx_snippet = _mod._ctx_snippet
extract_glossary = _mod.extract_glossary
generate_vtt = _mod.generate_vtt
generate_srt = _mod.generate_srt
_GENDER_LABEL_MAP = _mod._GENDER_LABEL_MAP
TTS_VOICE_FEMALE_HI = _mod.TTS_VOICE_FEMALE_HI
TTS_VOICE_MALE_HI = _mod.TTS_VOICE_MALE_HI
VOICE_POOL = _mod.VOICE_POOL
MAX_RATE_PCT = _mod.MAX_RATE_PCT
_LLM_BATCH = _mod._LLM_BATCH
_CTX_WINDOW = _mod._CTX_WINDOW


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


# ── _assign_genders ───────────────────────────────────────────────────────────

def test_assign_genders_high_confidence_female():
    # f_prob >= 0.70 → female regardless
    result = _assign_genders({"SP0": (0.92, 0.08)})
    assert result["SP0"] == "female"


def test_assign_genders_below_half_is_male():
    result = _assign_genders({"SP0": (0.40, 0.60)})
    assert result["SP0"] == "male"


def test_assign_genders_borderline_top_speaker_with_margin_is_female():
    # Tears-of-Steel scenario: SP1=0.54 (female), others clearly male
    probs = {"SP0": (0.04, 0.96), "SP1": (0.54, 0.45), "SP2": (0.20, 0.80)}
    result = _assign_genders(probs)
    assert result["SP1"] == "female"
    assert result["SP0"] == "male"
    assert result["SP2"] == "male"


def test_assign_genders_borderline_no_margin_is_male():
    # Two borderline speakers close together → both male (neither wins the margin)
    probs = {"SP0": (0.55, 0.45), "SP1": (0.52, 0.48)}
    result = _assign_genders(probs)
    assert result["SP0"] == "male"
    assert result["SP1"] == "male"


def test_assign_genders_elephants_dream_scenario():
    # SPEAKER_01=0.48 (below 0.5 → male), SPEAKER_00=0.97 (→ female)
    probs = {"SPEAKER_01": (0.48, 0.52), "SPEAKER_00": (0.97, 0.03),
             "SPEAKER_02": (0.42, 0.58)}
    result = _assign_genders(probs)
    assert result["SPEAKER_00"] == "female"
    assert result["SPEAKER_01"] == "male"
    assert result["SPEAKER_02"] == "male"


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


# ── _assign_voice_pool ────────────────────────────────────────────────────────

def _seg_timed(seg_id: int, start: float, speaker: str) -> dict:
    return {"id": seg_id, "start": start, "end": start + 1.0, "speaker": speaker}


def test_assign_voice_pool_single_male_speaker():
    genders = {"SPEAKER_00": "male"}
    segs = [_seg_timed(0, 0.0, "SPEAKER_00")]
    result = _assign_voice_pool(genders, segs, "hi")
    assert result["SPEAKER_00"] == VOICE_POOL["hi"]["male"][0]


def test_assign_voice_pool_male_female_distinct():
    genders = {"SPEAKER_00": "male", "SPEAKER_01": "female"}
    segs = [_seg_timed(0, 0.0, "SPEAKER_00"), _seg_timed(1, 1.0, "SPEAKER_01")]
    result = _assign_voice_pool(genders, segs, "hi")
    assert result["SPEAKER_00"] != result["SPEAKER_01"]


def test_assign_voice_pool_english_multi_male_distinct():
    # English pool has 4 male voices — first two speakers get different voices
    genders = {"SPEAKER_00": "male", "SPEAKER_01": "male"}
    segs = [_seg_timed(0, 0.0, "SPEAKER_00"), _seg_timed(1, 1.0, "SPEAKER_01")]
    result = _assign_voice_pool(genders, segs, "en")
    assert result["SPEAKER_00"] != result["SPEAKER_01"]
    assert result["SPEAKER_00"] == VOICE_POOL["en"]["male"][0]
    assert result["SPEAKER_01"] == VOICE_POOL["en"]["male"][1]


def test_assign_voice_pool_stable_order_by_first_utterance():
    # SPEAKER_01 appears first in timeline despite sorting by name last
    genders = {"SPEAKER_00": "male", "SPEAKER_01": "male"}
    segs = [_seg_timed(0, 5.0, "SPEAKER_00"), _seg_timed(1, 0.0, "SPEAKER_01")]
    result = _assign_voice_pool(genders, segs, "en")
    # SPEAKER_01 starts at 0.0 → gets pool index 0
    assert result["SPEAKER_01"] == VOICE_POOL["en"]["male"][0]
    assert result["SPEAKER_00"] == VOICE_POOL["en"]["male"][1]


def test_assign_voice_pool_cycles_when_pool_exhausted():
    # Hindi has only 1 male voice — third male speaker wraps back to index 0
    genders = {f"SPEAKER_0{i}": "male" for i in range(3)}
    segs = [_seg_timed(i, float(i), f"SPEAKER_0{i}") for i in range(3)]
    result = _assign_voice_pool(genders, segs, "hi")
    male_pool = VOICE_POOL["hi"]["male"]
    assert result["SPEAKER_00"] == male_pool[0 % len(male_pool)]
    assert result["SPEAKER_01"] == male_pool[1 % len(male_pool)]
    assert result["SPEAKER_02"] == male_pool[2 % len(male_pool)]


# ── _ctx_snippet ──────────────────────────────────────────────────────────────

def test_ctx_snippet_extracts_id_en_hi():
    segs = [{"id": 1, "en_text": "Hello", "hi_text": "नमस्ते", "duration": 1.5}]
    result = _ctx_snippet(segs)
    assert result == [{"id": 1, "en_text": "Hello", "hi_text": "नमस्ते"}]


def test_ctx_snippet_empty_returns_empty():
    assert _ctx_snippet([]) == []


# ── context window slicing ────────────────────────────────────────────────────

def test_ctx_window_first_batch_has_no_before():
    n = 5
    start = 0
    ctx_before = list(range(n))[max(0, start - _CTX_WINDOW):start]
    assert ctx_before == []


def test_ctx_window_last_batch_has_no_after():
    segments = list(range(10))
    end = len(segments)
    ctx_after = segments[end:end + _CTX_WINDOW]
    assert ctx_after == []


def test_ctx_window_middle_batch_has_both():
    segments = list(range(20))
    batch_size = 5
    i = 1  # second batch
    start = i * batch_size
    end = start + batch_size
    ctx_before = segments[max(0, start - _CTX_WINDOW):start]
    ctx_after  = segments[end:end + _CTX_WINDOW]
    assert len(ctx_before) == _CTX_WINDOW
    assert len(ctx_after) == _CTX_WINDOW


# ── extract_glossary ──────────────────────────────────────────────────────────

def _make_segs(*texts: str) -> list[dict]:
    return [{"id": i, "en_text": t, "hi_text": ""} for i, t in enumerate(texts)]


def test_extract_glossary_returns_cached_when_file_exists(tmp_path: Path):
    cache = tmp_path / "glossary.json"
    cache.write_text('{"Tom": "टॉम"}', encoding="utf-8")
    result = extract_glossary(_make_segs("Hello Tom"), cache_path=cache)
    assert result == {"Tom": "टॉम"}


def test_extract_glossary_returns_empty_when_no_candidates():
    # All single-occurrence lower-case words → no candidates
    segs = _make_segs("we have main engine start", "four three two one")
    result = extract_glossary(segs, cache_path=None)
    assert result == {}


def test_extract_glossary_calls_claude_and_caches(tmp_path: Path):
    import json as _json
    from unittest.mock import MagicMock, patch

    cache = tmp_path / "glossary.json"
    segs = _make_segs("Tom said hello", "Tom left quickly")

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text='{"Tom": "टॉम"}')]

    with patch("pipeline_v2.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = fake_response
        result = extract_glossary(segs, target_lang="hi", cache_path=cache)

    assert result == {"Tom": "टॉम"}
    assert cache.exists()
    assert _json.loads(cache.read_text()) == {"Tom": "टॉम"}


# ── subtitle timing re-alignment ──────────────────────────────────────────────

def _timed_seg(seg_id: int, start: float, end: float, hi: str = "text") -> dict:
    return {"id": seg_id, "start": start, "end": end,
            "en_text": "src", "hi_text": hi}


def test_subtitle_end_clamped_to_original_window():
    seg = _timed_seg(0, 5.0, 8.0)
    eff_durs = {0: 10.0}  # TTS longer than window → clamp to window
    MIN_SUB_S = 0.5
    actual = eff_durs[seg["id"]]
    original_window = seg["end"] - seg["start"]
    seg["end"] = seg["start"] + max(MIN_SUB_S, min(actual, original_window))
    assert seg["end"] == 8.0  # clamped to original window


def test_subtitle_end_uses_actual_when_shorter():
    seg = _timed_seg(0, 5.0, 8.0)
    eff_durs = {0: 1.5}  # TTS shorter than window
    MIN_SUB_S = 0.5
    actual = eff_durs[seg["id"]]
    original_window = seg["end"] - seg["start"]
    seg["end"] = seg["start"] + max(MIN_SUB_S, min(actual, original_window))
    assert seg["end"] == 6.5  # 5.0 + 1.5


def test_subtitle_end_respects_minimum_floor():
    seg = _timed_seg(0, 5.0, 8.0)
    eff_durs = {0: 0.1}  # extremely short TTS
    MIN_SUB_S = 0.5
    actual = eff_durs[seg["id"]]
    original_window = seg["end"] - seg["start"]
    seg["end"] = seg["start"] + max(MIN_SUB_S, min(actual, original_window))
    assert seg["end"] == 5.5  # 5.0 + 0.5 floor


def test_subtitle_vtt_uses_updated_end_times():
    seg = _timed_seg(0, 1.0, 2.0)
    seg["end"] = 1.8  # simulated re-alignment
    vtt = generate_vtt([seg])
    assert "00:00:01.800" in vtt


def test_subtitle_srt_uses_updated_end_times():
    seg = _timed_seg(0, 1.0, 2.0)
    seg["end"] = 1.6
    srt = generate_srt([seg])
    assert "00:00:01,600" in srt
