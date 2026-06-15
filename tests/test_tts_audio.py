"""Tests for pure-logic functions in code/tts_audio.py.

No real audio synthesis or ffmpeg calls are made — subprocess.run and
AudioSegment I/O are mocked throughout.
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module load — avoids importing tts_audio at collection time (which would
# trigger edge_tts / gtts import errors if those aren't installed in the
# test environment).
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "tts_audio",
    Path(__file__).parent.parent / "code" / "tts_audio.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["tts_audio"] = _mod
_spec.loader.exec_module(_mod)

_emotion_tag       = _mod._emotion_tag
_voice_tag         = _mod._voice_tag
_atempo_compress   = _mod._atempo_compress
TTS_BASE_RATE_PCT  = _mod.TTS_BASE_RATE_PCT
MAX_RATE_PCT       = _mod.MAX_RATE_PCT
VOICE_POOL         = _mod.VOICE_POOL
get_effective_seg_durations = _mod.get_effective_seg_durations


# ===========================================================================
# _emotion_tag
# ===========================================================================

class TestEmotionTag:
    def test_emotion_tag_neutral_returns_empty(self):
        assert _emotion_tag("neutral") == ""

    def test_emotion_tag_none_returns_empty(self):
        assert _emotion_tag(None) == ""

    def test_emotion_tag_empty_string_returns_empty(self):
        assert _emotion_tag("") == ""

    def test_emotion_tag_happy_includes_r(self):
        result = _emotion_tag("happy")
        assert result.startswith("_r"), f"expected '_r…', got {result!r}"
        assert "hap" in result, f"expected 'hap' in {result!r}"

    def test_emotion_tag_angry(self):
        assert _emotion_tag("angry") == "_rang"

    def test_emotion_tag_cache_collision(self):
        """happy and angry must produce different tags so cache files don't collide."""
        assert _emotion_tag("happy") != _emotion_tag("angry")

    def test_emotion_tag_sad(self):
        assert _emotion_tag("sad") == "_rsad"


# ===========================================================================
# _voice_tag
# ===========================================================================

class TestVoiceTag:
    def test_voice_tag_male_hi(self):
        assert _voice_tag("hi-IN-MadhurNeural") == "m"

    def test_voice_tag_female_hi(self):
        assert _voice_tag("hi-IN-SwaraNeural") == "f"

    def test_voice_tag_male_en(self):
        assert _voice_tag("en-US-GuyNeural") == "m"

    def test_voice_tag_female_en(self):
        assert _voice_tag("en-US-JennyNeural") == "f"

    def test_voice_tag_unknown_defaults_to_female(self):
        # Any voice not in _MALE_VOICES should return "f"
        assert _voice_tag("xx-XX-UnknownNeural") == "f"


# ===========================================================================
# Rate calculation (inline replication of _synth_pass2 logic)
# ===========================================================================

def _calc_edge_rate(tts_s: float, target_s: float) -> int:
    """Replicate the needed_pct / edge_rate logic from _synth_pass2."""
    needed_pct = int((tts_s / target_s - 1) * 100) + TTS_BASE_RATE_PCT
    return max(TTS_BASE_RATE_PCT, min(needed_pct, MAX_RATE_PCT))


class TestRateCalculation:
    def test_rate_no_speedup_when_tts_fits(self):
        """TTS shorter than target → rate stays at base."""
        rate = _calc_edge_rate(tts_s=2.0, target_s=3.0)
        assert rate == TTS_BASE_RATE_PCT

    def test_rate_increases_when_tts_overflows(self):
        """TTS longer than target → rate must exceed base."""
        rate = _calc_edge_rate(tts_s=3.0, target_s=2.0)
        assert rate > TTS_BASE_RATE_PCT

    def test_rate_capped_at_max(self):
        """Extreme overflow → rate capped at MAX_RATE_PCT."""
        rate = _calc_edge_rate(tts_s=10.0, target_s=1.0)
        assert rate == MAX_RATE_PCT

    def test_rate_stays_at_base_when_barely_over(self):
        """Slight overflow → rate is within [TTS_BASE_RATE_PCT, MAX_RATE_PCT]."""
        rate = _calc_edge_rate(tts_s=2.1, target_s=2.0)
        assert TTS_BASE_RATE_PCT <= rate <= MAX_RATE_PCT

    def test_rate_exactly_equal(self):
        """tts_s == target_s → needed_pct == TTS_BASE_RATE_PCT → base rate."""
        rate = _calc_edge_rate(tts_s=2.0, target_s=2.0)
        assert rate == TTS_BASE_RATE_PCT

    def test_base_and_max_constants_sensible(self):
        assert TTS_BASE_RATE_PCT < MAX_RATE_PCT
        assert MAX_RATE_PCT > 0


# ===========================================================================
# _atempo_compress
# ===========================================================================

def _make_fake_audio(duration_ms: int) -> MagicMock:
    """Return a MagicMock that behaves like an AudioSegment of duration_ms."""
    audio = MagicMock()
    audio.__len__ = MagicMock(return_value=duration_ms)
    audio.__getitem__ = MagicMock(return_value=audio)
    return audio


class TestAtempoCompress:
    def test_no_compress_when_fits(self):
        """Audio already ≤ target * 1.05 → returned unchanged, no subprocess call."""
        audio = _make_fake_audio(2000)  # 2 s
        target_s = 2.0

        with patch("tts_audio.subprocess.run") as mock_run, \
             patch("tts_audio.AudioSegment.from_mp3") as mock_from_mp3:
            result = _atempo_compress(audio, target_s)

        mock_run.assert_not_called()
        mock_from_mp3.assert_not_called()
        assert result is audio

    def test_no_compress_within_5pct_margin(self):
        """Audio 4 % over target (within 5 % margin) → returned unchanged."""
        audio = _make_fake_audio(2080)  # 2.08 s vs target 2.0 s → 4 % over
        target_s = 2.0

        with patch("tts_audio.subprocess.run") as mock_run:
            result = _atempo_compress(audio, target_s)

        mock_run.assert_not_called()
        assert result is audio

    def test_atempo_single_filter_when_factor_le_2(self):
        """Factor ≤ 2.0 → ffmpeg called with a single atempo=X filter."""
        audio = _make_fake_audio(3000)   # 3.0 s → factor = 3 / 2 = 1.5
        target_s = 2.0
        fake_result = _make_fake_audio(2000)

        with patch("tts_audio.subprocess.run") as mock_run, \
             patch("tts_audio.AudioSegment.from_mp3", return_value=fake_result), \
             patch.object(audio, "export"), \
             patch("tts_audio.tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("tts_audio.os.unlink"), \
             patch("tts_audio.os.path.exists", return_value=True):

            fake_file = MagicMock()
            fake_file.name = "/tmp/test_in.mp3"
            mock_tmp.return_value = fake_file

            _atempo_compress(audio, target_s)

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        filter_str = cmd[cmd.index("-filter:a") + 1]
        # Single filter — no comma
        assert "," not in filter_str, f"expected single filter, got: {filter_str!r}"
        assert filter_str.startswith("atempo=")

    def test_atempo_chains_filters_when_factor_gt_2(self):
        """Factor = 3.0 → two chained atempo filters: atempo=2.0,atempo=1.5."""
        audio = _make_fake_audio(6000)   # 6 s → factor = 6 / 2 = 3.0
        target_s = 2.0
        fake_result = _make_fake_audio(2000)

        with patch("tts_audio.subprocess.run") as mock_run, \
             patch("tts_audio.AudioSegment.from_mp3", return_value=fake_result), \
             patch.object(audio, "export"), \
             patch("tts_audio.tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("tts_audio.os.unlink"), \
             patch("tts_audio.os.path.exists", return_value=True):

            fake_file = MagicMock()
            fake_file.name = "/tmp/test_in.mp3"
            mock_tmp.return_value = fake_file

            _atempo_compress(audio, target_s)

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        filter_str = cmd[cmd.index("-filter:a") + 1]
        parts = filter_str.split(",")
        assert len(parts) == 2, f"expected 2 filters, got: {parts}"
        assert parts[0] == "atempo=2.0"
        assert parts[1].startswith("atempo=")

    def test_atempo_returns_trimmed_when_result_too_long(self):
        """If ffmpeg result is longer than target, the output is sliced."""
        audio = _make_fake_audio(3000)   # 3 s → factor = 1.5
        target_s = 2.0
        # Fake result is 2500 ms — still over target
        fake_result = MagicMock()
        fake_result.__len__ = MagicMock(return_value=2500)
        trimmed = MagicMock()
        fake_result.__getitem__ = MagicMock(return_value=trimmed)

        with patch("tts_audio.subprocess.run"), \
             patch("tts_audio.AudioSegment.from_mp3", return_value=fake_result), \
             patch.object(audio, "export"), \
             patch("tts_audio.tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("tts_audio.os.unlink"), \
             patch("tts_audio.os.path.exists", return_value=True):

            fake_file = MagicMock()
            fake_file.name = "/tmp/test_in.mp3"
            mock_tmp.return_value = fake_file

            result = _atempo_compress(audio, target_s)

        # Should have sliced the result
        fake_result.__getitem__.assert_called_once_with(slice(None, 2000, None))
        assert result is trimmed


# ===========================================================================
# get_effective_seg_durations
# ===========================================================================

class TestGetEffectiveSegDurations:
    def test_get_effective_no_files(self, tmp_path: Path):
        """No effective mp3 files → returns empty dict."""
        seg_dir = tmp_path / "hi_segments_sample"
        seg_dir.mkdir()
        original_prepared = _mod.PREPARED
        _mod.PREPARED = tmp_path
        try:
            segments = [{"id": 0}, {"id": 1}]
            result = get_effective_seg_durations(segments, "sample", "hi")
        finally:
            _mod.PREPARED = original_prepared

        assert result == {}

    def test_get_effective_finds_files(self, tmp_path: Path):
        """Finds effective mp3 → returns {seg_id: duration_s}."""
        seg_dir = tmp_path / "hi_segments_sample"
        seg_dir.mkdir()
        eff_file = seg_dir / "seg_000_f_effective.mp3"
        eff_file.write_bytes(b"fake mp3 content")

        fake_audio = MagicMock()
        fake_audio.__len__ = MagicMock(return_value=2000)

        original_prepared = _mod.PREPARED
        _mod.PREPARED = tmp_path
        try:
            with patch("tts_audio.AudioSegment.from_mp3", return_value=fake_audio):
                segments = [{"id": 0}]
                result = get_effective_seg_durations(segments, "sample", "hi")
        finally:
            _mod.PREPARED = original_prepared

        assert result == {0: 2.0}

    def test_get_effective_mixed_presence(self, tmp_path: Path):
        """Only segments with effective files appear in the result."""
        seg_dir = tmp_path / "hi_segments_sample"
        seg_dir.mkdir()
        (seg_dir / "seg_001_m_effective.mp3").write_bytes(b"fake")

        fake_audio = MagicMock()
        fake_audio.__len__ = MagicMock(return_value=3500)

        original_prepared = _mod.PREPARED
        _mod.PREPARED = tmp_path
        try:
            with patch("tts_audio.AudioSegment.from_mp3", return_value=fake_audio):
                segments = [{"id": 0}, {"id": 1}, {"id": 2}]
                result = get_effective_seg_durations(segments, "sample", "hi")
        finally:
            _mod.PREPARED = original_prepared

        assert 0 not in result
        assert result[1] == pytest.approx(3.5)
        assert 2 not in result

    def test_get_effective_seg_dir_missing(self, tmp_path: Path):
        """Non-existent seg_dir → returns empty dict without error."""
        original_prepared = _mod.PREPARED
        _mod.PREPARED = tmp_path
        try:
            segments = [{"id": 0}]
            result = get_effective_seg_durations(segments, "nonexistent", "hi")
        finally:
            _mod.PREPARED = original_prepared

        assert result == {}
