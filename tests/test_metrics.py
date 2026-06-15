"""Tests for code/metrics.py — no GPU, no network, no real Claude calls."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module bootstrap — load metrics.py without touching sys.path
# ---------------------------------------------------------------------------
_METRICS_PATH = Path(__file__).parent.parent / "code" / "metrics.py"
_spec = importlib.util.spec_from_file_location("metrics", _METRICS_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["metrics"] = _mod
_spec.loader.exec_module(_mod)

_isochrony_fit            = _mod._isochrony_fit
compute_text_bleu         = _mod.compute_text_bleu
compute_segment_isochrony = _mod.compute_segment_isochrony
_merge_segment_quality    = _mod._merge_segment_quality
score_mos_rubric          = _mod.score_mos_rubric
grade_translations        = _mod.grade_translations


# ===========================================================================
# _isochrony_fit
# ===========================================================================

class TestIsochronyFit:
    def test_ratio_none_returns_3(self):
        assert _isochrony_fit(None) == 3

    def test_ratio_exactly_1_returns_5(self):
        assert _isochrony_fit(1.0) == 5

    def test_ratio_very_low_returns_3(self):
        # ratio <= 0.80 maps to 3 (under-run is "uncertain", not necessarily bad)
        assert _isochrony_fit(0.5) == 3

    def test_ratio_at_0_80_boundary_returns_3(self):
        assert _isochrony_fit(0.80) == 3

    def test_ratio_just_above_0_80_returns_5(self):
        # 0.81 falls in (0.80, 1.15] → 5
        assert _isochrony_fit(0.81) == 5

    def test_ratio_slightly_off_high_returns_5(self):
        # 0.95 is well within the (0.80, 1.15] band
        assert _isochrony_fit(0.95) == 5

    def test_ratio_at_1_15_boundary_returns_5(self):
        assert _isochrony_fit(1.15) == 5

    def test_ratio_just_above_1_15_returns_4(self):
        assert _isochrony_fit(1.16) == 4

    def test_ratio_at_1_30_boundary_returns_4(self):
        assert _isochrony_fit(1.30) == 4

    def test_ratio_just_above_1_30_returns_3(self):
        assert _isochrony_fit(1.31) == 3

    def test_ratio_at_1_50_boundary_returns_3(self):
        assert _isochrony_fit(1.50) == 3

    def test_ratio_just_above_1_50_returns_2(self):
        assert _isochrony_fit(1.51) == 2

    def test_ratio_at_2_00_boundary_returns_2(self):
        assert _isochrony_fit(2.00) == 2

    def test_ratio_very_high_returns_1(self):
        assert _isochrony_fit(2.01) == 1
        assert _isochrony_fit(5.0) == 1


# ===========================================================================
# compute_text_bleu
# ===========================================================================

class TestComputeTextBleu:
    """compute_text_bleu calls GoogleTranslator — mock it out."""

    def _make_segments(self, texts: list[str]) -> list[dict]:
        return [{"id": i + 1, "hi_text": t} for i, t in enumerate(texts)]

    def test_text_bleu_perfect(self):
        """Back-translation matches reference exactly → BLEU ~100."""
        reference = "the quick brown fox"
        segments  = self._make_segments(["कुछ हिंदी"])

        with patch.object(_mod, "GoogleTranslator") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.translate.return_value = reference
            mock_cls.return_value = mock_inst

            result = compute_text_bleu(segments, reference)

        assert result["bleu"] is not None
        assert result["bleu"] > 90
        assert result["n_segments"] == 1

    def test_text_bleu_partial(self):
        """Partial match produces a BLEU score between 0 and 100."""
        reference = "the quick brown fox jumps over the lazy dog"
        segments  = self._make_segments(["कुछ हिंदी"])

        with patch.object(_mod, "GoogleTranslator") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.translate.return_value = "the quick brown fox"
            mock_cls.return_value = mock_inst

            result = compute_text_bleu(segments, reference)

        assert result["bleu"] is not None
        assert 0 < result["bleu"] < 100

    def test_text_bleu_empty_segments(self):
        """No segments → function still returns a dict; bleu may be None or 0."""
        with patch.object(_mod, "GoogleTranslator") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.translate.return_value = ""
            mock_cls.return_value = mock_inst

            result = compute_text_bleu([], "some reference")

        assert isinstance(result, dict)
        # sacrebleu on empty hypothesis returns 0.0 or function may short-circuit
        assert "bleu" in result

    def test_text_bleu_no_reference_empty_string(self):
        """Empty reference string — should not crash."""
        segments = self._make_segments(["कुछ"])

        with patch.object(_mod, "GoogleTranslator") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.translate.return_value = "something"
            mock_cls.return_value = mock_inst

            result = compute_text_bleu(segments, "")

        assert isinstance(result, dict)

    def test_text_bleu_translator_raises(self):
        """If GoogleTranslator raises, function returns dict with bleu=None."""
        segments = self._make_segments(["कुछ"])

        with patch.object(_mod, "GoogleTranslator") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.translate.side_effect = RuntimeError("network error")
            mock_cls.return_value = mock_inst

            result = compute_text_bleu(segments, "some reference")

        assert result.get("bleu") is None
        assert "note" in result


# ===========================================================================
# compute_segment_isochrony
# ===========================================================================

class TestComputeSegmentIsochrony:
    """Mock AudioSegment.from_mp3 to control TTS duration."""

    def _make_seg(self, sid: int, start: float, end: float) -> dict:
        return {"id": sid, "start": start, "end": end}

    def _mock_audio(self, duration_ms: int):
        """Return a MagicMock whose len() equals duration_ms."""
        audio = MagicMock()
        audio.__len__ = MagicMock(return_value=duration_ms)
        return audio

    def test_isochrony_exact_fit(self, tmp_path: Path):
        """TTS == segment window → ratio 1.0."""
        seg = self._make_seg(1, start=0.0, end=5.0)
        # create a fake effective mp3 file
        mp3 = tmp_path / "seg_001_voice_effective.mp3"
        mp3.write_bytes(b"")

        with patch.object(_mod.AudioSegment, "from_mp3",
                          return_value=self._mock_audio(5000)):
            results = compute_segment_isochrony([seg], tmp_path)

        assert len(results) == 1
        assert results[0]["isochrony_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_isochrony_tts_longer(self, tmp_path: Path):
        """TTS longer than segment window → ratio > 1.0."""
        seg = self._make_seg(1, start=0.0, end=4.0)
        mp3 = tmp_path / "seg_001_voice_effective.mp3"
        mp3.write_bytes(b"")

        with patch.object(_mod.AudioSegment, "from_mp3",
                          return_value=self._mock_audio(6000)):
            results = compute_segment_isochrony([seg], tmp_path)

        assert results[0]["isochrony_ratio"] == pytest.approx(1.5, abs=0.01)

    def test_isochrony_no_effective_file(self, tmp_path: Path):
        """No mp3 found → tts_duration_s and isochrony_ratio are None."""
        seg = self._make_seg(2, start=0.0, end=3.0)
        results = compute_segment_isochrony([seg], tmp_path)

        assert len(results) == 1
        assert results[0]["tts_duration_s"] is None
        assert results[0]["isochrony_ratio"] is None

    def test_isochrony_returns_list(self, tmp_path: Path):
        """Output is a list of dicts with required keys."""
        seg = self._make_seg(1, start=1.0, end=4.0)
        mp3 = tmp_path / "seg_001_voice_effective.mp3"
        mp3.write_bytes(b"")

        with patch.object(_mod.AudioSegment, "from_mp3",
                          return_value=self._mock_audio(3000)):
            results = compute_segment_isochrony([seg], tmp_path)

        assert isinstance(results, list)
        row = results[0]
        assert "id" in row
        assert "isochrony_ratio" in row
        assert row["id"] == 1

    def test_isochrony_fallback_to_non_effective_mp3(self, tmp_path: Path):
        """When no *_effective.mp3 exists, falls back to any seg_NNN_*.mp3."""
        seg = self._make_seg(3, start=0.0, end=2.0)
        mp3 = tmp_path / "seg_003_voice.mp3"
        mp3.write_bytes(b"")

        with patch.object(_mod.AudioSegment, "from_mp3",
                          return_value=self._mock_audio(2000)):
            results = compute_segment_isochrony([seg], tmp_path)

        assert results[0]["isochrony_ratio"] == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# _merge_segment_quality
# ===========================================================================

class TestMergeSegmentQuality:
    """_merge_segment_quality(isochrony: list[dict], grades_by_id: dict[int, dict])."""

    def _iso_row(self, sid: int, ratio: float | None = 1.0) -> dict:
        return {
            "id": sid,
            "start": 0.0,
            "end": 3.0,
            "en_duration_s": 3.0,
            "tts_duration_s": 3.0 if ratio is not None else None,
            "isochrony_ratio": ratio,
        }

    def test_merge_adds_grade_fields(self):
        iso    = [self._iso_row(1, ratio=1.0)]
        grades = {1: {"fidelity": 4, "fluency": 5, "emotion_register": 3}}

        merged = _merge_segment_quality(iso, grades)

        assert len(merged) == 1
        row = merged[0]
        assert row["fidelity"] == 4
        assert row["fluency"] == 5
        assert row["emotion_register"] == 3
        # isochrony_fit also computed
        assert row["fit"] == 5  # ratio=1.0 → score 5

    def test_merge_preserves_ungraded_segment(self):
        iso    = [self._iso_row(1), self._iso_row(2)]
        grades = {1: {"fidelity": 3, "fluency": 3}}

        merged = _merge_segment_quality(iso, grades)

        assert len(merged) == 2
        seg2 = next(r for r in merged if r["id"] == 2)
        assert "fidelity" not in seg2
        assert "fluency"  not in seg2

    def test_merge_empty_grades(self):
        iso = [self._iso_row(1), self._iso_row(2)]
        merged = _merge_segment_quality(iso, {})

        assert len(merged) == 2
        for row in merged:
            assert "fidelity" not in row
            assert "fluency"  not in row

    def test_merge_fit_score_computed_correctly(self):
        iso = [self._iso_row(1, ratio=1.4)]  # 1.30 < 1.4 <= 1.50 → fit=3
        merged = _merge_segment_quality(iso, {})
        assert merged[0]["fit"] == 3

    def test_merge_note_propagated(self):
        iso    = [self._iso_row(1)]
        grades = {1: {"fidelity": 2, "fluency": 2, "note": "poor word choice"}}

        merged = _merge_segment_quality(iso, grades)
        assert merged[0]["note"] == "poor word choice"

    def test_merge_no_emotion_register_when_absent(self):
        """emotion_register must NOT be added when not in grade dict."""
        iso    = [self._iso_row(1)]
        grades = {1: {"fidelity": 5, "fluency": 5}}  # no emotion_register

        merged = _merge_segment_quality(iso, grades)
        assert "emotion_register" not in merged[0]


# ===========================================================================
# grade_translations  (mock Claude)
# ===========================================================================

class TestGradeTranslations:
    def _seg(self, sid: int, en: str, hi: str,
             start: float = 0.0, end: float = 3.0) -> dict:
        return {"id": sid, "en_text": en, "hi_text": hi,
                "start": start, "end": end}

    def _make_client_mock(self, response_text: str) -> MagicMock:
        content_block = MagicMock()
        content_block.text = response_text
        msg = MagicMock()
        msg.content = [content_block]
        client = MagicMock()
        client.messages.create.return_value = msg
        return client

    def test_grade_translations_empty_segments(self):
        """Empty list → returns [] immediately without calling Claude."""
        with patch.object(_mod.anthropic, "Anthropic") as mock_cls:
            result = grade_translations([])

        mock_cls.assert_not_called()
        assert result == []

    def test_grade_translations_segments_with_no_hi_text_filtered(self):
        """Segments with blank hi_text are silently dropped → returns []."""
        segs = [self._seg(1, "hello", "")]
        with patch.object(_mod.anthropic, "Anthropic") as mock_cls:
            result = grade_translations(segs)

        mock_cls.assert_not_called()
        assert result == []

    def test_grade_translations_returns_grades(self):
        """Happy path: Claude returns valid JSON grades."""
        segs = [
            self._seg(1, "Hello world", "नमस्ते दुनिया"),
            self._seg(2, "Good night",  "शुभ रात्रि"),
        ]
        fake_grades = [
            {"id": 1, "fidelity": 5, "fluency": 4},
            {"id": 2, "fidelity": 4, "fluency": 5},
        ]
        client_mock = self._make_client_mock(json.dumps(fake_grades))

        with patch.object(_mod.anthropic, "Anthropic", return_value=client_mock):
            result = grade_translations(segs)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["fidelity"] == 5
        assert result[1]["fluency"]  == 5

    def test_grade_translations_handles_markdown_fence(self):
        """Response wrapped in ```json ... ``` is unwrapped correctly."""
        segs = [self._seg(1, "Hi", "नमस्ते")]
        fake_grades = [{"id": 1, "fidelity": 5, "fluency": 5}]
        wrapped = f"```json\n{json.dumps(fake_grades)}\n```"
        client_mock = self._make_client_mock(wrapped)

        with patch.object(_mod.anthropic, "Anthropic", return_value=client_mock):
            result = grade_translations(segs)

        assert result[0]["id"] == 1

    def test_grade_translations_includes_emotion_register(self):
        """emotion field triggers emotion_register in the payload; grade returned."""
        segs = [self._seg(1, "Run!", "भागो!", start=0.0, end=1.0)]
        segs[0]["emotion"] = "fear"
        fake_grades = [{"id": 1, "fidelity": 5, "fluency": 5, "emotion_register": 4}]
        client_mock = self._make_client_mock(json.dumps(fake_grades))

        with patch.object(_mod.anthropic, "Anthropic", return_value=client_mock):
            result = grade_translations(segs)

        assert result[0]["emotion_register"] == 4


# ===========================================================================
# score_mos_rubric  (mock Claude)
# ===========================================================================

class TestScoreMosRubric:
    def _seg(self, sid: int, start: float = 0.0, end: float = 4.0) -> dict:
        return {
            "id": sid,
            "en_text": f"segment {sid}",
            "hi_text": f"सेगमेंट {sid}",
            "start": start,
            "end": end,
        }

    def _all_5_rating(self, sid: int) -> dict:
        return {"id": sid, "naturalness": 5, "fidelity": 5,
                "timing": 5, "emotion": 5, "names": 5}

    def _make_client_mock(self, ratings: list[dict]) -> MagicMock:
        content_block = MagicMock()
        content_block.text = json.dumps(ratings)
        msg = MagicMock()
        msg.content = [content_block]
        client = MagicMock()
        client.messages.create.return_value = msg
        return client

    def test_mos_rubric_empty_segments(self):
        """No segments → returns dict with mos=None and a note."""
        result = score_mos_rubric([])
        assert result["mos"] is None
        assert "note" in result

    def test_mos_rubric_segments_no_hi_text(self):
        """Segments with blank hi_text are filtered → mos=None."""
        segs = [{"id": 1, "en_text": "hi", "hi_text": "",
                  "start": 0.0, "end": 2.0}]
        result = score_mos_rubric(segs)
        assert result["mos"] is None

    def test_mos_rubric_returns_score(self):
        """All-5 ratings → mos == 100.0."""
        segs = [self._seg(i) for i in range(1, 4)]
        ratings = [self._all_5_rating(s["id"]) for s in segs]
        client_mock = self._make_client_mock(ratings)

        with patch.object(_mod.anthropic, "Anthropic", return_value=client_mock):
            result = score_mos_rubric(segs)

        assert result["mos"] == pytest.approx(100.0, abs=0.1)
        assert 0 <= result["mos"] <= 100
        assert result["n_sampled"] == 3
        assert "breakdown" in result
        assert "per_segment" in result

    def test_mos_rubric_score_in_range(self):
        """Mixed ratings → mos is in [0, 100]."""
        segs = [self._seg(i) for i in range(1, 6)]
        ratings = [
            {"id": 1, "naturalness": 3, "fidelity": 4, "timing": 3, "emotion": 2, "names": 5},
            {"id": 2, "naturalness": 4, "fidelity": 3, "timing": 4, "emotion": 3, "names": 4},
            {"id": 3, "naturalness": 2, "fidelity": 2, "timing": 3, "emotion": 2, "names": 3},
            {"id": 4, "naturalness": 5, "fidelity": 5, "timing": 5, "emotion": 4, "names": 5},
            {"id": 5, "naturalness": 1, "fidelity": 2, "timing": 1, "emotion": 1, "names": 2},
        ]
        client_mock = self._make_client_mock(ratings)

        with patch.object(_mod.anthropic, "Anthropic", return_value=client_mock):
            result = score_mos_rubric(segs)

        assert result["mos"] is not None
        assert 0 <= result["mos"] <= 100

    def test_mos_rubric_samples_up_to_n(self):
        """When there are more segments than n_sample, only n_sample are scored."""
        n = 5
        segs = [self._seg(i) for i in range(1, 21)]  # 20 segments
        captured_payloads: list = []

        def fake_create(**kwargs):
            payload = json.loads(kwargs["messages"][0]["content"])
            captured_payloads.append(payload)
            ratings = [self._all_5_rating(item["id"]) for item in payload]
            content_block = MagicMock()
            content_block.text = json.dumps(ratings)
            msg = MagicMock()
            msg.content = [content_block]
            return msg

        client_mock = MagicMock()
        client_mock.messages.create.side_effect = fake_create

        with patch.object(_mod.anthropic, "Anthropic", return_value=client_mock):
            result = score_mos_rubric(segs, n_sample=n)

        assert result["n_sampled"] <= n
        assert len(captured_payloads[0]) <= n

    def test_mos_rubric_breakdown_has_all_dims(self):
        """Breakdown dict contains all 5 MOS dimensions."""
        segs = [self._seg(1)]
        ratings = [self._all_5_rating(1)]
        client_mock = self._make_client_mock(ratings)

        with patch.object(_mod.anthropic, "Anthropic", return_value=client_mock):
            result = score_mos_rubric(segs)

        dims = {"naturalness", "fidelity", "timing", "emotion", "names"}
        assert dims == set(result["breakdown"].keys())

    def test_mos_rubric_claude_error_returns_none(self):
        """If Claude call raises, returns dict with mos=None."""
        segs = [self._seg(1)]
        client_mock = MagicMock()
        client_mock.messages.create.side_effect = RuntimeError("API down")

        with patch.object(_mod.anthropic, "Anthropic", return_value=client_mock):
            result = score_mos_rubric(segs)

        assert result["mos"] is None
        assert "note" in result

    def test_mos_rubric_glossary_appended_to_system(self):
        """Glossary terms are added to the system prompt."""
        segs = [self._seg(1)]
        ratings = [self._all_5_rating(1)]
        captured_calls: list = []

        def fake_create(**kwargs):
            captured_calls.append(kwargs)
            content_block = MagicMock()
            content_block.text = json.dumps(ratings)
            msg = MagicMock()
            msg.content = [content_block]
            return msg

        client_mock = MagicMock()
        client_mock.messages.create.side_effect = fake_create

        with patch.object(_mod.anthropic, "Anthropic", return_value=client_mock):
            score_mos_rubric(segs, glossary={"Netflix": "नेटफ्लिक्स"})

        system_used = captured_calls[0]["system"]
        assert "Netflix" in system_used
        assert "नेटफ्लिक्स" in system_used
