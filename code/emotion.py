"""Stage 2.5 — Per-segment emotion classification using text-based DistilRoBERTa.

Text-based SER avoids the vocal-projection false-positive that audio models produce
on broadcast/film speech (energetic delivery ≠ angry). superb/wav2vec2-base-superb-er
was evaluated and rejected: trained on acted speech (IEMOCAP), it misfires on broadcast
content and degrades emotion_register across all test clips. Revisit when a model
trained on film/broadcast audio is available.

Audio SER helpers (_classify_audio_segment, _fuse_emotion_dists, _compute_va) are
retained for score_tts_emotion_fidelity and future use.
"""
import math
import tempfile
from pathlib import Path

_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
_AUDIO_MODEL_ID = "superb/wav2vec2-base-superb-er"

# Model outputs: anger, disgust, fear, joy, neutral, sadness, surprise
_LABEL_MAP: dict[str, str] = {
    "anger":    "angry",
    "disgust":  "disgust",
    "fear":     "fearful",
    "joy":      "happy",
    "neutral":  "neutral",
    "sadness":  "sad",
    "surprise": "surprised",
    # superb/wav2vec2-base-superb-er labels
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
}

# Russell circumplex (valence, arousal) per emotion — used for soft similarity scoring
_VALENCE_AROUSAL: dict[str, tuple[float, float]] = {
    "neutral":   ( 0.00,  0.00),
    "happy":     ( 0.80,  0.60),
    "angry":     (-0.60,  0.80),
    "sad":       (-0.70, -0.50),
    "fearful":   (-0.50,  0.70),
    "disgust":   (-0.60,  0.20),
    "surprised": ( 0.10,  0.80),
    "calm":      ( 0.30, -0.40),
}

# SSML prosody per emotion — pitch, volume, and rate
# Rate is an independent per-segment modifier on top of global isochrony rate.
# Angry/fearful/surprised → faster (urgency). Sad → slower (weight). Disgust → slightly slow.
SSML_PROSODY: dict[str, dict[str, str]] = {
    "neutral":   {"pitch": "+0%",  "volume": "medium", "rate": "+0%"},
    "happy":     {"pitch": "+15%", "volume": "loud",   "rate": "+8%"},
    "angry":     {"pitch": "+8%",  "volume": "x-loud", "rate": "+12%"},
    "sad":       {"pitch": "-15%", "volume": "soft",   "rate": "-12%"},
    "fearful":   {"pitch": "+12%", "volume": "soft",   "rate": "+10%"},
    "disgust":   {"pitch": "-5%",  "volume": "medium", "rate": "-5%"},
    "surprised": {"pitch": "+22%", "volume": "loud",   "rate": "+10%"},
}

_PROSODY_CONFIG_PATH = Path(__file__).parent.parent / "config" / "emotion_prosody.yaml"

_text_pipeline  = None
_audio_pipeline = None


def load_prosody_config(lang: str = "hi") -> dict[str, dict[str, str]]:
    """Load SSML prosody params from config/emotion_prosody.yaml.

    Merges language-specific overrides on top of 'default'. Falls back to
    the hardcoded SSML_PROSODY dict if the file is missing or unreadable.
    """
    if not _PROSODY_CONFIG_PATH.exists():
        return SSML_PROSODY
    try:
        import yaml
        raw      = yaml.safe_load(_PROSODY_CONFIG_PATH.read_text())
        base     = {k: dict(v) for k, v in raw.get("default", {}).items()}
        for emo, params in raw.get(lang, {}).items():
            base.setdefault(emo, {}).update(params)
        return base or SSML_PROSODY
    except Exception as exc:
        print(f"  [warn] Could not load emotion_prosody.yaml: {exc}; using defaults")
        return SSML_PROSODY


def _get_text_pipeline():
    global _text_pipeline
    if _text_pipeline is None:
        from transformers import pipeline
        _text_pipeline = pipeline("text-classification", model=_MODEL_ID, device="cpu")
    return _text_pipeline


def _get_audio_pipeline():
    global _audio_pipeline
    if _audio_pipeline is None:
        from transformers import pipeline
        _audio_pipeline = pipeline(
            "audio-classification", model=_AUDIO_MODEL_ID, device="cpu"
        )
    return _audio_pipeline


_TEXT_WEIGHT  = 0.7
_AUDIO_WEIGHT = 0.3
_MIN_AUDIO_DURATION_S = 0.5   # audio SER unreliable below this
_TEXT_CONFIDENCE_GATE = 0.6   # only fuse audio when text top-1 score is below this


def _classify_audio_segment(
    audio,   # pydub.AudioSegment, pre-loaded at 16kHz mono
    start_s: float,
    end_s: float,
    pipe,
) -> dict[str, float]:
    """Slice source audio and run audio SER. Returns emotion→prob dict or {}."""
    import os
    start_ms = int(start_s * 1000)
    end_ms   = int(end_s   * 1000)
    chunk    = audio[start_ms:end_ms]
    if len(chunk) < int(_MIN_AUDIO_DURATION_S * 1000):
        return {}
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            chunk.export(tmp_path, format="wav")
        preds = pipe(tmp_path)
        return {_LABEL_MAP.get(p["label"], "neutral"): float(p["score"]) for p in preds}
    except Exception:
        return {}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _fuse_emotion_dists(
    text_dist: dict[str, float],
    audio_dist: dict[str, float],
) -> dict[str, float]:
    """Merge text (7-class) and audio (4-class) probability distributions.

    Audio covers {angry, happy, neutral, sad}. For shared emotions, fuse
    at _TEXT_WEIGHT/_AUDIO_WEIGHT. For text-only emotions (disgust, fearful,
    surprised), keep text contribution and renormalize so all probs sum to 1.
    """
    all_emotions = set(text_dist) | set(audio_dist)
    fused: dict[str, float] = {}
    for emo in all_emotions:
        t = text_dist.get(emo, 0.0)
        a = audio_dist.get(emo, 0.0)
        fused[emo] = t * _TEXT_WEIGHT + a * _AUDIO_WEIGHT
    total = sum(fused.values())
    if total > 0:
        fused = {k: round(v / total, 3) for k, v in fused.items()}
    return fused


def _compute_va(dist: dict[str, float]) -> tuple[float, float]:
    """Weighted average valence and arousal from an emotion probability distribution."""
    valence = sum(dist.get(emo, 0.0) * _VALENCE_AROUSAL.get(emo, (0.0, 0.0))[0]
                  for emo in dist)
    arousal = sum(dist.get(emo, 0.0) * _VALENCE_AROUSAL.get(emo, (0.0, 0.0))[1]
                  for emo in dist)
    return round(valence, 3), round(arousal, 3)


def _va_similarity(emo1: str, emo2: str) -> float:
    """Euclidean distance in valence/arousal space, normalised to [0, 1].

    Same emotion → 1.0. Opposites (happy vs sad) → ~0.07.
    Nearby emotions (neutral vs sad) → ~0.6.
    """
    va1 = _VALENCE_AROUSAL.get(emo1, (0.0, 0.0))
    va2 = _VALENCE_AROUSAL.get(emo2, (0.0, 0.0))
    dist = math.sqrt((va1[0] - va2[0]) ** 2 + (va1[1] - va2[1]) ** 2)
    return round(max(0.0, 1.0 - dist / 2.0), 3)


def classify_segment_emotions(audio_path: Path,
                               segments: list[dict]) -> list[dict]:
    """Add emotion fields to each segment using text SER (j-hartmann DistilRoBERTa).

    Adds: emotion, emotion_score, emotion_dist, arousal, valence.
    arousal/valence are derived from the Russell circumplex weighted by emotion_dist.
    """
    print(f"  Loading emotion model ({_MODEL_ID}) …")
    pipe = _get_text_pipeline()
    out: list[dict] = []
    for seg in segments:
        text = str(seg.get("en_text") or "").strip()
        if not text:
            out.append({**seg, "emotion": "neutral", "emotion_score": 1.0,
                        "emotion_dist": {"neutral": 1.0}, "arousal": 0.0, "valence": 0.0})
            continue
        try:
            preds = pipe(text, truncation=True, max_length=512, top_k=None)
            top   = max(preds, key=lambda x: x["score"])
            emotion = _LABEL_MAP.get(top["label"], "neutral")
            score   = round(float(top["score"]), 3)
            dist    = {
                _LABEL_MAP.get(p["label"], "neutral"): round(float(p["score"]), 3)
                for p in preds if p["score"] >= 0.05
            }
        except Exception:
            emotion, score, dist = "neutral", 1.0, {"neutral": 1.0}
        valence, arousal = _compute_va(dist)
        out.append({**seg, "emotion": emotion, "emotion_score": score,
                    "emotion_dist": dist, "arousal": arousal, "valence": valence})
    return out


def _back_translate(text: str, source_lang: str, target_lang: str = "en") -> str:
    """Translate text back to English for emotion classification.

    j-hartmann is English-only — running it directly on Hindi gives unreliable
    results. Back-translating first gives the model text it was trained on.
    """
    if source_lang == target_lang:
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text) or text
    except Exception:
        return text  # on failure return original; classifier degrades gracefully


def score_emotion_consistency(original_audio: Path,
                               dubbed_audio: Path,
                               segments: list[dict],
                               target_lang: str = "hi") -> dict:
    """Compare source emotion (en_text) vs dubbed emotion (hi_text back-translated to EN).

    Back-translates hi_text → English before classifying so j-hartmann operates on
    the language it was trained on. Returns binary match_pct AND avg_soft_score.
    original_audio / dubbed_audio accepted for API compatibility but unused.
    """
    pipe = _get_text_pipeline()
    records, matches, soft_total = [], 0, 0.0

    def _classify(text: str) -> tuple[str, float]:
        try:
            pred = pipe(text, truncation=True, max_length=512)[0]
            return _LABEL_MAP.get(pred["label"], "neutral"), round(float(pred["score"]), 3)
        except Exception:
            return "neutral", 1.0

    for seg in segments:
        src_text = str(seg.get("en_text") or "").strip()
        tgt_text = str(seg.get("hi_text") or "").strip()
        if not src_text or not tgt_text:
            continue

        tgt_en = _back_translate(tgt_text, source_lang=target_lang, target_lang="en")

        src_emo, src_score = _classify(src_text)
        dub_emo, dub_score = _classify(tgt_en)
        match      = src_emo == dub_emo
        similarity = _va_similarity(src_emo, dub_emo)
        if match:
            matches += 1
        soft_total += similarity
        records.append({
            "id":             seg["id"],
            "source_emotion": src_emo,
            "source_score":   src_score,
            "dubbed_emotion": dub_emo,
            "dubbed_score":   dub_score,
            "dubbed_back_en": tgt_en,
            "match":          match,
            "va_similarity":  similarity,
        })

    n = len(records)
    return {
        "match_pct":       round(matches / n * 100, 1) if n else None,
        "avg_soft_score":  round(soft_total / n * 100, 1) if n else None,
        "n_segments":      n,
        "segments":        records,
    }


def score_tts_emotion_fidelity(dubbed_audio_path: Path,
                                segments: list[dict]) -> dict:
    """Check whether the synthesised dubbed audio SOUNDS as intended.

    Slices the dubbed mp3 by segment timestamps, runs wav2vec2 audio SER on each
    slice, compares perceived audio emotion against the intended emotion stored in
    segments[*]["emotion"]. Returns avg_soft_score in [0, 100].
    """
    if not dubbed_audio_path.exists():
        return {"note": f"dubbed audio not found: {dubbed_audio_path.name}"}

    try:
        from pydub import AudioSegment as _AS
    except ImportError:
        return {"note": "pydub not installed"}

    print(f"  Loading audio emotion model ({_AUDIO_MODEL_ID}) …")
    try:
        audio_pipe = _get_audio_pipeline()
    except Exception as exc:
        return {"note": f"audio model load failed: {exc}"}

    try:
        audio = _AS.from_file(str(dubbed_audio_path)).set_channels(1).set_frame_rate(16000)
    except Exception as exc:
        return {"note": f"audio load failed: {exc}"}

    records, soft_total, n = [], 0.0, 0
    for seg in segments:
        intended = seg.get("emotion", "neutral")
        start_ms  = int(float(seg.get("start", 0)) * 1000)
        end_ms    = int(float(seg.get("end",   0)) * 1000)
        if end_ms <= start_ms:
            continue
        chunk = audio[start_ms:end_ms]
        if len(chunk) < 100:   # skip sub-100ms slivers
            continue
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                chunk.export(tmp.name, format="wav")
                preds = audio_pipe(tmp.name)
            top   = max(preds, key=lambda x: x["score"])
            perceived = _LABEL_MAP.get(top["label"], "neutral")
            similarity = _va_similarity(intended, perceived)
        except Exception:
            perceived, similarity = "neutral", _va_similarity(intended, "neutral")

        soft_total += similarity
        n          += 1
        records.append({
            "id":        seg["id"],
            "intended":  intended,
            "perceived": perceived,
            "va_similarity": similarity,
        })

    return {
        "avg_soft_score": round(soft_total / n * 100, 1) if n else None,
        "n_segments":     n,
        "segments":       records,
    }
