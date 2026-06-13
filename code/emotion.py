"""Stage 2.5 — Per-segment emotion classification using text-based DistilRoBERTa.

Text-based SER avoids the vocal-projection false-positive that audio models produce
on broadcast speech (energetic delivery ≠ angry). Runs on the ASR transcript.
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

# SSML prosody per emotion — pitch and volume only; rate handled by isochrony
SSML_PROSODY: dict[str, dict[str, str]] = {
    "neutral":   {"pitch": "+0%",  "volume": "medium"},
    "happy":     {"pitch": "+15%", "volume": "loud"},
    "angry":     {"pitch": "+5%",  "volume": "x-loud"},
    "sad":       {"pitch": "-12%", "volume": "soft"},
    "fearful":   {"pitch": "+10%", "volume": "soft"},
    "disgust":   {"pitch": "-5%",  "volume": "medium"},
    "surprised": {"pitch": "+20%", "volume": "loud"},
}

_text_pipeline  = None
_audio_pipeline = None


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
    """Add 'emotion' and 'emotion_score' to each segment using ASR text."""
    print(f"  Loading emotion model ({_MODEL_ID}) …")
    pipe = _get_text_pipeline()
    out  = []
    for seg in segments:
        text = str(seg.get("en_text") or "").strip()
        if not text:
            out.append({**seg, "emotion": "neutral", "emotion_score": 1.0})
            continue
        try:
            pred    = pipe(text, truncation=True, max_length=512)[0]
            emotion = _LABEL_MAP.get(pred["label"], "neutral")
            score   = round(float(pred["score"]), 3)
        except Exception:
            emotion, score = "neutral", 1.0
        out.append({**seg, "emotion": emotion, "emotion_score": score})
    return out


def score_emotion_consistency(original_audio: Path,
                               dubbed_audio: Path,
                               segments: list[dict]) -> dict:
    """Compare source emotion (from en_text) vs dubbed emotion (from hi_text).

    Returns binary match_pct AND avg_soft_score (valence/arousal similarity).
    dubbed_audio and original_audio args accepted for API compatibility but unused.
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

        src_emo, src_score = _classify(src_text)
        dub_emo, dub_score = _classify(tgt_text)
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
