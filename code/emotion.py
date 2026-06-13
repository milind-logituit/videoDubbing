"""Stage 2.5 — Per-segment emotion classification using text-based DistilRoBERTa.

Text-based SER avoids the vocal-projection false-positive that audio models produce
on broadcast speech (energetic delivery ≠ angry). Runs on the ASR transcript.
"""
from pathlib import Path

_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# Model outputs: anger, disgust, fear, joy, neutral, sadness, surprise
_LABEL_MAP: dict[str, str] = {
    "anger":    "angry",
    "disgust":  "disgust",
    "fear":     "fearful",
    "joy":      "happy",
    "neutral":  "neutral",
    "sadness":  "sad",
    "surprise": "surprised",
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

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("text-classification", model=_MODEL_ID, device="cpu")
    return _pipeline


def classify_segment_emotions(audio_path: Path,
                               segments: list[dict]) -> list[dict]:
    """Add 'emotion' and 'emotion_score' to each segment using ASR text."""
    print(f"  Loading emotion model ({_MODEL_ID}) …")
    pipe = _get_pipeline()
    out  = []
    for seg in segments:
        text = seg.get("en_text", "").strip()
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

    Uses the same text-based model on source and target text so consistency
    reflects whether the translation preserved emotional register.
    dubbed_audio and original_audio args are accepted for API compatibility
    but not used (text-based approach).
    """
    pipe = _get_pipeline()
    records, matches = [], 0

    for seg in segments:
        src_text = str(seg.get("en_text") or "").strip()
        tgt_text = str(seg.get("hi_text") or "").strip()
        if not src_text or not tgt_text:
            continue

        def _classify(text: str) -> tuple[str, float]:
            try:
                pred = pipe(text, truncation=True, max_length=512)[0]
                return _LABEL_MAP.get(pred["label"], "neutral"), round(float(pred["score"]), 3)
            except Exception:
                return "neutral", 1.0

        src_emo, src_score = _classify(src_text)
        dub_emo, dub_score = _classify(tgt_text)
        match = src_emo == dub_emo
        if match:
            matches += 1
        records.append({
            "id":             seg["id"],
            "source_emotion": src_emo,
            "source_score":   src_score,
            "dubbed_emotion": dub_emo,
            "dubbed_score":   dub_score,
            "match":          match,
        })

    n = len(records)
    return {
        "match_pct":  round(matches / n * 100, 1) if n else None,
        "n_segments": n,
        "segments":   records,
    }
