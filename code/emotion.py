"""Stage 2.5 — Per-segment emotion classification using wav2vec2 SER."""
from pathlib import Path

import numpy as np
from pydub import AudioSegment

_MODEL_ID    = "superb/wav2vec2-base-superb-er"
_SAMPLE_RATE = 16000
_MIN_DUR_S   = 0.5

_LABEL_MAP = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}

# SSML prosody per emotion — pitch and volume only; rate handled by isochrony
SSML_PROSODY: dict[str, dict[str, str]] = {
    "neutral": {"pitch": "+0%",  "volume": "medium"},
    "happy":   {"pitch": "+15%", "volume": "loud"},
    "angry":   {"pitch": "+5%",  "volume": "x-loud"},
    "sad":     {"pitch": "-12%", "volume": "soft"},
}

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("audio-classification", model=_MODEL_ID, device="cpu")
    return _pipeline


def _load_mono_f32(audio_path: Path) -> np.ndarray:
    audio = (
        AudioSegment.from_file(str(audio_path))
        .set_frame_rate(_SAMPLE_RATE)
        .set_channels(1)
    )
    return np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0


def classify_segment_emotions(audio_path: Path,
                               segments: list[dict]) -> list[dict]:
    """Add 'emotion' and 'emotion_score' fields to each segment dict."""
    print(f"  Loading SER model ({_MODEL_ID}) …")
    pipe   = _get_pipeline()
    samples = _load_mono_f32(audio_path)
    out    = []
    for seg in segments:
        dur = float(seg["end"]) - float(seg["start"])
        if dur < _MIN_DUR_S:
            out.append({**seg, "emotion": "neutral", "emotion_score": 1.0})
            continue
        start_i = int(float(seg["start"]) * _SAMPLE_RATE)
        end_i   = int(float(seg["end"])   * _SAMPLE_RATE)
        chunk   = samples[start_i:end_i]
        if len(chunk) < int(_SAMPLE_RATE * _MIN_DUR_S):
            out.append({**seg, "emotion": "neutral", "emotion_score": 1.0})
            continue
        try:
            preds   = pipe({"array": chunk, "sampling_rate": _SAMPLE_RATE})
            top     = preds[0]
            emotion = _LABEL_MAP.get(top["label"], "neutral")
            score   = round(float(top["score"]), 3)
        except Exception:
            emotion, score = "neutral", 1.0
        out.append({**seg, "emotion": emotion, "emotion_score": score})
    return out


def score_emotion_consistency(original_audio: Path,
                               dubbed_audio: Path,
                               segments: list[dict]) -> dict:
    """Compare SER labels on original vs dubbed audio per segment."""
    pipe     = _get_pipeline()
    orig_s   = _load_mono_f32(original_audio)
    dubbed_s = _load_mono_f32(dubbed_audio)

    records, matches = [], 0
    for seg in segments:
        if not seg.get("emotion"):
            continue
        start_i = int(float(seg["start"]) * _SAMPLE_RATE)
        end_i   = int(float(seg["end"])   * _SAMPLE_RATE)
        src_chunk = orig_s[start_i:end_i]
        dub_chunk = dubbed_s[start_i:end_i]

        def _classify(chunk):
            if len(chunk) < int(_SAMPLE_RATE * _MIN_DUR_S):
                return "neutral", 1.0
            try:
                preds = pipe({"array": chunk, "sampling_rate": _SAMPLE_RATE})
                top   = preds[0]
                return _LABEL_MAP.get(top["label"], "neutral"), round(float(top["score"]), 3)
            except Exception:
                return "neutral", 1.0

        src_emo,  src_score  = _classify(src_chunk)
        dub_emo,  dub_score  = _classify(dub_chunk)
        match = src_emo == dub_emo
        if match:
            matches += 1
        records.append({
            "id": seg["id"],
            "source_emotion":  src_emo,
            "source_score":    src_score,
            "dubbed_emotion":  dub_emo,
            "dubbed_score":    dub_score,
            "match":           match,
        })

    n = len(records)
    return {
        "match_pct":  round(matches / n * 100, 1) if n else None,
        "n_segments": n,
        "segments":   records,
    }
