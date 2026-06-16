"""Stage 2.5 — Per-segment emotion classification using text + audio SER fusion.

Text SER: j-hartmann/emotion-english-distilroberta-base (7-class categorical).
Audio SER: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (VA regression,
trained on MSP-PODCAST broadcast/film speech — avoids the IEMOCAP acted-speech
misfire that made superb/wav2vec2-base-superb-er degrade emotion_register).

Fusion: text 0.65 / audio 0.35 via shared Russell circumplex VA space.
classify_segment_emotions() fuses both modalities when audio_path is valid.
score_tts_emotion_fidelity() uses audeering VA regression (broadcast/film-trained).

Note: call smooth_emotion_arc() after Stage 2.5 (classify_segment_emotions) and
before Stage 4b to flag per-speaker one-off emotion outliers for dashboard highlighting.
"""
import math
import tempfile
from pathlib import Path

_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
_AUDIO_MODEL_ID = "superb/wav2vec2-base-superb-er"
_AUDEERING_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

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

_text_pipeline      = None
_audio_pipeline     = None
_audeering_model    = None
_audeering_processor = None

# ---------------------------------------------------------------------------
# Audeering custom model — must be at module level so from_pretrained works.
# Transformers 5.x removed all_tied_weights_keys from PreTrainedModel; we
# declare it explicitly on the subclass.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig as _AutoConfig
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model as _Wav2Vec2Model,
        Wav2Vec2PreTrainedModel as _Wav2Vec2PreTrainedModel,
    )

    class _RegressionHead(nn.Module):
        def __init__(self, config: _AutoConfig) -> None:
            super().__init__()
            self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout  = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            x = self.dropout(features)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            return self.out_proj(x)

    class _AudeeringEmotionModel(_Wav2Vec2PreTrainedModel):
        # transformers 5.x changed all_tied_weights_keys from list to dict;
        # mark_tied_weights_as_initialized calls .keys() on it.
        all_tied_weights_keys: dict[str, str] = {}
        _tied_weights_keys:    list[str]      = []

        def __init__(self, config: _AutoConfig) -> None:
            super().__init__(config)
            self.wav2vec2   = _Wav2Vec2Model(config)
            self.classifier = _RegressionHead(config)
            self.init_weights()

        def forward(self, input_values: "torch.Tensor") -> "torch.Tensor":
            hidden = self.wav2vec2(input_values).last_hidden_state
            hidden = torch.mean(hidden, dim=1)
            return self.classifier(hidden)

    _AUDEERING_MODEL_CLASS_AVAILABLE = True
except Exception:
    _AUDEERING_MODEL_CLASS_AVAILABLE = False


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


def _get_audeering_model():
    """Lazy-load audeering VA regression model (broadcast/film trained)."""
    global _audeering_model, _audeering_processor
    if _audeering_model is None:
        if not _AUDEERING_MODEL_CLASS_AVAILABLE:
            raise RuntimeError("audeering model class failed to initialise at import time")
        from transformers import Wav2Vec2Processor
        print(f"  Loading audio emotion model ({_AUDEERING_MODEL_ID}) …")
        _audeering_processor = Wav2Vec2Processor.from_pretrained(_AUDEERING_MODEL_ID)
        _audeering_model = _AudeeringEmotionModel.from_pretrained(_AUDEERING_MODEL_ID)
        _audeering_model.eval()
    return _audeering_model, _audeering_processor


def _va_to_dist(valence: float, arousal: float) -> dict[str, float]:
    """Convert continuous (valence, arousal) to a soft emotion probability distribution.

    Uses inverse squared distance from each emotion's Russell circumplex coordinates.
    Closest emotion gets the most probability mass.
    """
    weights: dict[str, float] = {}
    for emo, (v, a) in _VALENCE_AROUSAL.items():
        dist_sq = (valence - v) ** 2 + (arousal - a) ** 2
        weights[emo] = 1.0 / (dist_sq + 1e-6)
    total = sum(weights.values())
    return {emo: round(w / total, 3) for emo, w in weights.items()}


def _classify_audio_segment_audeering(
    audio,   # pydub.AudioSegment pre-loaded at 16 kHz mono
    start_s: float,
    end_s: float,
    model,
    processor,
) -> dict[str, float]:
    """Run audeering VA regression on an audio slice.

    Model outputs [arousal, dominance, valence] as continuous regression values
    (approx. [-1, 1]) via _AudeeringEmotionModel.forward(input_values).
    Values map directly into _VALENCE_AROUSAL [-1, 1] space — no sigmoid or
    rescaling needed.
    Returns {} on short segments or any runtime error.
    """
    import numpy as np
    import torch

    start_ms = int(start_s * 1000)
    end_ms   = int(end_s   * 1000)
    chunk    = audio[start_ms:end_ms]
    if len(chunk) < int(_MIN_AUDIO_DURATION_S * 1000):
        return {}
    try:
        samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / (2 ** 15)
        inputs  = processor(samples, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            values = model(inputs.input_values).squeeze().cpu().numpy()
        # audeering: index 0 = arousal, 1 = dominance, 2 = valence
        arousal = float(values[0])
        valence = float(values[2])
        return _va_to_dist(valence, arousal)
    except Exception:
        return {}


_TEXT_WEIGHT  = 0.7     # used by score_tts_emotion_fidelity / legacy superb path
_AUDIO_WEIGHT = 0.3
_MIN_AUDIO_DURATION_S  = 0.5   # audio SER unreliable below this
_TEXT_CONFIDENCE_GATE  = 0.6   # legacy confidence gate for superb model
_AUDEERING_TEXT_WEIGHT  = 0.65  # audeering is better — give audio slightly more weight
_AUDEERING_AUDIO_WEIGHT = 0.35


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
                               segments: list[dict],
                               use_audio_ser: bool = True) -> list[dict]:
    """Add emotion fields to each segment using text SER fused with audio SER.

    Text SER: j-hartmann/emotion-english-distilroberta-base (always on).
    Audio SER: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (fused at
    _AUDEERING_TEXT_WEIGHT/_AUDEERING_AUDIO_WEIGHT when audio_path exists and
    use_audio_ser=True).

    Adds: emotion, emotion_score, emotion_dist, arousal, valence.
    """
    print(f"  Loading text emotion model ({_MODEL_ID}) …")
    pipe = _get_text_pipeline()

    audio       = None
    aud_model   = None
    aud_proc    = None
    if use_audio_ser and audio_path.exists():
        try:
            from pydub import AudioSegment as _AS
            audio = _AS.from_file(str(audio_path)).set_channels(1).set_frame_rate(16_000)
            print(f"  Loading audio emotion model ({_AUDEERING_MODEL_ID}) …")
            aud_model, aud_proc = _get_audeering_model()
        except Exception as exc:
            print(f"  [warn] Audio SER setup failed: {exc}; using text-only")
            audio = None

    out: list[dict] = []
    for seg in segments:
        text = str(seg.get("en_text") or "").strip()
        if not text:
            out.append({**seg, "emotion": "neutral", "emotion_score": 1.0,
                        "emotion_dist": {"neutral": 1.0}, "arousal": 0.0, "valence": 0.0})
            continue
        try:
            preds   = pipe(text, truncation=True, max_length=512, top_k=None)
            top     = max(preds, key=lambda x: x["score"])
            emotion = _LABEL_MAP.get(top["label"], "neutral")
            score   = round(float(top["score"]), 3)
            dist: dict[str, float] = {
                _LABEL_MAP.get(p["label"], "neutral"): round(float(p["score"]), 3)
                for p in preds if p["score"] >= 0.05
            }
        except Exception:
            emotion, score, dist = "neutral", 1.0, {"neutral": 1.0}

        if audio is not None and aud_model is not None:
            audio_dist = _classify_audio_segment_audeering(
                audio, float(seg.get("start", 0)), float(seg.get("end", 0)),
                aud_model, aud_proc,
            )
            if audio_dist:
                all_emotions = set(dist) | set(audio_dist)
                fused: dict[str, float] = {}
                for emo in all_emotions:
                    fused[emo] = (dist.get(emo, 0.0) * _AUDEERING_TEXT_WEIGHT
                                  + audio_dist.get(emo, 0.0) * _AUDEERING_AUDIO_WEIGHT)
                total = sum(fused.values())
                if total > 0:
                    dist = {k: round(v / total, 3) for k, v in fused.items()}
                emotion = max(dist, key=lambda k: dist[k])
                score   = dist[emotion]

        valence, arousal = _compute_va(dist)
        out.append({**seg, "emotion": emotion, "emotion_score": score,
                    "emotion_dist": dist, "arousal": arousal, "valence": valence})
    return out


def smooth_emotion_arc(segments: list[dict], window: int = 5) -> list[dict]:
    """Flag per-speaker emotion outliers without overriding them.

    Groups segments by speaker, then within each group applies a centred majority-vote
    window to detect one-off emotions that differ from the surrounding context.
    Segments that are outliers get arc_flagged=True; all other fields are left untouched.
    """
    for seg in segments:
        seg.setdefault("arc_flagged", False)

    half = window // 2

    by_speaker: dict[str, list[int]] = {}
    for idx, seg in enumerate(segments):
        key = seg.get("speaker", "__anon__")
        by_speaker.setdefault(key, []).append(idx)

    for speaker_indices in by_speaker.values():
        # not enough context to make a reliable judgement
        if len(speaker_indices) < window:
            continue

        for pos in range(half, len(speaker_indices) - half):
            window_idxs = speaker_indices[pos - half: pos + half + 1]
            emotions = [segments[i]["emotion"] for i in window_idxs]
            centre_emotion = emotions[half]

            # majority vote; ties resolved arbitrarily — centre must be strictly dominant
            counts: dict[str, int] = {}
            for e in emotions:
                counts[e] = counts.get(e, 0) + 1
            dominant = max(counts, key=lambda e: counts[e])

            if centre_emotion != dominant:
                segments[speaker_indices[pos]]["arc_flagged"] = True

    return segments


def classify_face_emotions(video_path: Path, segments: list[dict]) -> list[dict]:
    """Detect per-segment face emotion from video frames using MediaPipe FaceMesh.

    For each segment, samples frames at ~2 fps, maps landmark geometry to a
    valence proxy (mouth curve) and arousal proxy (brow raise), then maps
    (valence, arousal) to the nearest Russell-circumplex emotion. Fuses with
    existing text SER emotion at 0.6/0.4 (text/face) weight in VA space.

    Skips silently if mediapipe is not installed or video cannot be opened.
    Requires: mediapipe, opencv-python
    """
    try:
        import cv2
        from mediapipe.python.solutions import face_mesh as mp_face  # 0.10+ moved solutions
    except ImportError:
        print("  [warn] mediapipe/cv2 not installed — skipping face emotion")
        return segments
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [warn] Could not open video for face emotion: {video_path}")
        return segments

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # ~2 fps gives enough signal without processing every frame
    sample_every = max(1, int(fps / 2))

    frames: list[tuple[float, object]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            frames.append((frame_idx / fps, frame))
        frame_idx += 1
    cap.release()

    print(f"  Face emotion: {len(frames)} sampled frames from {video_path.name}")

    _MOUTH_LEFT, _MOUTH_RIGHT = 61, 291
    _BROW_LEFT,  _BROW_RIGHT  = 70, 300
    _NOSE_TIP = 1

    def _frame_va(landmarks) -> tuple[float, float]:
        pts = {i: (landmarks[i].x, landmarks[i].y) for i in
               [_MOUTH_LEFT, _MOUTH_RIGHT, _BROW_LEFT, _BROW_RIGHT, _NOSE_TIP]}
        mouth_mid_y = (pts[_MOUTH_LEFT][1] + pts[_MOUTH_RIGHT][1]) / 2
        nose_y = pts[_NOSE_TIP][1]
        # mouth below nose → frown (negative valence); above → smile (positive)
        valence = max(-1.0, min(1.0, float(nose_y - mouth_mid_y) * 5.0))
        brow_y = (pts[_BROW_LEFT][1] + pts[_BROW_RIGHT][1]) / 2
        # raised brows (smaller y in image coords) relative to nose → high arousal
        arousal = max(-1.0, min(1.0, float(nose_y - brow_y) * 4.0))
        return valence, arousal

    def _va_to_emotion(valence: float, arousal: float) -> str:
        best, best_dist = "neutral", float("inf")
        for emo, (v, a) in _VALENCE_AROUSAL.items():
            d = (valence - v) ** 2 + (arousal - a) ** 2
            if d < best_dist:
                best_dist = d
                best = emo
        return best

    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                           refine_landmarks=False, min_detection_confidence=0.4) as fm:
        out = []
        for seg in segments:
            seg_frames = [(ts, f) for ts, f in frames
                          if seg["start"] <= ts < seg["end"]]
            if not seg_frames:
                out.append(seg)
                continue

            vas = []
            for _, frame in seg_frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    vas.append(_frame_va(lms))

            if not vas:
                out.append(seg)
                continue

            avg_v = sum(v for v, _ in vas) / len(vas)
            avg_a = sum(a for _, a in vas) / len(vas)
            face_emo = _va_to_emotion(avg_v, avg_a)

            text_emo = seg.get("emotion", "neutral")
            t_v, t_a = _VALENCE_AROUSAL.get(text_emo, (0.0, 0.0))
            fused_v = 0.6 * t_v + 0.4 * avg_v
            fused_a = 0.6 * t_a + 0.4 * avg_a
            fused = _va_to_emotion(fused_v, fused_a)

            out.append({**seg, "emotion": fused, "face_emotion": face_emo,
                        "face_valence": round(avg_v, 3), "face_arousal": round(avg_a, 3)})
        return out


def load_persona_map(path: Path) -> dict[str, dict[str, float]]:
    """Load optional per-speaker valence/arousal offsets from a JSON file.

    Expected format::

        {
          "SPEAKER_00": {"valence_offset": -0.2, "arousal_offset":  0.1},
          "SPEAKER_01": {"valence_offset":  0.3, "arousal_offset": -0.1}
        }

    Returns {} if the file does not exist or cannot be parsed.
    """
    if not path.exists():
        return {}
    try:
        import json
        raw = json.loads(path.read_text())
        validated: dict[str, dict[str, float]] = {}
        for speaker, offsets in raw.items():
            v = float(offsets.get("valence_offset", 0.0))
            a = float(offsets.get("arousal_offset", 0.0))
            validated[speaker] = {"valence_offset": v, "arousal_offset": a}
        return validated
    except Exception as exc:
        print(f"  [warn] Could not load persona_map {path.name}: {exc}")
        return {}


def apply_persona_offsets(segments: list[dict],
                           persona_map: dict[str, dict[str, float]]) -> list[dict]:
    """Shift each segment's (valence, arousal) by per-speaker offsets and re-derive emotion.

    Offsets are clamped so the result stays within [-1, 1]. The new emotion is the
    argmax of _va_to_dist() on the shifted coordinates. Adjusted segments gain
    persona_adjusted=True so the dashboard can surface them.
    """
    out: list[dict] = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        if not speaker or speaker not in persona_map:
            out.append(seg)
            continue
        offsets  = persona_map[speaker]
        new_v    = max(-1.0, min(1.0, float(seg.get("valence", 0.0))
                                  + offsets["valence_offset"]))
        new_a    = max(-1.0, min(1.0, float(seg.get("arousal", 0.0))
                                  + offsets["arousal_offset"]))
        dist     = _va_to_dist(new_v, new_a)
        new_emo  = max(dist, key=lambda k: dist[k])
        out.append({**seg,
                    "valence":          round(new_v, 3),
                    "arousal":          round(new_a, 3),
                    "emotion":          new_emo,
                    "emotion_score":    dist[new_emo],
                    "emotion_dist":     dist,
                    "persona_adjusted": True})
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

    Slices the dubbed audio by segment timestamps, runs audeering VA regression
    on each slice (broadcast/film-trained — replaces superb IEMOCAP model),
    compares perceived VA-derived emotion against the intended emotion in
    segments[*]["emotion"]. Returns avg_soft_score in [0, 100].
    """
    if not dubbed_audio_path.exists():
        return {"note": f"dubbed audio not found: {dubbed_audio_path.name}"}

    try:
        from pydub import AudioSegment as _AS
    except ImportError:
        return {"note": "pydub not installed"}

    try:
        aud_model, aud_proc = _get_audeering_model()
    except Exception as exc:
        return {"note": f"audio model load failed: {exc}"}

    try:
        audio = _AS.from_file(str(dubbed_audio_path)).set_channels(1).set_frame_rate(16000)
    except Exception as exc:
        return {"note": f"audio load failed: {exc}"}

    records, soft_total, n = [], 0.0, 0
    for seg in segments:
        intended = seg.get("emotion", "neutral")
        start_s  = float(seg.get("start", 0))
        end_s    = float(seg.get("end",   0))
        if end_s <= start_s:
            continue
        if (end_s - start_s) * 1000 < 100:   # skip sub-100ms slivers
            continue
        audio_dist = _classify_audio_segment_audeering(
            audio, start_s, end_s, aud_model, aud_proc
        )
        if audio_dist:
            perceived  = max(audio_dist, key=lambda k: audio_dist[k])
            similarity = _va_similarity(intended, perceived)
        else:
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
