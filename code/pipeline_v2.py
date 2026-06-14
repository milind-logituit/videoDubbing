"""
VideoDubbing v2 pipeline — video-aware
Inputs : any MP4/MOV with English audio (or uses the built-in sample clip)
Outputs:
  data/raw/sample_en.mp4          — synthetic demo video (dark BG + English TTS)
  model_outputs/subtitles_hi.vtt  — WebVTT Hindi subtitles
  model_outputs/subtitles_hi.srt  — SRT Hindi subtitles
  data/raw/dubbed_hi.mp4          — original video with Hindi audio track
  model_outputs/metrics_v2.json   — quality metrics

Run: uv run python code/pipeline_v2.py
"""
import argparse
import asyncio
import json
import subprocess
from pathlib import Path

import anthropic
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
from PIL import Image, ImageDraw
from pydub import AudioSegment

import edge_tts
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from eval_lipsync import compute_lipsync_score
from apply_lipsync import apply_wav2lip
from emotion import (classify_segment_emotions, score_emotion_consistency,
                     score_tts_emotion_fidelity)

# Re-exported from sub-modules so tests pulling from this module still work
from diarize import (                                         # noqa: E402
    diarize_speakers, assign_speakers,
    detect_speaker_genders,
    _assign_genders, _GENDER_LABEL_MAP,                      # noqa: F401
    _FEMALE_CONFIDENCE_THRESH,
)
from tts_audio import (                                       # noqa: E402
    synthesize_hindi_audio, _duck_original_audio,
    _seg_voice, _voice_tag, MAX_RATE_PCT,                    # noqa: F401
    VOICE_POOL,
    TTS_VOICE_FEMALE_HI, TTS_VOICE_MALE_HI,                    # noqa: F401
    TTS_VOICE_FEMALE_EN, TTS_VOICE_MALE_EN,                    # noqa: F401
    TTS_VOICE_HI, TTS_BASE_RATE_PCT,                         # noqa: F401
    BG_AUDIO_VOL, BG_AUDIO_VOL_SPEECH,
    separate_stems, get_effective_seg_durations,
)
from metrics import (                                         # noqa: E402
    compute_metrics, compute_text_bleu, compute_back_translation_bleu,
    compute_segment_isochrony, grade_translations,
    _merge_segment_quality,
)

ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data/raw"
PREPARED  = ROOT / "data/prepared"
MODEL_OUT = ROOT / "model_outputs"
for d in [RAW, PREPARED, MODEL_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SOURCE_SCRIPT = [
    {"id": 1, "speaker": "Narrator",
     "text": ("In a world where streaming has replaced the multiplex, "
               "content is king — and every second of screen time must "
               "earn its place.")},
    {"id": 2, "speaker": "Character A",
     "text": ("We built this platform from nothing. "
               "Forty million subscribers in five years. "
               "Nobody thought it was possible.")},
    {"id": 3, "speaker": "Character B",
     "text": ("The audience has changed. "
               "They want stories told in their own language, "
               "with voices that feel like home.")},
    {"id": 4, "speaker": "Narrator",
     "text": ("Today, artificial intelligence makes it possible "
               "to bring every story to every audience — "
               "instantly, accurately, and at scale.")},
]
FULL_SOURCE_TEXT = " ".join(s["text"] for s in SOURCE_SCRIPT)

REFERENCE_HINDI = (
    "एक ऐसी दुनिया में जहां स्ट्रीमिंग ने मल्टीप्लेक्स की जगह ले ली है, "
    "कंटेंट ही राजा है। "
    "हमने यह प्लेटफॉर्म कुछ नहीं से बनाया। "
    "पांच साल में चार करोड़ सब्सक्राइबर। "
    "दर्शक बदल गए हैं। "
    "वे अपनी भाषा में कहानियां सुनना चाहते हैं। "
    "आज, आर्टिफिशियल इंटेलिजेंस हर कहानी को हर दर्शक तक पहुंचाना संभव बनाता है।"
)

WHISPER_MODEL    = "large-v3-turbo"
WHISPER_MODEL_HI = "large-v3-turbo"
VIDEO_SIZE    = (1280, 720)
VIDEO_FPS     = 24
BG_COLOR      = (15, 23, 42)
ACCENT_COLOR  = (251, 191, 36)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Generate sample video
# ─────────────────────────────────────────────────────────────────────────────

def _generate_source_audio(force: bool = False) -> Path:
    audio_path = RAW / "source_en.mp3"
    if audio_path.exists() and not force:
        return audio_path
    print("  Generating English source audio …")
    async def _synth():
        communicate = edge_tts.Communicate(FULL_SOURCE_TEXT, "en-US-JennyNeural")
        await communicate.save(str(audio_path))
    try:
        asyncio.run(_synth())
    except Exception:
        gTTS(FULL_SOURCE_TEXT, lang="en", slow=False).save(str(audio_path))
    return audio_path


def _make_title_frame() -> Path:
    img = Image.new("RGB", VIDEO_SIZE, BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, VIDEO_SIZE[0], 6], fill=ACCENT_COLOR)
    try:
        from PIL import ImageFont
        font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 64)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        font_xs = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        font_lg = font_sm = font_xs = ImageFont.load_default()

    title = "OTT Content — AI Dubbing Demo"
    sub   = "English → Hindi  |  Whisper ASR + Google Translate + Neural TTS"
    tag   = "Built for Eros Now / SunNxt  |  Logituit AI Practice"

    def centered_x(text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return (VIDEO_SIZE[0] - (bbox[2] - bbox[0])) // 2

    draw.text((centered_x(title, font_lg), 270), title,
              font=font_lg, fill=(255, 255, 255))
    draw.text((centered_x(sub, font_sm), 370), sub,
              font=font_sm, fill=(148, 163, 184))
    draw.text((centered_x(tag, font_xs), 430), tag,
              font=font_xs, fill=(100, 116, 139))
    draw.rectangle(
        [0, VIDEO_SIZE[1] - 6, VIDEO_SIZE[0], VIDEO_SIZE[1]], fill=ACCENT_COLOR
    )
    frame_path = RAW / "title_frame.png"
    img.save(frame_path)
    return frame_path


def generate_sample_video(force: bool = False) -> Path:
    video_path = RAW / "sample_en.mp4"
    if video_path.exists() and not force:
        print(f"  Sample video already exists: {video_path.name}")
        return video_path

    audio_path = _generate_source_audio()
    frame_path = _make_title_frame()
    duration   = len(AudioSegment.from_mp3(str(audio_path))) / 1000

    print(f"  Building sample video ({duration:.1f}s) …")
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(frame_path),
        "-i", str(audio_path),
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
        "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k",
        "-r", str(VIDEO_FPS),
        "-t", str(duration),
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  Sample video saved: {video_path.name}")
    return video_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Extract audio from video
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio(video_path: Path) -> Path:
    audio_path = PREPARED / f"{video_path.stem}_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — ASR
# ─────────────────────────────────────────────────────────────────────────────

def transcribe(audio_path: Path, source_lang: str = "en") -> dict:
    print(f"  Loading Whisper '{WHISPER_MODEL}' …")
    model = whisper.load_model(WHISPER_MODEL)
    print(f"  Transcribing {audio_path.name} (lang={source_lang}) …")
    result = model.transcribe(str(audio_path), language=source_lang,
                               word_timestamps=True, verbose=False)
    print(f"  Transcript: {result['text'].strip()[:100]} …")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Translation
# ─────────────────────────────────────────────────────────────────────────────

_FILLER_MAPS: dict[str, dict[str, str]] = {
    "hi": {"hmm": "हाँ", "uh": "", "um": "", "ah": "अच्छा"},
    "en": {"hmm": "yeah", "uh": "", "um": "", "ah": "ah"},
}


def translate_segments(whisper_result: dict,
                       source_lang: str = "en",
                       target_lang: str = "hi") -> list[dict]:
    translator   = GoogleTranslator(source=source_lang, target=target_lang)
    filler_map   = _FILLER_MAPS.get(target_lang, {})
    segments_out = []
    for seg in whisper_result["segments"]:
        en = seg["text"].strip()
        if not en:
            continue
        hi = translator.translate(en) or en
        if en.lower().rstrip(".!?,") in filler_map:
            hi = filler_map[en.lower().rstrip(".!?,")]
        entry: dict = {
            "id":       seg["id"],
            "start":    round(seg["start"], 3),
            "end":      round(seg["end"],   3),
            "duration": round(seg["end"] - seg["start"], 3),
            "en_text":  en,
            "hi_text":  hi,
        }
        if "speaker" in seg:
            entry["speaker"] = seg["speaker"]
        segments_out.append(entry)
        print(f"    [{seg['start']:.1f}s]  {en[:55]}")
        print(f"           →  {hi[:55]}")
    print(f"  Translated {len(segments_out)} segments.")
    return segments_out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4b — LLM post-correction (Claude)
# ─────────────────────────────────────────────────────────────────────────────

_LLM_BATCH = 100

_LANG_NAMES = {"en": "English", "hi": "Hindi", "de": "German"}

_REFINE_TARGET_GUIDANCE = {
    "hi": (
        "     • Hindi TTS speaks at ~3.5 words/second — use this to judge "
        "length. A 2s window fits ~7 Hindi words maximum.\n"
        "     • Prefer shorter, natural phrasing over complete sentences when "
        "the window is tight. Cut filler and subordinate clauses first.\n"
        "     • Use common, everyday Hindi vocabulary (Hindustani/Bollywood register) "
        "that speech recognition systems reliably transcribe. "
        "Avoid rare, Sanskritised, or literary Hindi words — prefer their "
        "everyday equivalents (e.g. 'काम' over 'कार्य', 'बात' over 'वार्तालाप').\n"
        "     • Fillers: 'Hmm' → 'हाँ', 'Uh'/'Um' → empty string, 'Ah' → 'अच्छा'.\n"
    ),
    "en": (
        "     • English TTS speaks at ~3.0–3.5 words/second — a 2s window "
        "fits ~6–7 words maximum.\n"
        "     • Prefer shorter, natural phrasing when the window is tight.\n"
        "     • Use natural broadcast English — clear, idiomatic, suitable for "
        "OTT dubbing. Avoid overly literal translations.\n"
        "     • Fillers: 'Hmm' → 'Yeah', 'Uh'/'Um' → empty string, 'Ah' → 'Ah'.\n"
    ),
}

_EMOTION_GUIDANCE = {
    "hi": (
        "  3. EMOTION (hard constraint — not optional): If an 'emotion' field is present,\n"
        "     the rewritten Hindi MUST carry that emotional register through word choice.\n"
        "     Use these Hindi-specific cues:\n"
        "       • angry   → forceful verbs, exclamatory particles (अरे!, क्यों!, नहीं!),\n"
        "                    short urgent clauses, avoid soft conjunctions\n"
        "       • fearful → tense/hesitant phrasing, words like डर, खतरा, बचाओ,\n"
        "                    broken or incomplete clauses where natural\n"
        "       • sad     → soft conjunctions (लेकिन, मगर, पर), reduced energy,\n"
        "                    words like दुख, अफसोस, याद; avoid exclamations\n"
        "       • happy   → upbeat vocab, वाह!, हाँ!, warm qualifiers (बढ़िया, शानदार)\n"
        "       • surprised → ओह!, अरे वाह!, क्या!, wide-eyed reactive phrasing\n"
        "       • disgust → distancing language, words like घिनौना, बेकार, छी\n"
        "       • neutral → plain declarative; do NOT add emotion not in the source\n"
        "     A neutral source line MUST stay neutral. Do not dramatise.\n"
    ),
    "en": (
        "  3. EMOTION (hard constraint — not optional): If an 'emotion' field is present,\n"
        "     the rewritten English MUST carry that emotional register:\n"
        "       • angry   → forceful, clipped sentences; strong verbs; avoid hedging\n"
        "       • fearful → hesitant, halting phrasing; 'I can't', 'we have to'\n"
        "       • sad     → slower cadence implied by word length; 'I miss', 'it's gone'\n"
        "       • happy   → short energetic lines; upbeat qualifiers\n"
        "       • neutral → plain declarative; do NOT add emotion not in the source\n"
        "     A neutral source line MUST stay neutral. Do not dramatise.\n"
    ),
}

_DEFAULT_EMOTION_GUIDANCE = (
    "  3. EMOTION (hard constraint — not optional): If an 'emotion' field is present,\n"
    "     the rewritten translation MUST match that emotional register through word choice.\n"
    "     angry → forceful/urgent; fearful → tense/hesitant; sad → subdued/soft;\n"
    "     happy → upbeat/energetic; neutral → plain, no added drama.\n"
    "     A neutral source MUST stay neutral. Do not dramatise.\n"
)


def _make_refine_system(
    source_lang: str = "en",
    target_lang: str = "hi",
    glossary: dict[str, str] | None = None,
) -> str:
    src_name      = _LANG_NAMES.get(source_lang, source_lang.upper())
    tgt_name      = _LANG_NAMES.get(target_lang, target_lang.upper())
    pacing        = _REFINE_TARGET_GUIDANCE.get(
        target_lang,
        f"     • Use natural, idiomatic {tgt_name} suitable for OTT dubbing.\n",
    )
    emotion_rules = _EMOTION_GUIDANCE.get(target_lang, _DEFAULT_EMOTION_GUIDANCE)
    glossary_rule = ""
    if glossary:
        terms = "; ".join(f"{k}→{v}" for k, v in glossary.items())
        glossary_rule = (
            f"  0. GLOSSARY (hard constraint): These proper nouns MUST appear "
            f"exactly as given in every segment: {terms}.\n"
        )
    return (
        f"You are a professional {tgt_name} dubbing editor for OTT streaming content "
        f"(Eros Now / SunNxt).\n"
        f"Input: JSON array of segments, each with ASR {src_name} (en_text), "
        f"machine-translated {tgt_name} (hi_text), duration_s (seconds available "
        "to speak this line), and optionally emotion fields.\n"
        "When present, 'emotion' is the dominant label and 'emotion_blend' is the full "
        "probability distribution (e.g. {fearful: 0.65, disgust: 0.20, neutral: 0.15}). "
        "Use the blend to capture emotional undertones, not just the top label.\n"
        "The input may include optional 'context_before' and 'context_after' arrays "
        "with neighbouring segments in their already-refined form. Use these ONLY for "
        "contextual consistency: match character names, terminology, and emotional arc "
        "with adjacent lines. Do NOT output context segments — only the main segments.\n"
        + glossary_rule
        + "For each segment:\n"
        "  1. Fix ASR transcription errors in en_text "
        "(e.g. 'half is likely' → 'half as likely').\n"
        f"  2. Rewrite hi_text as natural spoken {tgt_name} for dubbing that fits "
        "within duration_s seconds.\n"
        + pacing +
        "     • Distinguish dinner vs supper, couch vs sofa, etc.\n"
        + emotion_rules +
        "Return ONLY a valid JSON array: "
        '[{"id": int, "en_text": str, "hi_text": str}, …]. '
        "Do NOT include duration_s or emotion in output. "
        "Same count and IDs as input. No markdown, no explanation."
    )


def _ctx_snippet(segs: list[dict]) -> list[dict]:
    return [{"id": s["id"], "en_text": s["en_text"], "hi_text": s["hi_text"]}
            for s in segs]


def _refine_batch(
    client: anthropic.Anthropic,
    batch: list[dict],
    refine_system: str,
    ctx_before: list[dict] | None = None,
    ctx_after: list[dict] | None = None,
) -> dict[int, dict]:
    payload: dict = {
        "segments": [
            {
                "id": s["id"],
                "en_text": s["en_text"],
                "hi_text": s["hi_text"],
                "duration_s": round(s["duration"], 2),
                **({"emotion": s["emotion"],
                    "emotion_blend": s["emotion_dist"]}
                   if s.get("emotion_dist") and s.get("emotion") != "neutral"
                   else {"emotion": s["emotion"]} if s.get("emotion") else {}),
            }
            for s in batch
        ]
    }
    if ctx_before:
        payload["context_before"] = _ctx_snippet(ctx_before)
    if ctx_after:
        payload["context_after"] = _ctx_snippet(ctx_after)

    content = json.dumps(payload, ensure_ascii=False)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=refine_system,
        messages=[{"role": "user", "content": content}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(raw)
    # Claude returns either the segments array directly or {"segments": [...]}
    if isinstance(parsed, dict):
        parsed = parsed.get("segments", [])
    return {r["id"]: r for r in parsed}


_CTX_WINDOW = 3   # segments of surrounding context passed to Claude per batch


def refine_segments(
    segments: list[dict],
    *,
    skip: bool = False,
    source_lang: str = "en",
    target_lang: str = "hi",
    glossary: dict[str, str] | None = None,
) -> list[dict]:
    """Fix ASR errors and rewrite target language as natural dubbing speech via Claude."""
    if skip:
        print("  Stage 4b skipped (--no-llm).")
        return segments

    refine_system = _make_refine_system(source_lang, target_lang, glossary)
    client = anthropic.Anthropic()
    n = len(segments)
    n_batches = (n + _LLM_BATCH - 1) // _LLM_BATCH
    tgt_name = _LANG_NAMES.get(target_lang, target_lang.upper())
    print(
        f"  Calling Claude (claude-sonnet-4-6) to refine {n} segments "
        f"→ {tgt_name} in {n_batches} batch(es) …"
    )

    refined: dict[int, dict] = {}
    for i in range(n_batches):
        start = i * _LLM_BATCH
        end   = min(start + _LLM_BATCH, n)
        batch = segments[start:end]
        ctx_before = segments[max(0, start - _CTX_WINDOW):start]
        ctx_after  = segments[end:end + _CTX_WINDOW]
        if n_batches > 1:
            print(f"    Batch {i + 1}/{n_batches} ({len(batch)} segments) …")
        refined.update(_refine_batch(client, batch, refine_system,
                                     ctx_before or None, ctx_after or None))

    out = []
    for seg in segments:
        r = refined.get(seg["id"])
        if r:
            seg = {**seg, "en_text": r["en_text"], "hi_text": r["hi_text"]}
        out.append(seg)

    changed = sum(
        1 for orig, new in zip(segments, out)
        if orig["en_text"] != new["en_text"] or orig["hi_text"] != new["hi_text"]
    )
    print(f"  LLM refined {changed}/{len(out)} segments.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4b.5 — Glossary extraction (proper nouns / character names)
# ─────────────────────────────────────────────────────────────────────────────

def extract_glossary(
    segments: list[dict],
    target_lang: str = "hi",
    cache_path: Path | None = None,
) -> dict[str, str]:
    """Return {en_term: target_transliteration} for proper nouns found in segments.

    Uses Claude Haiku (cheap) to identify character names, place names, and
    technical terms, then returns their canonical target-language forms.
    Results are cached to cache_path if provided so subsequent runs are free.
    """
    import re

    if cache_path and cache_path.exists():
        import json as _json
        glossary = _json.loads(cache_path.read_text())
        print(f"  Glossary cached ({len(glossary)} terms): {list(glossary.items())[:5]}")
        return glossary

    # Heuristic: capitalized tokens that appear 2+ times are likely proper nouns
    all_text = " ".join(s.get("en_text", "") for s in segments)
    tokens = re.findall(r"\b([A-Z][a-z]{1,20})\b", all_text)
    from collections import Counter
    candidates = [t for t, c in Counter(tokens).items() if c >= 2]

    if not candidates:
        return {}

    tgt_name = _LANG_NAMES.get(target_lang, target_lang.upper())
    client = anthropic.Anthropic()
    prompt = (
        f"From this list of words extracted from a video transcript, identify only "
        f"proper nouns (character names, place names, brand names, technical terms). "
        f"For each proper noun, provide its standard {tgt_name} transliteration.\n\n"
        f"Words: {', '.join(candidates)}\n\n"
        f'Return ONLY valid JSON: {{"Tom": "टॉम", "Celia": "सेलिया", ...}}. '
        f"Omit common English words. If no proper nouns, return {{}}."
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    import json as _json
    try:
        glossary = _json.loads(raw)
    except Exception:
        glossary = {}

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps(glossary, ensure_ascii=False, indent=2))

    print(f"  Glossary extracted ({len(glossary)} terms): {list(glossary.items())[:5]}")
    return glossary


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4c — Emotion register repair loop
# ─────────────────────────────────────────────────────────────────────────────

def _make_repair_system(source_lang: str = "en", target_lang: str = "hi") -> str:
    tgt_name      = _LANG_NAMES.get(target_lang, target_lang.upper())
    emotion_rules = _EMOTION_GUIDANCE.get(target_lang, _DEFAULT_EMOTION_GUIDANCE)
    return (
        f"You are rewriting {tgt_name} dubbing segments whose emotional register "
        f"failed quality review.\n"
        "Each segment includes: en_text (source), hi_text (current translation that "
        "failed), emotion (dominant label), emotion_blend (full probability "
        "distribution), and grade_note (the reviewer's diagnosis).\n"
        "Your ONLY job: rewrite hi_text so it carries the intended emotional register "
        "through word choice. Do NOT change meaning, timing, or sentence structure "
        "unless essential for emotional impact.\n"
        + emotion_rules +
        "Return ONLY a valid JSON array: "
        '[{"id": int, "en_text": str, "hi_text": str}, …]. '
        "Same count and IDs as input. No markdown, no explanation."
    )


def repair_emotion_register(segments: list[dict], *,
                             skip: bool = False,
                             source_lang: str = "en",
                             target_lang: str = "hi",
                             threshold: int = 3) -> list[dict]:
    """Stage 4c — re-refine segments where Haiku grades emotion_register < threshold."""
    from metrics import grade_translations

    emotional = [s for s in segments
                 if s.get("emotion") and s["emotion"] != "neutral"]
    if not emotional:
        print("  No non-neutral segments — skipping repair.")
        return segments
    if skip:
        print("  Stage 4c skipped (--no-llm).")
        return segments

    print(f"  Pre-grading {len(emotional)} emotional segment(s) …")
    grades      = grade_translations(emotional)
    grades_by_id = {g["id"]: g for g in grades}

    to_repair = [
        {**s, "grade_note": grades_by_id.get(s["id"], {}).get("note", "register too flat")}
        for s in emotional
        if grades_by_id.get(s["id"], {}).get("emotion_register", threshold) < threshold
    ]

    if not to_repair:
        print(f"  All emotional segments pass emotion_register ≥ {threshold}.")
        return segments

    print(f"  {len(to_repair)} segment(s) below threshold — rewriting with repair prompt …")
    client       = anthropic.Anthropic()
    repair_system = _make_repair_system(source_lang, target_lang)
    repair_payload = [
        {
            "id":           s["id"],
            "en_text":      s["en_text"],
            "hi_text":      s["hi_text"],
            "emotion":      s["emotion"],
            "emotion_blend": s.get("emotion_dist", {}),
            "grade_note":   s["grade_note"],
        }
        for s in to_repair
    ]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=repair_system,
        messages=[{"role": "user",
                   "content": json.dumps(repair_payload, ensure_ascii=False)}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    repaired = {r["id"]: r for r in json.loads(raw)}

    out = []
    for seg in segments:
        r = repaired.get(seg["id"])
        if r:
            seg = {**seg, "hi_text": r["hi_text"]}
            print(f"    [{seg['start']:.1f}s] repaired ({seg['emotion']})")
        out.append(seg)
    print(f"  Repaired {len(repaired)} segment(s).")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3.5 helpers — voice pool assignment
# ─────────────────────────────────────────────────────────────────────────────

def _assign_voice_pool(
    genders: dict[str, str],
    segments: list[dict],
    target_lang: str,
) -> dict[str, str]:
    """Assign a distinct voice from VOICE_POOL to each speaker.

    Speakers are sorted by first utterance so assignment is stable across runs.
    When the pool has only one voice (e.g. Hindi male), all same-gender speakers
    share it — but the mechanism is ready for expansion.
    """
    pool = VOICE_POOL.get(target_lang, VOICE_POOL["hi"])

    def _first_start(sp: str) -> float:
        for s in segments:
            if s.get("speaker") == sp:
                return float(s.get("start", 0))
        return 0.0

    counters: dict[str, int] = {"male": 0, "female": 0}
    result: dict[str, str] = {}
    for sp in sorted(genders, key=_first_start):
        gender = genders[sp]
        voice_list = pool.get(gender, pool.get("male", []))
        if not voice_list:
            continue
        result[sp] = voice_list[counters[gender] % len(voice_list)]
        counters[gender] += 1
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Subtitle files
# ─────────────────────────────────────────────────────────────────────────────

def _vtt_time(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"

def _srt_time(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    ms = int((sec % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{ms:03d}"


def generate_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines += [f"{_vtt_time(seg['start'])} --> {_vtt_time(seg['end'])}",
                  seg["hi_text"], ""]
    return "\n".join(lines)


def generate_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i),
                  f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}",
                  seg["hi_text"], ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Create dubbed video
# ─────────────────────────────────────────────────────────────────────────────

def create_dubbed_video(
    video_path: Path,
    hindi_audio_path: Path,
    segments: list[dict] | None = None,
    force: bool = False,
    target_lang: str = "hi",
    no_vocals_path: Path | None = None,
) -> Path:
    dubbed_video = RAW / f"{video_path.stem}_dubbed_{target_lang}.mp4"
    if dubbed_video.exists() and not force:
        print(f"  Dubbed video already exists: {dubbed_video.name}")
        return dubbed_video

    orig_audio = PREPARED / f"{video_path.stem}_audio.wav"

    if no_vocals_path and no_vocals_path.exists():
        # Stem separation available: mix dubbed TTS over music/SFX track at full volume
        print("  Mixing dubbed TTS with instrumental/SFX stem (full volume) …")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(hindi_audio_path),
            "-i", str(no_vocals_path),
            "-filter_complex",
            "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[aout]",
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", str(dubbed_video),
        ]
    elif segments and orig_audio.exists():
        print(f"  Ducking original audio "
              f"(speech={BG_AUDIO_VOL_SPEECH:.0%}, gaps={BG_AUDIO_VOL:.0%}) …")
        bg_path = _duck_original_audio(orig_audio, segments)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(hindi_audio_path),
            "-i", str(bg_path),
            "-filter_complex",
            "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[aout]",
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", str(dubbed_video),
        ]
    else:
        print(f"  Mixing Hindi TTS with original audio (bg={BG_AUDIO_VOL:.0%}) …")
        af = (f"[0:a]volume={BG_AUDIO_VOL}[bg];"
              "[1:a][bg]amix=inputs=2:duration=first:normalize=0[aout]")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path), "-i", str(hindi_audio_path),
            "-filter_complex", af,
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", str(dubbed_video),
        ]

    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  Dubbed video saved: {dubbed_video.name}")
    return dubbed_video


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(segments, vtt, srt, metrics, src_audio, dubbed_audio,
                 stem: str = "sample", target_lang: str = "hi"):
    pd.DataFrame(segments).to_csv(
        PREPARED / f"transcript_bilingual_{stem}.csv", index=False
    )
    (MODEL_OUT / f"subtitles_{stem}_{target_lang}.vtt").write_text(vtt, encoding="utf-8")
    (MODEL_OUT / f"subtitles_{stem}_{target_lang}.srt").write_text(srt, encoding="utf-8")
    with open(MODEL_OUT / f"metrics_{stem}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved outputs to {MODEL_OUT}/")
    for out in sorted(MODEL_OUT.iterdir()):
        print(f"  {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoDubbing v2 pipeline")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to an existing MP4/MOV to dub. "
                             "Skips synthetic sample-video generation.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Stage 4b Claude post-correction.")
    parser.add_argument("--no-diarize", action="store_true",
                        help="Skip Stage 3.5 speaker diarization.")
    parser.add_argument("--no-stems", action="store_true",
                        help="Skip Stage 2.1 demucs stem separation; "
                             "fall back to audio ducking.")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for pyannote models. "
                             "Reads HF_TOKEN env var if not provided.")
    parser.add_argument("--gender-thresh", type=float,
                        default=_FEMALE_CONFIDENCE_THRESH,
                        help="Min female probability to assign female voice "
                             f"(default {_FEMALE_CONFIDENCE_THRESH}).")
    parser.add_argument("--lipsync", action="store_true",
                        help="Run Wav2Lip (Stage 7b) to re-generate mouth movements. "
                             "Slow — adds ~5 min per 2 min of video.")
    parser.add_argument("--source-lang", type=str, default="en",
                        help="Source language for ASR (Whisper language code, default: en).")
    parser.add_argument("--target-lang", type=str, default="hi",
                        help="Target language for translation and TTS (default: hi).")
    args = parser.parse_args()
    source_lang = args.source_lang
    target_lang = args.target_lang

    if args.input:
        if not args.input.exists():
            raise FileNotFoundError(f"Input video not found: {args.input}")
        video_path = args.input
        print(f"Stage 1 — Using provided video: {video_path.name}")
    else:
        print("Stage 1 — Generating sample video …")
        video_path = generate_sample_video()

    print("\nStage 2 — Extracting audio …")
    audio_path = extract_audio(video_path)

    print("\nStage 2.1 — Stem separation (demucs) …")
    no_vocals_path: Path | None = None
    if args.no_stems:
        print("  Skipped (--no-stems).")
    else:
        _vocals, no_vocals_path = separate_stems(audio_path)
        if no_vocals_path == audio_path:
            no_vocals_path = None  # fallback triggered — use ducking path

    print("\nStage 3 — ASR (Whisper) …")
    whisper_result = transcribe(audio_path, source_lang=source_lang)

    print("\nStage 3.5 — Speaker diarization …")
    speaker_voices: dict[str, str] = {}
    if args.no_diarize:
        print("  Skipped (--no-diarize).")
    else:
        import os
        hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("  No HF token found — skipping diarization. "
                  "Pass --hf-token or set HF_TOKEN.")
        else:
            turns = diarize_speakers(audio_path, hf_token)
            whisper_result["segments"] = assign_speakers(
                whisper_result["segments"], turns
            )
            genders = detect_speaker_genders(audio_path,
                                             whisper_result["segments"])
            speaker_voices = _assign_voice_pool(
                genders, whisper_result["segments"], target_lang
            )
            print(f"  Voice map: {speaker_voices}")

    print("\nStage 4 — Translation (Google Translate) …")
    segments = translate_segments(whisper_result,
                                  source_lang=source_lang, target_lang=target_lang)

    print("\nStage 4a — Glossary extraction …")
    glossary: dict[str, str] = {}
    if not args.no_llm:
        glossary_cache = PREPARED / f"glossary_{video_path.stem}_{target_lang}.json"
        glossary = extract_glossary(segments, target_lang=target_lang,
                                    cache_path=glossary_cache)

    print("\nStage 4b — LLM post-correction (Claude) …")
    segments = refine_segments(segments, skip=args.no_llm,
                               source_lang=source_lang, target_lang=target_lang,
                               glossary=glossary or None)

    print("\nStage 2.5 — Emotion analysis (SER) …")
    segments = classify_segment_emotions(audio_path, segments)
    emo_counts: dict[str, int] = {}
    for s in segments:
        emo_counts[s.get("emotion", "neutral")] = emo_counts.get(s.get("emotion", "neutral"), 0) + 1
    print(f"  Emotion distribution: {emo_counts}")

    print("\nStage 4c — Emotion register repair …")
    segments = repair_emotion_register(segments, skip=args.no_llm,
                                       source_lang=source_lang, target_lang=target_lang)

    tgt_name = _LANG_NAMES.get(target_lang, target_lang.upper())
    print(f"\nStage 6 — {tgt_name} TTS …")
    hindi_audio = synthesize_hindi_audio(
        segments, stem=video_path.stem, src_audio=audio_path,
        speaker_voices=speaker_voices, target_lang=target_lang,
    )

    print("\nStage 5 — Generating subtitle files (aligned to dubbed audio) …")
    eff_durs = get_effective_seg_durations(segments, video_path.stem, target_lang)
    if eff_durs:
        MIN_SUB_S = 0.5
        for seg in segments:
            if seg["id"] in eff_durs:
                actual = eff_durs[seg["id"]]
                original_window = seg["end"] - seg["start"]
                seg["end"] = seg["start"] + max(MIN_SUB_S, min(actual, original_window))
    vtt = generate_vtt(segments)
    srt = generate_srt(segments)
    print(f"  VTT: {len(vtt)} chars  |  SRT: {len(srt)} chars")

    print("\nStage 7 — Creating dubbed video …")
    dubbed_video = create_dubbed_video(video_path, hindi_audio, segments=segments,
                                       target_lang=target_lang,
                                       no_vocals_path=no_vocals_path)

    if args.lipsync:
        print("\nStage 7b — Wav2Lip lip-sync …")
        ls_out = RAW / f"{video_path.stem}_lipsync_{target_lang}.mp4"
        wav2lip_result = apply_wav2lip(video_path, hindi_audio, ls_out)
        if wav2lip_result["success"]:
            print(f"  Lip-synced video: {ls_out.name}")
            metrics_lipsync_video = ls_out
        else:
            print(f"  [warn] Wav2Lip failed: {wav2lip_result['note']}")
            metrics_lipsync_video = dubbed_video
    else:
        metrics_lipsync_video = dubbed_video

    print("\nStage 8 — Computing metrics …")
    has_ref   = args.input is None
    src_audio = RAW / "source_en.mp3" if has_ref else audio_path
    metrics   = compute_metrics(whisper_result, segments, src_audio,
                                hindi_audio, has_reference=has_ref)
    metrics["tts"]["voice_map"] = speaker_voices

    print("\nStage 8b — Segment quality …")
    seg_dir = PREPARED / f"hi_segments_{video_path.stem}"
    iso = compute_segment_isochrony(segments, seg_dir)
    grades = grade_translations(segments) if not args.no_llm else []
    if grades:
        print(f"  LLM graded {len(grades)} segments.")
    metrics["segment_quality"] = _merge_segment_quality(
        iso, {g["id"]: g for g in grades}
    )

    print("\nStage 8c — Back-translation BLEU …")
    original_src = whisper_result["text"].strip()
    metrics["text_bleu"] = compute_text_bleu(
        segments, original_src, source_lang=source_lang, target_lang=target_lang,
    )
    print(f"  Text-level BLEU: {metrics['text_bleu'].get('bleu')}")
    metrics["back_translation"] = compute_back_translation_bleu(
        hindi_audio, original_src,
        source_lang=source_lang, target_lang=target_lang,
    )

    print("\nStage 8d — Lip-sync score …")
    metrics["lipsync"] = compute_lipsync_score(metrics_lipsync_video)
    metrics["lipsync"]["wav2lip_applied"] = args.lipsync
    ls = metrics["lipsync"]
    if ls.get("sync_score") is not None:
        print(f"  sync_score={ls['sync_score']}  pearson_r={ls['pearson_r']}  "
              f"lag={ls['best_lag_ms']}ms  faces={ls['faces_pct']}%")
    else:
        print(f"  Lip-sync score unavailable: {ls.get('note')}")

    print("\nStage 8e — Emotion consistency score …")
    try:
        emo = score_emotion_consistency(audio_path, hindi_audio, segments,
                                        target_lang=target_lang)
        ep  = emo
        print(f"  Text match: {ep['match_pct']}%  soft: {ep['avg_soft_score']}%  "
              f"over {ep['n_segments']} segments")
        print("  Stage 8e-ii — TTS emotion fidelity (audio SER) …")
        emo["tts_fidelity"] = score_tts_emotion_fidelity(hindi_audio, segments)
        tf = emo["tts_fidelity"]
        print(f"  TTS fidelity soft score: {tf.get('avg_soft_score')}% "
              f"over {tf.get('n_segments')} segments")
        metrics["emotion"] = emo
    except Exception as exc:
        metrics["emotion"] = {"match_pct": None, "note": str(exc)}
        print(f"  [warn] Emotion scoring failed: {exc}")

    save_outputs(segments, vtt, srt, metrics, src_audio, hindi_audio,
                 stem=video_path.stem, target_lang=target_lang)

    print("\n── Quality metrics ──")
    wer_pct    = metrics["asr"]["wer_pct"]
    bleu       = metrics["translation"]["bleu"]
    txt_bleu   = metrics.get("text_bleu", {}).get("bleu")
    bt_bleu    = metrics["back_translation"].get("bleu")
    sync_score = metrics["lipsync"].get("sync_score")
    _emo_data    = metrics.get("emotion", {})
    emo_match    = _emo_data.get("match_pct")
    emo_soft     = _emo_data.get("avg_soft_score")
    emo_tts      = _emo_data.get("tts_fidelity", {}).get("avg_soft_score")
    print(f"  ASR WER              : {f'{wer_pct:.1f}%' if wer_pct is not None else 'N/A'}")
    print(f"  Translation BLEU     : {f'{bleu:.1f}' if bleu is not None else 'N/A'}")
    print(f"  Text-level BLEU      : {f'{txt_bleu:.1f}' if txt_bleu is not None else 'N/A'}")
    print(f"  Back-translation BLEU: {f'{bt_bleu:.1f}' if bt_bleu is not None else 'N/A'}")
    print(f"  Lip-sync score       : {f'{sync_score:.3f}' if sync_score is not None else 'N/A'}")
    print(f"  Emotion match (bin)  : {f'{emo_match:.1f}%' if emo_match is not None else 'N/A'}")
    print(f"  Emotion soft score   : {f'{emo_soft:.1f}%' if emo_soft is not None else 'N/A'}")
    print(f"  TTS emotion fidelity : {f'{emo_tts:.1f}%' if emo_tts is not None else 'N/A'}")
    print(f"  Duration ratio       : {metrics['alignment']['duration_ratio']:.3f}")
    print("\nDone.")

