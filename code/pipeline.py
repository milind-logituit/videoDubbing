"""
VideoDubbing pipeline: ASR → Translation → TTS
Three stages:
  1. generate_source_clip  — create a synthetic English MP3 with edge-tts (our "original content")
  2. transcribe            — Whisper ASR → English transcript + word-level timestamps
  3. translate             — Helsinki-NLP opus-mt-en-hi → Hindi text per segment
  4. synthesize            — edge-tts → Hindi MP3 (natural-sounding voice)
  5. compute_metrics       — WER, BLEU, duration alignment
  6. save_outputs          — metrics.json, bilingual transcript CSV, audio manifest

Run: uv run python code/pipeline.py
"""
import asyncio
import json
from pathlib import Path

from deep_translator import GoogleTranslator
import whisper
from gtts import gTTS
from pydub import AudioSegment

import edge_tts
import jiwer
import sacrebleu
import pandas as pd

ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data/raw"
PREPARED  = ROOT / "data/prepared"
MODEL_OUT = ROOT / "model_outputs"
for d in [RAW, PREPARED, MODEL_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Source content ────────────────────────────────────────────────────────────
# Short entertainment-style English script representative of OTT content.
SOURCE_SCRIPT = [
    {
        "id": 1,
        "speaker": "Narrator",
        "text": (
            "In a world where streaming has replaced the multiplex, "
            "content is king — and every second of screen time must earn its place."
        ),
    },
    {
        "id": 2,
        "speaker": "Character A",
        "text": (
            "We built this platform from nothing. "
            "Forty million subscribers in five years. "
            "Nobody thought it was possible."
        ),
    },
    {
        "id": 3,
        "speaker": "Character B",
        "text": (
            "The audience has changed. "
            "They want stories told in their own language, "
            "with voices that feel like home."
        ),
    },
    {
        "id": 4,
        "speaker": "Narrator",
        "text": (
            "Today, artificial intelligence makes it possible "
            "to bring every story to every audience — "
            "instantly, accurately, and at scale."
        ),
    },
]

FULL_SOURCE_TEXT = " ".join(s["text"] for s in SOURCE_SCRIPT)

# Human-quality reference Hindi translation for BLEU scoring
REFERENCE_HINDI = (
    "एक ऐसी दुनिया में जहां स्ट्रीमिंग ने मल्टीप्लेक्स की जगह ले ली है, "
    "कंटेंट ही राजा है। "
    "हमने यह प्लेटफॉर्म कुछ नहीं से बनाया। "
    "पांच साल में चार करोड़ सब्सक्राइबर। "
    "दर्शक बदल गए हैं। "
    "वे अपनी भाषा में कहानियां सुनना चाहते हैं। "
    "आज, आर्टिफिशियल इंटेलिजेंस हर कहानी को हर दर्शक तक पहुंचाना संभव बनाता है।"
)

WHISPER_MODEL     = "base"
TRANSLATION_MODEL = "Google Translate (en→hi)"
TTS_VOICE_HI      = "hi-IN-SwaraNeural"
TTS_VOICE_EN      = "en-US-JennyNeural"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Generate source English clip
# ─────────────────────────────────────────────────────────────────────────────

def generate_source_clip(force: bool = False) -> Path:
    out_path = RAW / "source_en.mp3"
    if out_path.exists() and not force:
        print(f"  Source clip already exists: {out_path.name}")
        return out_path

    print(f"  Generating source English clip ({len(FULL_SOURCE_TEXT)} chars) …")
    try:
        async def _synth():
            communicate = edge_tts.Communicate(FULL_SOURCE_TEXT, TTS_VOICE_EN)
            await communicate.save(str(out_path))
        asyncio.run(_synth())
        print(f"  Saved (edge-tts): {out_path.name}")
    except Exception as e:
        print(f"  edge-tts failed ({e}), falling back to gTTS …")
        gTTS(FULL_SOURCE_TEXT, lang="en", slow=False).save(str(out_path))
        print(f"  Saved (gTTS): {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — ASR: Whisper transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcribe(audio_path: Path) -> dict:
    print(f"  Loading Whisper '{WHISPER_MODEL}' model …")
    model = whisper.load_model(WHISPER_MODEL)
    print(f"  Transcribing {audio_path.name} …")
    result = model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
        verbose=False,
    )
    preview = result["text"].strip()[:120]
    print(f"  Transcript preview: {preview} …")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Translation: English → Hindi via Google Translate (free)
# ─────────────────────────────────────────────────────────────────────────────

def translate_segments(whisper_result: dict) -> list[dict]:
    translator   = GoogleTranslator(source="en", target="hi")
    segments_out = []

    for seg in whisper_result["segments"]:
        en = seg["text"].strip()
        if not en:
            continue

        hi = translator.translate(en)

        segments_out.append({
            "id":       seg["id"],
            "start":    round(seg["start"], 3),
            "end":      round(seg["end"], 3),
            "duration": round(seg["end"] - seg["start"], 3),
            "en_text":  en,
            "hi_text":  hi,
        })
        print(f"    [{seg['start']:.1f}s] {en[:60]}")
        print(f"          → {hi[:60]}")

    print(f"  Translated {len(segments_out)} segments.")
    return segments_out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — TTS: Hindi synthesis
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_hindi(segments: list[dict], force: bool = False) -> Path:
    dubbed_path = RAW / "dubbed_hi.mp3"
    if dubbed_path.exists() and not force:
        print(f"  Hindi dubbed audio already exists: {dubbed_path.name}")
        return dubbed_path

    seg_dir = PREPARED / "hi_segments"
    seg_dir.mkdir(exist_ok=True)
    seg_paths = []

    combined_hi = " ".join(s["hi_text"] for s in segments)
    print(f"  Synthesising Hindi audio ({len(combined_hi)} chars) …")

    for seg in segments:
        seg_path = seg_dir / f"seg_{seg['id']:03d}.mp3"
        if not seg_path.exists():
            try:
                async def _synth(text=seg["hi_text"], path=seg_path):
                    communicate = edge_tts.Communicate(text, TTS_VOICE_HI)
                    await communicate.save(str(path))
                asyncio.run(_synth())
            except Exception:
                gTTS(seg["hi_text"], lang="hi", slow=False).save(str(seg_path))
        seg_paths.append(seg_path)

    combined = AudioSegment.empty()
    for p in seg_paths:
        combined += AudioSegment.from_mp3(str(p))
    combined.export(str(dubbed_path), format="mp3")
    print(f"  Hindi dubbed audio: {dubbed_path.name}  ({len(combined)/1000:.1f}s)")
    return dubbed_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    whisper_result: dict,
    segments:       list[dict],
    source_path:    Path,
    dubbed_path:    Path,
) -> dict:
    # WER: Whisper transcript vs ground-truth source script
    hypothesis = whisper_result["text"].strip().lower()
    reference  = FULL_SOURCE_TEXT.strip().lower()
    wer_score  = jiwer.wer(reference, hypothesis)

    # BLEU: machine Hindi vs reference Hindi
    machine_hindi = " ".join(s["hi_text"] for s in segments)
    bleu = sacrebleu.corpus_bleu([machine_hindi], [[REFERENCE_HINDI]])

    # Duration alignment ratio
    try:
        src_audio = AudioSegment.from_mp3(str(source_path))
        dub_audio = AudioSegment.from_mp3(str(dubbed_path))
        src_dur   = len(src_audio) / 1000
        dub_dur   = len(dub_audio) / 1000
        dar       = dub_dur / src_dur
    except Exception:
        src_dur = dub_dur = dar = 0.0

    en_chars = len(FULL_SOURCE_TEXT.replace(" ", ""))
    hi_chars = len(machine_hindi.replace(" ", ""))
    spr_en   = en_chars / src_dur if src_dur > 0 else 0
    spr_hi   = hi_chars / dub_dur if dub_dur > 0 else 0

    return {
        "asr": {
            "model":   WHISPER_MODEL,
            "wer":     round(wer_score, 4),
            "wer_pct": round(wer_score * 100, 2),
        },
        "translation": {
            "model":      TRANSLATION_MODEL,
            "bleu":       round(bleu.score, 2),
            "n_segments": len(segments),
        },
        "tts": {
            "voice": TTS_VOICE_HI,
        },
        "alignment": {
            "source_duration_s": round(src_dur, 2),
            "dubbed_duration_s": round(dub_dur, 2),
            "duration_ratio":    round(dar, 3),
            "en_chars_per_sec":  round(spr_en, 1),
            "hi_chars_per_sec":  round(spr_hi, 1),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(segments, metrics, source_path, dubbed_path):
    pd.DataFrame(segments).to_csv(PREPARED / "transcript_bilingual.csv", index=False)

    with open(MODEL_OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(MODEL_OUT / "audio_manifest.json", "w") as f:
        json.dump({"source_en": str(source_path), "dubbed_hi": str(dubbed_path)}, f, indent=2)

    print(f"\nSaved:")
    print(f"  {PREPARED / 'transcript_bilingual.csv'}")
    print(f"  {MODEL_OUT / 'metrics.json'}")
    print(f"  {MODEL_OUT / 'audio_manifest.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(segments, metrics):
    print("\n── Bilingual transcript ──")
    for s in segments:
        print(f"  [{s['start']:.1f}s – {s['end']:.1f}s]  {s['en_text']}")
        print(f"    → {s['hi_text']}")

    m = metrics
    print(f"\n── Quality metrics ──")
    print(f"  ASR WER          : {m['asr']['wer_pct']:.1f}%")
    print(f"  Translation BLEU : {m['translation']['bleu']:.1f}")
    print(f"  Duration ratio   : {m['alignment']['duration_ratio']:.3f}  "
          f"({m['alignment']['source_duration_s']:.1f}s EN → "
          f"{m['alignment']['dubbed_duration_s']:.1f}s HI)")
    print(f"  Speech pace EN   : {m['alignment']['en_chars_per_sec']:.1f} ch/s")
    print(f"  Speech pace HI   : {m['alignment']['hi_chars_per_sec']:.1f} ch/s")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Stage 1 — Generating source English clip …")
    source_path = generate_source_clip()

    print("\nStage 2 — ASR transcription (Whisper) …")
    whisper_result = transcribe(source_path)

    print("\nStage 3 — Translation (Google Translate en→hi) …")
    segments = translate_segments(whisper_result)

    print("\nStage 4 — Hindi TTS synthesis …")
    dubbed_path = synthesize_hindi(segments)

    print("\nStage 5 — Computing quality metrics …")
    metrics = compute_metrics(whisper_result, segments, source_path, dubbed_path)

    save_outputs(segments, metrics, source_path, dubbed_path)
    print_summary(segments, metrics)
    print("\nDone.")
