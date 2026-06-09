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
import asyncio
import json
import subprocess
from pathlib import Path

import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
from PIL import Image, ImageDraw
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

# ── Constants ─────────────────────────────────────────────────────────────────
SOURCE_SCRIPT = [
    {"id": 1, "speaker": "Narrator",
     "text": ("In a world where streaming has replaced the multiplex, "
               "content is king — and every second of screen time must earn its place.")},
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

WHISPER_MODEL = "base"
TTS_VOICE_HI  = "hi-IN-SwaraNeural"
TTS_VOICE_EN  = "en-US-JennyNeural"
VIDEO_SIZE    = (1280, 720)
VIDEO_FPS     = 24
BG_COLOR      = (15, 23, 42)      # #0f172a — dark slate
ACCENT_COLOR  = (251, 191, 36)    # #fbbf24 — amber


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Generate sample video
# ─────────────────────────────────────────────────────────────────────────────

def _generate_source_audio(force: bool = False) -> Path:
    audio_path = RAW / "source_en.mp3"
    if audio_path.exists() and not force:
        return audio_path
    print("  Generating English source audio …")
    async def _synth():
        communicate = edge_tts.Communicate(FULL_SOURCE_TEXT, TTS_VOICE_EN)
        await communicate.save(str(audio_path))
    try:
        asyncio.run(_synth())
    except Exception:
        gTTS(FULL_SOURCE_TEXT, lang="en", slow=False).save(str(audio_path))
    return audio_path


def _make_title_frame() -> Path:
    """Pillow: branded dark title card for the sample video."""
    img = Image.new("RGB", VIDEO_SIZE, BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Accent bar at top
    draw.rectangle([0, 0, VIDEO_SIZE[0], 6], fill=ACCENT_COLOR)

    # Main title text (system font fallback)
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

    draw.text((centered_x(title, font_lg), 270), title, font=font_lg, fill=(255, 255, 255))
    draw.text((centered_x(sub, font_sm),   370), sub,   font=font_sm, fill=(148, 163, 184))
    draw.text((centered_x(tag, font_xs),   430), tag,   font=font_xs, fill=(100, 116, 139))
    draw.rectangle([0, VIDEO_SIZE[1] - 6, VIDEO_SIZE[0], VIDEO_SIZE[1]], fill=ACCENT_COLOR)

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

def transcribe(audio_path: Path) -> dict:
    print(f"  Loading Whisper '{WHISPER_MODEL}' …")
    model = whisper.load_model(WHISPER_MODEL)
    print(f"  Transcribing {audio_path.name} …")
    result = model.transcribe(str(audio_path), language="en",
                               word_timestamps=True, verbose=False)
    print(f"  Transcript: {result['text'].strip()[:100]} …")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Translation
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
            "end":      round(seg["end"],   3),
            "duration": round(seg["end"] - seg["start"], 3),
            "en_text":  en,
            "hi_text":  hi,
        })
        print(f"    [{seg['start']:.1f}s]  {en[:55]}")
        print(f"           →  {hi[:55]}")
    print(f"  Translated {len(segments_out)} segments.")
    return segments_out


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
# Stage 6 — Hindi TTS
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_hindi_audio(segments: list[dict], force: bool = False) -> Path:
    dubbed_path = RAW / "dubbed_hi_v2.mp3"
    if dubbed_path.exists() and not force:
        print(f"  Hindi audio already exists: {dubbed_path.name}")
        return dubbed_path

    seg_dir = PREPARED / "hi_segments_v2"
    seg_dir.mkdir(exist_ok=True)
    combined_hi = " ".join(s["hi_text"] for s in segments)
    print(f"  Synthesising Hindi audio ({len(combined_hi)} chars) …")

    seg_paths = []
    for seg in segments:
        p = seg_dir / f"seg_{seg['id']:03d}.mp3"
        if not p.exists():
            try:
                async def _s(text=seg["hi_text"], path=p):
                    communicate = edge_tts.Communicate(text, TTS_VOICE_HI)
                    await communicate.save(str(path))
                asyncio.run(_s())
            except Exception:
                gTTS(seg["hi_text"], lang="hi", slow=False).save(str(p))
        seg_paths.append(p)

    combined = AudioSegment.empty()
    for p in seg_paths:
        combined += AudioSegment.from_mp3(str(p))
    combined.export(str(dubbed_path), format="mp3")
    print(f"  Hindi dubbed audio: {dubbed_path.name}  ({len(combined)/1000:.1f}s)")
    return dubbed_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Create dubbed video
# ─────────────────────────────────────────────────────────────────────────────

def create_dubbed_video(video_path: Path, hindi_audio_path: Path,
                        force: bool = False) -> Path:
    dubbed_video = RAW / f"{video_path.stem}_dubbed_hi.mp4"
    if dubbed_video.exists() and not force:
        print(f"  Dubbed video already exists: {dubbed_video.name}")
        return dubbed_video

    print(f"  Replacing audio track …")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(hindi_audio_path),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(dubbed_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  Dubbed video saved: {dubbed_video.name}")
    return dubbed_video


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8 — Quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(whisper_result: dict, segments: list[dict],
                    src_audio: Path, dubbed_audio: Path) -> dict:
    wer_score = jiwer.wer(FULL_SOURCE_TEXT.lower(), whisper_result["text"].lower())
    machine_hindi = " ".join(s["hi_text"] for s in segments)
    bleu = sacrebleu.corpus_bleu([machine_hindi], [[REFERENCE_HINDI]])

    try:
        src_dur = len(AudioSegment.from_mp3(str(src_audio))) / 1000
        dub_dur = len(AudioSegment.from_mp3(str(dubbed_audio))) / 1000
        dar     = dub_dur / src_dur
    except Exception:
        src_dur = dub_dur = dar = 0.0

    return {
        "asr":         {"model": WHISPER_MODEL, "wer": round(wer_score, 4),
                        "wer_pct": round(wer_score * 100, 2)},
        "translation": {"model": "Google Translate (en→hi)",
                        "bleu": round(bleu.score, 2),
                        "n_segments": len(segments)},
        "tts":         {"voice": TTS_VOICE_HI},
        "alignment":   {"source_duration_s": round(src_dur, 2),
                        "dubbed_duration_s": round(dub_dur, 2),
                        "duration_ratio": round(dar, 3),
                        "en_chars_per_sec": round(len(FULL_SOURCE_TEXT.replace(" ","")) / src_dur, 1) if src_dur else 0,
                        "hi_chars_per_sec": round(len(machine_hindi.replace(" ","")) / dub_dur, 1) if dub_dur else 0},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(segments, vtt, srt, metrics, src_audio, dubbed_audio):
    pd.DataFrame(segments).to_csv(PREPARED / "transcript_bilingual_v2.csv", index=False)
    (MODEL_OUT / "subtitles_hi.vtt").write_text(vtt, encoding="utf-8")
    (MODEL_OUT / "subtitles_hi.srt").write_text(srt, encoding="utf-8")
    with open(MODEL_OUT / "metrics_v2.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved outputs to {MODEL_OUT}/")
    for f in sorted(MODEL_OUT.iterdir()):
        print(f"  {f.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Stage 1 — Generating sample video …")
    video_path = generate_sample_video()

    print("\nStage 2 — Extracting audio …")
    audio_path = extract_audio(video_path)

    print("\nStage 3 — ASR (Whisper) …")
    whisper_result = transcribe(audio_path)

    print("\nStage 4 — Translation (Google Translate) …")
    segments = translate_segments(whisper_result)

    print("\nStage 5 — Generating subtitle files …")
    vtt = generate_vtt(segments)
    srt = generate_srt(segments)
    print(f"  VTT: {len(vtt)} chars  |  SRT: {len(srt)} chars")

    print("\nStage 6 — Hindi TTS …")
    hindi_audio = synthesize_hindi_audio(segments)

    print("\nStage 7 — Creating dubbed video …")
    dubbed_video = create_dubbed_video(video_path, hindi_audio)

    print("\nStage 8 — Computing metrics …")
    src_audio = RAW / "source_en.mp3"
    metrics   = compute_metrics(whisper_result, segments, src_audio, hindi_audio)

    save_outputs(segments, vtt, srt, metrics, src_audio, hindi_audio)

    print("\n── Quality metrics ──")
    print(f"  ASR WER          : {metrics['asr']['wer_pct']:.1f}%")
    print(f"  Translation BLEU : {metrics['translation']['bleu']:.1f}")
    print(f"  Duration ratio   : {metrics['alignment']['duration_ratio']:.3f}")
    print("\nDone.")
