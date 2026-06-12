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
    TTS_VOICE_FEMALE_HI, TTS_VOICE_MALE_HI,
    TTS_VOICE_HI, TTS_BASE_RATE_PCT,                         # noqa: F401
    BG_AUDIO_VOL, BG_AUDIO_VOL_SPEECH,
)
from metrics import (                                         # noqa: E402
    compute_metrics, compute_back_translation_bleu,
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

WHISPER_MODEL    = "base"
WHISPER_MODEL_HI = "medium"
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

FILLER_MAP = {"hmm": "हाँ", "uh": "", "um": "", "ah": "अच्छा"}


def translate_segments(whisper_result: dict) -> list[dict]:
    translator   = GoogleTranslator(source="en", target="hi")
    segments_out = []
    for seg in whisper_result["segments"]:
        en = seg["text"].strip()
        if not en:
            continue
        hi = translator.translate(en)
        if en.lower().rstrip(".!?,") in FILLER_MAP:
            hi = FILLER_MAP[en.lower().rstrip(".!?,")]
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

_REFINE_SYSTEM = (
    "You are a professional Hindi dubbing editor for OTT streaming content "
    "(Eros Now / SunNxt).\n"
    "Input: JSON array of segments, each with ASR English (en_text), "
    "Google-Translate Hindi (hi_text), and duration_s (seconds available "
    "to speak this line).\n"
    "For each segment:\n"
    "  1. Fix ASR transcription errors in en_text "
    "(e.g. 'half is likely' → 'half as likely').\n"
    "  2. Rewrite hi_text as natural spoken Hindi for dubbing that fits "
    "within duration_s seconds.\n"
    "     • Hindi TTS speaks at ~3.5 words/second — use this to judge "
    "length. A 2s window fits ~7 Hindi words maximum.\n"
    "     • Prefer shorter, natural phrasing over complete sentences when "
    "the window is tight. Cut filler and subordinate clauses first.\n"
    "     • Use common, everyday Hindi vocabulary (Hindustani/Bollywood register) "
    "that speech recognition systems reliably transcribe. "
    "Avoid rare, Sanskritised, or literary Hindi words — prefer their "
    "everyday equivalents (e.g. 'काम' over 'कार्य', 'बात' over 'वार्तालाप').\n"
    "     • Distinguish dinner vs supper, couch vs sofa, etc.\n"
    "     • Fillers: 'Hmm' → 'हाँ', 'Uh'/'Um' → empty string, "
    "'Ah' → 'अच्छा'.\n"
    "Return ONLY a valid JSON array: "
    '[{"id": int, "en_text": str, "hi_text": str}, …]. '
    "Do NOT include duration_s in output. "
    "Same count and IDs as input. No markdown, no explanation."
)


def _refine_batch(client: anthropic.Anthropic, batch: list[dict]) -> dict[int, dict]:
    payload = [
        {
            "id": s["id"],
            "en_text": s["en_text"],
            "hi_text": s["hi_text"],
            "duration_s": round(s["duration"], 2),
        }
        for s in batch
    ]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=_REFINE_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return {r["id"]: r for r in json.loads(raw)}


def refine_segments(segments: list[dict], *, skip: bool = False) -> list[dict]:
    """Fix ASR errors and rewrite Hindi as natural dubbing-quality speech via Claude."""
    if skip:
        print("  Stage 4b skipped (--no-llm).")
        return segments

    client = anthropic.Anthropic()
    n = len(segments)
    n_batches = (n + _LLM_BATCH - 1) // _LLM_BATCH
    print(
        f"  Calling Claude (claude-sonnet-4-6) to refine {n} segments "
        f"in {n_batches} batch(es) …"
    )

    refined: dict[int, dict] = {}
    for i in range(n_batches):
        batch = segments[i * _LLM_BATCH : (i + 1) * _LLM_BATCH]
        if n_batches > 1:
            print(f"    Batch {i + 1}/{n_batches} ({len(batch)} segments) …")
        refined.update(_refine_batch(client, batch))

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
) -> Path:
    dubbed_video = RAW / f"{video_path.stem}_dubbed_hi.mp4"
    if dubbed_video.exists() and not force:
        print(f"  Dubbed video already exists: {dubbed_video.name}")
        return dubbed_video

    orig_audio = PREPARED / f"{video_path.stem}_audio.wav"
    if segments and orig_audio.exists():
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
                 stem: str = "sample"):
    pd.DataFrame(segments).to_csv(
        PREPARED / f"transcript_bilingual_{stem}.csv", index=False
    )
    (MODEL_OUT / f"subtitles_{stem}_hi.vtt").write_text(vtt, encoding="utf-8")
    (MODEL_OUT / f"subtitles_{stem}_hi.srt").write_text(srt, encoding="utf-8")
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
    args = parser.parse_args()

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

    print("\nStage 3 — ASR (Whisper) …")
    whisper_result = transcribe(audio_path)

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
            speaker_voices = {
                sp: (TTS_VOICE_MALE_HI if g == "male" else TTS_VOICE_FEMALE_HI)
                for sp, g in genders.items()
            }
            print(f"  Voice map: {speaker_voices}")

    print("\nStage 4 — Translation (Google Translate) …")
    segments = translate_segments(whisper_result)

    print("\nStage 4b — LLM post-correction (Claude) …")
    segments = refine_segments(segments, skip=args.no_llm)

    print("\nStage 5 — Generating subtitle files …")
    vtt = generate_vtt(segments)
    srt = generate_srt(segments)
    print(f"  VTT: {len(vtt)} chars  |  SRT: {len(srt)} chars")

    print("\nStage 6 — Hindi TTS …")
    hindi_audio = synthesize_hindi_audio(
        segments, stem=video_path.stem, src_audio=audio_path,
        speaker_voices=speaker_voices,
    )

    print("\nStage 7 — Creating dubbed video …")
    dubbed_video = create_dubbed_video(video_path, hindi_audio, segments=segments)

    if args.lipsync:
        print("\nStage 7b — Wav2Lip lip-sync …")
        ls_out = RAW / f"{video_path.stem}_lipsync_hi.mp4"
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
    original_en = whisper_result["text"].strip()
    metrics["back_translation"] = compute_back_translation_bleu(
        hindi_audio, original_en
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

    save_outputs(segments, vtt, srt, metrics, src_audio, hindi_audio,
                 stem=video_path.stem)

    print("\n── Quality metrics ──")
    wer_pct    = metrics["asr"]["wer_pct"]
    bleu       = metrics["translation"]["bleu"]
    bt_bleu    = metrics["back_translation"].get("bleu")
    sync_score = metrics["lipsync"].get("sync_score")
    print(f"  ASR WER              : {f'{wer_pct:.1f}%' if wer_pct is not None else 'N/A'}")
    print(f"  Translation BLEU     : {f'{bleu:.1f}' if bleu is not None else 'N/A'}")
    print(f"  Back-translation BLEU: {f'{bt_bleu:.1f}' if bt_bleu is not None else 'N/A'}")
    print(f"  Lip-sync score       : {f'{sync_score:.3f}' if sync_score is not None else 'N/A'}")
    print(f"  Duration ratio       : {metrics['alignment']['duration_ratio']:.3f}")
    print("\nDone.")

