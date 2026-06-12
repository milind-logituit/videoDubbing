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
import jiwer
import sacrebleu
import pandas as pd

from eval_lipsync import compute_lipsync_score

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

WHISPER_MODEL       = "base"
TTS_VOICE_FEMALE_HI = "hi-IN-SwaraNeural"
TTS_VOICE_MALE_HI   = "hi-IN-MadhurNeural"
TTS_VOICE_HI        = TTS_VOICE_FEMALE_HI   # default / backwards-compat
TTS_VOICE_EN        = "en-US-JennyNeural"
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
# Stage 3.5 — Speaker diarization + gender detection
# ─────────────────────────────────────────────────────────────────────────────

def diarize_speakers(audio_path: Path, hf_token: str) -> list[dict]:
    """Run pyannote speaker-diarization-3.1. Result cached as JSON beside audio."""
    cache = audio_path.with_suffix(".diarization.json")
    if cache.exists():
        print(f"  Loaded cached diarization: {cache.name}")
        return json.loads(cache.read_text())

    from pyannote.audio import Pipeline as _DPipeline
    print("  Loading pyannote/speaker-diarization-3.1 …")
    dia_pipeline = _DPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )
    raw = dia_pipeline(str(audio_path))
    annotation = (
        raw.speaker_diarization if hasattr(raw, "speaker_diarization") else raw
    )
    turns = [
        {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": label}
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    cache.write_text(json.dumps(turns, indent=2))
    print(f"  Diarization: {len(turns)} turns, "
          f"{len({t['speaker'] for t in turns})} speaker(s)")
    return turns


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Tag each segment with the speaker that overlaps it most.

    Falls back to the nearest turn (by midpoint distance) when no turn
    overlaps a segment — this handles silence gaps at turn boundaries.
    Runs in O(n log m) via sorted turns and early exit.
    """
    sorted_turns = sorted(turns, key=lambda t: t["start"])
    out = []
    for seg in segments:
        # Pass A — find best overlap using early-exit (O(log m) amortised)
        best: str | None = None
        best_overlap = 0.0
        for turn in sorted_turns:
            if turn["start"] > seg["end"]:
                break
            overlap = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if overlap > best_overlap:
                best_overlap, best = overlap, turn["speaker"]

        # Pass B — no overlap (silence gap): nearest turn by midpoint distance
        if best is None:
            seg_mid = (seg["start"] + seg["end"]) / 2
            best_dist = float("inf")
            for turn in sorted_turns:
                dist = abs((turn["start"] + turn["end"]) / 2 - seg_mid)
                if dist < best_dist:
                    best_dist, best = dist, turn["speaker"]

        if best is None:
            best = "SPEAKER_00"
            print(f"  [warn] no speaker found for segment "
                  f"{seg['start']:.1f}–{seg['end']:.1f}s; defaulting to {best}")
        out.append({**seg, "speaker": best})
    return out


_GENDER_MODEL_ID = "audeering/wav2vec2-large-robust-24-ft-age-gender"
# {0: female, 1: male, 2: child} — child maps to female voice
_GENDER_LABEL_MAP: dict[int, str] = {0: "female", 1: "male", 2: "female"}
_gender_model_cache: tuple | None = None  # (model, processor) loaded once per process
_FEMALE_CONFIDENCE_THRESH = 0.70  # require 70% confidence to assign female voice
BG_AUDIO_VOL = 0.20       # original audio vol during silence gaps (ambience)
BG_AUDIO_VOL_SPEECH = 0.04  # original audio vol during Hindi TTS (suppress EN dialogue)


def _load_gender_model() -> tuple:
    """Load audeering age-gender model, caching it for the process lifetime."""
    global _gender_model_cache
    if _gender_model_cache is not None:
        return _gender_model_cache

    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor, AutoConfig
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
    from huggingface_hub import hf_hub_download

    class _Head(nn.Module):
        def __init__(self, config: object, n: int) -> None:
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # type: ignore[arg-type]
            self.dropout = nn.Dropout(config.final_dropout)  # type: ignore[arg-type]
            self.out_proj = nn.Linear(config.hidden_size, n)  # type: ignore[arg-type]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out_proj(torch.tanh(self.dense(self.dropout(x))))

    class _AgeGenderModel(nn.Module):
        def __init__(self, config: object) -> None:
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model(config)  # type: ignore[arg-type]
            self.age = _Head(config, 1)
            self.gender = _Head(config, 3)

        def forward(self, input_values: torch.Tensor) -> torch.Tensor:
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return torch.softmax(self.gender(hidden), dim=1)

    print(f"  Loading gender classifier ({_GENDER_MODEL_ID}) …")
    config = AutoConfig.from_pretrained(_GENDER_MODEL_ID)  # nosec B615
    model = _AgeGenderModel(config)
    ckpt = hf_hub_download(_GENDER_MODEL_ID, "pytorch_model.bin")  # nosec B615
    missing, unexpected = model.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True), strict=False
    )
    if missing or unexpected:
        print(f"  [warn] gender model: missing={missing}, unexpected={unexpected}")
    model.eval()
    processor = Wav2Vec2Processor.from_pretrained(_GENDER_MODEL_ID)  # nosec B615
    _gender_model_cache = (model, processor)
    return _gender_model_cache


def _collect_speaker_chunks(
    audio: list, sr: int, segments: list[dict]
) -> dict[str, list]:
    speaker_chunks: dict[str, list] = {}
    for seg in segments:
        sp = seg.get("speaker", "SPEAKER_00")
        s_idx = int(seg["start"] * sr)
        e_idx = int(seg["end"] * sr)
        if e_idx > s_idx:
            speaker_chunks.setdefault(sp, []).append(audio[s_idx:e_idx])
    return speaker_chunks


def _assign_genders(
    speaker_probs: dict[str, tuple[float, float]],
    female_thresh: float = _FEMALE_CONFIDENCE_THRESH,
    relative_margin: float = 0.20,
) -> dict[str, str]:
    """
    Assign gender labels given per-speaker (f_prob, m_prob) tuples.

    Rules (in order):
    1. f_prob >= female_thresh          → female  (high confidence)
    2. f_prob < 0.50                    → male    (majority male)
    3. 0.50 ≤ f_prob < female_thresh    → borderline: female only if this speaker
       has the highest f_prob among all speakers by >= relative_margin, else male.
       Handles mixed-cast clips where one character is clearly "more female" than
       the rest without reaching the absolute threshold.
    """
    sorted_by_f = sorted(speaker_probs.items(), key=lambda x: -x[1][0])
    f_probs_desc = [v[0] for _, v in sorted_by_f]

    genders: dict[str, str] = {}
    for rank, (sp, (f_prob, _)) in enumerate(sorted_by_f):
        if f_prob >= female_thresh:
            genders[sp] = "female"
        elif f_prob < 0.50:
            genders[sp] = "male"
        else:
            # borderline: assign female only to the top-ranked speaker and only
            # if they lead the next speaker by at least relative_margin
            next_f = f_probs_desc[rank + 1] if rank + 1 < len(f_probs_desc) else 0.0
            if rank == 0 and (f_prob - next_f) >= relative_margin:
                genders[sp] = "female"
            else:
                genders[sp] = "male"
    return genders


def detect_speaker_genders(
    audio_path: Path,
    segments: list[dict],
    female_thresh: float = _FEMALE_CONFIDENCE_THRESH,
) -> dict[str, str]:
    """Classify gender per speaker using a wav2vec2 age-gender model."""
    import numpy as np
    import torch
    import librosa

    model, processor = _load_gender_model()
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    speaker_chunks = _collect_speaker_chunks(audio, sr, segments)

    speaker_probs: dict[str, tuple[float, float]] = {}
    for sp, chunks in speaker_chunks.items():
        chunk = np.concatenate(chunks).astype(np.float32)
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt",
                           padding=True)
        with torch.no_grad():
            probs = model(inputs["input_values"])
        speaker_probs[sp] = (float(probs[0, 0]), float(probs[0, 1]))

    genders = _assign_genders(speaker_probs, female_thresh=female_thresh)
    for sp, gender in genders.items():
        f_prob, m_prob = speaker_probs[sp]
        print(f"    {sp}: {gender} (f={f_prob:.2f} m={m_prob:.2f})")
    return genders


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Translation
# ─────────────────────────────────────────────────────────────────────────────

FILLER_MAP = {"hmm": "हाँ", "uh": "", "um": "", "ah": "अच्छा"}

_GRADE_SYSTEM = """\
You are a professional Hindi dubbing quality assessor.
Rate each segment on two dimensions (1 = poor, 5 = excellent):
- fidelity: semantic accuracy of Hindi vs English
- fluency: naturalness of the Hindi phrasing (grammar, word choice, register)

Respond ONLY with a JSON array in input order:
[{"id": <int>, "fidelity": <1-5>, "fluency": <1-5>}, ...]
Add "note": "..." only for scores ≤ 2. Omit otherwise.\
"""


def _isochrony_fit(ratio: float | None) -> int:
    """Map isochrony ratio to 1-5 fit score (algorithmic, reliable)."""
    if ratio is None:
        return 3
    if ratio <= 0.80:
        return 3  # too short — may sound slow
    if ratio <= 1.15:
        return 5  # perfect
    if ratio <= 1.30:
        return 4  # slightly over
    if ratio <= 1.50:
        return 3  # noticeably rushed
    if ratio <= 2.00:
        return 2  # significantly rushed
    return 1     # badly overflow


def translate_segments(whisper_result: dict) -> list[dict]:
    translator   = GoogleTranslator(source="en", target="hi")
    segments_out = []
    for seg in whisper_result["segments"]:
        en = seg["text"].strip()
        if not en:
            continue
        hi = translator.translate(en)
        # Post-process: replace or drop filler-only segments
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

_LLM_BATCH = 100  # max segments per Claude call (~2 min of audio at ~50 seg/min)

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
# Stage 6 — Hindi TTS  (timestamp-aligned + rate-controlled re-synthesis)
# ─────────────────────────────────────────────────────────────────────────────

MAX_RATE_PCT = 40  # cap edge-tts speed-up; 40% keeps speech intelligible


def _synth_segment(text: str, path: Path, rate: str = "+0%",
                   voice: str = TTS_VOICE_FEMALE_HI) -> None:
    """Synthesize one segment via edge-tts at the given rate; gTTS fallback."""
    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(str(path))
    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"  [warn] edge-tts failed ({exc}); falling back to gTTS")
        gTTS(text, lang="hi", slow=False).save(str(path))


def _seg_voice(seg: dict, voices: dict[str, str]) -> str:
    return voices.get(seg.get("speaker", ""), TTS_VOICE_FEMALE_HI)


def _voice_tag(voice: str) -> str:
    """Single-char cache key distinguishing male/female segment files."""
    return "m" if voice == TTS_VOICE_MALE_HI else "f"


def _synth_pass1(valid: list[dict], seg_dir: Path,
                 voices: dict[str, str]) -> float:
    """Synthesize all segments at normal rate; return total TTS duration in s."""
    for seg in valid:
        v = _seg_voice(seg, voices)
        p = seg_dir / f"seg_{seg['id']:03d}_{_voice_tag(v)}.mp3"
        if not p.exists():
            _synth_segment(seg["hi_text"], p, voice=v)
    return sum(
        len(AudioSegment.from_mp3(
            str(seg_dir / f"seg_{s['id']:03d}_{_voice_tag(_seg_voice(s, voices))}.mp3")
        )) / 1000
        for s in valid
    )


def _synth_pass2(valid: list[dict], seg_dir: Path, voices: dict[str, str],
                 global_rate: int, total_ms: int) -> AudioSegment:
    """Re-synthesize at global_rate and overlay each segment at its timestamp."""
    combined = AudioSegment.silent(duration=total_ms)
    for seg in valid:
        v = _seg_voice(seg, voices)
        tag = _voice_tag(v)
        if global_rate > 5:
            p_fast = seg_dir / f"seg_{seg['id']:03d}_{tag}_r{global_rate}.mp3"
            if not p_fast.exists():
                _synth_segment(seg["hi_text"], p_fast,
                               rate=f"+{global_rate}%", voice=v)
            seg_audio = AudioSegment.from_mp3(str(p_fast))
        else:
            seg_audio = AudioSegment.from_mp3(
                str(seg_dir / f"seg_{seg['id']:03d}_{tag}.mp3")
            )
        combined = combined.overlay(seg_audio, position=int(seg["start"] * 1000))
    return combined


def synthesize_hindi_audio(segments: list[dict], stem: str = "sample",
                           src_audio: Path | None = None,
                           speaker_voices: dict[str, str] | None = None,
                           force: bool = False) -> Path:
    dubbed_path = RAW / f"dubbed_hi_{stem}.mp3"
    if dubbed_path.exists() and not force:
        print(f"  Hindi audio already exists: {dubbed_path.name}")
        return dubbed_path

    seg_dir = PREPARED / f"hi_segments_{stem}"
    seg_dir.mkdir(exist_ok=True)
    voices = speaker_voices or {}

    if src_audio and src_audio.exists():
        total_ms = len(AudioSegment.from_file(str(src_audio)))
    else:
        total_ms = int(segments[-1]["end"] * 1000) + 500 if segments else 5000

    valid = [s for s in segments if s["hi_text"].strip()]
    combined_hi = " ".join(s["hi_text"] for s in valid)
    print(f"  Synthesising Hindi audio ({len(combined_hi)} chars, "
          f"base={total_ms / 1000:.1f}s) …")

    total_tts_s = _synth_pass1(valid, seg_dir, voices)
    total_speech_s = sum(s["duration"] for s in valid)
    raw_rate = int((total_tts_s / total_speech_s - 1) * 100)
    global_rate = max(0, min(raw_rate, MAX_RATE_PCT))
    print(f"  TTS {total_tts_s:.1f}s over {total_speech_s:.1f}s speech "
          f"→ uniform rate: +{global_rate}%")

    combined = _synth_pass2(valid, seg_dir, voices, global_rate, total_ms)
    combined.export(str(dubbed_path), format="mp3")
    print(f"  Hindi dubbed audio: {dubbed_path.name}  ({len(combined) / 1000:.1f}s)")
    return dubbed_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Create dubbed video
# ─────────────────────────────────────────────────────────────────────────────

def _duck_original_audio(
    audio_path: Path, segments: list[dict],
) -> Path:
    """Build ducked original audio: quiet during speech segments, fuller in gaps.

    pydub `+N` means +N dB. Convert linear ratios via 20*log10(ratio).
    """
    import math
    db_gap    = 20 * math.log10(BG_AUDIO_VOL)         # -14 dB @ 20%
    db_speech = 20 * math.log10(BG_AUDIO_VOL_SPEECH)  # -28 dB @ 4%

    orig = AudioSegment.from_file(str(audio_path))
    ducked = AudioSegment.silent(duration=len(orig))

    intervals = sorted(
        (int(s["start"] * 1000), int(s["end"] * 1000))
        for s in segments if s.get("hi_text", "").strip()
    )

    prev_end = 0
    for start_ms, end_ms in intervals:
        if start_ms > prev_end:
            ducked = ducked.overlay(
                orig[prev_end:start_ms] + db_gap, position=prev_end
            )
        ducked = ducked.overlay(
            orig[start_ms:end_ms] + db_speech, position=start_ms
        )
        prev_end = end_ms
    if prev_end < len(orig):
        ducked = ducked.overlay(orig[prev_end:] + db_gap, position=prev_end)

    out = audio_path.with_suffix(".ducked.wav")
    ducked.export(str(out), format="wav")
    return out


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
# Stage 8 — Quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(whisper_result: dict, segments: list[dict],
                    src_audio: Path, dubbed_audio: Path,
                    has_reference: bool = True) -> dict:
    machine_hindi = " ".join(s["hi_text"] for s in segments)

    if has_reference:
        wer_score = jiwer.wer(FULL_SOURCE_TEXT.lower(),
                               whisper_result["text"].lower())
        bleu_obj  = sacrebleu.corpus_bleu([machine_hindi], [[REFERENCE_HINDI]])
        wer_val: float | None       = round(wer_score, 4)
        wer_pct_val: float | None   = round(wer_score * 100, 2)
        bleu_val: float | None      = round(bleu_obj.score, 2)
        wer_note  = None
        bleu_note = None
    else:
        wer_val = wer_pct_val = bleu_val = None
        wer_note  = "N/A — no reference text for this clip"
        bleu_note = "N/A — no reference text for this clip"

    try:
        src_dur = len(AudioSegment.from_file(str(src_audio))) / 1000
        dub_dur = len(AudioSegment.from_mp3(str(dubbed_audio))) / 1000
        dar     = dub_dur / src_dur
    except Exception:
        src_dur = dub_dur = dar = 0.0

    asr_block: dict = {"model": WHISPER_MODEL, "wer": wer_val,
                       "wer_pct": wer_pct_val}
    if wer_note is not None:
        asr_block["note"] = wer_note

    trans_block: dict = {"model": "Google Translate (en→hi)",
                         "bleu": bleu_val,
                         "n_segments": len(segments)}
    if bleu_note is not None:
        trans_block["note"] = bleu_note

    return {
        "asr":         asr_block,
        "translation": trans_block,
        "tts":         {"voice": TTS_VOICE_HI},
        "alignment":   {"source_duration_s": round(src_dur, 2),
                        "dubbed_duration_s": round(dub_dur, 2),
                        "duration_ratio": round(dar, 3),
                        "en_chars_per_sec": round(
                            len(FULL_SOURCE_TEXT.replace(" ", "")) / src_dur, 1
                        ) if src_dur else 0,
                        "hi_chars_per_sec": round(
                            len(machine_hindi.replace(" ", "")) / dub_dur, 1
                        ) if dub_dur else 0},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8c — Back-translation BLEU
# ─────────────────────────────────────────────────────────────────────────────

def compute_back_translation_bleu(
    dubbed_audio: Path,
    original_en_text: str,
) -> dict:
    """ASR the dubbed Hindi audio, back-translate to English, score with BLEU.

    Returns a dict ready to merge into metrics["back_translation"].
    Uses the same Whisper model already loaded for Stage 3.
    """
    try:
        print("  Transcribing dubbed Hindi audio …")
        hi_model = whisper.load_model(WHISPER_MODEL)
        hi_result = hi_model.transcribe(str(dubbed_audio), language="hi")
        hi_transcript = hi_result["text"].strip()

        print("  Back-translating Hindi → English …")
        bt_en = GoogleTranslator(source="hi", target="en").translate(hi_transcript)

        bleu_obj = sacrebleu.corpus_bleu([bt_en], [[original_en_text]])
        bleu_score = round(bleu_obj.score, 2)

        return {
            "hi_transcript_chars": len(hi_transcript),
            "back_translated_en": bt_en,
            "bleu": bleu_score,
        }
    except Exception as exc:
        return {"bleu": None, "note": f"Failed: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8b — Segment-level quality (isochrony + LLM grading)
# ─────────────────────────────────────────────────────────────────────────────

def compute_segment_isochrony(
    segments: list[dict], seg_dir: Path
) -> list[dict]:
    """Per-segment isochrony ratio: TTS duration / EN window duration."""
    results = []
    for seg in segments:
        sid = int(seg["id"])
        en_dur = round(float(seg["end"]) - float(seg["start"]), 2)
        candidates = list(seg_dir.glob(f"seg_{sid:03d}_*.mp3"))
        if candidates:
            tts_dur = round(len(AudioSegment.from_mp3(str(candidates[0]))) / 1000.0, 2)
            ratio = round(tts_dur / en_dur, 3) if en_dur > 0 else None
        else:
            tts_dur = ratio = None
        results.append({
            "id": sid,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "en_duration_s": en_dur,
            "tts_duration_s": tts_dur,
            "isochrony_ratio": ratio,
        })
    return results


def grade_translations(segments: list[dict]) -> list[dict]:
    """Call Claude Haiku to rate fidelity / fluency / fit per segment."""
    payload = [
        {"id": int(seg["id"]), "en": seg["en_text"], "hi": seg["hi_text"],
         "duration_s": round(float(seg["end"]) - float(seg["start"]), 1)}
        for seg in segments
        if seg.get("hi_text", "").strip()
    ]
    if not payload:
        return []
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=_GRADE_SYSTEM,
        messages=[{"role": "user",
                   "content": json.dumps(payload, ensure_ascii=False)}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except Exception as exc:
        # Retry once: ask Claude to return only the JSON array
        print(f"  [warn] grade_translations parse failed ({exc}), retrying …")
        retry = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Return ONLY the JSON array, no markdown, no explanation."},
            ],
            system=_GRADE_SYSTEM,
        )
        raw2 = retry.content[0].text.strip()
        if raw2.startswith("```"):
            raw2 = raw2.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(raw2)
        except Exception as exc2:
            print(f"  [warn] grade_translations retry also failed ({exc2}), skipping grades.")
            return []


def _merge_segment_quality(
    isochrony: list[dict], grades_by_id: dict[int, dict]
) -> list[dict]:
    out = []
    for row in isochrony:
        g = grades_by_id.get(row["id"], {})
        merged = dict(row)
        # fit is always algorithmic; fidelity/fluency come from LLM
        merged["fit"] = _isochrony_fit(row.get("isochrony_ratio"))
        if g:
            merged["fidelity"] = g.get("fidelity")
            merged["fluency"]  = g.get("fluency")
            if "note" in g:
                merged["note"] = g["note"]
        out.append(merged)
    return out


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
    metrics["lipsync"] = compute_lipsync_score(dubbed_video)
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
