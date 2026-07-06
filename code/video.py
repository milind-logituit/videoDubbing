"""Stages 1/5/7/7c — Sample video, subtitle files, dubbed-video mux, loudnorm."""
import asyncio
import subprocess
from pathlib import Path

from gtts import gTTS
from PIL import Image, ImageDraw
from pydub import AudioSegment

import edge_tts

from tts_audio import (
    _duck_original_audio,
    BG_AUDIO_VOL, BG_AUDIO_VOL_SPEECH,
)

ROOT     = Path(__file__).parent.parent
RAW      = ROOT / "data/raw"
PREPARED = ROOT / "data/prepared"

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

VIDEO_SIZE    = (1280, 720)
VIDEO_FPS     = 24
BG_COLOR      = (15, 23, 42)
ACCENT_COLOR  = (251, 191, 36)


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
        speaker = seg.get("speaker", "")
        text    = seg["hi_text"]
        if speaker:
            text = f"<v {speaker}>{text}</v>"
        lines += [f"{_vtt_time(seg['start'])} --> {_vtt_time(seg['end'])}",
                  text, ""]
    return "\n".join(lines)


def generate_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        speaker = seg.get("speaker", "")
        text    = f"[{speaker}] {seg['hi_text']}" if speaker else seg["hi_text"]
        lines += [str(i),
                  f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}",
                  text, ""]
    return "\n".join(lines)


def apply_loudnorm(
    input_path: Path,
    target_lufs: float = -27.0,
    true_peak: float = -1.5,
    lra: float = 11.0,
    force: bool = False,
) -> Path:
    """Normalise audio loudness to Netflix standard (−27 LKFS ±2 LU, TP −1.5 dBTP).

    Two-pass loudnorm: pass 1 measures, pass 2 applies with linear gain so the
    video track is never re-encoded.  Output is written alongside the input as
    <stem>_norm<suffix>.
    """
    out_path = input_path.with_stem(input_path.stem + "_norm")
    if out_path.exists() and not force:
        print(f"  Loudnorm output already exists: {out_path.name}")
        return out_path

    filter_str = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:print_format=json"

    # Pass 1 — measure
    r1 = subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-af", filter_str,
         "-vn", "-f", "null", "/dev/null"],
        capture_output=True, text=True,
    )
    # loudnorm prints JSON to stderr
    import json as _json
    import re as _re
    m = _re.search(r"\{[^{}]+\}", r1.stderr, _re.DOTALL)
    if not m:
        raise RuntimeError(f"loudnorm pass 1 failed to produce JSON:\n{r1.stderr[-500:]}")
    meas = _json.loads(m.group())

    # Pass 2 — apply with measured values (linear=true for accurate gain)
    filter2 = (
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}"
        f":measured_I={meas['input_i']}"
        f":measured_TP={meas['input_tp']}"
        f":measured_LRA={meas['input_lra']}"
        f":measured_thresh={meas['input_thresh']}"
        f":offset={meas['target_offset']}"
        f":linear=true"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-af", filter2, "-c:v", "copy", str(out_path)],
        check=True, capture_output=True,
    )
    print(f"  Loudnorm applied (target {target_lufs} LUFS): {out_path.name}")
    return out_path


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
