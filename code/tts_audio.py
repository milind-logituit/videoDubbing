"""Stage 6 — Hindi TTS synthesis and audio ducking/mixing."""
import asyncio
import math
from pathlib import Path

from gtts import gTTS
from pydub import AudioSegment

import edge_tts

ROOT     = Path(__file__).parent.parent
RAW      = ROOT / "data/raw"
PREPARED = ROOT / "data/prepared"

TTS_VOICE_FEMALE_HI = "hi-IN-SwaraNeural"
TTS_VOICE_MALE_HI   = "hi-IN-MadhurNeural"
TTS_VOICE_HI        = TTS_VOICE_FEMALE_HI   # default / backwards-compat
TTS_BASE_RATE_PCT   = -10
MAX_RATE_PCT        = 40   # cap edge-tts speed-up; 40% keeps speech intelligible
BG_AUDIO_VOL        = 0.20       # original audio vol during silence gaps
BG_AUDIO_VOL_SPEECH = 0.04       # original audio vol during Hindi TTS


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
            _synth_segment(seg["hi_text"], p, rate=f"{TTS_BASE_RATE_PCT:+d}%", voice=v)
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
        effective_rate = TTS_BASE_RATE_PCT + global_rate
        if effective_rate != 0:
            p_fast = seg_dir / f"seg_{seg['id']:03d}_{tag}_r{effective_rate}.mp3"
            if not p_fast.exists():
                _synth_segment(seg["hi_text"], p_fast,
                               rate=f"{effective_rate:+d}%", voice=v)
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


def _duck_original_audio(audio_path: Path, segments: list[dict]) -> Path:
    """Build ducked original audio: quiet during speech segments, fuller in gaps."""
    db_gap    = 20 * math.log10(BG_AUDIO_VOL)
    db_speech = 20 * math.log10(BG_AUDIO_VOL_SPEECH)

    orig   = AudioSegment.from_file(str(audio_path))
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
