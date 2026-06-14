"""Stage 6 — Hindi TTS synthesis and audio ducking/mixing."""
import asyncio
import math
import os
import subprocess
import tempfile
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
TTS_VOICE_FEMALE_EN = "en-US-JennyNeural"
TTS_VOICE_MALE_EN   = "en-US-GuyNeural"
TTS_VOICE_EN        = TTS_VOICE_FEMALE_EN
_TTS_DEFAULT_VOICE  = {"hi": TTS_VOICE_FEMALE_HI, "en": TTS_VOICE_FEMALE_EN}

# Ordered pools for per-speaker voice assignment. Speakers are sorted by first
# utterance then assigned cyclically — first male speaker gets pool[0], second
# gets pool[1], etc. Add voices here as they become available.
VOICE_POOL: dict[str, dict[str, list[str]]] = {
    "hi": {
        "male":   ["hi-IN-MadhurNeural"],
        "female": ["hi-IN-SwaraNeural"],
    },
    "en": {
        "male":   [
            "en-US-GuyNeural",
            "en-US-AndrewNeural",
            "en-US-ChristopherNeural",
            "en-US-EricNeural",
        ],
        "female": [
            "en-US-JennyNeural",
            "en-US-AriaNeural",
            "en-US-SaraNeural",
            "en-US-NancyNeural",
        ],
    },
}

_MALE_VOICES: set[str] = {
    v
    for lang_pools in VOICE_POOL.values()
    for v in lang_pools.get("male", [])
} | {TTS_VOICE_MALE_HI, TTS_VOICE_MALE_EN}
TTS_BASE_RATE_PCT   = -10
MAX_RATE_PCT        = 40    # cap edge-tts speed-up; 40% keeps speech intelligible
MAX_ATEMPO_FACTOR   = 4.0   # ffmpeg atempo cap; beyond this we hard-trim
MIN_DUB_DURATION_S  = 0.25  # skip micro-segments shorter than this when ratio > 4
BG_AUDIO_VOL        = 0.20       # original audio vol during silence gaps
BG_AUDIO_VOL_SPEECH = 0.04       # original audio vol during TTS


def _atempo_compress(audio: AudioSegment, target_s: float) -> AudioSegment:
    """Compress audio to target_s via ffmpeg atempo (≤4x); hard-trim beyond that."""
    actual_s = len(audio) / 1000.0
    if actual_s <= target_s * 1.05:
        return audio

    factor = min(actual_s / target_s, MAX_ATEMPO_FACTOR)
    filters: list[str] = []
    rem = factor
    while rem > 2.0 + 1e-6:
        filters.append("atempo=2.0")
        rem /= 2.0
    if rem > 1.001:
        filters.append(f"atempo={rem:.5f}")

    tmp_in = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_out = tmp_in.name.replace(".mp3", "_out.mp3")
    try:
        audio.export(tmp_in.name, format="mp3")
        tmp_in.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in.name, "-filter:a", ",".join(filters), tmp_out],
            check=True, capture_output=True,
        )
        result = AudioSegment.from_mp3(tmp_out)
    finally:
        os.unlink(tmp_in.name)
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)

    target_ms = int(target_s * 1000)
    return result[:target_ms] if len(result) > target_ms else result


def _emotion_tag(emotion: str | None) -> str:
    """Short suffix for cache filename so neutral/angry/sad files don't collide.

    v2 adds rate modulation to prosody — suffix includes 'r' to invalidate
    any cached files built without rate.
    """
    if not emotion or emotion == "neutral":
        return ""
    return f"_r{emotion[:3]}"  # _rhap / _rang / _rsad  (r = rate-aware prosody)


def _synth_segment(text: str, path: Path, rate: str = "+0%",
                   voice: str = TTS_VOICE_FEMALE_HI,
                   emotion: str | None = None) -> None:
    """Synthesize one segment with SSML prosody (pitch + volume + rate) for emotion."""
    from emotion import SSML_PROSODY
    prosody = SSML_PROSODY.get(emotion or "neutral", SSML_PROSODY["neutral"])
    if emotion and emotion != "neutral":
        content = (
            f'<prosody pitch="{prosody["pitch"]}" '
            f'volume="{prosody["volume"]}" '
            f'rate="{prosody["rate"]}">'
            f"{text}"
            f"</prosody>"
        )
    else:
        content = text

    async def _run() -> None:
        communicate = edge_tts.Communicate(content, voice, rate=rate)
        await communicate.save(str(path))
    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"  [warn] edge-tts failed ({exc}); falling back to gTTS")
        lang = voice[:2]
        gTTS(text, lang=lang, slow=False).save(str(path))


def _seg_voice(seg: dict, voices: dict[str, str],
               default: str = TTS_VOICE_FEMALE_HI) -> str:
    return voices.get(seg.get("speaker", ""), default)


def _voice_tag(voice: str) -> str:
    """Single-char cache key distinguishing male/female segment files."""
    return "m" if voice in _MALE_VOICES else "f"


def _synth_pass1(valid: list[dict], seg_dir: Path, voices: dict[str, str],
                 default_voice: str = TTS_VOICE_FEMALE_HI) -> dict[int, float]:
    """Synthesize all segments at base rate; return {seg_id: tts_duration_s}."""
    durations: dict[int, float] = {}
    for seg in valid:
        v   = _seg_voice(seg, voices, default_voice)
        emo = seg.get("emotion")
        p   = seg_dir / f"seg_{seg['id']:03d}_{_voice_tag(v)}{_emotion_tag(emo)}.mp3"
        if not p.exists():
            _synth_segment(seg["hi_text"], p, rate=f"{TTS_BASE_RATE_PCT:+d}%",
                           voice=v, emotion=emo)
        durations[seg["id"]] = len(AudioSegment.from_mp3(str(p))) / 1000.0
    return durations


def _synth_pass2(valid: list[dict], seg_dir: Path, voices: dict[str, str],
                 seg_durations: dict[int, float], total_ms: int,
                 default_voice: str = TTS_VOICE_FEMALE_HI) -> AudioSegment:
    """Re-synthesize each segment at its own rate; atempo-compress if still overflowing."""
    combined = AudioSegment.silent(duration=total_ms)
    for seg in valid:
        v        = _seg_voice(seg, voices, default_voice)
        tag      = _voice_tag(v)
        emo      = seg.get("emotion")
        etag     = _emotion_tag(emo)
        tts_s    = seg_durations[seg["id"]]
        target_s = seg["end"] - seg["start"]

        if target_s < MIN_DUB_DURATION_S and tts_s / target_s > 4.0:
            # micro-segment: write silence so metrics reads effective duration correctly
            effective_path = seg_dir / f"seg_{seg['id']:03d}_{tag}_effective.mp3"
            AudioSegment.silent(duration=int(target_s * 1000) or 20).export(
                str(effective_path), format="mp3"
            )
            continue

        needed_pct = int((tts_s / target_s - 1) * 100) + TTS_BASE_RATE_PCT
        edge_rate  = max(TTS_BASE_RATE_PCT, min(needed_pct, MAX_RATE_PCT))

        if edge_rate != TTS_BASE_RATE_PCT:
            p_fast = seg_dir / f"seg_{seg['id']:03d}_{tag}_r{edge_rate}{etag}.mp3"
            if not p_fast.exists():
                _synth_segment(seg["hi_text"], p_fast,
                               rate=f"{edge_rate:+d}%", voice=v, emotion=emo)
            seg_audio = AudioSegment.from_mp3(str(p_fast))
        else:
            seg_audio = AudioSegment.from_mp3(
                str(seg_dir / f"seg_{seg['id']:03d}_{tag}{etag}.mp3")
            )

        if len(seg_audio) / 1000.0 > target_s * 1.05:
            seg_audio = _atempo_compress(seg_audio, target_s)

        # persist the effective (post-atempo) audio so metrics read the right duration
        effective_path = seg_dir / f"seg_{seg['id']:03d}_{tag}_effective.mp3"
        seg_audio.export(str(effective_path), format="mp3")

        combined = combined.overlay(seg_audio, position=int(seg["start"] * 1000))
    return combined


def synthesize_hindi_audio(segments: list[dict], stem: str = "sample",
                            src_audio: Path | None = None,
                            speaker_voices: dict[str, str] | None = None,
                            force: bool = False,
                            target_lang: str = "hi") -> Path:
    default_voice = _TTS_DEFAULT_VOICE.get(target_lang, TTS_VOICE_FEMALE_HI)
    dubbed_path = RAW / f"dubbed_{target_lang}_{stem}.mp3"
    if dubbed_path.exists() and not force:
        # Invalidate cache when segment count differs from what was synthesised.
        # Segment count is embedded in the seg_dir file count; a mismatch means
        # the dubbed audio was built for a different segmentation.
        seg_dir_check = PREPARED / f"{target_lang}_segments_{stem}"
        valid_segs    = [s for s in segments if str(s.get("hi_text") or "").strip()]
        cached_count  = len(list(seg_dir_check.glob("seg_*.mp3"))) if seg_dir_check.exists() else 0
        if cached_count > 0 and abs(cached_count - len(valid_segs)) > 2:
            print(f"  Segment count changed ({cached_count} cached → {len(valid_segs)} new) "
                  f"— invalidating TTS cache.")
        else:
            print(f"  {target_lang.upper()} audio already exists: {dubbed_path.name}")
            return dubbed_path

    seg_dir = PREPARED / f"{target_lang}_segments_{stem}"
    seg_dir.mkdir(exist_ok=True)
    voices = speaker_voices or {}

    if src_audio and src_audio.exists():
        total_ms = len(AudioSegment.from_file(str(src_audio)))
    else:
        total_ms = int(segments[-1]["end"] * 1000) + 500 if segments else 5000

    valid = [s for s in segments if s["hi_text"].strip()]
    combined_tgt = " ".join(s["hi_text"] for s in valid)
    print(f"  Synthesising {target_lang.upper()} audio ({len(combined_tgt)} chars, "
          f"base={total_ms / 1000:.1f}s) …")

    seg_durations  = _synth_pass1(valid, seg_dir, voices, default_voice)
    total_tts_s    = sum(seg_durations.values())
    total_speech_s = sum(s["duration"] for s in valid)
    print(f"  TTS {total_tts_s:.1f}s over {total_speech_s:.1f}s speech "
          f"→ per-segment rate control + atempo")

    combined = _synth_pass2(valid, seg_dir, voices, seg_durations, total_ms, default_voice)
    combined.export(str(dubbed_path), format="mp3")
    print(f"  {target_lang.upper()} dubbed audio: {dubbed_path.name}  "
          f"({len(combined) / 1000:.1f}s)")
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
