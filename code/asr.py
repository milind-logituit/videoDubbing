"""Stages 2–3 — Audio extraction and Whisper ASR."""
import subprocess
from pathlib import Path

import whisper

ROOT     = Path(__file__).parent.parent
PREPARED = ROOT / "data/prepared"

WHISPER_MODEL    = "large-v3-turbo"
WHISPER_MODEL_HI = "large-v3-turbo"


def extract_audio(video_path: Path) -> Path:
    audio_path = PREPARED / f"{video_path.stem}_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def transcribe(audio_path: Path, source_lang: str = "en") -> dict:
    print(f"  Loading Whisper '{WHISPER_MODEL}' …")
    model = whisper.load_model(WHISPER_MODEL)
    print(f"  Transcribing {audio_path.name} (lang={source_lang}) …")
    result = model.transcribe(str(audio_path), language=source_lang,
                               word_timestamps=True, verbose=False)
    print(f"  Transcript: {result['text'].strip()[:100]} …")
    return result
