"""
Calibrate emotion_prosody.yaml parameters without a full pipeline rerun.

Strategy:
  1. Load existing bilingual transcript CSV (emotion labels already in it).
  2. For each non-neutral emotion, re-synthesize the segment with a small
     pitch × rate grid using edge-tts.
  3. Score each variant via score_tts_emotion_fidelity (audio SER on TTS output).
  4. Pick the params with the highest tts_fidelity per emotion.
  5. Write winning params to config/emotion_prosody.yaml.

Usage:
  uv run python code/calibrate_prosody.py --clip test_clips/tears_of_steel_2min.mp4
  uv run python code/calibrate_prosody.py --clip test_clips/tears_of_steel_2min.mp4 --lang hi --dry-run
"""
import argparse
import asyncio
import itertools
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from emotion import SSML_PROSODY, load_prosody_config, score_tts_emotion_fidelity
import edge_tts

ROOT      = Path(__file__).parent.parent
PREPARED  = ROOT / "data/prepared"
CONFIG    = ROOT / "config" / "emotion_prosody.yaml"

# Grid to search — 5×7 = 35 combos per emotion (audeering scorer reliable on broadcast)
_PITCH_VARIANTS = ["-15%", "-7%", "+0%", "+10%", "+20%"]
_RATE_VARIANTS  = ["-20%", "-14%", "-7%", "+0%", "+7%", "+14%", "+20%"]
_VOLUME_FIXED   = {
    "neutral":   "medium",
    "happy":     "loud",
    "angry":     "x-loud",
    "sad":       "soft",
    "fearful":   "soft",
    "disgust":   "medium",
    "surprised": "loud",
}
_TARGET_EMOTIONS = ["happy", "angry", "sad", "fearful", "disgust", "surprised"]


def _synth(text: str, voice: str, pitch: str, volume: str, rate: str,
           base_rate: str = "-10%") -> Path:
    """Synthesize one TTS segment with given prosody to a temp WAV. Returns path."""
    content = (
        f'<prosody pitch="{pitch}" volume="{volume}" rate="{rate}">'
        f"{text}"
        f"</prosody>"
    )

    async def _run(out: str) -> None:
        comm = edge_tts.Communicate(content, voice, rate=base_rate)
        await comm.save(out)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    asyncio.run(_run(tmp.name))
    return Path(tmp.name)


def calibrate(clip_path: Path, lang: str = "hi", dry_run: bool = False) -> None:
    stem = clip_path.stem
    csv_path = PREPARED / f"transcript_bilingual_{stem}.csv"
    if not csv_path.exists():
        print(f"Transcript not found: {csv_path}")
        print("Run the full pipeline on this clip first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "emotion" not in df.columns:
        print("No emotion column in transcript — rerun pipeline to add it.")
        sys.exit(1)

    # Default voice for this language
    from tts_audio import _TTS_DEFAULT_VOICE
    voice = _TTS_DEFAULT_VOICE.get(lang, "hi-IN-SwaraNeural")

    print(f"\nCalibrating prosody for lang={lang} on {stem}")
    print(f"Voice: {voice}  |  Grid: {len(_PITCH_VARIANTS)}×{len(_RATE_VARIANTS)} = {len(_PITCH_VARIANTS)*len(_RATE_VARIANTS)} combos per emotion\n")

    current_config = load_prosody_config(lang)
    best_params: dict[str, dict[str, str]] = {}

    for emotion in _TARGET_EMOTIONS:
        segs = df[df["emotion"] == emotion].to_dict("records")
        if not segs:
            print(f"  [{emotion}] no segments in transcript — skipping")
            continue

        # Use up to 5 representative segments (audeering is reliable — more averaging is better)
        segs = segs[:5]
        volume = _VOLUME_FIXED.get(emotion, "medium")
        current = current_config.get(emotion, SSML_PROSODY.get(emotion, {}))
        print(f"  [{emotion}] {len(segs)} segment(s)  current: pitch={current.get('pitch')} rate={current.get('rate')}")

        if dry_run:
            best_params[emotion] = current
            continue

        best_score = -1.0
        best_combo = (current.get("pitch", "+0%"), current.get("rate", "+0%"))
        total      = len(_PITCH_VARIANTS) * len(_RATE_VARIANTS)
        tried      = 0

        for pitch, rate in itertools.product(_PITCH_VARIANTS, _RATE_VARIANTS):
            # Build fake segment list pointing to temp synthesized files
            temp_paths = []
            fake_segs  = []
            for seg in segs:
                hi_text = str(seg.get("hi_text") or "").strip()
                if not hi_text:
                    continue
                p = _synth(hi_text, voice, pitch, volume, rate)
                temp_paths.append(p)
                fake_segs.append({**seg, "emotion": emotion,
                                  "start": 0.0, "end": 30.0})

            if not fake_segs:
                continue

            # Concatenate temp segments into one file for scoring
            from pydub import AudioSegment as _AS
            combined = _AS.empty()
            for p in temp_paths:
                combined += _AS.from_file(str(p))

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as combined_f:
                combined.export(combined_f.name, format="wav")
                combined_path = Path(combined_f.name)

            # Patch segment timestamps to match combined duration
            combined_dur = len(combined) / 1000.0
            step         = combined_dur / len(fake_segs)
            for i, fs in enumerate(fake_segs):
                fs["start"] = i * step
                fs["end"]   = (i + 1) * step

            result = score_tts_emotion_fidelity(combined_path, fake_segs)
            score  = result.get("avg_soft_score") or 0.0

            # Cleanup
            for p in temp_paths:
                p.unlink(missing_ok=True)
            combined_path.unlink(missing_ok=True)

            tried += 1
            if score > best_score:
                best_score = score
                best_combo = (pitch, rate)

        best_params[emotion] = {"pitch": best_combo[0], "volume": volume,
                                "rate":  best_combo[1]}
        print(f"    best: pitch={best_combo[0]} rate={best_combo[1]}  "
              f"tts_fidelity={best_score:.1f}%  ({tried}/{total} combos tried)")

    if dry_run:
        print("\n[dry-run] No changes written.")
        return

    # Write winning params into YAML under language-specific block
    if CONFIG.exists():
        existing = yaml.safe_load(CONFIG.read_text()) or {}
    else:
        existing = {"default": {k: dict(v) for k, v in SSML_PROSODY.items()}}

    existing.setdefault(lang, {})
    for emotion, params in best_params.items():
        existing[lang][emotion] = params

    CONFIG.write_text(yaml.dump(existing, allow_unicode=True, sort_keys=False))
    print(f"\nWrote calibrated params to {CONFIG}")
    print(json.dumps(best_params, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate emotion prosody parameters")
    parser.add_argument("--clip",    type=Path, required=True,
                        help="Clip that has already been processed (needs transcript CSV)")
    parser.add_argument("--lang",    type=str,  default="hi",
                        help="Target language code (default: hi)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print grid plan without synthesizing or writing")
    args = parser.parse_args()
    calibrate(args.clip, lang=args.lang, dry_run=args.dry_run)
