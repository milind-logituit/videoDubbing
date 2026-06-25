"""
Calibrate emotion_prosody.yaml by scoring the FULL dubbed clip for each param combo.

Strategy:
  1. Load the cached bilingual transcript CSV (ASR + translation already done).
  2. For each prosody param combo in the grid:
       a. Temporarily write the params to emotion_prosody.yaml.
       b. Re-synthesize TTS (Stage 5) using those params.
       c. Mix the dubbed audio with the source video (Stage 7, audio-only).
       d. Score the final mixed audio with score_tts_emotion_fidelity (audeering).
  3. Pick the params with the highest avg_soft_score per emotion.
  4. Write winning params back to config/emotion_prosody.yaml.

Stages 1–4a (ASR, translation, glossary) are skipped — they are read from the
cached transcript CSV so each iteration only pays for TTS + scoring (~30 s/run).

Usage:
  uv run python code/calibrate_prosody.py --clip test_clips/tears_of_steel_2min.mp4
  uv run python code/calibrate_prosody.py --clip test_clips/tears_of_steel_2min.mp4 --dry-run
  uv run python code/calibrate_prosody.py --clip test_clips/tears_of_steel_2min.mp4 --emotions happy angry
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from emotion import SSML_PROSODY, score_tts_emotion_fidelity

ROOT     = Path(__file__).parent.parent
PREPARED = ROOT / "data/prepared"
RAW      = ROOT / "data/raw"
CONFIG   = ROOT / "config" / "emotion_prosody.yaml"

# 5 × 7 = 35 combos per emotion — same grid as before
_PITCH_VARIANTS = ["-15%", "-7%", "+0%", "+10%", "+20%"]
_RATE_VARIANTS  = ["-20%", "-14%", "-7%", "+0%", "+7%", "+14%", "+20%"]
_VOLUME_FIXED: dict[str, str] = {
    "neutral":   "medium",
    "happy":     "loud",
    "angry":     "x-loud",
    "sad":       "soft",
    "fearful":   "soft",
    "disgust":   "medium",
    "surprised": "loud",
}
_ALL_EMOTIONS = ["happy", "angry", "sad", "fearful", "disgust", "surprised"]
_MAX_SEGS_PER_EMOTION = 8  # cap to keep each run fast


def _load_config() -> dict:
    if CONFIG.exists():
        return yaml.safe_load(CONFIG.read_text()) or {}
    return {"default": {k: dict(v) for k, v in SSML_PROSODY.items()}}


def _write_config(cfg: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False))


def _synthesize_with_params(
    segs: list[dict],
    stem: str,
    lang: str,
    pitch: str,
    volume: str,
    rate: str,
    tmp_dir: Path,
) -> Path:
    """
    Synthesize TTS for the given segments using the specified prosody params.

    Temporarily patches emotion_prosody.yaml so tts_audio.synthesize_hindi_audio
    picks up the overridden params, then restores the original config.
    Returns path to the mixed dubbed audio WAV.
    """
    from tts_audio import synthesize_hindi_audio

    # Patch: override the emotion in every segment so params apply uniformly
    patched_segs = [{**s, "emotion": list(_VOLUME_FIXED.keys())[0]} for s in segs]

    # Temporarily write a single-emotion config so prosody is deterministic
    original_cfg = _load_config()
    probe_emotion = "calibration_probe"
    temp_cfg = dict(original_cfg)
    temp_cfg.setdefault(lang, {})
    temp_cfg[lang][probe_emotion] = {"pitch": pitch, "volume": volume, "rate": rate}
    _write_config(temp_cfg)

    # Override emotion label to our probe emotion
    patched_segs = [{**s, "emotion": probe_emotion} for s in segs]

    try:
        dubbed_audio = synthesize_hindi_audio(
            patched_segs,
            stem=f"{stem}_probe",
            lang=lang,
            force=True,
            out_dir=tmp_dir,
        )
    finally:
        # Always restore original config
        _write_config(original_cfg)

    return dubbed_audio


def _score(dubbed_audio: Path, segs: list[dict]) -> float:
    """Return avg_soft_score from audeering on the full dubbed audio."""
    result = score_tts_emotion_fidelity(dubbed_audio, segs)
    return float(result.get("avg_soft_score") or 0.0)


def calibrate(
    clip_path: Path,
    lang: str = "hi",
    target_emotions: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    stem     = clip_path.stem
    csv_path = PREPARED / f"transcript_bilingual_{stem}.csv"

    if not csv_path.exists():
        print(f"Transcript not found: {csv_path}")
        print("Run the full pipeline on this clip first to generate the cached transcript.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "emotion" not in df.columns:
        print("No emotion column in transcript — rerun pipeline with --face-emotion or emotion stage enabled.")
        sys.exit(1)

    emotions = target_emotions or _ALL_EMOTIONS
    total_combos = len(_PITCH_VARIANTS) * len(_RATE_VARIANTS)

    print(f"\nCalibrating prosody for lang={lang} on {stem}")
    print(f"Emotions: {emotions}")
    print(f"Grid: {len(_PITCH_VARIANTS)} pitch × {len(_RATE_VARIANTS)} rate = {total_combos} combos per emotion")
    print("Scoring: full dubbed audio (audeering SER) — not isolated TTS clips\n")

    if dry_run:
        for emotion in emotions:
            segs = df[df["emotion"] == emotion].to_dict("records")
            print(f"  [{emotion}] {len(segs)} segment(s) — would run {total_combos} combos")
        print("\n[dry-run] No synthesis or config changes.")
        return

    cfg = _load_config()
    best_params: dict[str, dict[str, str]] = {}
    results_log: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for emotion in emotions:
            segs = df[df["emotion"] == emotion].to_dict("records")
            if not segs:
                print(f"  [{emotion}] no segments — skipping")
                continue

            segs = segs[:_MAX_SEGS_PER_EMOTION]
            volume  = _VOLUME_FIXED.get(emotion, "medium")
            current = cfg.get(lang, {}).get(emotion, SSML_PROSODY.get(emotion, {}))
            print(f"\n  [{emotion}] {len(segs)} seg(s) | "
                  f"current pitch={current.get('pitch')} rate={current.get('rate')}")

            best_score = -1.0
            best_combo = (current.get("pitch", "+0%"), current.get("rate", "+0%"))
            tried = 0

            for pitch, rate in itertools.product(_PITCH_VARIANTS, _RATE_VARIANTS):
                try:
                    dubbed = _synthesize_with_params(
                        segs, stem, lang, pitch, volume, rate, tmp_dir
                    )
                    score = _score(dubbed, segs)
                except Exception as exc:
                    print(f"    [warn] pitch={pitch} rate={rate} failed: {exc}")
                    score = 0.0

                results_log.append({
                    "emotion": emotion, "pitch": pitch, "rate": rate,
                    "volume": volume, "score": round(score, 3),
                })
                tried += 1

                if score > best_score:
                    best_score = score
                    best_combo = (pitch, rate)

                print(f"    pitch={pitch:5s} rate={rate:5s}  score={score:.1f}%"
                      f"  {'← best' if (pitch, rate) == best_combo else ''}")

            best_params[emotion] = {
                "pitch":  best_combo[0],
                "volume": volume,
                "rate":   best_combo[1],
            }
            print(f"  [{emotion}] WINNER: pitch={best_combo[0]} rate={best_combo[1]} "
                  f"score={best_score:.1f}%  ({tried}/{total_combos} combos)")

    # Write results
    cfg.setdefault(lang, {})
    for emotion, params in best_params.items():
        cfg[lang][emotion] = params
    _write_config(cfg)

    results_path = ROOT / "model_outputs" / f"prosody_calibration_{stem}_{lang}.json"
    results_path.write_text(json.dumps(results_log, indent=2))

    print(f"\nWrote calibrated params → {CONFIG}")
    print(f"Full results log       → {results_path}")
    print(json.dumps(best_params, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate emotion prosody by scoring full dubbed clips"
    )
    parser.add_argument("--clip", type=Path, required=True,
                        help="Source clip (needs cached transcript CSV)")
    parser.add_argument("--lang", type=str, default="hi",
                        help="Target language code (default: hi)")
    parser.add_argument("--emotions", nargs="+", choices=_ALL_EMOTIONS,
                        help="Limit to specific emotions (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without synthesizing or writing")
    args = parser.parse_args()
    calibrate(args.clip, lang=args.lang, target_emotions=args.emotions, dry_run=args.dry_run)
