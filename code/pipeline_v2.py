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
import os
os.environ.setdefault("MallocStackLogging", "0")  # suppress macOS malloc noise in subprocesses

import argparse
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from eval_lipsync import compute_lipsync_score
from apply_lipsync import apply_wav2lip
from emotion import (classify_segment_emotions, smooth_emotion_arc,
                     score_emotion_consistency, score_tts_emotion_fidelity,
                     score_tts_prosody_transfer,
                     classify_face_emotions, load_persona_map, apply_persona_offsets)

# Re-exported from sub-modules so tests pulling from this module still work
from asr import (                                              # noqa: E402
    extract_audio, transcribe,
    WHISPER_MODEL, WHISPER_MODEL_HI,                           # noqa: F401
)
from translation import (                                      # noqa: E402
    translate_segments, refine_segments, extract_glossary,
    repair_emotion_register,
    _make_refine_system, _ctx_snippet, _refine_batch,          # noqa: F401
    _make_repair_system,                                       # noqa: F401
    _FILLER_MAPS, _LANG_NAMES,                                 # noqa: F401
    _LLM_BATCH, _CTX_WINDOW, _ASR_SKIP_THRESH,                 # noqa: F401
)
from video import (                                            # noqa: E402
    generate_sample_video, create_dubbed_video, apply_loudnorm,
    generate_vtt, generate_srt,
    _generate_source_audio, _make_title_frame,                 # noqa: F401
    _vtt_time, _srt_time,                                      # noqa: F401
    SOURCE_SCRIPT, FULL_SOURCE_TEXT, REFERENCE_HINDI,          # noqa: F401
    VIDEO_SIZE, VIDEO_FPS, BG_COLOR, ACCENT_COLOR,             # noqa: F401
)
from reporting import (                                        # noqa: E402
    save_outputs, _print_quality, _metrics_summary_row,
)
from diarize import (                                          # noqa: E402
    diarize_speakers, assign_speakers,
    detect_speaker_genders,
    _assign_genders, _GENDER_LABEL_MAP,                        # noqa: F401
    _FEMALE_CONFIDENCE_THRESH,
)
from tts_audio import (                                        # noqa: E402
    synthesize_hindi_audio, _duck_original_audio,              # noqa: F401
    _assign_voice_pool,
    _seg_voice, _voice_tag, MAX_RATE_PCT,                      # noqa: F401
    VOICE_POOL,                                                # noqa: F401
    TTS_VOICE_FEMALE_HI, TTS_VOICE_MALE_HI,                    # noqa: F401
    TTS_VOICE_FEMALE_EN, TTS_VOICE_MALE_EN,                    # noqa: F401
    TTS_VOICE_HI, TTS_BASE_RATE_PCT,                           # noqa: F401
    BG_AUDIO_VOL, BG_AUDIO_VOL_SPEECH,                         # noqa: F401
    separate_stems, get_effective_seg_durations,
)
from metrics import (                                          # noqa: E402
    compute_metrics, compute_text_bleu, compute_back_translation_bleu,
    compute_segment_isochrony, grade_translations,
    _merge_segment_quality, score_mos_rubric,
)

ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data/raw"
PREPARED  = ROOT / "data/prepared"
MODEL_OUT = ROOT / "model_outputs"
for d in [RAW, PREPARED, MODEL_OUT]:
    d.mkdir(parents=True, exist_ok=True)


def run_pipeline(video_path: Path, args, *, has_ref: bool = False) -> dict:
    """Run all pipeline stages on one video. Returns the metrics dict."""
    import os
    source_lang = args.source_lang
    target_lang = args.target_lang

    print("\nStage 2 — Extracting audio …")
    audio_path = extract_audio(video_path)

    print("\nStage 2.1 — Stem separation (demucs) …")
    no_vocals_path: Path | None = None
    vocals_path: Path | None = None
    if args.no_stems:
        print("  Skipped (--no-stems).")
    else:
        vocals_path, no_vocals_path = separate_stems(audio_path)
        if no_vocals_path == audio_path:
            no_vocals_path = None
            vocals_path = None

    print("\nStage 3 — ASR (Whisper) …")
    whisper_result = transcribe(audio_path, source_lang=source_lang)

    print("\nStage 3.5 — Speaker diarization …")
    speaker_voices: dict[str, str] = {}
    if args.no_diarize:
        print("  Skipped (--no-diarize).")
    else:
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
            speaker_voices = _assign_voice_pool(
                genders, whisper_result["segments"], target_lang
            )
            print(f"  Voice map: {speaker_voices}")

    print("\nStage 4 — Translation (Google Translate) …")
    segments = translate_segments(whisper_result,
                                  source_lang=source_lang, target_lang=target_lang)

    print("\nStage 4a — Glossary extraction …")
    glossary: dict[str, str] = {}
    if not args.no_llm:
        glossary_cache = PREPARED / f"glossary_{video_path.stem}_{target_lang}.json"
        glossary = extract_glossary(segments, target_lang=target_lang,
                                    cache_path=glossary_cache)

    print("\nStage 4b — LLM post-correction (Claude) …")
    segments = refine_segments(segments, skip=args.no_llm,
                               source_lang=source_lang, target_lang=target_lang,
                               glossary=glossary or None)

    print("\nStage 2.5 — Emotion analysis (SER) …")
    segments = classify_segment_emotions(audio_path, segments)
    emo_counts: dict[str, int] = {}
    for s in segments:
        emo_counts[s.get("emotion", "neutral")] = (
            emo_counts.get(s.get("emotion", "neutral"), 0) + 1
        )
    print(f"  Emotion distribution: {emo_counts}")
    segments = smooth_emotion_arc(segments)
    arc_flagged = sum(1 for s in segments if s.get("arc_flagged"))
    if arc_flagged:
        print(f"  Arc smoother flagged {arc_flagged} outlier segment(s)")

    if args.face_emotion:
        print("\nStage 2.6 — Face emotion detection (MediaPipe) …")
        segments = classify_face_emotions(video_path, segments)
        face_count = sum(1 for s in segments if s.get("face_emotion"))
        print(f"  Face emotion detected on {face_count}/{len(segments)} segments")

    persona_map = load_persona_map(video_path.parent / "persona_map.json")
    if persona_map:
        print(f"\nStage 2.7 — Persona map ({len(persona_map)} speaker(s)) …")
        segments = apply_persona_offsets(segments, persona_map)
        adj = sum(1 for s in segments if s.get("persona_adjusted"))
        print(f"  Adjusted {adj}/{len(segments)} segments")

    print("\nStage 4c — Emotion register repair …")
    segments = repair_emotion_register(segments, skip=args.no_llm,
                                       source_lang=source_lang, target_lang=target_lang)

    tgt_name = _LANG_NAMES.get(target_lang, target_lang.upper())
    print(f"\nStage 6 — {tgt_name} TTS …")
    hindi_audio = synthesize_hindi_audio(
        segments, stem=video_path.stem, src_audio=audio_path,
        speaker_voices=speaker_voices, target_lang=target_lang,
    )

    print("\nStage 5 — Generating subtitle files (aligned to dubbed audio) …")
    eff_durs = get_effective_seg_durations(segments, video_path.stem, target_lang)
    if eff_durs:
        MIN_SUB_S = 0.5
        for seg in segments:
            if seg["id"] in eff_durs:
                actual = eff_durs[seg["id"]]
                original_window = seg["end"] - seg["start"]
                seg["end"] = seg["start"] + max(MIN_SUB_S, min(actual, original_window))
    vtt = generate_vtt(segments)
    srt = generate_srt(segments)
    print(f"  VTT: {len(vtt)} chars  |  SRT: {len(srt)} chars")

    print("\nStage 7 — Creating dubbed video …")
    dubbed_video = create_dubbed_video(video_path, hindi_audio, segments=segments,
                                       target_lang=target_lang,
                                       no_vocals_path=no_vocals_path)

    print("\nStage 7c — LUFS normalisation (−27 LKFS, Netflix standard) …")
    try:
        dubbed_video = apply_loudnorm(dubbed_video, force=args.force)
    except Exception as e:
        print(f"  [warn] Loudnorm failed, using unnormalised video: {e}")

    if args.lipsync:
        print("\nStage 7b — Wav2Lip lip-sync …")
        ls_out = RAW / f"{video_path.stem}_lipsync_{target_lang}.mp4"
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
    src_audio = RAW / "source_en.mp3" if has_ref else audio_path
    metrics   = compute_metrics(whisper_result, segments, src_audio,
                                hindi_audio, has_reference=has_ref)
    metrics["tts"]["voice_map"] = speaker_voices

    print("\nStage 8b — Segment quality …")
    seg_dir = PREPARED / f"hi_segments_{video_path.stem}"
    iso    = compute_segment_isochrony(segments, seg_dir)
    grades = grade_translations(segments) if not args.no_llm else []
    if grades:
        print(f"  LLM graded {len(grades)} segments.")
    metrics["segment_quality"] = _merge_segment_quality(
        iso, {g["id"]: g for g in grades}
    )

    print("\nStage 8c — Back-translation BLEU …")
    original_src = whisper_result["text"].strip()
    metrics["text_bleu"] = compute_text_bleu(
        segments, original_src, source_lang=source_lang, target_lang=target_lang,
    )
    print(f"  Text-level BLEU: {metrics['text_bleu'].get('bleu')}")
    metrics["back_translation"] = compute_back_translation_bleu(
        hindi_audio, original_src,
        source_lang=source_lang, target_lang=target_lang,
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

    print("\nStage 8e — Emotion consistency score …")
    try:
        emo = score_emotion_consistency(audio_path, hindi_audio, segments,
                                        target_lang=target_lang)
        print(f"  Text match: {emo['match_pct']}%  soft: {emo['avg_soft_score']}%  "
              f"over {emo['n_segments']} segments")
        print("  Stage 8e-ii — TTS emotion fidelity (audio SER) …")
        emo["tts_fidelity"] = score_tts_emotion_fidelity(hindi_audio, segments)
        tf = emo["tts_fidelity"]
        print(f"  TTS fidelity soft score: {tf.get('avg_soft_score')}% "
              f"over {tf.get('n_segments')} segments")

        print("  Stage 8e-iii — Prosody-transfer fidelity (source↔dub contour) …")
        prosody_src = vocals_path if (vocals_path and vocals_path.exists()) else audio_path
        emo["prosody_transfer"] = score_tts_prosody_transfer(
            prosody_src, hindi_audio, segments
        )
        pt = emo["prosody_transfer"]
        if pt.get("avg_soft_score") is not None:
            print(f"  Prosody-transfer soft score: {pt['avg_soft_score']}% "
                  f"over {pt['n_segments']} segments")
        else:
            print(f"  Prosody-transfer unavailable: {pt.get('note')}")
        metrics["emotion"] = emo
    except Exception as exc:
        metrics["emotion"] = {"match_pct": None, "note": str(exc)}
        print(f"  [warn] Emotion scoring failed: {exc}")

    print("\nStage 8f — MOS rubric (5-dim spot-check) …")
    if not args.no_llm:
        mos_result = score_mos_rubric(segments, glossary=glossary or None)
        metrics["mos_rubric"] = mos_result
        mos = mos_result.get("mos")
        breakdown = mos_result.get("breakdown", {})
        if mos is not None:
            print(f"  MOS: {mos}/100  (n={mos_result['n_sampled']})  "
                  + "  ".join(f"{k}={v}" for k, v in breakdown.items()))
        else:
            print(f"  MOS unavailable: {mos_result.get('note')}")
    else:
        metrics["mos_rubric"] = {"mos": None, "note": "skipped (--no-llm)"}
        print("  Skipped (--no-llm).")

    save_outputs(segments, vtt, srt, metrics, src_audio, hindi_audio,
                 stem=video_path.stem, target_lang=target_lang)
    _print_quality(metrics)
    return metrics


if __name__ == "__main__":
    import datetime
    import sys

    parser = argparse.ArgumentParser(description="VideoDubbing v2 pipeline")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to an existing MP4/MOV to dub.")
    parser.add_argument("--batch", type=Path, default=None,
                        help="Directory of MP4/MOV files to dub in sequence; "
                             "writes a batch_summary CSV to model_outputs/.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Stage 4b Claude post-correction.")
    parser.add_argument("--no-diarize", action="store_true",
                        help="Skip Stage 3.5 speaker diarization.")
    parser.add_argument("--no-stems", action="store_true",
                        help="Skip Stage 2.1 demucs stem separation; "
                             "fall back to audio ducking.")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for pyannote models. "
                             "Reads HF_TOKEN env var if not provided.")
    parser.add_argument("--gender-thresh", type=float,
                        default=_FEMALE_CONFIDENCE_THRESH,
                        help="Min female probability to assign female voice "
                             f"(default {_FEMALE_CONFIDENCE_THRESH}).")
    parser.add_argument("--face-emotion", action="store_true",
                        help="Run MediaPipe face emotion detection and fuse with text SER (Stage 2.6, adds ~45s/clip).")
    parser.add_argument("--lipsync", action="store_true",
                        help="Run Wav2Lip (Stage 7b) to re-generate mouth movements.")
    parser.add_argument("--source-lang", type=str, default="en",
                        help="Source language for ASR (default: en).")
    parser.add_argument("--target-lang", type=str, default="hi",
                        help="Target language for TTS (default: hi).")
    args = parser.parse_args()

    if args.batch:
        clips = sorted(args.batch.glob("*.mp4")) + sorted(args.batch.glob("*.mov"))
        if not clips:
            print(f"No .mp4/.mov files found in {args.batch}")
            sys.exit(1)
        rows = []
        for clip in clips:
            print(f"\n{'=' * 60}\nProcessing: {clip.name}\n{'=' * 60}")
            m = run_pipeline(clip, args, has_ref=False)
            rows.append(_metrics_summary_row(clip.name, m))
            print("\nDone.")
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = MODEL_OUT / f"batch_summary_{ts}.csv"
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nBatch complete ({len(rows)} clips). Summary: {csv_path}")
    else:
        if args.input:
            if not args.input.exists():
                raise FileNotFoundError(f"Input video not found: {args.input}")
            video_path = args.input
            print(f"Stage 1 — Using provided video: {video_path.name}")
            has_ref = False
        else:
            print("Stage 1 — Generating sample video …")
            video_path = generate_sample_video()
            has_ref = True
        run_pipeline(video_path, args, has_ref=has_ref)
        print("\nDone.")
