# VideoDubbing Pipeline — Design Document

**Version**: v2  
**Last updated**: 2026-06-12  
**Authors**: Logituit AI Practice

---

## Purpose

This document describes the architecture of the VideoDubbing v2 pipeline, the rationale behind key design decisions, and the constraints those decisions were made under. It is intended for engineers continuing this work and for technical reviewers at Eros / SunNxt.

---

## Problem Statement

Translate a short-form English OTT video into dubbed Hindi output — replacing the English audio track with natural-sounding Hindi TTS — while preserving the original video frame and keeping the dubbed speech time-aligned to the original speaker segments. Measure output quality with objective metrics so progress can be tracked across iterations.

**Constraints**:
- No access to real OTT content at POC stage → pipeline must synthesise its own sample clip
- Must run on a developer Mac (Apple Silicon) without GPU dependency
- Demo-able to Eros / SunNxt stakeholders via Streamlit Cloud without cloud pipeline execution
- Anthropic API available; HuggingFace available for open-weight models

---

## Pipeline Overview

```
[Video / synthetic clip]
        │
        ▼
Stage 1  generate_sample_video()     ← if no --input provided
        │
        ▼
Stage 2  extract_audio()             ← ffmpeg; 16kHz mono WAV
        │
        ▼
Stage 3  transcribe()                ← Whisper ASR (base model)
        │
        ▼
Stage 3.5  diarize_speakers()        ← pyannote 3.1 (optional; needs HF_TOKEN)
           assign_speakers()         ← overlap-based label assignment
           detect_speaker_genders()  ← wav2vec2 age-gender model
        │
        ▼
Stage 4  translate_segments()        ← Google Translate (free tier)
        │
        ▼
Stage 4b  refine_segments()          ← Claude Sonnet (post-correction + ASR fix)
        │
        ▼
Stage 5  generate_vtt() / srt()      ← subtitle file generation
        │
        ▼
Stage 6  synthesize_hindi_audio()    ← edge-tts (neural) with gTTS fallback
           two-pass isochrony fit    ← Pass 1: normal rate; Pass 2: uniform speedup
        │
        ▼
Stage 7  create_dubbed_video()       ← ffmpeg; Hindi TTS + ducked original audio
        │
        ▼
Stage 7b  apply_wav2lip()            ← Wav2Lip lip-sync (optional; --lipsync flag)
        │
        ▼
Stage 8   compute_metrics()          ← WER, BLEU, duration ratio
Stage 8b  compute_segment_isochrony() / grade_translations()
Stage 8c  compute_back_translation_bleu()
Stage 8d  compute_lipsync_score()    ← Haar cascade + audio RMS correlation
        │
        ▼
[model_outputs/metrics_<stem>.json + subtitles_<stem>_hi.{vtt,srt}]
```

---

## Module Structure

After the Option B refactor (commit `6c7bcdc`), the code is split across five files:

| File | Responsibility |
|---|---|
| `pipeline_v2.py` | Orchestration (main), ASR, translation, subtitle gen, sample-video generation, CLI |
| `diarize.py` | Stage 3.5 — pyannote diarization, speaker assignment, gender detection |
| `tts_audio.py` | Stage 6 — edge-tts synthesis, two-pass isochrony, audio ducking |
| `metrics.py` | Stages 8/8b/8c — WER, BLEU, isochrony, LLM grading, back-translation |
| `eval_lipsync.py` | Stage 8d — audio-visual sync scoring (Haar cascade + Pearson r) |
| `apply_lipsync.py` | Stage 7b — Wav2Lip inference wrapper |

`pipeline_v2.py` re-exports all symbols from the sub-modules so that `tests/test_pipeline_v2.py` (which pulls symbols from the `pipeline_v2` namespace via `importlib`) requires no changes when the module layout changes.

---

## Stage-by-Stage Design Notes

### Stage 1 — Synthetic sample video

No real OTT content is available at POC stage. Rather than use a generic clip, the pipeline generates its own 4-segment English source video: a static 1280×720 title card with neural TTS audio (edge-tts `en-US-JennyNeural`, gTTS fallback). This lets the pipeline self-validate end-to-end without data-access dependencies. The `SOURCE_SCRIPT` and `REFERENCE_HINDI` constants define the ground truth for metric computation against this clip.

When `--input <video>` is supplied the stage is skipped and reference-based scoring is disabled (WER and BLEU set to `N/A`).

### Stage 2 — Audio extraction

ffmpeg extracts a 16kHz mono WAV. The 16kHz constraint is set by Whisper's expected sample rate; pyannote and the wav2vec2 gender model also work at 16kHz.

### Stage 3 — ASR

Whisper `base` is used for the English ASR pass. This was chosen over `small`/`medium` because:
- The synthetic clip has clean TTS audio, so a larger model doesn't improve accuracy
- `base` loads in <1s on Apple Silicon; `medium` takes ~6s
- Back-translation (Stage 8c) uses Whisper `medium` for Hindi because Hindi accuracy at `base` is poor

Whisper's `word_timestamps=True` is set to give segment-level timing, which is the unit all downstream stages operate on.

### Stage 3.5 — Speaker diarization and gender detection

**Diarization is optional** (`--no-diarize` flag or absent `HF_TOKEN`). pyannote 3.1 requires a HuggingFace token gated model. When unavailable, all segments fall back to a single default voice.

**Speaker assignment algorithm**: overlap-based — each Whisper segment is tagged with the pyannote speaker turn that overlaps it most (by seconds). When no turn overlaps (silence gap between turns), the nearest turn by midpoint distance is used. This O(n log m) approach handles the common edge case of a segment that straddles a speaker-change boundary.

**Gender detection**: `audeering/wav2vec2-large-robust-24-ft-age-gender` returns per-class probabilities (female, male, child). Child maps to female voice. The model is loaded once and cached for the process lifetime.

**Gender assignment decision rule** (`_assign_genders`): three tiers in order:
1. `f_prob >= 0.70` → female (high confidence threshold)
2. `f_prob < 0.50` → male (majority-male probability)
3. `0.50 ≤ f_prob < 0.70` → borderline: female only if this speaker has the highest `f_prob` among all speakers by ≥ 0.20 margin; otherwise male

The 0.20 relative-margin rule was added after observing that multi-speaker clips with one clearly female speaker and several male speakers often produce borderline scores (0.52–0.58) due to cross-contamination from adjacent male speech. Without the margin, multiple speakers could be assigned female incorrectly.

### Stage 4 — Translation

Google Translate (free tier via `deep-translator`) is used as a first-pass machine translation. It is fast, free, and generally acceptable for news/editorial Hindi. Known weaknesses:
- Literal over natural phrasing
- ASR errors in English propagate into Hindi
- No awareness of dubbing timing constraints

Both weaknesses are addressed by Stage 4b.

**Filler handling**: common ASR fillers (`uh`, `um`, `hmm`, `ah`) are mapped to Hindi equivalents or empty strings at the translation stage rather than in the LLM pass, to avoid wasting LLM tokens on them.

### Stage 4b — LLM post-correction (Claude Sonnet)

This is the highest-value stage for output quality. Claude Sonnet (`claude-sonnet-4-6`) receives each segment's English text, Google-translated Hindi, and available window duration in seconds, and is asked to:

1. Fix ASR transcription errors in the English (e.g. "half is likely" → "half as likely")
2. Rewrite Hindi as natural spoken Hindustani/Bollywood register that fits within `duration_s`

**Key prompt decisions**:
- Explicitly tell the model Hindi TTS speaks at ~3.5 words/second so it can reason about length
- Instruct it to prefer common, everyday vocabulary over Sanskritised literary Hindi — this directly improves back-translation BLEU because ASR models reliably transcribe common words
- Batch processing in groups of 100 segments per API call to stay within context limits

The model is Sonnet not Haiku because register quality (natural dubbing voice vs. functional translation) requires stronger instruction-following. Haiku produced noticeably more literal translations in testing.

### Stage 5 — Subtitle files

Both WebVTT and SRT formats are generated from the segment list. No design complexity here — standard format strings from the Whisper segment timestamps.

### Stage 6 — Hindi TTS with isochrony control

**Voice selection**: Microsoft `hi-IN-SwaraNeural` (female) and `hi-IN-MadhurNeural` (male) via edge-tts. These are the highest-quality free neural Hindi voices available without a paid API key. gTTS is the fallback if edge-tts fails (network or rate limit).

**Two-pass isochrony fit**: the central challenge is that Hindi TTS audio is often longer than the English source window. A naive approach (synthesise and overlay at the original timestamp) causes segments to bleed into each other.

The two-pass approach:
1. Pass 1: synthesise all segments at a base rate of `-10%` (slightly slower than default, for better naturalness). Measure total TTS duration.
2. Compute the ratio of total TTS duration to total speech window duration. Cap the required speedup at `MAX_RATE_PCT = 40%`.
3. Pass 2: re-synthesise all segments at the computed uniform rate, then overlay each at its original `start` timestamp.

The uniform-rate approach (rather than per-segment) was chosen because per-segment rates sound unnatural when two adjacent segments require very different speeds. A uniform rate preserves relative speech rhythm. The 40% cap was chosen empirically — above ~40%, edge-tts intelligibility degrades noticeably.

**Audio ducking**: when diarization has run, the original English audio is included in the output at reduced volume (20% during silence gaps, 4% during Hindi speech segments). This preserves ambient audio and avoids a "dead" background track, which is important for demo naturalness. Without diarization the original is mixed at a flat 20%.

### Stage 7 — Dubbed video assembly

ffmpeg is used to mux the original video stream with the Hindi TTS audio (and optionally ducked original). Video stream is copied (`-c:v copy`) without re-encoding to avoid quality loss and keep processing fast.

### Stage 7b — Wav2Lip lip-sync (optional)

Wav2Lip (`--lipsync` flag) re-generates mouth movements from the dubbed audio using the `wav2lip_gan.pth` checkpoint. This is a ~5 min process per 2 min of video and requires face detection.

**Known limitation**: Wav2Lip fails with `ValueError: Face not detected!` on clips with wide shots or non-face frames (e.g. `tears_of_steel`). The pipeline handles this gracefully — `apply_wav2lip` returns `{"success": False, "note": "..."}` and the stage falls back to the non-lipsynced dubbed video for metric computation. Per-shot face-presence gating (skip Wav2Lip for no-face frames) is on the backlog.

### Stages 8/8b/8c/8d — Quality metrics

Four metric categories:

| Metric | Method | Notes |
|---|---|---|
| ASR WER | `jiwer.wer` against reference text | Only available for synthetic clip |
| Translation BLEU | `sacrebleu.corpus_bleu` vs. reference Hindi | Only for synthetic clip |
| Back-translation BLEU | Whisper `medium` (hi) → Google Translate → `sacrebleu` vs. original EN | Works for any clip |
| Duration ratio | dubbed audio length / source audio length | Target: 0.90–1.10 |
| Segment isochrony | TTS file duration / EN window duration per segment | Stored per-segment |
| LLM fidelity/fluency | Claude Haiku grades 1–5 on each segment | Haiku used here (grading not generation) |
| Lip-sync score | Haar cascade mouth activity vs. audio RMS, Pearson r | 0.0–1.0; target ≥ 0.70 |

**Back-translation BLEU as the primary accuracy signal**: WER and forward BLEU require reference text (only available for the synthetic clip). Back-translation BLEU works on any clip — dub to Hindi, ASR the Hindi with Whisper medium, back-translate to English, measure BLEU against the original English. This is the metric that generalises to real OTT content from Eros / SunNxt.

**Lip-sync scoring without SyncNet**: SyncNet is the standard approach but requires PyTorch GPU inference and ~2GB of model weights. Instead, `eval_lipsync.py` uses OpenCV's Haar cascade (bundled, no download) to detect mouth regions and correlates mouth-area frame-diff variance against audio RMS energy, with Pearson r evaluated at lags -1/0/+1 frames. This runs in under 10 seconds on CPU and produces a `sync_score` in [0, 1]. The current score on `tears_of_steel_2min` is ~0.51; target is ≥ 0.70.

---

## Data Flow and File Conventions

```
data/
  raw/
    sample_en.mp4              ← synthetic source clip
    source_en.mp3              ← English TTS audio (for synthetic clip)
    dubbed_hi_<stem>.mp3       ← final Hindi TTS audio track
    <stem>_dubbed_hi.mp4       ← dubbed video (original video + Hindi audio)
    <stem>_lipsync_hi.mp4      ← lip-synced video (if --lipsync)
  prepared/
    <stem>_audio.wav           ← 16kHz mono WAV extracted from video
    <stem>_audio.ducked.wav    ← ducked original audio (during speech)
    hi_segments_<stem>/        ← per-segment TTS MP3 files (cache for two-pass)
    transcript_bilingual_<stem>.csv

model_outputs/
  subtitles_<stem>_hi.vtt
  subtitles_<stem>_hi.srt
  metrics_<stem>.json
```

The `<stem>` is the input video filename without extension (e.g. `tears_of_steel_2min`). Using stems as cache keys means different clips coexist in the same directory and Stage 6 can skip re-synthesis if segment files already exist.

---

## CLI Flags

| Flag | Effect |
|---|---|
| `--input <path>` | Use existing video; skips sample generation; disables reference scoring |
| `--no-llm` | Skip Stage 4b (Claude) and LLM grading in Stage 8b |
| `--no-diarize` | Skip Stage 3.5 (pyannote + gender detection); all segments use female voice |
| `--hf-token <token>` | HuggingFace token for pyannote; falls back to `HF_TOKEN` env var |
| `--gender-thresh <float>` | Min female probability to assign female voice (default 0.70) |
| `--lipsync` | Run Wav2Lip Stage 7b (slow; ~5 min per 2 min) |

---

## Testing Strategy

Tests live in `tests/test_pipeline_v2.py`. They cover only pure logic that does not require GPU, network, or file I/O:

- `assign_speakers` — overlap logic, gap fallback, empty turns, field preservation
- `_assign_genders` — confidence thresholds, borderline margin rule, multi-speaker scenarios
- `_seg_voice` / `_voice_tag` — voice selection and cache key generation
- `MAX_RATE_PCT` — TTS rate cap arithmetic
- `_LLM_BATCH` — batch splitting correctness

The test file loads `pipeline_v2.py` via `importlib.util.spec_from_file_location` and pulls symbols from the module namespace. This approach means tests survive module restructuring as long as `pipeline_v2.py` re-exports the tested symbols — which it does via explicit `from <module> import ...` at the top of the file.

**What is not tested**: GPU-dependent stages (Wav2Lip), network-dependent stages (Google Translate, Claude API, edge-tts), and ffmpeg-dependent stages (audio extraction, video assembly). Those require integration testing against actual media files.

---

## Design Decisions Log

| Decision | Chosen approach | Rejected alternative | Reason |
|---|---|---|---|
| Translation pipeline | Google Translate → Claude post-correction | Claude end-to-end translation | Cost and latency; Google handles bulk, Claude fixes quality |
| LLM model for post-correction | Claude Sonnet | Claude Haiku | Haiku produced noticeably more literal translations in testing |
| LLM model for grading | Claude Haiku | Claude Sonnet | Grading is structured classification, not generation — Haiku is sufficient and cheaper |
| TTS engine | edge-tts (Microsoft neural) | Google TTS / AWS Polly | Best free Hindi neural voice quality; no API key required |
| Isochrony strategy | Uniform rate speedup (capped at 40%) | Per-segment rate adjustment | Uniform rate preserves natural speech rhythm; per-segment sounds jarring |
| Audio background | Ducked original at 4%/20% | Silent background | Preserves ambient audio; avoids "dead" background track in demo |
| Lip-sync scoring | Haar cascade + Pearson r (CPU) | SyncNet / AV-HuBERT | SyncNet needs GPU + 2GB weights; Haar cascade runs in <10s, no extra deps |
| Module structure | Option B: extract 3 heavy modules | Option A: full per-stage split | Option B required zero test file changes (via re-exports); Option A is the backlog item |
| Test loading strategy | `importlib.util.spec_from_file_location` | Direct `import pipeline_v2` | Avoids executing `__main__` block on import; allows test isolation without package install |

---

## Known Limitations and Backlog

- **Wav2Lip on wide-angle clips**: fails when faces are absent. Backlog: per-shot face-presence gating.
- **Reference metrics require synthetic clip**: WER and forward BLEU are unavailable for real OTT content until a gold-standard reference translation is obtained from Eros/SunNxt.
- **Isochrony cap at 40%**: segments that genuinely require >40% speedup still overflow their window. Long-term fix: shorten the Hindi text in Stage 4b rather than relying on rate increase.
- **Single audio track output**: pipeline outputs one Hindi audio track. Multi-track output (original EN + Hindi) for broadcaster delivery is not yet implemented.
- **Option A refactor**: `pipeline_v2.py` (now ~580 lines) can be further split into `asr.py`, `translation.py`, `tts.py`, `video.py` — tracked in TODO.md.
