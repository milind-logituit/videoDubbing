# TODO — VideoDubbing v3.0

Tracking open work to meet v3 emotion-matching targets.
Spec: `docs/emotion_matching_spec.docx`

---

## P0 — Must-have for v3.0

### Multi-modal emotion detection
**Target:** avg `emotion_register` ≥ 4.0 (current ~3.1)

- [x] **Expose `arousal` and `valence` per segment** — eebfc8b (derived from text SER circumplex, passed to Stage 4b)
- [ ] **Fuse audio SER into `classify_segment_emotions()`**
  - Evaluated `superb/wav2vec2-base-superb-er`: degraded emotion_register and timing on 2/3 clips (trained on acted speech / IEMOCAP, misfires on broadcast content)
  - Both ungated (0.3 weight) and confidence-gated (text < 0.6) variants tested — neither beat text-only baseline
  - **Blocked:** needs a model trained on film/broadcast audio (e.g. `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` or similar) before this can help

### Emotion-controlled TTS calibration
**Target:** TTS fidelity soft score ≥ 75% (current ~62%)

- [x] **Externalise `SSML_PROSODY` to `emotion_prosody.yaml`** — done 2026-06-15
  - `load_prosody_config(lang)` in `emotion.py` loads `config/emotion_prosody.yaml` (`default` block + optional per-lang overrides); falls back to hardcoded `SSML_PROSODY` if missing/unreadable
  - `tts_audio.py` derives `lang` from `voice[:2]` and pulls prosody via the loader; `pyyaml>=6.0.3` added to `pyproject.toml`

- [~] **Calibrate prosody params against MOS rubric** — harness built 2026-06-15, calibration run pending
  - `code/calibrate_prosody.py` grid-searches pitch × rate per non-neutral emotion, scores via `score_tts_emotion_fidelity`, writes winners to `emotion_prosody.yaml`
  - First run on `tears_of_steel_2min` (hi) confirmed working but slow (~35–45 min for the full 7×5 grid); **next run: narrow to 4×3 grid (~15 min)** then commit the calibrated YAML
  - Owner: QA + ML | Target: Week 3

---

## P1 — Ship in v3.0, can slip to v3.1

### Emotion arc continuity
**Target:** Repair rate ≤ 8% (current ~15%)

- [ ] **Add `emotion_arc` smoothing pass before Stage 4b**
  - For each speaker, scan consecutive segments; if a single segment's emotion is an outlier within a run of 5+, flag it rather than override
  - Write `arc_flagged: true` into segment dict; surface in dashboard transcript view as yellow highlight
  - File: new function `smooth_emotion_arc()` in `emotion.py`

- [ ] **Colour-band emotion in dashboard bilingual transcript (tab 4)**
  - Red = angry/fearful/disgust, Blue = sad, Green = happy/surprised, Grey = neutral
  - One-line CSS injection per row based on `emotion` column in transcript CSV

### Regression test gate

- [ ] **Add `tests/test_emotion_regression.py`**
  - Full pipeline run on `test_clips/tears_of_steel_2min.mp4` (with LLM, with emotion, no diarize for speed)
  - Assert `avg_emotion_register ≥ 3.8` and `tts_fidelity_soft_score ≥ 65%`
  - Mark as `pytest.mark.slow`; run in CI on any PR touching `emotion.py`, `pipeline_v2.py`, or LLM prompts

---

## P2 — v3.1

- [ ] **Per-speaker persona map (`persona_map.json`)**
  - After diarization, load optional `persona_map.json` for valence/arousal offsets per speaker
  - Apply offsets before passing emotion to Stage 4b and SSML prosody layer
  - Initially hand-authored; eventually auto-extracted from first 60s of each speaker's audio

- [ ] **ElevenLabs TTS provider gate**
  - Add `--tts-provider=elevenlabs` flag
  - Build provider abstraction in `tts_audio.py` so edge-tts vs ElevenLabs is a config switch
  - Blocks on pricing negotiation ($0.30/1K chars); BD dependency

- [ ] **Face emotion detection (`--face-emotion` flag)**
  - MediaPipe FaceMesh → per-frame AU scores → aggregate over segment
  - Fuse with audio+text at 0.6 / 0.2 / 0.2
  - Opt-in only (adds ~45s/clip on CPU)

---

## Blocked / External dependencies

- [ ] **OTT A/B test framework** — viewer drop-off metric at emotional peaks
  - Owner: Eros Now partner
  - Target: v3.1

- [ ] **ElevenLabs Emotion API access** — needed for P2 TTS upgrade
  - Owner: BD
  - Target: Week 4

---

## Done ✅

- SSML prosody control per emotion (pitch/volume/rate) — `emotion.py`, `tts_audio.py`
- Valence/arousal circumplex for soft similarity scoring — `emotion.py`
- Text-based SER (j-hartmann) + back-translation for Hindi scoring — `emotion.py`
- TTS emotion fidelity scoring (audio SER on dubbed output) — `emotion.py`
- Stage 4c repair loop for low-scoring segments — `pipeline_v2.py`
- MOS rubric (5-dim, Claude Sonnet) — `metrics.py`
- MOS rubric panel in dashboard — `dashboard_v2.py`
- Tamil language support (voices, filler map, emotion guidance) — `tts_audio.py`, `pipeline_v2.py`
- Wav2Lip integration + dashboard UX — `apply_lipsync.py`, `dashboard_v2.py`
