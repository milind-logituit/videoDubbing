# TODO — VideoDubbing v3.0

Tracking open work to meet v3 emotion-matching targets.
Spec: `docs/emotion_matching_spec.docx`

---

## P0 — Must-have for v3.0

### Multi-modal emotion detection
**Target:** avg `emotion_register` ≥ 4.0 (current ~3.1)

- [x] **Expose `arousal` and `valence` per segment** — eebfc8b (derived from text SER circumplex, passed to Stage 4b)
- [x] **Fuse audio SER into `classify_segment_emotions()`** — done 2026-06-16
  - Switched from `superb/wav2vec2-base-superb-er` (IEMOCAP-trained, misfired on broadcast) to `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (MSP-PODCAST-trained, broadcast/film)
  - Audeering outputs continuous [arousal, dominance, valence]; mapped to Russell circumplex via `_va_to_dist()` → fused at 0.65 text / 0.35 audio using shared VA space
  - `use_audio_ser=True` by default; falls back to text-only gracefully on model-load or audio failure
  - 11 new unit tests covering `_va_to_dist`, text-only path, audio fusion, and fallback — all 159 tests passing

### Emotion-controlled TTS calibration
**Target:** TTS fidelity soft score ≥ 75% (current ~62%)

- [x] **Externalise `SSML_PROSODY` to `emotion_prosody.yaml`** — done 2026-06-15
  - `load_prosody_config(lang)` in `emotion.py` loads `config/emotion_prosody.yaml` (`default` block + optional per-lang overrides); falls back to hardcoded `SSML_PROSODY` if missing/unreadable
  - `tts_audio.py` derives `lang` from `voice[:2]` and pulls prosody via the loader; `pyyaml>=6.0.3` added to `pyproject.toml`

- [x] **Calibrate prosody params against MOS rubric** — done 2026-06-15
  - 3×3 grid (pitch × rate) per emotion on `tears_of_steel_2min` (hi); winners written to `config/emotion_prosody.yaml` under `hi:` block
  - MOS: 83.2 → **85.2** (+2.0); TTS fidelity scorer unreliable (wav2vec2 misfires on broadcast) — wider grid deferred until better model available
  - Do not expand to 5×7 grid until `audeering/wav2vec2-large-robust` or equivalent is available as scorer

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
