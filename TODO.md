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

- [x] **Expand calibration to 5×7 prosody grid** — done 2026-06-16
  - `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` now in `score_tts_emotion_fidelity()` — gate cleared
  - `_PITCH_VARIANTS` 3→5, `_RATE_VARIANTS` 3→7 (35 combos/emotion); max segments per emotion 3→5
  - **Calibration result: inconclusive** — all emotions converge to pitch=-15%/rate=-20%; scores 7–35% vs 74.7% on full-clip
  - Root cause (confirmed 2026-06-26): audeering predicts "surprised"/"happy" for 18/20 segments regardless of intended emotion; speech content dominates over prosody in VA prediction
  - Fixed per-segment cache bug in `calibrate_prosody.py` (scores now vary across combos); but calibrated params made tts_fidelity WORSE (65.7%) than default hand-crafted params (74.7%)
  - YAML `hi:` block reverted to defaults; calibration approach (vary pitch/rate → score with audeering) cannot reliably push above 75% because audeering is content-sensitive not prosody-sensitive
  - To clear 75% reliably: need ElevenLabs (more expressive emotion control, blocked on BD) OR a different TTS fidelity scorer that isolates prosody from content

---

## P1 — Ship in v3.0, can slip to v3.1

### Emotion arc continuity
**Target:** Repair rate ≤ 8% (current ~15%)

- [x] **Add `emotion_arc` smoothing pass before Stage 4b** — done (prior session)
  - `smooth_emotion_arc()` in `emotion.py`; wired at `pipeline_v2.py:925`
  - Writes `arc_flagged: true` into segment dict; surfaced as yellow highlight in dashboard tab 4

- [x] **Colour-band emotion in dashboard bilingual transcript (tab 4)** — done 2026-06-16
  - Red = angry/fearful/disgust (`#FEE2E2`), Blue = sad (`#DBEAFE`), Green = happy/surprised (`#DCFCE7`), Grey = neutral (`#F3F4F6`)
  - Yellow (`#FEF9C3`) for `arc_flagged` rows (overrides emotion colour)
  - Implemented via `_emotion_row_style()` in `dashboard_v2.py`

### Regression test gate

- [x] **Add `tests/test_emotion_regression.py`** — done; gate passed 2026-06-16
  - Full pipeline run on `test_clips/tears_of_steel_2min.mp4` (with LLM, with emotion, no diarize)
  - **Results:** `avg_emotion_register` 4.00/5 (gate ≥3.8 ✅, target ≥4.0 ✅) · `tts_fidelity` 74.7% (gate ≥65% ✅, target ≥75% ⚠️ just below)
  - Thresholds aligned to `docs/emotion_matching_spec.docx`; prior test was checking `emotion.avg_soft_score` (wrong metric)
  - Marked `pytest.mark.slow`; run with `uv run pytest -m slow`

---

## P2 — v3.1

- [x] **Per-speaker persona map (`persona_map.json`)** — done 2026-06-16
  - `load_persona_map(path)` + `apply_persona_offsets(segments, map)` in `emotion.py`
  - Wired as Stage 2.7 in `pipeline_v2.py` (after face emotion, before repair); loads `persona_map.json` from the input video's directory
  - Format: `{"SPEAKER_00": {"valence_offset": -0.2, "arousal_offset": 0.1}, …}`; example in `config/persona_map_example.json`
  - Adjusted segments get `persona_adjusted: True`; 11 new tests — 170 total passing

- [ ] **ElevenLabs TTS provider gate** — blocked (see below)

- [x] **Face emotion detection (`--face-emotion` flag)** — done (prior session)
  - `classify_face_emotions()` in `emotion.py`; wired as Stage 2.6 in `pipeline_v2.py`
  - MediaPipe FaceMesh → VA proxy → fused with text SER at 0.6/0.4; opt-in via `--face-emotion`

---

## Blocked / External dependencies

- [ ] **OTT A/B test framework** — viewer drop-off metric at emotional peaks
  - Owner: Eros Now partner
  - Target: v3.1

- [ ] **ElevenLabs Emotion API access** — needed for P2 TTS upgrade
  - Owner: BD
  - Target: Week 4

- [ ] **ElevenLabs TTS provider gate** — waiting on BD pricing ($0.30/1K chars) + API access
  - Add `--tts-provider=elevenlabs` flag; provider abstraction in `tts_audio.py`

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
