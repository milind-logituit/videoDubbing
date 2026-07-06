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
  - emotion2vec+ Large (`iic/emotion2vec_plus_large`) evaluated 2026-06-26 — scored 67.6% (worse: biases toward "angry" while audeering biases toward "surprised", which inadvertently scores neutral segments better)
  - Direct VA comparison tried 2026-06-26 — bypassed argmax step, compared intended VA directly to audeering raw (valence, arousal). Scored 56.0% (worse: audeering outputs cluster in high-arousal/positive-valence region ~(0.4, 0.85) regardless of intended emotion — different normalization from `_VALENCE_AROUSAL` [-1, 1] space; argmax step correctly normalizes to our canonical space)
  - Praat/parselmouth prosody-only scorer tried 2026-06-26 (`score_tts_prosody_fidelity` in emotion.py) — scored 39.5%. Root cause: natural sentence-level F0 variance (±15-22%, driven by intonation) is larger than SSML pitch offsets (8-22%). Azure TTS does apply SSML pitch (surprised=289 Hz vs neutral=254 Hz baseline), but the content-dependent variance (~25 Hz std) swamps the signal. Would require paired neutral/emotional recordings of identical text to isolate SSML contribution.
  - **All local scorer alternatives exhausted.** 74.7% is the practical ceiling for audeering + Azure Hindi TTS. To exceed 75%: need ElevenLabs TTS (blocked on BD) or paired neutral reference audio to enable Praat-based differential scoring.

- [x] **Prosody-*transfer* fidelity scorer** — done 2026-07-02
  - `score_tts_prosody_transfer(source_audio, dubbed_audio, segments)` in `emotion.py`; wired as Stage 8e-iii in `pipeline_v2.py` (added **alongside** `tts_fidelity`, does not replace it or the regression gate)
  - Compares z-normalised F0 + energy **contour shapes** of dubbed Hindi vs the original English **vocal stem** (captured from demucs `vocals.wav`, falls back to source mix), per aligned segment. F0 weighted 0.6 / energy 0.4; correlation `[-1,1]→[0,1]`; segments with <10% voiced source skipped
  - Sidesteps both prior failure modes: references the *actual source performance* (not a content-sensitive SER argmax) and measures contour *correlation* (not SSML-offset detection, which the earlier Praat scorer showed is swamped by content variance)
  - **First run on `tears_of_steel_2min` (hi): 51.7% over 19 segments** — most per-segment F0 correlations near-zero/negative. Diagnostic confirms the 74.7% ceiling is NOT a scorer artifact: the current SSML-preset pipeline genuinely does **not** transfer the source performance's dynamics into the dub. This is the target metric for source-conditioned prosody warping (next lever) — expect it to rise substantially once the Hindi TTS is contour-warped to the English source
- [~] **Source-conditioned prosody warping (PoC)** — `code/warp_prosody.py`, 2026-07-02
  - WORLD (pyworld) analysis→resynth of dubbed audio; per segment, blends dubbed log-F0 *shape* toward source log-F0 shape (dubbed mean/range retained), leaving spectral envelope + aperiodicity untouched. `blend` in [0,1]. Deps: `pyworld`, `soundfile`
  - **tears_of_steel, prosody-transfer score:** baseline 51.7% → blend0.4 54.4 → **blend0.6 57.6** → blend0.8 59.7 → blend1.0 61.5. Monotonic — warping causally lifts the stuck metric
  - ⚠️ **Partial circularity:** warp-toward-source then score-vs-source; the gain proves the mechanism works, NOT that output sounds better. pyworld resynth is lossy (vocoder artifacts even at blend0). **Not yet validated for naturalness** — existing MOS rubric grades TEXT not audio, so no automated arbiter; needs a listening test (baseline vs blend0.6 vs blend1.0) before shipping. Warped WAVs in `scratchpad/tears_warped_b*.wav`
  - Next: A/B listen → if natural, wire as pipeline stage + sweep all clips; if artifacty, try warping only high-arousal segments or a gentler blend
  - **6-clip validation sweep 2026-07-02 (mean 50.6%):** tears_of_steel 51.7 (n19) · sintel_climax 51.5 (n19) · sintel_shaman 55.7 (n18) · elephants_dream 49.2 (n17) · advanced_english 53.5 (n17) · cosmos_laundromat 41.8 (**n2** — 17 segs skipped, music-dominant clip w/ <10% voiced source). Five of six cluster 49–56% with median F0 r≈0 → **no transfer, consistently, across content types**. Justifies building source-conditioned prosody warping (pyworld). Sweep script: `scratchpad/sweep_prosody_transfer.py`
  - Added `praat-parselmouth>=0.4.7` to deps; 5 new unit tests (`tests/test_emotion_scoring.py`) — matching/opposite/unvoiced/missing-file paths; 52 passing

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

## Tech debt — flagged during pipeline_v2 modularisation (2026-07-06, branch `refactor/pipeline-v2-modules`)

- [ ] **Loudnorm silently never runs from the CLI** — `run_pipeline` calls `apply_loudnorm(dubbed_video, force=args.force)` but argparse defines no `--force` flag; the `AttributeError` is swallowed by the Stage 7c `try/except`, so every CLI run prints the "[warn] Loudnorm failed" path and ships unnormalised audio. Fix: add `--force` (or drop the kwarg). Pre-existing bug, found not introduced by the refactor
- [ ] **`SOURCE_SCRIPT`/`FULL_SOURCE_TEXT` duplicated** — `code/metrics.py` keeps a manual copy ("kept in sync" comment), now also defined in `code/video.py`; consolidate into one shared constant
- [ ] **`save_outputs` dead parameters** — takes `src_audio` / `dubbed_audio` and never uses them (`code/reporting.py`); drop them (touches call site in `run_pipeline`)
- [ ] **Stage numbering non-monotonic in `run_pipeline`** — Stages 2.5/2.6/2.7 (emotion) run after 4b; renumber or reorder logs for readability
- [ ] **Per-module test coverage** — `asr.py`, `translation.py`, `video.py`, `reporting.py` are only covered indirectly via `tests/test_pipeline_v2.py`; add dedicated test files before COMET-QE / LatentSync / EMO-SIM work lands

---

## Lipsync (LatentSync) — pick up Monday 2026-06-30

- [ ] **Confirm inference params are being applied server-side** — output file size identical across default and HQ settings (inference_steps=35, guidance_scale=2.2, enable_deepcache=false); Atharv to verify params are wired into the API handler
- [ ] **Face detector fails on animated/VFX content** — `RuntimeError: Face not detected` on all our dubbed clips (sintel_climax, sample_en); need real human-face footage to test end-to-end
- [ ] **VPN access via Sandeep** — GPU machine only reachable over internal VPN; email drafted, pending send
- [ ] **Quality bar** — Atharv's `test_output2.mp4` is noticeably better than API output even with same clip; investigate if server is running different settings locally vs API

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
