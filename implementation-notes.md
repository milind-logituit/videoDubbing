# pipeline_v2 modularisation — implementation notes

Branch: `refactor/pipeline-v2-modules` (2026-07-06)

Pure structural refactor of `code/pipeline_v2.py` (1215 → ~350 lines). Every
function body was copied byte-for-byte; only the module it lives in and the
imports changed. `run_pipeline`'s signature, CLI args, and return shape are
unchanged.

## What moved where

| New location | Symbols |
|---|---|
| `code/asr.py` | `extract_audio`, `transcribe`, `WHISPER_MODEL`, `WHISPER_MODEL_HI` |
| `code/translation.py` | `translate_segments`, `refine_segments`, `extract_glossary`, `repair_emotion_register`, `_make_refine_system`, `_ctx_snippet`, `_refine_batch`, `_make_repair_system`, `_FILLER_MAPS`, `_LANG_NAMES`, `_REFINE_TARGET_GUIDANCE`, `_EMOTION_GUIDANCE`, `_DEFAULT_EMOTION_GUIDANCE`, `_LLM_BATCH`, `_CTX_WINDOW`, `_ASR_SKIP_THRESH` |
| `code/video.py` | `generate_sample_video`, `_generate_source_audio`, `_make_title_frame`, `generate_vtt`, `generate_srt`, `_vtt_time`, `_srt_time`, `create_dubbed_video`, `apply_loudnorm`, `SOURCE_SCRIPT`, `FULL_SOURCE_TEXT`, `REFERENCE_HINDI`, `VIDEO_SIZE`, `VIDEO_FPS`, `BG_COLOR`, `ACCENT_COLOR` |
| `code/reporting.py` | `save_outputs`, `_print_quality`, `_metrics_summary_row` |
| `code/tts_audio.py` (existing) | `_assign_voice_pool` appended |
| `code/pipeline_v2.py` | `run_pipeline` + `__main__` CLI only, plus a re-export block for backwards compatibility |

## Deviations from the proposed split

1. **Flat modules instead of `asr/`, `translation/`, … packages.** The
   codebase convention in `code/` is flat sibling modules (`diarize.py`,
   `tts_audio.py`, `metrics.py`) imported via a `sys.path` insert, and the
   test suite loads modules by file path (`spec_from_file_location`).
   Packages would break both patterns for no gain at this size.
2. **`metrics/` renamed to `reporting.py`.** A `metrics` module already
   exists (`code/metrics.py` — `compute_metrics`, `grade_translations`, …)
   and `pipeline_v2` imports from it. A package of the same name would
   shadow it. `save_outputs` / `_print_quality` / `_metrics_summary_row`
   are output/reporting concerns anyway, not metric computation.
3. **No `tts.py` module.** The proposed grouping had low cohesion:
   - `_assign_voice_pool` → appended to existing `tts_audio.py`, next to
     the `VOICE_POOL` table it consumes.
   - `_generate_source_audio` → `video.py`; its only caller is
     `generate_sample_video` and it exists solely to build the sample clip.
   - `apply_loudnorm` → `video.py`; it is ffmpeg post-processing of the
     dubbed video, pipelined directly after `create_dubbed_video`.
4. **Three `patch()` targets updated in `tests/test_pipeline_v2.py`**
   (`pipeline_v2.GoogleTranslator` → `translation.GoogleTranslator`,
   `pipeline_v2._refine_batch` → `translation._refine_batch`,
   `pipeline_v2.anthropic.Anthropic` → `translation.anthropic.Anthropic` ×2).
   `unittest.mock.patch` rebinds names in the module where the function
   *lives*, so the targets must follow the moved functions. All other test
   access goes through `_mod.<name>` attribute reads, which the re-export
   block preserves.
5. **Constants moved with their consumers.** `SOURCE_SCRIPT` /
   `FULL_SOURCE_TEXT` / `REFERENCE_HINDI` are sample-clip data → `video.py`.
   `WHISPER_MODEL*` → `asr.py`. All are re-exported from `pipeline_v2`.
6. **`RAW`/`PREPARED`/`MODEL_OUT` path constants duplicated per module**
   (matching how `tts_audio.py` / `metrics.py` already do it). The
   `mkdir` side effect stays in `pipeline_v2.py`, which is imported before
   any stage runs.

## Pre-existing issues noticed (NOT fixed — out of scope)

- `run_pipeline` calls `apply_loudnorm(dubbed_video, force=args.force)` but
  the CLI defines no `--force` argument. The resulting `AttributeError` is
  swallowed by the surrounding `try/except`, so **loudnorm silently never
  runs** on CLI invocations ("[warn] Loudnorm failed … 'Namespace' object
  has no attribute 'force'").
- `code/metrics.py` keeps a manual copy of `SOURCE_SCRIPT`/`FULL_SOURCE_TEXT`
  ("kept in sync with pipeline_v2.py" comment) — now duplicated with
  `video.py`; candidate for a shared constant later.
- `save_outputs` takes `src_audio` / `dubbed_audio` parameters it never uses.
- Stage numbering in `run_pipeline` is non-monotonic (2.5/2.6/2.7 run after
  4b) — preserved as-is.

## Follow-up (separate tasks, not done here)

- `asr.py`, `translation.py`, `video.py`, `reporting.py` have no dedicated
  unit-test files (they are covered indirectly via `test_pipeline_v2.py`).
  Direct per-module tests would help before the COMET-QE / LatentSync /
  EMO-SIM work lands.
