"""Stages 8/8b/8c — Quality metrics, segment grading, back-translation BLEU."""
import json
from pathlib import Path

import anthropic
import jiwer
import sacrebleu
import whisper
from deep_translator import GoogleTranslator
from pydub import AudioSegment

ROOT      = Path(__file__).parent.parent
PREPARED  = ROOT / "data/prepared"
MODEL_OUT = ROOT / "model_outputs"

WHISPER_MODEL    = "large-v3-turbo"
WHISPER_MODEL_HI = "large-v3-turbo"
TTS_VOICE_HI     = "hi-IN-SwaraNeural"

# Reference data — kept in sync with pipeline_v2.py SOURCE_SCRIPT / REFERENCE_HINDI
_SOURCE_SCRIPT = [
    {"id": 1, "speaker": "Narrator",
     "text": ("In a world where streaming has replaced the multiplex, "
               "content is king — and every second of screen time must "
               "earn its place.")},
    {"id": 2, "speaker": "Character A",
     "text": ("We built this platform from nothing. "
               "Forty million subscribers in five years. "
               "Nobody thought it was possible.")},
    {"id": 3, "speaker": "Character B",
     "text": ("The audience has changed. "
               "They want stories told in their own language, "
               "with voices that feel like home.")},
    {"id": 4, "speaker": "Narrator",
     "text": ("Today, artificial intelligence makes it possible "
               "to bring every story to every audience — "
               "instantly, accurately, and at scale.")},
]
FULL_SOURCE_TEXT = " ".join(s["text"] for s in _SOURCE_SCRIPT)

REFERENCE_HINDI = (
    "एक ऐसी दुनिया में जहां स्ट्रीमिंग ने मल्टीप्लेक्स की जगह ले ली है, "
    "कंटेंट ही राजा है। "
    "हमने यह प्लेटफॉर्म कुछ नहीं से बनाया। "
    "पांच साल में चार करोड़ सब्सक्राइबर। "
    "दर्शक बदल गए हैं। "
    "वे अपनी भाषा में कहानियां सुनना चाहते हैं। "
    "आज, आर्टिफिशियल इंटेलिजेंस हर कहानी को हर दर्शक तक पहुंचाना संभव बनाता है।"
)

_GRADE_SYSTEM = """\
You are a professional Hindi dubbing quality assessor.
Rate each segment on three dimensions (1 = poor, 5 = excellent):
- fidelity: semantic accuracy of Hindi vs English
- fluency: naturalness of the Hindi phrasing (grammar, word choice, register)
- emotion_register: ONLY when an "emotion" field is provided — does the Hindi
  carry that emotional weight through word choice and phrasing?
  (1 = flat/wrong register, 3 = partially preserved, 5 = fully preserved)
  Omit this key entirely when no "emotion" field is in the input segment.

Respond ONLY with a JSON array in input order:
[{"id": <int>, "fidelity": <1-5>, "fluency": <1-5>, "emotion_register": <1-5 or omit>}, ...]
Add "note": "..." only for scores ≤ 2. Omit otherwise.\
"""


def _isochrony_fit(ratio: float | None) -> int:
    """Map isochrony ratio to 1-5 fit score (algorithmic, reliable)."""
    if ratio is None:
        return 3
    if ratio <= 0.80:
        return 3
    if ratio <= 1.15:
        return 5
    if ratio <= 1.30:
        return 4
    if ratio <= 1.50:
        return 3
    if ratio <= 2.00:
        return 2
    return 1


def compute_metrics(whisper_result: dict, segments: list[dict],
                    src_audio: Path, dubbed_audio: Path,
                    has_reference: bool = True) -> dict:
    machine_hindi = " ".join(s["hi_text"] for s in segments)

    if has_reference:
        wer_score = jiwer.wer(FULL_SOURCE_TEXT.lower(),
                               whisper_result["text"].lower())
        bleu_obj  = sacrebleu.corpus_bleu([machine_hindi], [[REFERENCE_HINDI]])
        wer_val: float | None     = round(wer_score, 4)
        wer_pct_val: float | None = round(wer_score * 100, 2)
        bleu_val: float | None    = round(bleu_obj.score, 2)
        wer_note  = None
        bleu_note = None
    else:
        wer_val = wer_pct_val = bleu_val = None
        wer_note  = "N/A — no reference text for this clip"
        bleu_note = "N/A — no reference text for this clip"

    try:
        src_dur = len(AudioSegment.from_file(str(src_audio))) / 1000
        dub_dur = len(AudioSegment.from_mp3(str(dubbed_audio))) / 1000
        dar     = dub_dur / src_dur
    except Exception:
        src_dur = dub_dur = dar = 0.0

    asr_block: dict = {"model": WHISPER_MODEL, "wer": wer_val,
                       "wer_pct": wer_pct_val}
    if wer_note is not None:
        asr_block["note"] = wer_note

    trans_block: dict = {"model": "Google Translate (en→hi)",
                         "bleu": bleu_val,
                         "n_segments": len(segments)}
    if bleu_note is not None:
        trans_block["note"] = bleu_note

    return {
        "asr":         asr_block,
        "translation": trans_block,
        "tts":         {"voice": TTS_VOICE_HI},
        "alignment":   {"source_duration_s": round(src_dur, 2),
                        "dubbed_duration_s": round(dub_dur, 2),
                        "duration_ratio": round(dar, 3),
                        "en_chars_per_sec": round(
                            len(FULL_SOURCE_TEXT.replace(" ", "")) / src_dur, 1
                        ) if src_dur else 0,
                        "hi_chars_per_sec": round(
                            len(machine_hindi.replace(" ", "")) / dub_dur, 1
                        ) if dub_dur else 0},
    }


def compute_text_bleu(segments: list[dict],
                      original_source_text: str,
                      source_lang: str = "en",
                      target_lang: str = "hi") -> dict:
    """Back-translate joined target text directly — no TTS/ASR noise in the loop."""
    try:
        tgt_text = " ".join(
            str(s.get(f"{target_lang}_text") or s.get("hi_text") or "").strip()
            for s in segments
            if str(s.get(f"{target_lang}_text") or s.get("hi_text") or "").strip()
        )
        bt_text = GoogleTranslator(source=target_lang, target=source_lang).translate(tgt_text)
        bleu_obj = sacrebleu.corpus_bleu([bt_text], [[original_source_text]])
        return {
            "bleu": round(bleu_obj.score, 2),
            "back_translated": bt_text,
            "n_segments": len(segments),
        }
    except Exception as exc:
        return {"bleu": None, "note": f"Failed: {exc}"}


def compute_back_translation_bleu(dubbed_audio: Path,
                                   original_source_text: str,
                                   source_lang: str = "en",
                                   target_lang: str = "hi") -> dict:
    """ASR dubbed audio, back-translate to source language, score with BLEU."""
    try:
        print(f"  Transcribing dubbed {target_lang.upper()} audio …")
        model_size = WHISPER_MODEL_HI if target_lang == "hi" else WHISPER_MODEL
        tgt_model = whisper.load_model(model_size)
        tgt_result = tgt_model.transcribe(str(dubbed_audio), language=target_lang)
        tgt_transcript = tgt_result["text"].strip()

        print(f"  Back-translating {target_lang.upper()} → {source_lang.upper()} …")
        bt_text = GoogleTranslator(source=target_lang, target=source_lang).translate(tgt_transcript)

        bleu_obj = sacrebleu.corpus_bleu([bt_text], [[original_source_text]])
        result = {
            "tgt_transcript_chars": len(tgt_transcript),
            "back_translated": bt_text,
            "bleu": round(bleu_obj.score, 2),
        }
        if target_lang == "hi":
            result["hi_transcript_chars"] = len(tgt_transcript)
            result["back_translated_en"] = bt_text
        return result
    except Exception as exc:
        return {"bleu": None, "note": f"Failed: {exc}"}


def compute_segment_isochrony(segments: list[dict], seg_dir: Path) -> list[dict]:
    """Per-segment isochrony ratio: TTS duration / EN window duration."""
    results = []
    for seg in segments:
        sid = int(seg["id"])
        en_dur = round(float(seg["end"]) - float(seg["start"]), 2)
        effective = list(seg_dir.glob(f"seg_{sid:03d}_*_effective.mp3"))
        candidates = effective or list(seg_dir.glob(f"seg_{sid:03d}_*.mp3"))
        if candidates:
            tts_dur = round(len(AudioSegment.from_mp3(str(candidates[0]))) / 1000.0, 2)
            ratio = round(tts_dur / en_dur, 3) if en_dur > 0 else None
        else:
            tts_dur = ratio = None
        results.append({
            "id": sid,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "en_duration_s": en_dur,
            "tts_duration_s": tts_dur,
            "isochrony_ratio": ratio,
        })
    return results


_GRADE_BATCH_SIZE = 20
_GRADE_TOKENS_PER_SEGMENT = 120  # ~80 tokens output + 40 headroom per segment


def _parse_grade_response(raw: str) -> list[dict]:
    """Strip markdown fences and parse JSON; raises ValueError on failure."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, list):
        raise ValueError(f"Expected JSON array, got {type(result).__name__}")
    return result


def _grade_batch(client: anthropic.Anthropic, batch: list[dict]) -> list[dict]:
    """Grade one batch; retries once with explicit repair prompt on parse failure."""
    max_tokens = max(512, len(batch) * _GRADE_TOKENS_PER_SEGMENT)
    content = json.dumps(batch, ensure_ascii=False)

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=_GRADE_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    raw = msg.content[0].text
    try:
        return _parse_grade_response(raw)
    except Exception as exc:
        print(f"  [warn] grade batch parse failed ({exc}), retrying …")
        retry = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=_GRADE_SYSTEM,
            messages=[
                {"role": "user", "content": content},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Return ONLY the JSON array, no markdown, no explanation."},
            ],
        )
        try:
            return _parse_grade_response(retry.content[0].text)
        except Exception as exc2:
            print(f"  [warn] grade batch retry also failed ({exc2}), skipping batch.")
            return []


def grade_translations(segments: list[dict]) -> list[dict]:
    """Call Claude Haiku to rate fidelity / fluency / fit per segment.

    Batches into groups of _GRADE_BATCH_SIZE to avoid max_tokens truncation
    on long clips.
    """
    payload = [
        {
            "id": int(seg["id"]),
            "en": str(seg.get("en_text") or ""),
            "hi": str(seg.get("hi_text") or ""),
            "duration_s": round(float(seg["end"]) - float(seg["start"]), 1),
            **({"emotion": seg["emotion"]} if seg.get("emotion") and seg["emotion"] != "neutral" else {}),
        }
        for seg in segments
        if str(seg.get("hi_text") or "").strip()
    ]
    if not payload:
        return []

    client = anthropic.Anthropic()
    results: list[dict] = []
    for i in range(0, len(payload), _GRADE_BATCH_SIZE):
        batch = payload[i : i + _GRADE_BATCH_SIZE]
        print(f"  Grading segments {i + 1}–{i + len(batch)} of {len(payload)} …")
        results.extend(_grade_batch(client, batch))
    return results


def _merge_segment_quality(
    isochrony: list[dict], grades_by_id: dict[int, dict]
) -> list[dict]:
    out = []
    for row in isochrony:
        g = grades_by_id.get(row["id"], {})
        merged = dict(row)
        merged["fit"] = _isochrony_fit(row.get("isochrony_ratio"))
        if g:
            merged["fidelity"] = g.get("fidelity")
            merged["fluency"]  = g.get("fluency")
            if "emotion_register" in g:
                merged["emotion_register"] = g["emotion_register"]
            if "note" in g:
                merged["note"] = g["note"]
        out.append(merged)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8f — MOS-style rubric (5-dimension spot-check via Claude Sonnet)
# ─────────────────────────────────────────────────────────────────────────────

_MOS_SYSTEM = """\
You are a professional dubbing quality evaluator for OTT streaming content.
Rate each segment on five dimensions (1 = poor, 5 = excellent):
- naturalness: Does the Hindi sound like natural spoken dialogue, not literal translation?
- fidelity: Is the core meaning of the English accurately preserved?
- timing: Does the Hindi length fit the duration_s window? Shorter or equal is better.
- emotion: Does the Hindi carry the right emotional register? Assume neutral if no emotion field.
- names: Are proper nouns / character names correctly transliterated? Rate 5 if none present.

Return ONLY a JSON array in input order:
[{"id": <int>, "naturalness": <1-5>, "fidelity": <1-5>, "timing": <1-5>, "emotion": <1-5>, "names": <1-5>}, ...]
No markdown, no explanation."""

_MOS_DIMS = ["naturalness", "fidelity", "timing", "emotion", "names"]


def score_mos_rubric(
    segments: list[dict],
    glossary: dict[str, str] | None = None,
    n_sample: int = 10,
) -> dict:
    """Sample up to n_sample segments evenly and rate on 5 dimensions via Claude Sonnet.

    Returns {"mos": float (0-100), "n_sampled": int, "breakdown": {...}, "per_segment": [...]}.
    """
    valid = [s for s in segments if str(s.get("hi_text") or "").strip()]
    if not valid:
        return {"mos": None, "note": "No segments to evaluate"}

    step   = max(1, len(valid) // n_sample)
    sample = valid[::step][:n_sample]

    payload = [
        {
            "id": int(s["id"]),
            "en": str(s.get("en_text", "")),
            "hi": str(s.get("hi_text", "")),
            "duration_s": round(float(s["end"]) - float(s["start"]), 1),
            **({"emotion": s["emotion"]}
               if s.get("emotion") and s["emotion"] != "neutral" else {}),
        }
        for s in sample
    ]

    system = _MOS_SYSTEM
    if glossary:
        terms = "; ".join(f"{k}={v}" for k, v in glossary.items())
        system += f"\n\nExpected transliterations (use for 'names' score): {terms}"

    client = anthropic.Anthropic()
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user",
                       "content": json.dumps(payload, ensure_ascii=False)}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        ratings = json.loads(raw)
    except Exception as exc:
        return {"mos": None, "note": f"Failed: {exc}"}

    totals = {d: 0.0 for d in _MOS_DIMS}
    count  = 0
    for r in ratings:
        for d in _MOS_DIMS:
            if d in r:
                totals[d] += float(r[d])
        count += 1

    if count == 0:
        return {"mos": None, "note": "No ratings returned"}

    averages = {d: round(totals[d] / count, 2) for d in _MOS_DIMS}
    mos      = round(sum(averages.values()) / len(_MOS_DIMS) / 5 * 100, 1)

    return {
        "mos":         mos,
        "n_sampled":   count,
        "breakdown":   averages,
        "per_segment": ratings,
    }
