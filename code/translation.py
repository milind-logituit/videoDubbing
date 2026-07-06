"""Stages 4/4b/4b.5/4c — Translation, LLM refinement, glossary, emotion repair."""
import json
from pathlib import Path

import anthropic
from deep_translator import GoogleTranslator

_FILLER_MAPS: dict[str, dict[str, str]] = {
    "hi": {"hmm": "हाँ", "uh": "", "um": "", "ah": "अच्छा"},
    "en": {"hmm": "yeah", "uh": "", "um": "", "ah": "ah"},
    "ta": {"hmm": "ஆம்", "uh": "", "um": "", "ah": "ஆ"},
}


def translate_segments(whisper_result: dict,
                       source_lang: str = "en",
                       target_lang: str = "hi") -> list[dict]:
    translator   = GoogleTranslator(source=source_lang, target=target_lang)
    filler_map   = _FILLER_MAPS.get(target_lang, {})
    segments_out = []
    for seg in whisper_result["segments"]:
        en = seg["text"].strip()
        if not en:
            continue
        hi = translator.translate(en) or en
        if en.lower().rstrip(".!?,") in filler_map:
            hi = filler_map[en.lower().rstrip(".!?,")]
        entry: dict = {
            "id":       seg["id"],
            "start":    round(seg["start"], 3),
            "end":      round(seg["end"],   3),
            "duration": round(seg["end"] - seg["start"], 3),
            "en_text":  en,
            "hi_text":  hi,
        }
        if "speaker" in seg:
            entry["speaker"] = seg["speaker"]
        if "no_speech_prob" in seg:
            entry["no_speech_prob"] = round(seg["no_speech_prob"], 3)
        segments_out.append(entry)
        print(f"    [{seg['start']:.1f}s]  {en[:55]}")
        print(f"           →  {hi[:55]}")
    print(f"  Translated {len(segments_out)} segments.")
    return segments_out


_LLM_BATCH = 100
_ASR_SKIP_THRESH = 0.70   # skip LLM refinement for segments where Whisper is unsure

_LANG_NAMES = {"en": "English", "hi": "Hindi", "de": "German", "ta": "Tamil"}

_REFINE_TARGET_GUIDANCE = {
    "hi": (
        "     • Hindi TTS speaks at ~3.5 words/second — use this to judge "
        "length. A 2s window fits ~7 Hindi words maximum.\n"
        "     • Prefer shorter, natural phrasing over complete sentences when "
        "the window is tight. Cut filler and subordinate clauses first.\n"
        "     • Use common, everyday Hindi vocabulary (Hindustani/Bollywood register) "
        "that speech recognition systems reliably transcribe. "
        "Avoid rare, Sanskritised, or literary Hindi words — prefer their "
        "everyday equivalents (e.g. 'काम' over 'कार्य', 'बात' over 'वार्तालाप').\n"
        "     • Fillers: 'Hmm' → 'हाँ', 'Uh'/'Um' → empty string, 'Ah' → 'अच्छा'.\n"
    ),
    "en": (
        "     • English TTS speaks at ~3.0–3.5 words/second — a 2s window "
        "fits ~6–7 words maximum.\n"
        "     • Prefer shorter, natural phrasing when the window is tight.\n"
        "     • Use natural broadcast English — clear, idiomatic, suitable for "
        "OTT dubbing. Avoid overly literal translations.\n"
        "     • Fillers: 'Hmm' → 'Yeah', 'Uh'/'Um' → empty string, 'Ah' → 'Ah'.\n"
    ),
    "ta": (
        "     • Tamil TTS speaks at ~3.0 words/second — a 2s window fits ~6 "
        "Tamil words maximum.\n"
        "     • Prefer shorter, colloquial Tamil (spoken/cinematic register) over "
        "literary or Sentamil forms. Cut subordinate clauses when the window is tight.\n"
        "     • Use everyday vocabulary that mainstream Tamil audiences recognise: "
        "prefer 'வீடு' over 'இல்லம்', 'பேசு' over 'சொல்', 'நான்' over 'யான்'.\n"
        "     • Fillers: 'Hmm' → 'ஆம்', 'Uh'/'Um' → empty string, 'Ah' → 'ஆ'.\n"
    ),
}

_EMOTION_GUIDANCE = {
    "hi": (
        "  3. EMOTION (hard constraint — not optional): If an 'emotion' field is present,\n"
        "     the rewritten Hindi MUST carry that emotional register through word choice.\n"
        "     Use these Hindi-specific cues:\n"
        "       • angry   → forceful verbs, exclamatory particles (अरे!, क्यों!, नहीं!),\n"
        "                    short urgent clauses, avoid soft conjunctions\n"
        "       • fearful → tense/hesitant phrasing, words like डर, खतरा, बचाओ,\n"
        "                    broken or incomplete clauses where natural\n"
        "       • sad     → soft conjunctions (लेकिन, मगर, पर), reduced energy,\n"
        "                    words like दुख, अफसोस, याद; avoid exclamations\n"
        "       • happy   → upbeat vocab, वाह!, हाँ!, warm qualifiers (बढ़िया, शानदार)\n"
        "       • surprised → ओह!, अरे वाह!, क्या!, wide-eyed reactive phrasing\n"
        "       • disgust → distancing language, words like घिनौना, बेकार, छी\n"
        "       • neutral → plain declarative; do NOT add emotion not in the source\n"
        "     A neutral source line MUST stay neutral. Do not dramatise.\n"
    ),
    "en": (
        "  3. EMOTION (hard constraint — not optional): If an 'emotion' field is present,\n"
        "     the rewritten English MUST carry that emotional register:\n"
        "       • angry   → forceful, clipped sentences; strong verbs; avoid hedging\n"
        "       • fearful → hesitant, halting phrasing; 'I can't', 'we have to'\n"
        "       • sad     → slower cadence implied by word length; 'I miss', 'it's gone'\n"
        "       • happy   → short energetic lines; upbeat qualifiers\n"
        "       • neutral → plain declarative; do NOT add emotion not in the source\n"
        "     A neutral source line MUST stay neutral. Do not dramatise.\n"
    ),
    "ta": (
        "  3. EMOTION (hard constraint — not optional): If an 'emotion' field is present,\n"
        "     the rewritten Tamil MUST carry that emotional register through word choice.\n"
        "     Use these Tamil-specific cues:\n"
        "       • angry   → forceful verbs, exclamatory particles (ஏன்!, வேண்டாம்!, இல்லை!),\n"
        "                    short urgent clauses, avoid soft conjunctions\n"
        "       • fearful → tense/hesitant phrasing, words like பயம், ஆபத்து, காப்பாற்று\n"
        "       • sad     → soft conjunctions (ஆனால், மட்டும்), reduced energy,\n"
        "                    words like துக்கம், வருத்தம், நினைவு; avoid exclamations\n"
        "       • happy   → upbeat vocab, வாழ்க!, ஆம்!, warm qualifiers (அருமை, சிறப்பு)\n"
        "       • surprised → ஓ!, அடேங்கப்பா!, என்ன!, wide-eyed reactive phrasing\n"
        "       • disgust → distancing language, words like அருவருப்பு, பயனற்றது, சீ\n"
        "       • neutral → plain declarative; do NOT add emotion not in the source\n"
        "     A neutral source line MUST stay neutral. Do not dramatise.\n"
    ),
}

_DEFAULT_EMOTION_GUIDANCE = (
    "  3. EMOTION (hard constraint — not optional): If an 'emotion' field is present,\n"
    "     the rewritten translation MUST match that emotional register through word choice.\n"
    "     angry → forceful/urgent; fearful → tense/hesitant; sad → subdued/soft;\n"
    "     happy → upbeat/energetic; neutral → plain, no added drama.\n"
    "     A neutral source MUST stay neutral. Do not dramatise.\n"
)


def _make_refine_system(
    source_lang: str = "en",
    target_lang: str = "hi",
    glossary: dict[str, str] | None = None,
) -> str:
    src_name      = _LANG_NAMES.get(source_lang, source_lang.upper())
    tgt_name      = _LANG_NAMES.get(target_lang, target_lang.upper())
    pacing        = _REFINE_TARGET_GUIDANCE.get(
        target_lang,
        f"     • Use natural, idiomatic {tgt_name} suitable for OTT dubbing.\n",
    )
    emotion_rules = _EMOTION_GUIDANCE.get(target_lang, _DEFAULT_EMOTION_GUIDANCE)
    glossary_rule = ""
    if glossary:
        terms = "; ".join(f"{k}→{v}" for k, v in glossary.items())
        glossary_rule = (
            f"  0. GLOSSARY (hard constraint): These proper nouns MUST appear "
            f"exactly as given in every segment: {terms}.\n"
        )
    return (
        f"You are a professional {tgt_name} dubbing editor for OTT streaming content "
        f"(Eros Now / SunNxt).\n"
        f"Input: JSON array of segments, each with ASR {src_name} (en_text), "
        f"machine-translated {tgt_name} (hi_text), duration_s (seconds available "
        "to speak this line), and optionally emotion fields.\n"
        "When present, 'emotion' is the dominant label and 'emotion_blend' is the full "
        "probability distribution (e.g. {fearful: 0.65, disgust: 0.20, neutral: 0.15}). "
        "Use the blend to capture emotional undertones, not just the top label. "
        "When 'arousal' (0=calm, 1=excited) and 'valence' (-1=negative, +1=positive) are "
        "present, use them to calibrate intensity: high arousal → urgent/energetic phrasing; "
        "low valence → heavier, darker word choices.\n"
        "The input may include optional 'context_before' and 'context_after' arrays "
        "with neighbouring segments in their already-refined form. Use these ONLY for "
        "contextual consistency: match character names, terminology, and emotional arc "
        "with adjacent lines. Do NOT output context segments — only the main segments.\n"
        + glossary_rule
        + "For each segment:\n"
        "  1. Fix ASR transcription errors in en_text "
        "(e.g. 'half is likely' → 'half as likely').\n"
        f"  2. Rewrite hi_text as natural spoken {tgt_name} for dubbing that fits "
        "within duration_s seconds.\n"
        + pacing +
        "     • Distinguish dinner vs supper, couch vs sofa, etc.\n"
        + emotion_rules +
        "Return ONLY a valid JSON array: "
        '[{"id": int, "en_text": str, "hi_text": str}, …]. '
        "Do NOT include duration_s or emotion in output. "
        "Same count and IDs as input. No markdown, no explanation."
    )


def _ctx_snippet(segs: list[dict]) -> list[dict]:
    return [{"id": s["id"], "en_text": s["en_text"], "hi_text": s["hi_text"]}
            for s in segs]


def _refine_batch(
    client: anthropic.Anthropic,
    batch: list[dict],
    refine_system: str,
    ctx_before: list[dict] | None = None,
    ctx_after: list[dict] | None = None,
) -> dict[int, dict]:
    payload: dict = {
        "segments": [
            {
                "id": s["id"],
                "en_text": s["en_text"],
                "hi_text": s["hi_text"],
                "duration_s": round(s["duration"], 2),
                **({"emotion":       s["emotion"],
                    "emotion_blend":  s["emotion_dist"],
                    **( {"arousal": s["arousal"], "valence": s["valence"]}
                        if s.get("arousal") is not None else {} )}
                   if s.get("emotion_dist") and s.get("emotion") != "neutral"
                   else {"emotion": s["emotion"]} if s.get("emotion") else {}),
            }
            for s in batch
        ]
    }
    if ctx_before:
        payload["context_before"] = _ctx_snippet(ctx_before)
    if ctx_after:
        payload["context_after"] = _ctx_snippet(ctx_after)

    content = json.dumps(payload, ensure_ascii=False)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=refine_system,
        messages=[{"role": "user", "content": content}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(raw)
    # Claude returns either the segments array directly or {"segments": [...]}
    if isinstance(parsed, dict):
        parsed = parsed.get("segments", [])
    return {r["id"]: r for r in parsed}


_CTX_WINDOW = 3   # segments of surrounding context passed to Claude per batch


def refine_segments(
    segments: list[dict],
    *,
    skip: bool = False,
    source_lang: str = "en",
    target_lang: str = "hi",
    glossary: dict[str, str] | None = None,
) -> list[dict]:
    """Fix ASR errors and rewrite target language as natural dubbing speech via Claude."""
    if skip:
        print("  Stage 4b skipped (--no-llm).")
        return segments

    # Gate: skip segments where Whisper itself was unsure (likely silence or noise)
    skipped_ids: set[int] = {
        s["id"] for s in segments
        if s.get("no_speech_prob", 0) > _ASR_SKIP_THRESH
    }
    to_refine = [s for s in segments if s["id"] not in skipped_ids]
    if skipped_ids:
        print(f"  Skipping {len(skipped_ids)} low-confidence ASR segment(s) "
              f"(no_speech_prob > {_ASR_SKIP_THRESH}).")

    refine_system = _make_refine_system(source_lang, target_lang, glossary)
    client = anthropic.Anthropic()
    n = len(to_refine)
    n_batches = (n + _LLM_BATCH - 1) // _LLM_BATCH
    tgt_name = _LANG_NAMES.get(target_lang, target_lang.upper())
    print(
        f"  Calling Claude (claude-sonnet-4-6) to refine {n} segments "
        f"→ {tgt_name} in {n_batches} batch(es) …"
    )

    refined: dict[int, dict] = {}
    for i in range(n_batches):
        start = i * _LLM_BATCH
        end   = min(start + _LLM_BATCH, n)
        batch = to_refine[start:end]
        ctx_before = to_refine[max(0, start - _CTX_WINDOW):start]
        ctx_after  = to_refine[end:end + _CTX_WINDOW]
        if n_batches > 1:
            print(f"    Batch {i + 1}/{n_batches} ({len(batch)} segments) …")
        refined.update(_refine_batch(client, batch, refine_system,
                                     ctx_before or None, ctx_after or None))

    out = []
    for seg in segments:
        r = refined.get(seg["id"])
        if r:
            seg = {**seg, "en_text": r["en_text"], "hi_text": r["hi_text"]}
        out.append(seg)

    changed = sum(
        1 for orig, new in zip(segments, out)
        if orig["en_text"] != new["en_text"] or orig["hi_text"] != new["hi_text"]
    )
    print(f"  LLM refined {changed}/{len(out)} segments.")
    return out


def extract_glossary(
    segments: list[dict],
    target_lang: str = "hi",
    cache_path: Path | None = None,
) -> dict[str, str]:
    """Return {en_term: target_transliteration} for proper nouns found in segments.

    Uses Claude Haiku (cheap) to identify character names, place names, and
    technical terms, then returns their canonical target-language forms.
    Results are cached to cache_path if provided so subsequent runs are free.
    """
    import re

    if cache_path and cache_path.exists():
        import json as _json
        glossary = _json.loads(cache_path.read_text())
        print(f"  Glossary cached ({len(glossary)} terms): {list(glossary.items())[:5]}")
        return glossary

    # Heuristic: capitalized tokens that appear 2+ times are likely proper nouns
    all_text = " ".join(s.get("en_text", "") for s in segments)
    tokens = re.findall(r"\b([A-Z][a-z]{1,20})\b", all_text)
    from collections import Counter
    candidates = [t for t, c in Counter(tokens).items() if c >= 2]

    if not candidates:
        return {}

    tgt_name = _LANG_NAMES.get(target_lang, target_lang.upper())
    client = anthropic.Anthropic()
    prompt = (
        f"From this list of words extracted from a video transcript, identify only "
        f"proper nouns (character names, place names, brand names, technical terms). "
        f"For each proper noun, provide its standard {tgt_name} transliteration.\n\n"
        f"Words: {', '.join(candidates)}\n\n"
        f'Return ONLY valid JSON: {{"Tom": "टॉम", "Celia": "सेलिया", ...}}. '
        f"Omit common English words. If no proper nouns, return {{}}."
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    import json as _json
    try:
        glossary = _json.loads(raw)
    except Exception:
        glossary = {}

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps(glossary, ensure_ascii=False, indent=2))

    print(f"  Glossary extracted ({len(glossary)} terms): {list(glossary.items())[:5]}")
    return glossary


def _make_repair_system(source_lang: str = "en", target_lang: str = "hi") -> str:
    tgt_name      = _LANG_NAMES.get(target_lang, target_lang.upper())
    emotion_rules = _EMOTION_GUIDANCE.get(target_lang, _DEFAULT_EMOTION_GUIDANCE)
    return (
        f"You are rewriting {tgt_name} dubbing segments whose emotional register "
        f"failed quality review.\n"
        "Each segment includes: en_text (source), hi_text (current translation that "
        "failed), emotion (dominant label), emotion_blend (full probability "
        "distribution), and grade_note (the reviewer's diagnosis).\n"
        "Your ONLY job: rewrite hi_text so it carries the intended emotional register "
        "through word choice. Do NOT change meaning, timing, or sentence structure "
        "unless essential for emotional impact.\n"
        + emotion_rules +
        "Return ONLY a valid JSON array: "
        '[{"id": int, "en_text": str, "hi_text": str}, …]. '
        "Same count and IDs as input. No markdown, no explanation."
    )


def repair_emotion_register(segments: list[dict], *,
                             skip: bool = False,
                             source_lang: str = "en",
                             target_lang: str = "hi",
                             threshold: int = 4) -> list[dict]:
    """Stage 4c — re-refine segments where Haiku grades emotion_register < threshold.

    Default raised from 3 to 4 so segments scoring exactly 3/5 are repaired.
    The spec target is avg_emotion_register ≥ 4.0; leaving er=3 segments unrepaired
    pulls the average below target on clips with several near-miss segments.
    """
    from metrics import grade_translations

    emotional = [s for s in segments
                 if s.get("emotion") and s["emotion"] != "neutral"]
    if not emotional:
        print("  No non-neutral segments — skipping repair.")
        return segments
    if skip:
        print("  Stage 4c skipped (--no-llm).")
        return segments

    print(f"  Pre-grading {len(emotional)} emotional segment(s) …")
    grades      = grade_translations(emotional)
    grades_by_id = {g["id"]: g for g in grades}

    to_repair = [
        {**s, "grade_note": grades_by_id.get(s["id"], {}).get("note", "register too flat")}
        for s in emotional
        if grades_by_id.get(s["id"], {}).get("emotion_register", threshold) < threshold
    ]

    if not to_repair:
        print(f"  All emotional segments pass emotion_register ≥ {threshold}.")
        return segments

    print(f"  {len(to_repair)} segment(s) below threshold — rewriting with repair prompt …")
    client       = anthropic.Anthropic()
    repair_system = _make_repair_system(source_lang, target_lang)
    repair_payload = [
        {
            "id":           s["id"],
            "en_text":      s["en_text"],
            "hi_text":      s["hi_text"],
            "emotion":      s["emotion"],
            "emotion_blend": s.get("emotion_dist", {}),
            "grade_note":   s["grade_note"],
        }
        for s in to_repair
    ]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=repair_system,
        messages=[{"role": "user",
                   "content": json.dumps(repair_payload, ensure_ascii=False)}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    repaired = {r["id"]: r for r in json.loads(raw)}

    out = []
    for seg in segments:
        r = repaired.get(seg["id"])
        if r:
            seg = {**seg, "hi_text": r["hi_text"]}
            print(f"    [{seg['start']:.1f}s] repaired ({seg['emotion']})")
        out.append(seg)
    print(f"  Repaired {len(repaired)} segment(s).")
    return out
