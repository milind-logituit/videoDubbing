"""Output persistence and console quality summaries for pipeline_v2."""
import json
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).parent.parent
PREPARED  = ROOT / "data/prepared"
MODEL_OUT = ROOT / "model_outputs"


def save_outputs(segments, vtt, srt, metrics, src_audio, dubbed_audio,
                 stem: str = "sample", target_lang: str = "hi"):
    pd.DataFrame(segments).to_csv(
        PREPARED / f"transcript_bilingual_{stem}.csv", index=False
    )
    (MODEL_OUT / f"subtitles_{stem}_{target_lang}.vtt").write_text(vtt, encoding="utf-8")
    (MODEL_OUT / f"subtitles_{stem}_{target_lang}.srt").write_text(srt, encoding="utf-8")
    with open(MODEL_OUT / f"metrics_{stem}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved outputs to {MODEL_OUT}/")
    for out in sorted(MODEL_OUT.iterdir()):
        print(f"  {out.name}")


def _print_quality(metrics: dict) -> None:
    wer_pct    = metrics["asr"]["wer_pct"]
    bleu       = metrics["translation"]["bleu"]
    txt_bleu   = metrics.get("text_bleu", {}).get("bleu")
    bt_bleu    = metrics["back_translation"].get("bleu")
    sync_score = metrics["lipsync"].get("sync_score")
    _emo       = metrics.get("emotion", {})
    emo_match  = _emo.get("match_pct")
    emo_soft   = _emo.get("avg_soft_score")
    emo_tts    = _emo.get("tts_fidelity", {}).get("avg_soft_score")
    mos        = metrics.get("mos_rubric", {}).get("mos")
    print("\n── Quality metrics ──")
    print(f"  ASR WER              : {f'{wer_pct:.1f}%' if wer_pct is not None else 'N/A'}")
    print(f"  Translation BLEU     : {f'{bleu:.1f}' if bleu is not None else 'N/A'}")
    print(f"  Text-level BLEU      : {f'{txt_bleu:.1f}' if txt_bleu is not None else 'N/A'}")
    print(f"  Back-translation BLEU: {f'{bt_bleu:.1f}' if bt_bleu is not None else 'N/A'}")
    print(f"  Lip-sync score       : {f'{sync_score:.3f}' if sync_score is not None else 'N/A'}")
    print(f"  Emotion match (bin)  : {f'{emo_match:.1f}%' if emo_match is not None else 'N/A'}")
    print(f"  Emotion soft score   : {f'{emo_soft:.1f}%' if emo_soft is not None else 'N/A'}")
    print(f"  TTS emotion fidelity : {f'{emo_tts:.1f}%' if emo_tts is not None else 'N/A'}")
    print(f"  Duration ratio       : {metrics['alignment']['duration_ratio']:.3f}")
    print(f"  MOS rubric (0-100)   : {f'{mos:.1f}' if mos is not None else 'N/A'}")


def _metrics_summary_row(clip: str, m: dict) -> dict:
    return {
        "clip":           clip,
        "text_bleu":      m.get("text_bleu", {}).get("bleu"),
        "lip_sync":       m.get("lipsync", {}).get("sync_score"),
        "emotion_soft":   m.get("emotion", {}).get("avg_soft_score"),
        "mos":            m.get("mos_rubric", {}).get("mos"),
        "duration_ratio": m.get("alignment", {}).get("duration_ratio"),
        "n_segments":     m.get("translation", {}).get("n_segments"),
    }
