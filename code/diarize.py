"""Stage 3.5 — Speaker diarization and gender detection."""
import json
from pathlib import Path

_GENDER_MODEL_ID = "audeering/wav2vec2-large-robust-24-ft-age-gender"
# {0: female, 1: male, 2: child} — child maps to female voice
_GENDER_LABEL_MAP: dict[int, str] = {0: "female", 1: "male", 2: "female"}
_gender_model_cache: tuple | None = None
_FEMALE_CONFIDENCE_THRESH = 0.70


def diarize_speakers(audio_path: Path, hf_token: str) -> list[dict]:
    """Run pyannote speaker-diarization-3.1. Result cached as JSON beside audio."""
    cache = audio_path.with_suffix(".diarization.json")
    if cache.exists():
        print(f"  Loaded cached diarization: {cache.name}")
        return json.loads(cache.read_text())

    from pyannote.audio import Pipeline as _DPipeline
    print("  Loading pyannote/speaker-diarization-3.1 …")
    dia_pipeline = _DPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )
    raw = dia_pipeline(str(audio_path))
    annotation = (
        raw.speaker_diarization if hasattr(raw, "speaker_diarization") else raw
    )
    turns = [
        {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": label}
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    cache.write_text(json.dumps(turns, indent=2))
    print(f"  Diarization: {len(turns)} turns, "
          f"{len({t['speaker'] for t in turns})} speaker(s)")
    return turns


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Tag each segment with the speaker that overlaps it most.

    Falls back to the nearest turn (by midpoint distance) when no turn
    overlaps a segment — this handles silence gaps at turn boundaries.
    Runs in O(n log m) via sorted turns and early exit.
    """
    sorted_turns = sorted(turns, key=lambda t: t["start"])
    out = []
    for seg in segments:
        best: str | None = None
        best_overlap = 0.0
        for turn in sorted_turns:
            if turn["start"] > seg["end"]:
                break
            overlap = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if overlap > best_overlap:
                best_overlap, best = overlap, turn["speaker"]

        if best is None:
            seg_mid = (seg["start"] + seg["end"]) / 2
            best_dist = float("inf")
            for turn in sorted_turns:
                dist = abs((turn["start"] + turn["end"]) / 2 - seg_mid)
                if dist < best_dist:
                    best_dist, best = dist, turn["speaker"]

        if best is None:
            best = "SPEAKER_00"
            print(f"  [warn] no speaker found for segment "
                  f"{seg['start']:.1f}–{seg['end']:.1f}s; defaulting to {best}")
        out.append({**seg, "speaker": best})
    return out


def _load_gender_model() -> tuple:
    """Load audeering age-gender model, caching it for the process lifetime."""
    global _gender_model_cache
    if _gender_model_cache is not None:
        return _gender_model_cache

    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor, AutoConfig
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
    from huggingface_hub import hf_hub_download

    class _Head(nn.Module):
        def __init__(self, config: object, n: int) -> None:
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # type: ignore[arg-type]
            self.dropout = nn.Dropout(config.final_dropout)  # type: ignore[arg-type]
            self.out_proj = nn.Linear(config.hidden_size, n)  # type: ignore[arg-type]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out_proj(torch.tanh(self.dense(self.dropout(x))))

    class _AgeGenderModel(nn.Module):
        def __init__(self, config: object) -> None:
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model(config)  # type: ignore[arg-type]
            self.age = _Head(config, 1)
            self.gender = _Head(config, 3)

        def forward(self, input_values: torch.Tensor) -> torch.Tensor:
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return torch.softmax(self.gender(hidden), dim=1)

    print(f"  Loading gender classifier ({_GENDER_MODEL_ID}) …")
    config = AutoConfig.from_pretrained(_GENDER_MODEL_ID)  # nosec B615
    model = _AgeGenderModel(config)
    ckpt = hf_hub_download(_GENDER_MODEL_ID, "pytorch_model.bin")  # nosec B615
    missing, unexpected = model.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True), strict=False
    )
    if missing or unexpected:
        print(f"  [warn] gender model: missing={missing}, unexpected={unexpected}")
    model.eval()
    processor = Wav2Vec2Processor.from_pretrained(_GENDER_MODEL_ID)  # nosec B615
    _gender_model_cache = (model, processor)
    return _gender_model_cache


def _collect_speaker_chunks(
    audio: list, sr: int, segments: list[dict]
) -> dict[str, list]:
    speaker_chunks: dict[str, list] = {}
    for seg in segments:
        sp = seg.get("speaker", "SPEAKER_00")
        s_idx = int(seg["start"] * sr)
        e_idx = int(seg["end"] * sr)
        if e_idx > s_idx:
            speaker_chunks.setdefault(sp, []).append(audio[s_idx:e_idx])
    return speaker_chunks


def _assign_genders(
    speaker_probs: dict[str, tuple[float, float]],
    female_thresh: float = _FEMALE_CONFIDENCE_THRESH,
    relative_margin: float = 0.20,
) -> dict[str, str]:
    """
    Assign gender labels given per-speaker (f_prob, m_prob) tuples.

    Rules (in order):
    1. f_prob >= female_thresh          → female  (high confidence)
    2. f_prob < 0.50                    → male    (majority male)
    3. 0.50 ≤ f_prob < female_thresh    → borderline: female only if this speaker
       has the highest f_prob among all speakers by >= relative_margin, else male.
    """
    sorted_by_f = sorted(speaker_probs.items(), key=lambda x: -x[1][0])
    f_probs_desc = [v[0] for _, v in sorted_by_f]

    genders: dict[str, str] = {}
    for rank, (sp, (f_prob, _)) in enumerate(sorted_by_f):
        if f_prob >= female_thresh:
            genders[sp] = "female"
        elif f_prob < 0.50:
            genders[sp] = "male"
        else:
            next_f = f_probs_desc[rank + 1] if rank + 1 < len(f_probs_desc) else 0.0
            if rank == 0 and (f_prob - next_f) >= relative_margin:
                genders[sp] = "female"
            else:
                genders[sp] = "male"
    return genders


def detect_speaker_genders(
    audio_path: Path,
    segments: list[dict],
    female_thresh: float = _FEMALE_CONFIDENCE_THRESH,
) -> dict[str, str]:
    """Classify gender per speaker using a wav2vec2 age-gender model."""
    import numpy as np
    import torch
    import librosa

    model, processor = _load_gender_model()
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    speaker_chunks = _collect_speaker_chunks(audio, sr, segments)

    speaker_probs: dict[str, tuple[float, float]] = {}
    for sp, chunks in speaker_chunks.items():
        chunk = np.concatenate(chunks).astype(np.float32)
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt",
                           padding=True)
        with torch.no_grad():
            probs = model(inputs["input_values"])
        speaker_probs[sp] = (float(probs[0, 0]), float(probs[0, 1]))

    genders = _assign_genders(speaker_probs, female_thresh=female_thresh)
    for sp, gender in genders.items():
        f_prob, m_prob = speaker_probs[sp]
        print(f"    {sp}: {gender} (f={f_prob:.2f} m={m_prob:.2f})")
    return genders
