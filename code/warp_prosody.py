"""Source-conditioned prosody warping (proof-of-concept).

Warps a dubbed track's F0 contour to follow the original-language performance's
pitch dynamics, per aligned segment, while preserving the dubbed speaker's own
mean pitch and range (so it stays natural and language-appropriate).

Pipeline: WORLD (pyworld) analysis of the dubbed audio → for each segment,
blend the dubbed log-F0 *shape* toward the source log-F0 shape → WORLD
resynthesis. Spectral envelope and aperiodicity are untouched, so timbre and
phonemes are preserved; only intonation moves.

This is the intervention targeting the r≈0 prosody-transfer measurement — see
`score_tts_prosody_transfer` in emotion.py.
"""
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf

_FRAME_PERIOD = 5.0        # ms — WORLD default
_MIN_VOICED_FRAMES = 4     # per segment, each side, to compute a shape
_MIN_SOURCE_VOICED_FRAC = 0.10


def _load_mono(path: Path, target_fs: int | None = None) -> tuple[np.ndarray, int]:
    x, fs = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if target_fs and fs != target_fs:
        # Linear resample is adequate for F0 shape extraction from the source.
        n = int(round(len(x) * target_fs / fs))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
        fs = target_fs
    return x, fs


def _harvest_f0(x: np.ndarray, fs: int) -> np.ndarray:
    f0, t = pw.harvest(x, fs, frame_period=_FRAME_PERIOD)
    return pw.stonemask(x, f0, t, fs)


def _segment_frames(f0: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Frame indices (into an F0 array on the 5 ms grid) covering [start, end]."""
    lo = int(np.floor(start_s * 1000.0 / _FRAME_PERIOD))
    hi = int(np.ceil(end_s * 1000.0 / _FRAME_PERIOD))
    return np.arange(max(0, lo), min(len(f0), hi))


def _logf0_shape(f0_slice: np.ndarray) -> np.ndarray | None:
    """z-normalised log-F0 over voiced frames, interpolated across gaps."""
    voiced = f0_slice > 0
    if voiced.sum() < _MIN_VOICED_FRAMES:
        return None
    logf0 = np.zeros_like(f0_slice)
    xp = np.flatnonzero(voiced)
    logf0_voiced = np.log(f0_slice[xp])
    logf0 = np.interp(np.arange(len(f0_slice)), xp, logf0_voiced)
    std = logf0.std()
    if std < 1e-6:
        return None
    return (logf0 - logf0.mean()) / std


def warp_dub_to_source(
    source_audio: Path,
    dubbed_audio: Path,
    segments: list[dict],
    output_path: Path,
    blend: float = 0.6,
) -> dict:
    """Warp `dubbed_audio` F0 toward `source_audio` per segment; write to
    `output_path`.

    `blend` in [0, 1]: 0 = unchanged dub, 1 = fully adopt the source contour
    shape (dubbed mean/range retained). ~0.6 tracks the source while keeping
    natural Hindi intonation.

    Returns dict with counts of warped/skipped segments.
    """
    source_audio, dubbed_audio = Path(source_audio), Path(dubbed_audio)
    output_path = Path(output_path)
    if not source_audio.exists():
        return {"success": False, "note": f"source not found: {source_audio.name}"}
    if not dubbed_audio.exists():
        return {"success": False, "note": f"dubbed not found: {dubbed_audio.name}"}

    dub_x, fs = _load_mono(dubbed_audio)
    src_x, _ = _load_mono(source_audio, target_fs=fs)

    dub_f0 = _harvest_f0(dub_x, fs)
    _, t = pw.harvest(dub_x, fs, frame_period=_FRAME_PERIOD)
    sp = pw.cheaptrick(dub_x, dub_f0, t, fs)
    ap = pw.d4c(dub_x, dub_f0, t, fs)
    src_f0 = _harvest_f0(src_x, fs)

    new_f0 = dub_f0.copy()
    warped, skipped = 0, 0
    for seg in segments:
        start_s, end_s = float(seg.get("start", 0)), float(seg.get("end", 0))
        if end_s <= start_s:
            skipped += 1
            continue

        d_idx = _segment_frames(dub_f0, start_s, end_s)
        s_idx = _segment_frames(src_f0, start_s, end_s)
        if len(d_idx) == 0 or len(s_idx) == 0:
            skipped += 1
            continue

        src_slice = src_f0[s_idx]
        if (src_slice > 0).mean() < _MIN_SOURCE_VOICED_FRAC:
            skipped += 1          # source is silence/music — nothing to transfer
            continue

        dub_slice = dub_f0[d_idx]
        dub_shape = _logf0_shape(dub_slice)
        src_shape = _logf0_shape(src_slice)
        if dub_shape is None or src_shape is None:
            skipped += 1
            continue

        # Resample source shape onto the dubbed segment's frame grid.
        src_on_dub = np.interp(
            np.linspace(0, len(src_shape) - 1, len(dub_shape)),
            np.arange(len(src_shape)),
            src_shape,
        )
        blended_shape = blend * src_on_dub + (1.0 - blend) * dub_shape

        voiced = dub_slice > 0
        logf0_voiced = np.log(dub_slice[voiced])
        mu, sigma = logf0_voiced.mean(), logf0_voiced.std()
        target_log = mu + sigma * blended_shape[voiced]
        seg_new = dub_slice.copy()
        seg_new[voiced] = np.exp(target_log)   # unvoiced frames stay 0
        new_f0[d_idx] = seg_new
        warped += 1

    y = pw.synthesize(new_f0, sp, ap, fs, frame_period=_FRAME_PERIOD)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), y, fs)
    return {
        "success": True,
        "output_path": str(output_path),
        "warped": warped,
        "skipped": skipped,
        "blend": blend,
    }


if __name__ == "__main__":
    import json
    import re
    import sys

    if len(sys.argv) < 5:
        print("Usage: uv run python code/warp_prosody.py <source.wav> "
              "<dubbed.mp3> <srt> <out.wav> [blend]")
        sys.exit(1)

    def _to_s(ts: str) -> float:
        h, mm, rest = ts.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(mm) * 60 + int(s) + int(ms) / 1000

    srt_text = Path(sys.argv[3]).read_text()
    blocks = re.findall(r"(\d+)\s*\n(\d\d:\d\d:\d\d,\d+) --> (\d\d:\d\d:\d\d,\d+)", srt_text)
    segs = [{"id": int(n) - 1, "start": _to_s(a), "end": _to_s(b)} for n, a, b in blocks]
    blend = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
    result = warp_dub_to_source(Path(sys.argv[1]), Path(sys.argv[2]), segs,
                                Path(sys.argv[4]), blend=blend)
    print(json.dumps(result, indent=2))
