"""
Audio-visual lip-sync correlation scorer.

Uses OpenCV's bundled Haar cascade face detector (no model downloads) to locate
the mouth region per frame, then correlates mouth-region activity with audio RMS.

Returns sync_score in [0, 1] — higher means better audio/visual sync.

Algorithm:
  1. Haar cascade face detection → mouth region (lower 40% of face bbox)
  2. Mouth-region frame-diff variance per frame (proxy for lip movement)
  3. Audio RMS energy per matching window
  4. Normalise both signals, Pearson r at lags -1/0/+1 frames
  5. sync_score = max(r) mapped from [-1,1] to [0,1]
"""
from pathlib import Path

import cv2
import numpy as np
from pydub import AudioSegment


_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _extract_mouth_activity(video_path: Path) -> tuple[np.ndarray, float, float]:
    """Return (activity_per_frame, fps, faces_pct)."""
    face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    activity = []
    face_count = 0
    prev_mouth = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=4, minSize=(40, 40))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            # Lower 40% of face bbox = mouth region
            my = y + int(h * 0.60)
            mh = int(h * 0.40)
            mouth = gray[my: my + mh, x: x + w].astype(np.float32)
            if prev_mouth is not None and mouth.shape == prev_mouth.shape:
                diff = np.abs(mouth - prev_mouth)
                activity.append(float(diff.mean()))
            else:
                activity.append(0.0)
            prev_mouth = mouth
            face_count += 1
        else:
            activity.append(0.0)
            prev_mouth = None

    cap.release()
    arr = np.array(activity, dtype=np.float32)
    total = max(len(arr), 1)
    faces_pct = round(face_count / total * 100, 1)

    rng = arr.max() - arr.min()
    if rng > 0:
        arr = (arr - arr.min()) / rng
    return arr, fps, faces_pct


def _extract_audio_rms(video_path: Path, fps: float, n_frames: int) -> np.ndarray:
    """Return RMS energy per frame window, normalised to [0,1]."""
    audio = AudioSegment.from_file(str(video_path))
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)

    frame_len = max(1, int(audio.frame_rate / fps))
    rms = []
    for i in range(n_frames):
        start = i * frame_len
        end = start + frame_len
        chunk = samples[start:end] if start < len(samples) else np.zeros(frame_len)
        rms.append(float(np.sqrt(np.mean(chunk ** 2) + 1e-8)))

    arr = np.array(rms, dtype=np.float32)
    rng = arr.max() - arr.min()
    if rng > 0:
        arr = (arr - arr.min()) / rng
    return arr


def compute_lipsync_score(video_path: Path) -> dict:
    """
    Compute audio-visual lip-sync score for a dubbed video.

    Returns dict:
      sync_score   : float [0,1] — higher is better
      pearson_r    : raw Pearson r at best lag
      best_lag_ms  : offset at peak correlation (ms)
      faces_pct    : % frames with face detected
      note         : str or None
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {"sync_score": None, "note": f"File not found: {video_path}"}

    try:
        activity, fps, faces_pct = _extract_mouth_activity(video_path)
    except Exception as exc:
        return {"sync_score": None, "note": f"Face extraction failed: {exc}"}

    n_frames = len(activity)
    if n_frames == 0:
        return {"sync_score": None, "note": "No frames extracted"}

    if faces_pct < 5:
        return {
            "sync_score": None,
            "faces_pct": faces_pct,
            "note": f"Face detected in only {faces_pct}% of frames — not enough for scoring",
        }

    try:
        rms = _extract_audio_rms(video_path, fps, n_frames)
    except Exception as exc:
        return {"sync_score": None, "note": f"Audio extraction failed: {exc}"}

    best_r, best_lag = -1.0, 0
    for lag in (-1, 0, 1):
        if lag == 0:
            a, b = activity, rms
        elif lag > 0:
            a, b = activity[lag:], rms[:-lag]
        else:
            a, b = activity[:lag], rms[-lag:]
        if len(a) < 10:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best_r:
            best_r, best_lag = r, lag

    sync_score = round(max(0.0, (best_r + 1) / 2), 3)
    lag_ms = round(best_lag * 1000 / fps, 1)

    return {
        "sync_score": sync_score,
        "pearson_r": round(best_r, 4),
        "best_lag_ms": lag_ms,
        "faces_pct": faces_pct,
        "note": None,
    }


if __name__ == "__main__":
    import sys
    import json
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not path:
        print("Usage: uv run python code/eval_lipsync.py <video.mp4>")
        sys.exit(1)
    print(json.dumps(compute_lipsync_score(path), indent=2))
