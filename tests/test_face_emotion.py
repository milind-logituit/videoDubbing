"""Tests for classify_face_emotions in code/emotion.py."""
import builtins
import importlib.util
import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock, patch

_spec = importlib.util.spec_from_file_location(
    "emotion", Path(__file__).parent.parent / "code" / "emotion.py"
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["emotion"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

classify_face_emotions = _mod.classify_face_emotions
_VALENCE_AROUSAL = _mod._VALENCE_AROUSAL

FakeLandmark = namedtuple("FakeLandmark", ["x", "y"])

_VIDEO = Path("/fake/video.mp4")


def _seg(**kwargs) -> dict:
    base = {"id": 0, "start": 0.0, "end": 1.0, "en_text": "hello", "hi_text": "नमस्ते", "emotion": "neutral"}
    base.update(kwargs)
    return base


def _make_landmarks(n: int = 478) -> list:
    lms = [FakeLandmark(0.5, 0.5)] * n
    lms = list(lms)
    lms[1] = FakeLandmark(0.5, 0.4)    # nose tip
    lms[61] = FakeLandmark(0.3, 0.35)  # mouth left — above nose → positive valence
    lms[291] = FakeLandmark(0.7, 0.35)
    lms[70] = FakeLandmark(0.3, 0.2)   # brow high (small y) → high arousal
    lms[300] = FakeLandmark(0.7, 0.2)
    return lms


_orig_import = builtins.__import__


def _real_import_no_mediapipe(name, *args, **kwargs):
    if "mediapipe" in name:
        raise ImportError(f"No module named '{name}'")
    return _orig_import(name, *args, **kwargs)


def _make_cv2_mock(fps: float = 2.0, frame_timestamps: list[float] | None = None):
    if frame_timestamps is None:
        frame_timestamps = [0.0]

    cv2 = MagicMock()
    cv2.CAP_PROP_FPS = 5
    cv2.COLOR_BGR2RGB = 4

    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = fps

    frames = [MagicMock() for _ in frame_timestamps]
    read_calls = iter([(True, f) for f in frames] + [(False, None)])
    cap.read.side_effect = lambda: next(read_calls)

    cv2.VideoCapture.return_value = cap
    cv2.cvtColor.side_effect = lambda frame, _: frame
    return cv2


def _make_face_mesh_mock(landmarks=None):
    result = MagicMock()
    if landmarks is None:
        result.multi_face_landmarks = None
    else:
        face = MagicMock()
        face.landmark = landmarks
        result.multi_face_landmarks = [face]

    fm_instance = MagicMock()
    fm_instance.process.return_value = result
    fm_instance.__enter__.return_value = fm_instance

    mp_face = MagicMock()
    mp_face.FaceMesh.return_value = fm_instance
    return mp_face


def _patch_deps(cv2_mock, mp_face_mock):
    mp_solutions = MagicMock()
    mp_solutions.face_mesh = mp_face_mock
    return patch.dict(sys.modules, {
        "cv2": cv2_mock,
        "mediapipe": MagicMock(),
        "mediapipe.python": MagicMock(),
        "mediapipe.python.solutions": mp_solutions,
        "mediapipe.python.solutions.face_mesh": mp_face_mock,
    })


# ── tests ──────────────────────────────────────────────────────────────────────

def test_graceful_fallback_when_mediapipe_not_installed():
    segs = [_seg()]
    with patch("builtins.__import__", side_effect=_real_import_no_mediapipe):
        result = classify_face_emotions(_VIDEO, segs)
    assert result == segs


def test_graceful_fallback_when_video_cannot_open():
    cv2 = MagicMock()
    cap = MagicMock()
    cap.isOpened.return_value = False
    cv2.VideoCapture.return_value = cap
    mp_face = _make_face_mesh_mock()
    segs = [_seg()]
    with _patch_deps(cv2, mp_face):
        result = classify_face_emotions(_VIDEO, segs)
    assert result == segs


def test_no_faces_detected_leaves_segments_unchanged():
    cv2 = _make_cv2_mock(fps=2.0, frame_timestamps=[0.0])
    mp_face = _make_face_mesh_mock(landmarks=None)
    segs = [_seg()]
    with _patch_deps(cv2, mp_face):
        result = classify_face_emotions(_VIDEO, segs)
    assert "face_emotion" not in result[0]
    assert result[0]["emotion"] == "neutral"


def test_face_emotion_fuses_with_text_ser():
    # fps=2, sample_every=1, frame_idx=0 → ts=0/2=0.0, within segment [0.0, 1.0)
    cv2 = _make_cv2_mock(fps=2.0, frame_timestamps=[0.0])
    lms = _make_landmarks()
    mp_face = _make_face_mesh_mock(landmarks=lms)
    segs = [_seg(emotion="neutral")]
    with _patch_deps(cv2, mp_face):
        result = classify_face_emotions(_VIDEO, segs)
    assert "face_emotion" in result[0]
    assert "face_valence" in result[0]
    assert "face_arousal" in result[0]
    assert result[0]["emotion"] in _VALENCE_AROUSAL


def test_frames_outside_segment_window_are_ignored():
    # fps=1, sample_every=1; skip 5 frames (False), then 1 real frame → ts=5/1=5.0
    # segment [0.0, 1.0) won't capture ts=5.0
    cv2 = MagicMock()
    cv2.CAP_PROP_FPS = 5
    cv2.COLOR_BGR2RGB = 4
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = 1.0  # fps=1, sample_every=1
    fake_frame = MagicMock()
    # 5 empty reads so frame_idx reaches 5, then one real frame at idx 5 → ts=5.0
    calls = [(False, None)] * 5 + [(True, fake_frame), (False, None)]
    call_iter = iter(calls)
    cap.read.side_effect = lambda: next(call_iter)
    cv2.VideoCapture.return_value = cap
    cv2.cvtColor.side_effect = lambda f, _: f
    mp_face = _make_face_mesh_mock(landmarks=_make_landmarks())
    segs = [_seg(start=0.0, end=1.0)]
    with _patch_deps(cv2, mp_face):
        result = classify_face_emotions(_VIDEO, segs)
    assert "face_emotion" not in result[0]


def test_original_fields_preserved_after_fusion():
    cv2 = _make_cv2_mock(fps=2.0, frame_timestamps=[0.0])
    lms = _make_landmarks()
    mp_face = _make_face_mesh_mock(landmarks=lms)
    segs = [_seg(id=7, start=0.0, end=1.0, en_text="hello", hi_text="नमस्ते", emotion="happy")]
    with _patch_deps(cv2, mp_face):
        result = classify_face_emotions(_VIDEO, segs)
    out = result[0]
    assert out["id"] == 7
    assert out["start"] == 0.0
    assert out["end"] == 1.0
    assert out["en_text"] == "hello"
    assert out["hi_text"] == "नमस्ते"
