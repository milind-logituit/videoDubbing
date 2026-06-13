"""
Wav2Lip wrapper — re-generates mouth movements in a dubbed video.

Takes the original face video + dubbed audio, produces a new MP4 where the
mouth moves in sync with the Hindi audio.

Calls vendor/Wav2Lip/inference.py as a subprocess so its sys.path manipulation
stays isolated from the main pipeline.
"""
import subprocess
import sys
from pathlib import Path

_WAV2LIP_DIR = Path(__file__).parent.parent / "vendor" / "Wav2Lip"
_CHECKPOINT  = _WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"


def apply_wav2lip(
    face_video: Path,
    dubbed_audio: Path,
    output_path: Path,
    resize_factor: int = 2,
    pads: tuple[int, int, int, int] = (0, 20, 0, 0),
) -> dict:
    """
    Run Wav2Lip inference on face_video + dubbed_audio → output_path.

    Returns dict:
      success     : bool
      output_path : str or None
      note        : str or None
    """
    face_video   = Path(face_video).resolve()
    dubbed_audio = Path(dubbed_audio).resolve()
    output_path  = Path(output_path).resolve()

    if not face_video.exists():
        return {"success": False, "output_path": None, "note": f"Face video not found: {face_video}"}
    if not dubbed_audio.exists():
        return {"success": False, "output_path": None, "note": f"Audio not found: {dubbed_audio}"}
    if not _CHECKPOINT.exists():
        return {"success": False, "output_path": None,
                "note": f"Wav2Lip checkpoint not found: {_CHECKPOINT}"}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_WAV2LIP_DIR / "inference.py"),
        "--checkpoint_path", str(_CHECKPOINT),
        "--face",            str(face_video),
        "--audio",           str(dubbed_audio),
        "--outfile",         str(output_path),
        "--resize_factor",   str(resize_factor),
        "--pads",            str(pads[0]), str(pads[1]), str(pads[2]), str(pads[3]),
        "--wav2lip_batch_size", "32",
        "--nosmooth",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_WAV2LIP_DIR),
            capture_output=True,
            text=True,
            timeout=3600,  # 60 min; MPS is slow without CUDA
        )
        if result.returncode != 0:
            return {
                "success": False,
                "output_path": None,
                "note": f"Wav2Lip failed (exit {result.returncode}): {result.stderr[-500:]}",
            }
        if not output_path.exists():
            return {"success": False, "output_path": None,
                    "note": "Wav2Lip exited 0 but output file not created"}
        return {"success": True, "output_path": str(output_path), "note": None}
    except subprocess.TimeoutExpired:
        return {"success": False, "output_path": None, "note": "Wav2Lip timed out (>10 min)"}
    except Exception as exc:
        return {"success": False, "output_path": None, "note": f"Subprocess error: {exc}"}


if __name__ == "__main__":
    import json
    if len(sys.argv) < 4:
        print("Usage: uv run python code/apply_lipsync.py <face.mp4> <audio.mp3> <output.mp4> [resize_factor]")
        sys.exit(1)
    rf = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    result = apply_wav2lip(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), resize_factor=rf)
    print(json.dumps(result, indent=2))
