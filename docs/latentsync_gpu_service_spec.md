# LatentSync GPU Service — Setup Spec

## What it does

Takes a dubbed video (original faces + visual) and dubbed audio, and re-generates
mouth movements so the lips visually match the new audio. LatentSync significantly
outperforms Wav2Lip on temporal consistency and realism.

---

## Setup on the GPU machine

1. Clone the repo:
   ```bash
   git clone https://github.com/bytedance/LatentSync
   cd LatentSync
   pip install -r requirements.txt
   ```

2. Download the checkpoint (~1.5 GB) — follow the HuggingFace download instructions
   in the repo README.

3. Verify it runs on one test clip before starting the server:
   ```bash
   python inference.py \
     --video_path test.mp4 \
     --audio_path test.wav \
     --output_path out.mp4
   ```

---

## REST endpoint

A minimal Flask server with one endpoint:

```
POST /lipsync
Content-Type: multipart/form-data

Fields:
  video  — MP4 file (dubbed video, original faces)
  audio  — WAV or MP3 file (dubbed audio)

Returns:
  200 OK, Content-Type: video/mp4, body = lip-synced MP4
  500 { "error": "<message>" } on failure
```

### Server code

```python
from flask import Flask, request, send_file
import subprocess, tempfile
from pathlib import Path

LATENTSYNC_DIR = "/path/to/LatentSync"  # update this

app = Flask(__name__)

@app.route("/lipsync", methods=["POST"])
def lipsync():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "input.mp4"
        audio_path = Path(tmp) / "audio.wav"
        out_path   = Path(tmp) / "output.mp4"

        request.files["video"].save(video_path)
        request.files["audio"].save(audio_path)

        result = subprocess.run([
            "python", "inference.py",
            "--video_path", str(video_path),
            "--audio_path", str(audio_path),
            "--output_path", str(out_path),
        ], cwd=LATENTSYNC_DIR, capture_output=True)

        if result.returncode != 0:
            return {"error": result.stderr.decode()[-500:]}, 500

        return send_file(out_path, mimetype="video/mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765)
```

Start with:
```bash
pip install flask
python server.py
```

---

## What to send back

Once the server is running, share:
- Server IP + port (e.g. `http://10.0.1.X:8765`)
- Approximate turnaround time per 2-minute clip
