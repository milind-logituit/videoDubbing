# AI Video Dubbing — POC Demo

End-to-end pipeline: English video → Hindi subtitles + Hindi dubbed audio.

**Built for Eros Now / SunNxt** | Logituit AI Practice

---

## What it does

| Use Case | Description |
|---|---|
| 🇮🇳 Hindi Subtitles | Original video plays with Hindi subtitle overlay (WebVTT, toggle CC on/off) |
| 🔊 Hindi Dubbed Audio | Same video with English audio replaced by AI-generated Hindi speech |

Pipeline: `Whisper ASR` → `Google Translate` → `Microsoft Neural TTS (hi-IN-SwaraNeural)`

---

## Prerequisites

**1. Python 3.12**
```bash
# Check your version
python3 --version
```
If not installed: https://www.python.org/downloads/

**2. uv** (fast Python package manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**3. ffmpeg**
```bash
# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

---

## Setup & Run

```bash
# 1. Clone / download this folder, then navigate into it
cd VideoDubbing

# 2. Install Python dependencies (~1 min, one-time)
uv sync

# 3. Generate the sample video and run the full pipeline (~60s, one-time)
uv run python code/pipeline_v2.py

# 4. Launch the demo dashboard
uv run streamlit run code/dashboard_v2.py
```

The browser will open automatically at **http://localhost:8501**

---

## One-command launcher (after first setup)

```bash
./run_demo.sh
```

---

## Try it with your own video

In the dashboard sidebar, switch **Video Source → Upload your own** and drop in any MP4 or MOV file with English speech. Processing takes ~30-60 seconds.

---

## Dashboard tabs

| Tab | What you'll see |
|---|---|
| 🇮🇳 Use Case 1 — Hindi Subtitles | Video player with native CC overlay |
| 🔊 Use Case 2 — Hindi Dubbed Audio | Video with Hindi voice track |
| ⚖️ Side by Side | Original vs subtitled / original vs dubbed |
| 📝 Bilingual Transcript | Timestamped EN↔HI side-by-side |
| ⚙️ How It Works | Pipeline diagram + component table + 8-language roadmap |

---

## Quality metrics (sample clip)

| Metric | Value |
|---|---|
| ASR Word Error Rate | 6.5% |
| Translation BLEU | 26.5 |
| Source duration | 31.9s |
| Dubbed duration | 41.9s (+31%) |

---

## Production upgrade path

| Stage | Current (POC) | Production |
|---|---|---|
| ASR | Whisper base (local) | Whisper large-v3 / Azure Speech |
| Translation | Google Translate (free) | Google Cloud Translation API |
| TTS | edge-tts hi-IN-SwaraNeural | Azure Neural TTS / ElevenLabs voice clone |
| Timing | Manual concat | Duration-matched TTS with time-stretch |
| Languages | Hindi | Tamil, Telugu, Kannada, Malayalam, Bengali + more |
