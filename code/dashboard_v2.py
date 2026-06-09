"""
VideoDubbing v2 dashboard — video-aware demo for Eros / SunNxt
Use Case 1: original video + Hindi subtitle overlay (WebVTT)
Use Case 2: original video with Hindi audio track (dubbed MP4)

Run: uv run streamlit run code/dashboard_v2.py
"""
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent

st.set_page_config(
    page_title="AI Video Dubbing v2",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_VIDEO   = ROOT / "data/raw/sample_en.mp4"
DUBBED_VIDEO   = ROOT / "data/raw/sample_en_dubbed_hi.mp4"
VTT_PATH       = ROOT / "model_outputs/subtitles_hi.vtt"
SRT_PATH       = ROOT / "model_outputs/subtitles_hi.srt"
TRANSCRIPT_CSV = ROOT / "data/prepared/transcript_bilingual_v2.csv"
METRICS_JSON   = ROOT / "model_outputs/metrics_v2.json"


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_sample_outputs():
    transcript = pd.read_csv(TRANSCRIPT_CSV)
    with open(METRICS_JSON) as f:
        metrics = json.load(f)
    vtt = VTT_PATH.read_text(encoding="utf-8")
    srt = SRT_PATH.read_text(encoding="utf-8")
    return transcript, metrics, vtt, srt


@st.cache_data(show_spinner=False)
def process_uploaded_video(video_bytes: bytes, filename: str):
    """Run the full v2 pipeline on an uploaded video, return (vtt, srt, dubbed_bytes, transcript_df, metrics)."""
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from pipeline_v2 import (
        extract_audio, transcribe, translate_segments,
        generate_vtt, generate_srt, synthesize_hindi_audio,
        create_dubbed_video, compute_metrics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video_path = tmp / filename
        video_path.write_bytes(video_bytes)

        audio_path   = extract_audio(video_path)
        w_result     = transcribe(audio_path)
        segments     = translate_segments(w_result)
        vtt          = generate_vtt(segments)
        srt          = generate_srt(segments)

        hindi_audio  = tmp / "hindi.mp3"
        # synthesize directly into temp dir
        from pydub import AudioSegment
        import asyncio, edge_tts, gtts
        combined = AudioSegment.empty()
        for seg in segments:
            seg_path = tmp / f"seg_{seg['id']}.mp3"
            try:
                async def _s(t=seg["hi_text"], p=seg_path):
                    communicate = edge_tts.Communicate(t, "hi-IN-SwaraNeural")
                    await communicate.save(str(p))
                asyncio.run(_s())
            except Exception:
                gtts.gTTS(seg["hi_text"], lang="hi").save(str(seg_path))
            combined += AudioSegment.from_mp3(str(seg_path))
        combined.export(str(hindi_audio), format="mp3")

        dubbed_path  = create_dubbed_video(video_path, hindi_audio)
        dubbed_bytes = dubbed_path.read_bytes()

        src_audio    = ROOT / "data/raw/source_en.mp3"
        metrics      = compute_metrics(w_result, segments, src_audio, hindi_audio)
        transcript   = pd.DataFrame(segments)

    return vtt, srt, dubbed_bytes, transcript, metrics


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Video Source")
source_mode = st.sidebar.radio("", ["Sample clip", "Upload your own"], index=0)

uploaded_file = None
if source_mode == "Upload your own":
    uploaded_file = st.sidebar.file_uploader(
        "Drop an MP4 or MOV", type=["mp4", "mov", "avi"],
        help="Max ~100MB. Processing takes ~30-60s.",
    )

st.sidebar.divider()
st.sidebar.markdown("**Pipeline**")
st.sidebar.code("Whisper (base)\n→ Google Translate\n→ edge-tts (hi-IN-Swara)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 AI Video Dubbing — v2")
st.caption(
    "English video → Hindi subtitles & Hindi dubbed audio  |  "
    "Whisper ASR + Google Translate + Microsoft Neural TTS  |  "
    "Built for Eros Now / SunNxt"
)

# ── Load / process ────────────────────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner(f"Processing **{uploaded_file.name}** — transcribing, translating, dubbing…"):
        vtt, srt, dubbed_bytes, transcript, metrics = process_uploaded_video(
            uploaded_file.read(), uploaded_file.name
        )
    video_src      = uploaded_file  # bytes-based, pass directly
    dubbed_src     = dubbed_bytes
    using_sample   = False
else:
    transcript, metrics, vtt, srt = load_sample_outputs()
    # Read as bytes — avoids path-with-spaces issues in Streamlit's file server
    video_src    = SAMPLE_VIDEO.read_bytes()
    dubbed_src   = DUBBED_VIDEO.read_bytes()
    using_sample = True

# ── KPI row ───────────────────────────────────────────────────────────────────
al = metrics["alignment"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("ASR Accuracy",    f"{100 - metrics['asr']['wer_pct']:.1f}%")
k2.metric("BLEU Score",      f"{metrics['translation']['bleu']:.1f}")
k3.metric("Segments",        str(metrics["translation"]["n_segments"]))
k4.metric("Source",          f"{al['source_duration_s']:.1f}s")
k5.metric("Dubbed (HI)",     f"{al['dubbed_duration_s']:.1f}s",
          f"{(al['duration_ratio'] - 1)*100:+.1f}% vs original", delta_color="off")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🇮🇳  Use Case 1 — Hindi Subtitles",
    "🔊  Use Case 2 — Hindi Dubbed Audio",
    "⚖️  Side by Side",
    "📝  Bilingual Transcript",
    "⚙️  How It Works",
])


# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Original video + Hindi subtitle overlay (WebVTT)")
    st.caption(
        "Subtitles are rendered natively in the browser — no re-encoding of the video. "
        "Toggle captions on/off using the CC button in the video player."
    )

    if video_src is not None:
        st.video(video_src, subtitles={"हिंदी": vtt})
    else:
        st.warning("Sample video not found — run `pipeline_v2.py` first.")

    with st.expander("⬇️  Download subtitle files"):
        c1, c2 = st.columns(2)
        c1.download_button("Download .vtt (WebVTT)", vtt.encode(),
                           "subtitles_hi.vtt", "text/vtt")
        c2.download_button("Download .srt", srt.encode(),
                           "subtitles_hi.srt", "text/plain")

    st.markdown("#### Generated WebVTT")
    st.code(vtt, language="text")


# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Original video with Hindi audio track (dubbed)")
    st.caption(
        "The English audio has been replaced with AI-generated Hindi speech "
        "using Microsoft's hi-IN-SwaraNeural voice."
    )

    if dubbed_src is not None:
        st.video(dubbed_src)
    else:
        st.warning("Dubbed video not found — run `pipeline_v2.py` first.")

    with st.expander("⬇️  Download dubbed video"):
        dl_bytes = dubbed_src if isinstance(dubbed_src, bytes) else DUBBED_VIDEO.read_bytes()
        st.download_button("Download dubbed MP4", dl_bytes,
                           "video_dubbed_hindi.mp4", "video/mp4")


# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Side-by-Side Comparison")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🇬🇧 Original — English audio")
        if video_src is not None:
            st.video(video_src)
        st.caption(f"Duration: {al['source_duration_s']:.1f}s  |  "
                   f"Pace: {al['en_chars_per_sec']:.1f} ch/s")

    with c2:
        st.markdown("#### 🇮🇳 Use Case 1 — Hindi subtitles")
        if video_src is not None:
            st.video(video_src, subtitles={"हिंदी": vtt})
        st.caption("Same video · Hindi CC overlay · no re-encode")

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 🇬🇧 Original — English audio")
        if video_src is not None:
            st.video(video_src)

    with c4:
        st.markdown("#### 🇮🇳 Use Case 2 — Hindi dubbed audio")
        if dubbed_src is not None:
            st.video(dubbed_src)
        st.caption(f"Duration: {al['dubbed_duration_s']:.1f}s  |  "
                   f"Pace: {al['hi_chars_per_sec']:.1f} ch/s  |  "
                   f"Voice: hi-IN-SwaraNeural")


# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Bilingual Transcript")
    st.caption("Whisper ASR → Google Translate · timestamps from Whisper segmentation")

    for _, row in transcript.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"🇬🇧 **[{row['start']:.1f}s – {row['end']:.1f}s]**")
                st.markdown(f"> {row['en_text']}")
            with c2:
                st.markdown("🇮🇳 **हिंदी**")
                st.markdown(f"> {row['hi_text']}")

    st.divider()
    st.download_button(
        "⬇️  Download bilingual transcript (CSV)",
        transcript.to_csv(index=False).encode("utf-8"),
        "transcript_bilingual.csv", "text/csv",
    )

    # Quality metrics section
    st.subheader("Quality Metrics")
    m1, m2, m3 = st.columns(3)

    wer = metrics["asr"]["wer_pct"]
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number", value=100 - wer,
        title={"text": "ASR Accuracy (1 − WER)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#00CC96"},
               "steps": [{"range": [0, 70], "color": "#EF553B"},
                         {"range": [70, 85], "color": "#FFA15A"},
                         {"range": [85, 100], "color": "#00CC96"}]},
        number={"suffix": "%"},
    ))
    fig1.update_layout(height=240, margin=dict(t=40, b=10))

    bleu = metrics["translation"]["bleu"]
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number", value=bleu,
        title={"text": "Translation BLEU"},
        gauge={"axis": {"range": [0, 50]}, "bar": {"color": "#636EFA"},
               "steps": [{"range": [0, 10],  "color": "#EF553B"},
                         {"range": [10, 20], "color": "#FFA15A"},
                         {"range": [20, 50], "color": "#00CC96"}]},
    ))
    fig2.update_layout(height=240, margin=dict(t=40, b=10))

    fig3 = go.Figure(go.Indicator(
        mode="gauge+number", value=al["duration_ratio"],
        title={"text": "Duration Ratio (HI / EN)"},
        gauge={"axis": {"range": [0.5, 2.0]}, "bar": {"color": "#FFA15A"},
               "steps": [{"range": [0.5, 0.9],  "color": "#EF553B"},
                         {"range": [0.9, 1.1],  "color": "#00CC96"},
                         {"range": [1.1, 2.0],  "color": "#FFA15A"}],
               "threshold": {"line": {"color": "black", "width": 3},
                             "thickness": 0.75, "value": 1.0}},
        number={"valueformat": ".3f"},
    ))
    fig3.update_layout(height=240, margin=dict(t=40, b=10))

    m1.plotly_chart(fig1, use_container_width=True)
    m2.plotly_chart(fig2, use_container_width=True)
    m3.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("How the Pipeline Works")
    st.markdown("""
    ```
    Input video (MP4/MOV)
          │
          ├── ffmpeg ──────────────────────► WAV audio (16 kHz mono)
          │                                        │
          │                               OpenAI Whisper (base)
          │                                        │
          │                            EN transcript + timestamps
          │                                        │
          │                            Google Translate (en→hi)
          │                                        │
          │                  ┌─────────────────────┴──────────────────┐
          │                  ▼                                         ▼
          │            WebVTT / SRT                          Hindi TTS audio
          │            subtitle file                    (hi-IN-SwaraNeural)
          │                  │                                         │
          │                  ▼                                         ▼
          └──── st.video(subtitles=vtt) ──────── ffmpeg -map 0:v -map 1:a ──►
                Use Case 1: subtitled video              Use Case 2: dubbed video
    ```
    """)

    st.markdown("#### Component breakdown")
    components = pd.DataFrame({
        "Stage": ["ASR", "Translation", "TTS", "Subtitle", "Dubbing"],
        "Model / Tool": ["OpenAI Whisper base", "Google Translate (free)", "edge-tts hi-IN-SwaraNeural",
                         "WebVTT → st.video()", "ffmpeg audio track replace"],
        "Cost": ["Free (local)", "Free", "Free (Microsoft Edge)", "Free", "Free (local)"],
        "Production upgrade": ["Whisper large-v3 / Azure Speech", "Google Cloud Translation API",
                               "Azure Neural TTS / ElevenLabs voice clone",
                               "Burn-in via ffmpeg subtitles filter",
                               "Time-stretch TTS to match original duration"],
        "Status": ["✅ Live", "✅ Live", "✅ Live", "✅ Live", "✅ Live"],
    })
    st.dataframe(components, use_container_width=True, hide_index=True)

    st.markdown("#### Supported language pairs (roadmap)")
    lang_df = pd.DataFrame({
        "Source": ["English"] * 5 + ["Hindi"] * 3,
        "Target": ["Hindi", "Tamil", "Telugu", "Kannada", "Malayalam",
                   "Tamil", "Telugu", "Kannada"],
        "TTS voice (edge-tts)": [
            "hi-IN-SwaraNeural", "ta-IN-PallaviNeural", "te-IN-ShrutiNeural",
            "kn-IN-SapnaNeural", "ml-IN-SobhanaNeural",
            "ta-IN-PallaviNeural", "te-IN-ShrutiNeural", "kn-IN-SapnaNeural",
        ],
        "Status": ["✅ Live in v2"] + ["🔧 Add in v3"] * 7,
    })
    st.dataframe(lang_df, use_container_width=True, hide_index=True)
