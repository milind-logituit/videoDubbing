"""
VideoDubbing v2 dashboard — video-aware demo for Eros / SunNxt
Use Case 1: original video + Hindi subtitle overlay (WebVTT)
Use Case 2: original video with Hindi audio track (dubbed MP4)

Run: uv run streamlit run code/dashboard_v2.py
"""
import json
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

RAW       = ROOT / "data/raw"
MODEL_OUT = ROOT / "model_outputs"


def _find_processed_clips() -> dict[str, tuple[Path, Path, Path, Path, Path, Path]]:
    """Return {stem: (orig, dubbed, metrics, vtt, transcript)} for ready clips."""
    clips: dict[str, tuple[Path, Path, Path, Path, Path]] = {}
    for d in [RAW, ROOT / "test_clips"]:
        if not d.exists():
            continue
        for mp4 in sorted(d.glob("*.mp4")):
            if mp4.stem.endswith("_dubbed_hi") or mp4.stem == "sample_en":
                continue
            stem   = mp4.stem
            dubbed = RAW / f"{stem}_dubbed_hi.mp4"
            metrics_path = MODEL_OUT / f"metrics_{stem}.json"
            vtt_path     = MODEL_OUT / f"subtitles_{stem}_hi.vtt"
            srt_path     = MODEL_OUT / f"subtitles_{stem}_hi.srt"
            transcript   = (
                ROOT / "data/prepared" / f"transcript_bilingual_{stem}.csv"
            )
            if dubbed.exists() and metrics_path.exists():
                clips[stem] = (mp4, dubbed, metrics_path, vtt_path, srt_path,
                               transcript)
    return clips


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_clip_outputs(stem: str, metrics_path: str, vtt_path: str,
                      srt_path: str, transcript_path: str):
    with open(metrics_path) as f:
        metrics = json.load(f)
    vtt = Path(vtt_path).read_text(encoding="utf-8")
    srt = Path(srt_path).read_text(encoding="utf-8")
    transcript = pd.read_csv(transcript_path)
    return transcript, metrics, vtt, srt


@st.cache_data(show_spinner=False)
def process_uploaded_video(video_bytes: bytes, filename: str):
    """Run the full v2 pipeline on an uploaded video."""
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from pipeline_v2 import (
        extract_audio, transcribe, translate_segments,
        generate_vtt, generate_srt,
        create_dubbed_video, compute_metrics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video_path = tmp / filename
        video_path.write_bytes(video_bytes)

        audio_path = extract_audio(video_path)
        w_result   = transcribe(audio_path)
        segments   = translate_segments(w_result)
        vtt        = generate_vtt(segments)
        srt        = generate_srt(segments)

        hindi_audio = tmp / "hindi.mp3"
        import asyncio
        import edge_tts
        import gtts
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for seg in segments:
            if not seg["hi_text"].strip():
                continue
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
        metrics      = compute_metrics(w_result, segments, src_audio,
                                       hindi_audio)
        transcript   = pd.DataFrame(segments)

    return vtt, srt, dubbed_bytes, transcript, metrics


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Video Source")
source_mode = st.sidebar.radio("", ["Sample clip", "Upload your own"], index=0)

uploaded_file     = None
selected_clip_key = None
processed_clips: dict = {}

if source_mode == "Upload your own":
    uploaded_file = st.sidebar.file_uploader(
        "Drop an MP4 or MOV", type=["mp4", "mov", "avi"],
        help="Max ~100MB. Processing takes ~30-60s.",
    )
else:
    processed_clips = _find_processed_clips()
    if processed_clips:
        selected_clip_key = st.sidebar.selectbox(
            "Choose a clip", list(processed_clips.keys()),
            format_func=lambda k: k.replace("_", " ").title(),
        )
    else:
        st.sidebar.warning(
            "No processed clips found. "
            "Run `pipeline_v2.py --input <video>` first."
        )

st.sidebar.divider()
st.sidebar.markdown("**Pipeline**")
st.sidebar.code(
    "Whisper (base)\n"
    "→ pyannote speaker diarization\n"
    "→ wav2vec2 gender classification\n"
    "→ Claude (isochrony constraints)\n"
    "→ edge-tts multi-voice\n"
    "  ♀ hi-IN-SwaraNeural\n"
    "  ♂ hi-IN-MadhurNeural"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 AI Video Dubbing — v2")
st.caption(
    "English video → Hindi subtitles & Hindi dubbed audio  |  "
    "Whisper ASR + Google Translate + Microsoft Neural TTS  |  "
    "Built for Eros Now / SunNxt"
)

# ── Load / process ────────────────────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner(
        f"Processing **{uploaded_file.name}** — transcribing, translating, dubbing…"
    ):
        vtt, srt, dubbed_bytes, transcript, metrics = process_uploaded_video(
            uploaded_file.read(), uploaded_file.name
        )
    video_src    = uploaded_file
    dubbed_src   = dubbed_bytes
elif selected_clip_key:
    orig, dubbed, metrics_p, vtt_p, srt_p, transcript_p = (
        processed_clips[selected_clip_key]
    )
    transcript, metrics, vtt, srt = load_clip_outputs(
        selected_clip_key,
        str(metrics_p), str(vtt_p), str(srt_p), str(transcript_p),
    )
    video_src  = orig.read_bytes()
    dubbed_src = dubbed.read_bytes()
else:
    st.info("Select a processed clip from the sidebar or upload your own video.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
al = metrics["alignment"]
k1, k2, k3, k4, k5 = st.columns(5)
_wer_pct = metrics["asr"]["wer_pct"]
_bleu    = metrics["translation"]["bleu"]
k1.metric("ASR Accuracy",
          f"{100 - _wer_pct:.1f}%" if _wer_pct is not None else "N/A")
k2.metric("BLEU Score",
          f"{_bleu:.1f}" if _bleu is not None else "N/A")
k3.metric("Segments",        str(metrics["translation"]["n_segments"]))
k4.metric("Source",          f"{al['source_duration_s']:.1f}s")
k5.metric("Dubbed (HI)",     f"{al['dubbed_duration_s']:.1f}s",
          f"{(al['duration_ratio'] - 1)*100:+.1f}% vs original",
          delta_color="off")

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
        "Subtitles are rendered natively in the browser — no re-encoding. "
        "Toggle captions on/off using the CC button in the video player."
    )
    st.video(video_src, subtitles={"हिंदी": vtt})

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
    _voice_display = metrics.get("tts", {}).get("voice", "multi-voice (diarized)")
    st.caption(
        "The English audio has been replaced with AI-generated Hindi speech. "
        f"Speaker diarization assigns per-character voices (♀ hi-IN-SwaraNeural, "
        f"♂ hi-IN-MadhurNeural). Recorded as: **{_voice_display}**."
    )
    st.video(dubbed_src)

    with st.expander("⬇️  Download dubbed video"):
        st.download_button("Download dubbed MP4",
                           dubbed_src if isinstance(dubbed_src, bytes)
                           else dubbed_src,
                           "video_dubbed_hindi.mp4", "video/mp4")


# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Side-by-Side Comparison")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🇬🇧 Original — English audio")
        st.video(video_src)
        st.caption(f"Duration: {al['source_duration_s']:.1f}s  |  "
                   f"Pace: {al['en_chars_per_sec']:.1f} ch/s")

    with c2:
        st.markdown("#### 🇮🇳 Use Case 1 — Hindi subtitles")
        st.video(video_src, subtitles={"हिंदी": vtt})
        st.caption("Same video · Hindi CC overlay · no re-encode")

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 🇬🇧 Original — English audio")
        st.video(video_src)

    with c4:
        st.markdown("#### 🇮🇳 Use Case 2 — Hindi dubbed audio")
        st.video(dubbed_src)
        _v = metrics.get("tts", {}).get("voice", "multi-voice")
        st.caption(f"Duration: {al['dubbed_duration_s']:.1f}s  |  "
                   f"Pace: {al['hi_chars_per_sec']:.1f} ch/s  |  "
                   f"Voice: {_v}")


# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Bilingual Transcript")
    st.caption(
        "Whisper ASR → Google Translate · timestamps from Whisper segmentation"
    )

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

    st.subheader("Quality Metrics")
    m1, m2, m3 = st.columns(3)

    _wer_pct_tab4 = metrics["asr"]["wer_pct"]
    _bleu_tab4    = metrics["translation"]["bleu"]

    if _wer_pct_tab4 is not None:
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number", value=100 - _wer_pct_tab4,
            title={"text": "ASR Accuracy (1 − WER)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#00CC96"},
                   "steps": [{"range": [0,  70], "color": "#EF553B"},
                             {"range": [70, 85], "color": "#FFA15A"},
                             {"range": [85, 100], "color": "#00CC96"}]},
            number={"suffix": "%"},
        ))
        fig1.update_layout(height=240, margin=dict(t=40, b=10))
        m1.plotly_chart(fig1, use_container_width=True)
    else:
        m1.metric("ASR Accuracy", "N/A — no reference")

    if _bleu_tab4 is not None:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number", value=_bleu_tab4,
            title={"text": "Translation BLEU"},
            gauge={"axis": {"range": [0, 50]}, "bar": {"color": "#636EFA"},
                   "steps": [{"range": [0,  10], "color": "#EF553B"},
                             {"range": [10, 20], "color": "#FFA15A"},
                             {"range": [20, 50], "color": "#00CC96"}]},
        ))
        fig2.update_layout(height=240, margin=dict(t=40, b=10))
        m2.plotly_chart(fig2, use_container_width=True)
    else:
        m2.metric("Translation BLEU", "N/A — no reference")

    fig3 = go.Figure(go.Indicator(
        mode="gauge+number", value=al["duration_ratio"],
        title={"text": "Duration Ratio (HI / EN)"},
        gauge={"axis": {"range": [0.5, 2.0]}, "bar": {"color": "#FFA15A"},
               "steps": [{"range": [0.5, 0.9], "color": "#EF553B"},
                         {"range": [0.9, 1.1], "color": "#00CC96"},
                         {"range": [1.1, 2.0], "color": "#FFA15A"}],
               "threshold": {"line": {"color": "black", "width": 3},
                             "thickness": 0.75, "value": 1.0}},
        number={"valueformat": ".3f"},
    ))
    fig3.update_layout(height=240, margin=dict(t=40, b=10))
    m3.plotly_chart(fig3, use_container_width=True)

    # ── Segment-level quality ─────────────────────────────────────────────────
    seg_quality = metrics.get("segment_quality", [])
    if seg_quality:
        st.divider()
        st.subheader("Segment-Level Quality")

        sq_df = pd.DataFrame(seg_quality)

        # Isochrony chart
        st.markdown("#### Isochrony — TTS duration / EN window")
        st.caption(
            "Ideal range 0.85–1.15 (green). "
            "Above 1.3 = speech will overflow its window and sound rushed."
        )
        _iso = sq_df[sq_df["isochrony_ratio"].notna()].copy()
        _iso["color"] = _iso["isochrony_ratio"].apply(
            lambda r: "#00CC96" if 0.85 <= r <= 1.15
            else ("#FFA15A" if r <= 1.30 else "#EF553B")
        )
        fig_iso = go.Figure(go.Bar(
            x=_iso["start"], y=_iso["isochrony_ratio"],
            marker_color=_iso["color"],
            hovertext=[
                f"[{r.start:.1f}–{r.end:.1f}s]  ratio={r.isochrony_ratio:.2f}"
                for _, r in _iso.iterrows()
            ],
            hoverinfo="text",
        ))
        fig_iso.add_hline(y=1.0, line_dash="dot", line_color="black",
                          annotation_text="1.0")
        fig_iso.update_layout(height=260, margin=dict(t=20, b=20),
                              yaxis_title="ratio", xaxis_title="segment start (s)")
        st.plotly_chart(fig_iso, use_container_width=True)

        # LLM scores table (only if grading was run)
        _grade_cols = [c for c in ["fidelity", "fluency", "fit"] if c in sq_df.columns]
        if _grade_cols:
            st.markdown("#### LLM Translation Grades (Claude Haiku, 1–5)")
            _display_cols = ["start", "end"] + _grade_cols + (
                ["note"] if "note" in sq_df.columns else []
            )
            _show = sq_df[_display_cols].rename(
                columns={"start": "Start (s)", "end": "End (s)",
                         "fidelity": "Fidelity", "fluency": "Fluency",
                         "fit": "Fit", "note": "Note"}
            )
            avg_scores = {c: sq_df[c].mean() for c in _grade_cols
                          if sq_df[c].notna().any()}
            sc1, sc2, sc3 = st.columns(3)
            for col_widget, (label, avg) in zip(
                [sc1, sc2, sc3],
                [("Avg Fidelity", avg_scores.get("fidelity")),
                 ("Avg Fluency",  avg_scores.get("fluency")),
                 ("Avg Fit",      avg_scores.get("fit"))]
            ):
                col_widget.metric(label, f"{avg:.1f} / 5" if avg else "N/A")
            st.dataframe(_show, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("How the Pipeline Works")
    st.markdown("""
    ```
    Input video (MP4/MOV)
          │
          ├── ffmpeg ──────────────────────► WAV audio (16 kHz mono)
          │                                        │
          │                    ┌───────────────────┤
          │                    │                   │
          │          OpenAI Whisper (base)   pyannote diarization 3.1
          │                    │                   │
          │          EN transcript + timestamps  speaker turns
          │                    │                   │
          │                    └───────────────────┤
          │                                        │
          │                         wav2vec2 gender classifier
          │                         (audeering/wav2vec2-large)
          │                                        │
          │                         speaker → voice map
          │                         ♀ SwaraNeural  ♂ MadhurNeural
          │                                        │
          │                    Claude (isochrony-constrained translation)
          │                                        │
          │                  ┌─────────────────────┴──────────────────┐
          │                  ▼                                         ▼
          │            WebVTT / SRT                      per-segment TTS audio
          │            subtitle file                  (voice matched per speaker)
          │                  │                                         │
          │                  ▼                                         ▼
          └──── st.video(subtitles=vtt) ──────── ffmpeg -map 0:v -map 1:a ──►
                Use Case 1: subtitled video              Use Case 2: dubbed video
    ```
    """)

    st.markdown("#### Component breakdown")
    components = pd.DataFrame({
        "Stage": ["ASR", "Diarization", "Translation", "TTS", "Subtitle", "Dubbing"],
        "Model / Tool": [
            "OpenAI Whisper base",
            "pyannote 3.1 + wav2vec2-large gender",
            "Claude (isochrony constraints)",
            "edge-tts hi-IN-Swara/MadhurNeural",
            "WebVTT → st.video()",
            "ffmpeg audio track replace",
        ],
        "Cost": [
            "Free (local)", "Free (HF model)", "Claude API",
            "Free (Microsoft Edge)", "Free", "Free (local)",
        ],
        "Production upgrade": [
            "Whisper large-v3 / Azure Speech",
            "pyannote cloud / AssemblyAI diarization",
            "Claude Opus with larger batches",
            "Azure Neural TTS / ElevenLabs voice clone",
            "Burn-in via ffmpeg subtitles filter",
            "Time-stretch TTS to match original duration",
        ],
        "Status": ["✅ Live"] * 6,
    })
    st.dataframe(components, use_container_width=True, hide_index=True)

    st.markdown("#### Supported language pairs (roadmap)")
    lang_df = pd.DataFrame({
        "Source": ["English"] * 5 + ["Hindi"] * 3,
        "Target": ["Hindi", "Tamil", "Telugu", "Kannada", "Malayalam",
                   "Tamil", "Telugu", "Kannada"],
        "TTS voice (edge-tts)": [
            "hi-IN-SwaraNeural", "ta-IN-PallaviNeural",
            "te-IN-ShrutiNeural", "kn-IN-SapnaNeural",
            "ml-IN-SobhanaNeural", "ta-IN-PallaviNeural",
            "te-IN-ShrutiNeural", "kn-IN-SapnaNeural",
        ],
        "Status": ["✅ Live in v2"] + ["🔧 Add in v3"] * 7,
    })
    st.dataframe(lang_df, use_container_width=True, hide_index=True)
