"""
VideoDubbing v2 dashboard — video-aware demo for Eros / SunNxt
Use Case 1: original video + Hindi subtitle overlay (WebVTT)
Use Case 2: original video with Hindi audio track (dubbed MP4)

Run: uv run streamlit run code/dashboard_v2.py
"""
import json
import re
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
    """Return {stem: (orig, dubbed, metrics, vtt, srt, transcript)} for ready clips."""
    clips: dict[str, tuple[Path, Path, Path, Path, Path, Path]] = {}
    _dubbed_re = re.compile(r"_dubbed_([a-z]{2})$")
    for d in [RAW, ROOT / "test_clips"]:
        if not d.exists():
            continue
        for mp4 in sorted(d.glob("*.mp4")):
            if _dubbed_re.search(mp4.stem) or mp4.stem == "sample_en":
                continue
            stem = mp4.stem
            # Find any dubbed version for this clip (hi, en, ta, …)
            for lang in ("hi", "en", "ta", "de"):
                dubbed = RAW / f"{stem}_dubbed_{lang}.mp4"
                if not dubbed.exists():
                    continue
                metrics_path = MODEL_OUT / f"metrics_{stem}.json"
                if not metrics_path.exists():
                    continue
                vtt_path   = MODEL_OUT / f"subtitles_{stem}_{lang}.vtt"
                srt_path   = MODEL_OUT / f"subtitles_{stem}_{lang}.srt"
                transcript = ROOT / "data/prepared" / f"transcript_bilingual_{stem}.csv"
                clips[stem] = (mp4, dubbed, metrics_path, vtt_path, srt_path,
                               transcript)
                break  # first matching lang wins
    return clips


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_clip_outputs(stem: str, metrics_path: str, vtt_path: str,
                      srt_path: str, transcript_path: str):
    with open(metrics_path) as f:
        metrics = json.load(f)
    vtt = Path(vtt_path).read_text(encoding="utf-8")
    srt = Path(srt_path).read_text(encoding="utf-8")
    transcript = pd.read_csv(transcript_path)
    return transcript, metrics, vtt, srt




# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Video Source")

processed_clips = _find_processed_clips()
selected_clip_key = None
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

st.sidebar.caption(
    "Pipeline runs on-premises — contact us to process your own content."
)

_WAV2LIP_CHECKPOINT = ROOT / "vendor" / "Wav2Lip" / "checkpoints" / "wav2lip_gan.pth"
st.sidebar.divider()
if _WAV2LIP_CHECKPOINT.exists():
    st.sidebar.success("👄 Wav2Lip: available — rerun with `--lipsync` to apply")
else:
    st.sidebar.info("👄 Wav2Lip: checkpoint not found (optional)")

st.sidebar.divider()
st.sidebar.markdown("**Pipeline**")
st.sidebar.code(
    "Whisper (large-v3-turbo)\n"
    "→ pyannote speaker diarization\n"
    "→ wav2vec2 gender classification\n"
    "→ Claude (isochrony + emotion)\n"
    "→ edge-tts multi-voice\n"
    "  Hindi ♀ hi-IN-SwaraNeural\n"
    "  Hindi ♂ hi-IN-MadhurNeural\n"
    "  Tamil ♀ ta-IN-PallaviNeural\n"
    "  Tamil ♂ ta-IN-ValluvarNeural"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 AI Video Dubbing — v2")
st.caption(
    "English video → Hindi / Tamil subtitles & dubbed audio  |  "
    "Whisper ASR + Google Translate + Claude + Microsoft Neural TTS  |  "
    "Built for Eros Now / SunNxt"
)

# ── Load / process ────────────────────────────────────────────────────────────
if selected_clip_key:
    orig, dubbed, metrics_p, vtt_p, srt_p, transcript_p = (
        processed_clips[selected_clip_key]
    )
    transcript, metrics, vtt, srt = load_clip_outputs(
        selected_clip_key,
        str(metrics_p), str(vtt_p), str(srt_p), str(transcript_p),
    )
    video_src  = orig.read_bytes()
    dubbed_src = dubbed.read_bytes()
    _dubbed_lang  = re.search(r"_dubbed_([a-z]{2})\.mp4$", str(dubbed)).group(1) if dubbed else "hi"
    _lipsync_path = RAW / f"{selected_clip_key}_lipsync_{_dubbed_lang}.mp4"
    lipsync_src = _lipsync_path.read_bytes() if _lipsync_path.exists() else None
else:
    st.info("Select a processed clip from the sidebar.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
al = metrics["alignment"]
_wer_pct = metrics["asr"]["wer_pct"]
_bleu    = metrics["translation"]["bleu"]
_bt        = metrics.get("back_translation", {})
_bt_bleu   = _bt.get("bleu")
_ls        = metrics.get("lipsync", {})
_sync      = _ls.get("sync_score")
_emo        = metrics.get("emotion", {})
_emo_match  = _emo.get("match_pct")
_emo_soft   = _emo.get("avg_soft_score")
_emo_tts    = _emo.get("tts_fidelity", {}).get("avg_soft_score")
k1, k2, k3 = st.columns(3)
k1.metric("Segments",    str(metrics["translation"]["n_segments"]))
k2.metric("Source",      f"{al['source_duration_s']:.1f}s")
k3.metric("Dubbed (HI)", f"{al['dubbed_duration_s']:.1f}s",
          f"{(al['duration_ratio'] - 1)*100:+.1f}% vs original",
          delta_color="off")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🇮🇳  Use Case 1 — Hindi Subtitles",
    "🔊  Use Case 2 — Hindi Dubbed Audio",
    "⚖️  Side by Side",
    "📝  Bilingual Transcript",
    "📊  Quality Metrics",
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

    if lipsync_src:
        _t2a, _t2b = st.tabs(["🔊 Dubbed (AI Audio)", "👄 Lip-Synced (Wav2Lip)"])
        with _t2a:
            st.video(dubbed_src)
            st.caption("AI-dubbed audio only — mouth movements unchanged from original.")
        with _t2b:
            st.video(lipsync_src)
            st.caption("Wav2Lip re-generates mouth movements to match the dubbed audio.")
    else:
        st.video(dubbed_src)
        if _WAV2LIP_CHECKPOINT.exists():
            st.info(
                "👄 **Lip sync available.** Rerun the pipeline with `--lipsync` "
                "to generate a Wav2Lip version with re-animated mouth movements."
            )

    with st.expander("⬇️  Download dubbed video"):
        st.download_button("Download dubbed MP4",
                           dubbed_src,
                           "video_dubbed_hindi.mp4", "video/mp4")
        if lipsync_src:
            st.download_button("Download lip-synced MP4",
                               lipsync_src,
                               "video_lipsync_hindi.mp4", "video/mp4")


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

    def _emotion_row_style(row):
        arc = row.get("arc_flagged") is True  # NaN/False/missing all → False
        if arc:
            colour = "#FEF9C3"
        else:
            emo = row.get("emotion", "neutral")
            colour = {
                "angry": "#FEE2E2", "fearful": "#FEE2E2", "disgust": "#FEE2E2",
                "sad": "#DBEAFE",
                "happy": "#DCFCE7", "surprised": "#DCFCE7",
                "neutral": "#F3F4F6",
            }.get(emo, "#F3F4F6")
        return [f"background-color: {colour}" if colour else "" for _ in row]

    st.dataframe(
        transcript.style.apply(_emotion_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.download_button(
        "⬇️  Download bilingual transcript (CSV)",
        transcript.to_csv(index=False).encode("utf-8"),
        "transcript_bilingual.csv", "text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Quality Metrics")

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

    if _wer_pct is not None:
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number", value=100 - _wer_pct,
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

    if _bleu is not None:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number", value=_bleu,
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

    if _bt_bleu is not None:
        fig4 = go.Figure(go.Indicator(
            mode="gauge+number", value=_bt_bleu,
            title={"text": "Back-Translation BLEU"},
            gauge={"axis": {"range": [0, 50]}, "bar": {"color": "#AB63FA"},
                   "steps": [{"range": [0,  10], "color": "#EF553B"},
                             {"range": [10, 20], "color": "#FFA15A"},
                             {"range": [20, 50], "color": "#00CC96"}]},
        ))
        fig4.update_layout(height=240, margin=dict(t=40, b=10))
        m4.plotly_chart(fig4, use_container_width=True)
        if _bt.get("back_translated_en") or _bt.get("back_translated"):
            with st.expander("Back-translated English (Hindi ASR → EN)"):
                st.write(_bt.get("back_translated_en") or _bt.get("back_translated"))
    else:
        m4.metric("Back-Translation BLEU", "N/A")

    if _sync is not None:
        fig5 = go.Figure(go.Indicator(
            mode="gauge+number", value=_sync,
            title={"text": "Lip-Sync Score"},
            gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#19D3F3"},
                   "steps": [{"range": [0,    0.45], "color": "#EF553B"},
                             {"range": [0.45, 0.60], "color": "#FFA15A"},
                             {"range": [0.60, 1.0],  "color": "#00CC96"}]},
            number={"valueformat": ".3f"},
        ))
        fig5.update_layout(height=240, margin=dict(t=40, b=10))
        m5.plotly_chart(fig5, use_container_width=True)
        if _ls.get("faces_pct") is not None:
            m5.caption(
                f"Pearson r={_ls.get('pearson_r', 'N/A')}  "
                f"lag={_ls.get('best_lag_ms', 'N/A')}ms  "
                f"face {_ls['faces_pct']}% of frames"
            )
    else:
        note = _ls.get("note", "not computed")
        m5.metric("Lip-Sync Score", "N/A")
        m5.caption(note)

    _emo_gauge_val = _emo_soft if _emo_soft is not None else _emo_match
    if _emo_gauge_val is not None:
        fig6 = go.Figure(go.Indicator(
            mode="gauge+number", value=_emo_gauge_val,
            title={"text": "Emotion Soft Score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#FF6692"},
                   "steps": [{"range": [0,  50], "color": "#EF553B"},
                             {"range": [50, 70], "color": "#FFA15A"},
                             {"range": [70, 100], "color": "#00CC96"}]},
            number={"suffix": "%"},
        ))
        fig6.update_layout(height=240, margin=dict(t=40, b=10))
        m6.plotly_chart(fig6, use_container_width=True)
        segs = _emo.get("segments", [])
        if segs:
            emo_dist = {}
            for s in segs:
                emo_dist[s["source_emotion"]] = emo_dist.get(s["source_emotion"], 0) + 1
            m6.caption("  ".join(f"{k}:{v}" for k, v in sorted(emo_dist.items())))
        if _emo_match is not None:
            m6.caption(f"Binary match: {_emo_match}%")
    else:
        m6.metric("Emotion Soft Score", "N/A")

    if _emo_tts is not None:
        fig7 = go.Figure(go.Indicator(
            mode="gauge+number", value=_emo_tts,
            title={"text": "TTS Emotion Fidelity"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#B6E880"},
                   "steps": [{"range": [0,  50], "color": "#EF553B"},
                             {"range": [50, 70], "color": "#FFA15A"},
                             {"range": [70, 100], "color": "#00CC96"}]},
            number={"suffix": "%"},
        ))
        fig7.update_layout(height=240, margin=dict(t=40, b=10))
        m7.plotly_chart(fig7, use_container_width=True)
        m7.caption("Audio SER: does dubbed voice sound as intended?")
    else:
        m7.metric("TTS Fidelity", "N/A")

    # ── MOS rubric ────────────────────────────────────────────────────────────
    _mos_data = metrics.get("mos_rubric", {})
    _mos_val  = _mos_data.get("mos")
    if _mos_val is not None:
        st.divider()
        st.subheader("MOS Rubric (AI Quality Score)")
        st.caption(
            f"Claude Sonnet spot-checks {_mos_data.get('n_sampled', '?')} segments "
            "on 5 dimensions (1–5 each), normalised to 0–100."
        )
        _mos_c1, _mos_c2 = st.columns([1, 2])
        with _mos_c1:
            _mos_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=_mos_val,
                title={"text": "Overall MOS"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#7C3AED"},
                    "steps": [
                        {"range": [0,  50], "color": "#EF553B"},
                        {"range": [50, 70], "color": "#FFA15A"},
                        {"range": [70, 100], "color": "#00CC96"},
                    ],
                },
                number={"suffix": "/100"},
            ))
            _mos_fig.update_layout(height=260, margin=dict(t=40, b=10))
            st.plotly_chart(_mos_fig, use_container_width=True)
        with _mos_c2:
            _breakdown = _mos_data.get("breakdown", {})
            if _breakdown:
                _dim_labels = {
                    "naturalness": "Naturalness",
                    "fidelity":    "Fidelity",
                    "timing":      "Timing fit",
                    "emotion":     "Emotion register",
                    "names":       "Proper nouns",
                }
                _dims  = [_dim_labels.get(k, k) for k in _breakdown]
                _vals  = [_breakdown[k] for k in _breakdown]
                _colors = [
                    "#00CC96" if v >= 4 else ("#FFA15A" if v >= 3 else "#EF553B")
                    for v in _vals
                ]
                _bar_fig = go.Figure(go.Bar(
                    x=_vals, y=_dims,
                    orientation="h",
                    marker_color=_colors,
                    text=[f"{v:.2f}/5" for v in _vals],
                    textposition="outside",
                ))
                _bar_fig.update_layout(
                    height=220, margin=dict(t=20, b=10, l=10, r=60),
                    xaxis={"range": [0, 5.5], "title": "Score (1–5)"},
                    yaxis={"title": ""},
                )
                st.plotly_chart(_bar_fig, use_container_width=True)
    elif _mos_data.get("note"):
        st.caption(f"MOS: {_mos_data['note']}")

    # ── Segment-level quality ─────────────────────────────────────────────────
    seg_quality = metrics.get("segment_quality", [])
    if seg_quality:
        st.divider()
        st.subheader("Segment-Level Quality")

        sq_df = pd.DataFrame(seg_quality)

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

        _grade_cols = [c for c in ["fidelity", "fluency", "fit", "emotion_register"]
                       if c in sq_df.columns]
        if _grade_cols:
            st.markdown("#### LLM Translation Grades (Claude Haiku, 1–5)")
            _display_cols = ["start", "end"] + _grade_cols + (
                ["note"] if "note" in sq_df.columns else []
            )
            _show = sq_df[_display_cols].rename(
                columns={"start": "Start (s)", "end": "End (s)",
                         "fidelity": "Fidelity", "fluency": "Fluency",
                         "fit": "Fit", "emotion_register": "Emotion Register",
                         "note": "Note"}
            )
            avg_scores = {c: sq_df[c].mean() for c in _grade_cols
                          if sq_df[c].notna().any()}
            _score_items = [
                ("Avg Fidelity",          avg_scores.get("fidelity")),
                ("Avg Fluency",           avg_scores.get("fluency")),
                ("Avg Fit",               avg_scores.get("fit")),
                ("Avg Emotion Register",  avg_scores.get("emotion_register")),
            ]
            _score_items = [(lbl, v) for lbl, v in _score_items if v is not None]
            _sc_cols = st.columns(len(_score_items))
            for col_widget, (label, avg) in zip(_sc_cols, _score_items):
                col_widget.metric(label, f"{avg:.1f} / 5")
            st.dataframe(_show, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
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
        "Status": ["✅ Live in v2", "✅ Live in v2"] + ["🔧 Add in v3"] * 6,
    })
    st.dataframe(lang_df, use_container_width=True, hide_index=True)
