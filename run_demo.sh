#!/usr/bin/env bash
# One-command launcher for the AI Video Dubbing demo.
# Run: ./run_demo.sh

set -e
cd "$(dirname "$0")"

echo "🎬 AI Video Dubbing Demo — Logituit AI Practice"
echo "================================================"

# Check prerequisites
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Install with:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg not found. Install with: brew install ffmpeg"
    exit 1
fi

# Install / sync dependencies
echo "📦 Syncing dependencies..."
uv sync --quiet

# Run pipeline if sample video doesn't exist yet
if [ ! -f "data/raw/sample_en.mp4" ] || [ ! -f "model_outputs/subtitles_hi.vtt" ]; then
    echo "🔧 Running pipeline (first-time setup, ~60s)..."
    uv run python code/pipeline_v2.py
else
    echo "✅ Pipeline outputs already exist — skipping."
fi

# Launch dashboard
echo ""
echo "✅ Launching dashboard at http://localhost:8501"
echo "   Press Ctrl+C to stop."
echo ""
uv run streamlit run code/dashboard_v2.py --server.port 8501
