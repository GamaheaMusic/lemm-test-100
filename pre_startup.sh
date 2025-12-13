#!/bin/bash

# Pre-startup script for HuggingFace Spaces
# This runs before the main application

echo "🚀 Initializing Music Generation Studio..."

# Verify espeak-ng installation
if command -v espeak-ng &> /dev/null; then
    echo "✅ espeak-ng is installed"
    espeak-ng --version
else
    echo "❌ espeak-ng not found"
    exit 1
fi

# Verify ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✅ ffmpeg is installed"
    ffmpeg -version | head -1
else
    echo "❌ ffmpeg not found"
fi

# Create necessary directories
mkdir -p outputs/music
mkdir -p outputs/mixed
mkdir -p models
mkdir -p logs

echo "✅ Directories created"

# Check Python version
python --version

# Verify key dependencies
echo "📦 Verifying Python packages..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || echo "❌ PyTorch not found"
python -c "import gradio; print(f'✅ Gradio {gradio.__version__}')" || echo "❌ Gradio not found"
python -c "import phonemizer; print('✅ phonemizer OK')" || echo "❌ phonemizer not found"

echo "✅ Pre-startup checks complete"
