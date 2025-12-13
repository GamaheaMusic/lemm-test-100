#!/bin/bash
# Setup script for HuggingFace Spaces
# Clones DiffRhythm2 source code if not present

set -e

echo "🔧 Setting up DiffRhythm2 source code..."

MODELS_DIR="models"
DR2_SRC_DIR="$MODELS_DIR/diffrhythm2_source"

# Create models directory
mkdir -p "$MODELS_DIR"

# Check if DiffRhythm2 source exists
if [ ! -d "$DR2_SRC_DIR" ]; then
    echo "📥 Cloning DiffRhythm2 source repository..."
    git clone https://github.com/ASLP-lab/DiffRhythm2.git "$DR2_SRC_DIR"
    echo "✅ DiffRhythm2 source cloned"
else
    echo "✅ DiffRhythm2 source already exists"
fi

echo "✅ Setup complete"
