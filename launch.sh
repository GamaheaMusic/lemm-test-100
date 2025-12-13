#!/usr/bin/env bash
#
# Music Generation App Launcher for Linux/WSL
# Automatically sets up models on first run and launches the application
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   🎵 Music Generation App - Launcher                 ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_progress() {
    echo -e "${YELLOW}⟳ $1${NC}"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if models are installed
check_models_installed() {
    if [ -d "models/diffrhythm2/models--ASLP-lab--DiffRhythm2" ]; then
        print_success "DiffRhythm2 model found"
        return 0
    fi
    
    print_info "Models not found - will download on first run"
    return 1
}

# Install models
install_models() {
    print_header
    print_info "First-time setup: Downloading AI models..."
    print_info "This will download approximately 5-8 GB of data"
    print_info "Progress will be shown during download"
    echo ""
    
    # Check for virtual environment
    VENV_PYTHON=".venv/bin/python"
    
    if [ ! -f "$VENV_PYTHON" ]; then
        print_error "Virtual environment not found. Please run:"
        print_error "  python3.11 -m venv .venv"
        print_error "  source .venv/bin/activate"
        print_error "  pip install -r requirements.txt"
        exit 1
    fi
    
    print_progress "Downloading models (this may take several minutes)..."
    echo ""
    
    # Run Python script to download models
    cat > /tmp/download_models.py << 'EOF'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm

def download_diffrhythm2():
    print('\n📥 Downloading DiffRhythm2 model...')
    models_dir = Path('models/diffrhythm2')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id='ASLP-lab/DiffRhythm2',
            local_dir=str(models_dir),
            local_dir_use_symlinks=False
        )
        print('✓ DiffRhythm2 downloaded successfully')
        return True
    except Exception as e:
        print(f'✗ Error downloading DiffRhythm2: {e}')
        return False

def download_fish_speech():
    print('\n📥 Downloading Fish Speech model...')
    models_dir = Path('models/fish_speech')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        
        files = [
            'config.json',
            'firefly-gan-vq-fsq-4x1024-42hz-generator.pth',
            'model.pth'
        ]
        
        for file in tqdm(files, desc='Fish Speech files'):
            hf_hub_download(
                repo_id='fishaudio/fish-speech-1.2',
                filename=file,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False
            )
        
        print('✓ Fish Speech downloaded successfully')
        return True
    except Exception as e:
        print(f'✗ Error downloading Fish Speech: {e}')
        return False

if __name__ == '__main__':
    success = True
    success = download_diffrhythm2() and success
    success = download_fish_speech() and success
    
    if success:
        print('\n✓ All models downloaded successfully!')
        sys.exit(0)
    else:
        print('\n✗ Some models failed to download')
        sys.exit(1)
EOF
    
    if $VENV_PYTHON /tmp/download_models.py; then
        echo ""
        print_success "Models installed successfully!"
        echo ""
        rm -f /tmp/download_models.py
        return 0
    else
        print_error "Model download failed"
        rm -f /tmp/download_models.py
        return 1
    fi
}

# Main
print_header

# Check for virtual environment
VENV_PYTHON=".venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    print_error "Virtual environment not found!"
    print_info "Please set up the environment first:"
    echo "  1. python3.11 -m venv .venv"
    echo "  2. source .venv/bin/activate"
    echo "  3. pip install -r requirements.txt"
    exit 1
fi

# Check and install models if needed
if ! check_models_installed; then
    read -p "Models not installed. Download now? (Y/n): " install
    if [[ ! "$install" =~ ^[Nn]$ ]]; then
        if ! install_models; then
            print_error "Failed to install models. Please check your internet connection."
            exit 1
        fi
    else
        print_info "Skipping model download. The app may not work properly without models."
    fi
fi

# Launch the application
print_info "Starting Music Generation App..."
print_info "Backend: http://localhost:7860"
print_info "Frontend: http://localhost:8000"
echo ""
print_info "Press Ctrl+C to stop the servers"
echo ""

# Trap to cleanup on exit
cleanup() {
    echo ""
    print_info "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

# Start backend in background
$VENV_PYTHON backend/run.py &
BACKEND_PID=$!

sleep 2

# Start frontend (this will block)
$VENV_PYTHON -m http.server 8000 --directory frontend

# Cleanup
cleanup
