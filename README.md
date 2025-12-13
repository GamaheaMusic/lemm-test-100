# Music Generation App

An AI-powered music generation application with **AMD GPU support** using DiffRhythm2 for music generation with built-in vocals and Fish Speech for TTS.

## ✨ Features

- 🎵 **Music Generation**: Generate music clips with vocals from text prompts using DiffRhythm2
-   **Style Consistency**: Automatic style matching - new clips inherit the musical character of existing ones
- 🎤 **Lyrics Integration**: Add your own lyrics or generate instrumental tracks
- 🎙️ **Built-in Vocals**: DiffRhythm2 generates vocals directly with the music
- 🎚️ **DAW-Style Timeline**: Professional horizontal timeline with tracks and playback controls
- 🎛️ **Advanced Mastering**: 32 professional presets + custom EQ, compression, and limiting
- 📍 **Flexible Positioning**: Place clips at Intro, Previous, Next, or Outro positions
- 💾 **Export/Download**: Merge and download your complete compositions
- 🎮 **AMD GPU Support**: Optimized for AMD GPUs via DirectML (Python 3.11)
- 🖥️ **Modern Web UI**: Clean, responsive interface with real-time updates

## 🎮 AMD GPU Support

This application is **optimized for AMD GPUs** using PyTorch DirectML:
- ✅ AMD Vega 8 (your current GPU)
- ✅ AMD Radeon RX series
- ✅ AMD Ryzen integrated graphics
- ✅ Automatic fallback to CPU if GPU unavailable

## Architecture

### Interface
- **Technology**: Gradio web interface
- **Access**: Browser-based at http://localhost:7860
- **Features**: Real-time updates, audio preview, timeline visualization

### Backend
- **Framework**: Python with Flask
- **GPU Backend**: DirectML for AMD GPU acceleration
- **Models**: 
  - DiffRhythm2 (ASLP-lab) for music generation with vocals
  - MuQ-MuLan for music style encoding
- **Features**: Local model caching, robust error handling, comprehensive logging

## Project Structure

```
Angen/
├── backend/
│   ├── app.py                 # Flask application entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py        # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Data models and schemas
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── generation.py      # Music and lyrics generation endpoints
│   │   ├── timeline.py        # Timeline management endpoints
│   │   └── export.py          # Export and download endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── diffrhythm_service.py    # DiffRhythm integration
│   │   ├── lyricmind_service.py     # LyricsMindAI integration
│   │   ├── fish_speech_service.py   # Fish Speech integration
│   │   ├── timeline_service.py      # Timeline management
│   │   └── export_service.py        # Audio export service
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          # Logging configuration
│       └── validators.py      # Request validation
├── frontend/
│   ├── index.html             # Main HTML page
│   ├── css/
│   │   └── style.css          # Styles
│   └── js/
│       └── app.js             # Frontend JavaScript
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore
└── README.md

```

## Installation & Quick Start

### Prerequisites

- **Python 3.11** (required for AMD GPU support via DirectML)
- pip
- Internet connection (for first-time model download ~5-8GB)

### Quick Setup

**Windows:**
```powershell
# 1. Clone/download the project
cd d:\2025-vibe-coding\Angen

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the app (downloads models on first run)
.\launch.ps1
```

**Linux/WSL:**
```bash
# 1. Clone/download the project
cd ~/Angen

# 2. Create virtual environment with Python 3.11
python3.11 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the app (downloads models on first run)
./launch.sh
```

### What the Launcher Does

The launcher script (`launch.ps1` or `launch.sh`) will:
1. Check if AI models are installed
2. If not, prompt to download them (~5-8GB):
   - DiffRhythm2 model (~5GB)
   - Fish Speech TTS (~2GB)
3. Start the backend server (Flask on port 7860)
4. Start the frontend server (HTTP server on port 8000)
5. Display progress and URLs

**Access the application at:**
- **Frontend**: http://localhost:8000
- **Backend API**: http://localhost:7860

## Usage

### 1. Generate Music

1. Enter a music prompt describing the style (e.g., "upbeat electronic dance music with heavy bass")
2. **(Optional)** Add lyrics in the lyrics box for vocal generation
3. Leave lyrics empty for instrumental music
5. Set duration (10-120 seconds, default 30)
6. Select timeline position (Intro/Previous/Next/Outro)
7. Click **"✨ Generate Music Clip"**
8. Preview the generated audio in the player

### 2. Manage Timeline

1. Switch to the **"Timeline"** tab
2. View all generated clips with their durations
3. Remove individual clips by entering clip number
4. Clear entire timeline if needed

### 3. Export

1. Open the **"Export"** tab
2. Enter a filename (without extension)
3. Choose format (WAV, MP3, or FLAC)
4. Click **"💾 Export Timeline"**
5. Download or play the merged audio

## AMD GPU Performance

- **First generation**: May take 30-60 seconds (model loading)
- **Subsequent generations**: 10-30 seconds on Vega 8
- **CPU fallback**: 1-3 minutes per clip

The app will automatically detect and use your AMD GPU via DirectML.

## Technical Details

### Models Used

1. **MusicGen-small** (Facebook)
   - Purpose: Music generation
   - Size: ~1.5GB
   - Generates up to 30 seconds of music from text

2. **Phi-2** (Microsoft)
   - Purpose: Lyrics generation
   - Size: ~5GB
   - Generates creative song lyrics from prompts

3. **Fish Speech** (TTS)
   - Purpose: Vocal synthesis
   - Size: ~2GB  
   - Converts lyrics to singing voice

### AMD GPU via DirectML

- Uses `torch-directml` backend
- Automatic device detection
- FP32/FP16 mixed precision
- Optimized for AMD Vega architecture

### Project Structure

## Troubleshooting

### AMD GPU Not Detected

1. Ensure `torch-directml` is installed:
   ```powershell
   pip install torch-directml
   ```

2. Check device in logs when app starts:
   ```
   ✅ AMD GPU detected via DirectML
   ```

3. If not detected, app will use CPU (slower but functional)

### Models Not Downloading

1. Check internet connection
2. Verify Hugging Face is accessible
3. Manual download option:
   ```powershell
   python setup_models.py
   ```

### Out of Memory Errors

1. Reduce clip duration
2. Close other applications
3. Use CPU instead (set `USE_DIRECTML=False` in `.env`)

### Slow Generation

- First run is slower (model loading)
- Vega 8: 10-30 seconds per clip is normal
- Check Task Manager to verify GPU usage

## Support

For issues:
1. Check `logs/app.log` for detailed error messages
2. Verify all dependencies installed: `pip list`
3. Ensure models downloaded: check `models/` folder
4. Test AMD GPU detection: `python -c "import torch_directml; print(torch_directml.is_available())"`

---

Built with ❤️ using AI-powered music generation
