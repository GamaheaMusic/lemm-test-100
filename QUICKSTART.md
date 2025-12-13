# 🚀 Quick Start Guide

## Running the Application

### **Windows**
```powershell
.\launch.ps1
```

### **Linux/Mac**
```bash
./launch.sh
```

## What the Launcher Does

The unified launcher script automatically handles everything:

1. **Environment Setup** ✓
   - Creates Python virtual environment (if not exists)
   - Verifies Python 3.11 installation

2. **Dependencies** ✓
   - Installs all required packages from requirements.txt
   - Installs DiffRhythm2 additional dependencies
   - Only reinstalls when requirements change

3. **Models** ✓
   - Checks for DiffRhythm2 model files
   - Downloads from HuggingFace on first generation (~5GB)
   - Models are cached locally after first download

4. **Servers** ✓
   - Starts backend API server (http://localhost:7860)
   - Starts frontend web server (http://localhost:8000)
   - Graceful shutdown on Ctrl+C

## First Run

On the first run, the launcher will:
- Create `.venv` directory
- Install ~2GB of Python packages
- Models will download when you generate your first music clip

**Total first-run time**: 5-10 minutes (depending on internet speed)

## Using the App

1. **Open your browser** to http://localhost:8000

2. **Generate music**:
   - Enter a text prompt (e.g., "upbeat electronic dance music")
   - Optionally add lyrics
   - Click "Generate Clip"
   - Wait ~1-2 minutes for generation (CPU mode)

3. **Build your composition**:
   - Generated clips appear on the timeline
   - Click to play individual clips
   - Generate more clips - they'll auto-match the style!
   - Use mastering presets to enhance your mix

4. **Export**:
   - Click "Export Composition"
   - Downloads as merged WAV file

## Troubleshooting

### Backend fails to start
- Check Python 3.11 is installed: `python --version`
- Check port 7860 is available
- Review backend logs in `backend/logs/`

### Dependencies fail to install
- Ensure internet connection is stable
- Try: `.\.venv\Scripts\pip.exe install -r requirements.txt --verbose`

### Out of memory
- Close other applications
- Reduce generation duration
- App runs in CPU mode - no GPU required

### Models fail to download
- Check internet connection
- Check disk space (need 10GB free)
- Models download from HuggingFace automatically

## Manual Setup (Advanced)

If you prefer manual setup:

```powershell
# Create venv
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac

# Install deps
pip install -r requirements.txt
pip install inflect

# Start backend
python backend/run.py

# Start frontend (new terminal)
python -m http.server 8000 --directory frontend
```

## System Requirements

- **Python**: 3.11 (required)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 10GB free space
- **CPU**: Multi-core processor
- **GPU**: Optional (NVIDIA CUDA supported)

## Ports Used

- **7860**: Backend API server
- **8000**: Frontend web server

Make sure these ports are not in use by other applications.

---

**Need help?** Check README.md for detailed documentation.
