# HuggingFace Spaces Deployment Guide

## Prerequisites

1. HuggingFace account
2. Git installed locally
3. HuggingFace CLI (optional): `pip install huggingface-hub`

## Deployment Steps

### Method 1: Using Git (Recommended)

1. **Clone the HuggingFace Space repository**:
   ```bash
   git clone https://huggingface.co/spaces/Gamahea/lemm-test-100
   cd lemm-test-100
   ```

2. **Copy required files** from this project:
   ```powershell
   # Core application files
   Copy-Item app.py lemm-test-100/
   Copy-Item hf_config.py lemm-test-100/
   Copy-Item requirements_hf.txt lemm-test-100/requirements.txt
   Copy-Item packages.txt lemm-test-100/
   Copy-Item README_HF.md lemm-test-100/README.md
   Copy-Item pre_startup.sh lemm-test-100/
   
   # Backend code
   Copy-Item -Recurse backend lemm-test-100/
   
   # Exclude heavy files (models will download automatically)
   # Do NOT copy: models/, outputs/, external/, .venv/
   ```

3. **Create `.gitignore`** in the Space directory:
   ```bash
   cd lemm-test-100
   cat > .gitignore << 'EOF'
   __pycache__/
   *.pyc
   *.pyo
   .Python
   *.log
   models/
   outputs/
   logs/
   .env
   EOF
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "Deploy Music Generation Studio"
   git push
   ```

5. **Wait for build**: HuggingFace will automatically build and deploy your Space

### Method 2: Using Web Interface

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces/Gamahea/lemm-test-100)
2. Click "Files and versions"
3. Upload files manually:
   - `app.py`
   - `hf_config.py`
   - `requirements.txt` (use `requirements_hf.txt` content)
   - `packages.txt`
   - `README.md` (use `README_HF.md` content)
   - `pre_startup.sh`
   - Entire `backend/` directory

## Files Overview

### Essential Files for HuggingFace Spaces

```
lemm-test-100/
├── app.py                    # Main Gradio application
├── hf_config.py             # Environment configuration for Spaces
├── requirements.txt         # Python dependencies
├── packages.txt             # System dependencies (espeak-ng, ffmpeg)
├── README.md                # Space description
├── pre_startup.sh           # Pre-launch verification script
└── backend/                 # Backend code
    ├── __init__.py
    ├── config/
    │   ├── __init__.py
    │   └── settings.py
    ├── models/
    │   ├── __init__.py
    │   └── schemas.py
    ├── services/
    │   ├── __init__.py
    │   ├── diffrhythm_service.py
    │   ├── lyricmind_service.py
    │   ├── timeline_service.py
    │   └── export_service.py
    └── utils/
        ├── __init__.py
        ├── logger.py
        ├── prompt_analyzer.py
        └── validators.py
```

### What NOT to Include

- `models/` directory (models download from HuggingFace automatically)
- `outputs/` directory (created at runtime)
- `external/` directory (espeak-ng installed via packages.txt)
- `.venv/` directory
- Development scripts (`launch.ps1`, `setup_*.py`, etc.)
- Local documentation (`CHANGES.md`, `INSTALL*.md`, etc.)

## Configuration for HuggingFace Spaces

### Space Settings

In your Space settings, configure:

- **SDK**: Gradio
- **SDK Version**: 4.44.0
- **Python version**: 3.11
- **Hardware**: CPU Basic (free tier) or upgrade for faster performance

### Environment Variables (Optional)

You can set these in Space settings if needed:

- `LOG_LEVEL`: `INFO` (default)
- `DEFAULT_CLIP_DURATION`: `30`
- `SAMPLE_RATE`: `44100`

## Troubleshooting

### Common Issues

1. **espeak-ng not found**:
   - Ensure `packages.txt` includes `espeak-ng`
   - Check pre_startup.sh runs successfully
   - View build logs for errors

2. **Model download fails**:
   - Check internet connectivity in Space
   - Models download from HuggingFace automatically
   - First run will be slower (downloading ~2-4GB)

3. **Out of memory**:
   - Reduce `DEFAULT_CLIP_DURATION` to 10-20 seconds
   - Use shorter prompts
   - Consider upgrading to paid hardware tier

4. **Slow generation**:
   - Expected on CPU (free tier): 2-4 minutes per 30s clip
   - Upgrade to GPU hardware for ~10x speedup
   - Generate shorter clips (10-20s) for faster results

### Checking Logs

View application logs in the Space's "Logs" tab to debug issues.

### Testing Locally Before Deployment

```bash
# Install dependencies
pip install -r requirements_hf.txt

# Run locally
python app.py
```

## Performance Expectations

### Free Tier (CPU Basic)

- **Music Generation**: 2-4 minutes per 30-second clip
- **Lyrics Generation**: 20-60 seconds
- **Model Loading**: 30-60 seconds (first time)
- **Concurrent Users**: Limited (1-2)

### Paid Tier (T4 Small GPU)

- **Music Generation**: 15-30 seconds per 30-second clip
- **Lyrics Generation**: 5-10 seconds
- **Model Loading**: 10-20 seconds (first time)
- **Concurrent Users**: Better support (5-10)

## Model Caching

Models are automatically cached by HuggingFace Spaces:

- **DiffRhythm2**: ~2GB (downloaded on first run)
- **MuQ-MuLan**: ~1GB (downloaded on first run)
- **Total storage**: ~3-4GB

Subsequent runs use cached models for faster startup.

## Security Notes

- All processing happens in HuggingFace's secure environment
- No sensitive data is stored
- Generated audio files are temporary
- Clear timeline before closing to free space

## Updating the Space

To update your deployment:

```bash
# Pull latest changes
git pull

# Make updates to code
# ... edit files ...

# Commit and push
git add .
git commit -m "Update: description of changes"
git push
```

HuggingFace will automatically rebuild and redeploy.

## Support

For issues with HuggingFace Spaces deployment:

1. Check HuggingFace Spaces documentation
2. Review build and runtime logs
3. Test locally first
4. Contact HuggingFace support if needed

## Additional Resources

- [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [DiffRhythm2 Model](https://huggingface.co/ASLP-lab/DiffRhythm2)
- [MuQ-MuLan Model](https://huggingface.co/OpenMuQ/MuQ-MuLan-large)

---

**Last Updated**: December 12, 2025
