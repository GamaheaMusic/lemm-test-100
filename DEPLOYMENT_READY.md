# 🚀 HuggingFace Spaces Deployment - Ready to Deploy

Your Music Generation Studio is ready for deployment to HuggingFace Spaces!

## ✅ Deployment Files Created

All necessary files have been created and configured:

1. **app.py** - Main Gradio application with:
   - Automatic DiffRhythm2 source setup
   - Lazy-loading models
   - Progress tracking
   - Emoji-enhanced UI
   - Complete music generation workflow

2. **hf_config.py** - Environment configuration:
   - Auto-detects HuggingFace Spaces
   - Configures espeak-ng paths
   - Sets up phonemizer environment

3. **requirements_hf.txt** - Python dependencies:
   - Gradio 4.44.0
   - PyTorch 2.4+ (CPU mode)
   - All DiffRhythm2 requirements
   - Audio processing libraries

4. **packages.txt** - System dependencies:
   - espeak-ng
   - ffmpeg
   - libsndfile1

5. **README_HF.md** - Space description:
   - Features overview
   - Usage instructions
   - Model information
   - Performance expectations

6. **pre_startup.sh** - Verification script:
   - Checks system dependencies
   - Creates required directories
   - Verifies Python packages

7. **setup_diffrhythm2_src.sh** - Source code setup:
   - Clones DiffRhythm2 repository
   - Sets up required directory structure
   - Runs automatically on first launch

8. **deploy_hf.ps1** - Deployment automation:
   - Clones Space repository
   - Copies all files
   - Creates .gitignore
   - Commits and pushes changes

9. **DEPLOY_HF.md** - Detailed deployment guide

## 🎯 Quick Deployment (3 Steps)

### Option A: Automated (PowerShell)

```powershell
# From project root directory
.\deploy_hf.ps1 -Clone -Push
```

This will:
1. Clone your HuggingFace Space repository
2. Copy all necessary files
3. Commit and push to HuggingFace

### Option B: Manual

1. **Clone your Space**:
   ```bash
   git clone https://huggingface.co/spaces/Gamahea/lemm-test-100
   cd lemm-test-100
   ```

2. **Copy files** (run from project root):
   ```powershell
   Copy-Item app.py lemm-test-100/
   Copy-Item hf_config.py lemm-test-100/
   Copy-Item requirements_hf.txt lemm-test-100/requirements.txt
   Copy-Item packages.txt lemm-test-100/
   Copy-Item README_HF.md lemm-test-100/README.md
   Copy-Item pre_startup.sh lemm-test-100/
   Copy-Item setup_diffrhythm2_src.sh lemm-test-100/
   Copy-Item -Recurse backend lemm-test-100/
   ```

3. **Deploy**:
   ```bash
   cd lemm-test-100
   git add .
   git commit -m "Deploy Music Generation Studio"
   git push
   ```

### Option C: Web Interface

1. Go to https://huggingface.co/spaces/Gamahea/lemm-test-100
2. Click "Files and versions"
3. Upload files manually:
   - app.py
   - hf_config.py
   - requirements.txt (use requirements_hf.txt)
   - packages.txt
   - README.md (use README_HF.md)
   - pre_startup.sh
   - setup_diffrhythm2_src.sh
   - backend/ directory (all contents)

## 📝 What Happens After Push

1. **Build Phase** (~5-10 minutes):
   - Installs system packages (espeak-ng, ffmpeg)
   - Installs Python dependencies
   - Runs setup scripts
   - Clones DiffRhythm2 source code

2. **First Run** (~5 minutes):
   - Downloads DiffRhythm2 model from HuggingFace
   - Downloads MuQ-MuLan model
   - Initializes services
   - Space becomes available

3. **Subsequent Runs**:
   - Models are cached
   - Faster startup (~1 minute)

## ⚠️ Important Notes

### Performance Expectations

- **Free Tier** (CPU-only):
  - 10-second clip: ~2 minutes
  - 30-second clip: ~4 minutes
  - 60-second clip: ~8 minutes

- **Recommended Settings**:
  - Start with 10-20 second clips
  - Use shorter durations for testing
  - Upgrade to GPU tier for faster generation

### Model Downloads

All models download automatically on first run:
- DiffRhythm2: ~1.5 GB
- MuQ-MuLan: ~500 MB
- DiffRhythm2 source: ~50 MB

HuggingFace caches these, so they only download once.

### System Dependencies

The following are automatically installed via `packages.txt`:
- **espeak-ng**: Text-to-phoneme conversion for lyrics
- **ffmpeg**: Audio format conversion
- **libsndfile1**: Audio file I/O

## 🐛 Troubleshooting

If the Space fails to build or run:

1. **Check Build Logs**:
   - Go to your Space on HuggingFace
   - Click "Logs" tab
   - Look for red error messages

2. **Common Issues**:

   **espeak-ng not found**:
   - Verify packages.txt is uploaded
   - Check pre_startup.sh output in logs

   **Module import errors**:
   - Verify requirements.txt matches requirements_hf.txt
   - Check Python version is 3.11

   **DiffRhythm2 source not found**:
   - Check setup_diffrhythm2_src.sh execution in logs
   - Verify git is available in Space

   **Model download timeout**:
   - Free tier has limited resources
   - May need to wait and retry

3. **Getting Help**:
   - Check full deployment guide: DEPLOY_HF.md
   - Review HuggingFace Spaces documentation
   - Check model repositories for issues

## 📚 Files Modified for HuggingFace Compatibility

The following backend files were updated to work in HuggingFace Spaces:

- **backend/services/diffrhythm_service.py**:
  - Now checks `PHONEMIZER_ESPEAK_PATH` environment variable
  - Falls back to local espeak-ng if not set
  - Compatible with both system and bundled espeak-ng

- **app.py**:
  - Imports hf_config first to set environment
  - Runs setup_diffrhythm2_src.sh automatically
  - Uses relative paths for HuggingFace compatibility

## 🎉 Next Steps

1. **Deploy** using one of the methods above
2. **Monitor** the build process in HuggingFace Logs
3. **Test** with a simple prompt:
   - Prompt: "upbeat pop song"
   - Duration: 10 seconds
   - Vocals: Instrumental (fastest)
4. **Share** your Space with others!

## 📖 Additional Resources

- **Deployment Guide**: DEPLOY_HF.md
- **HuggingFace Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **DiffRhythm2 Model**: https://huggingface.co/ASLP-lab/DiffRhythm2
- **MuQ-MuLan Model**: https://huggingface.co/OpenMuQ/MuQ-MuLan-large

---

**Ready to deploy?** Run `.\deploy_hf.ps1 -Clone -Push` and watch your Space come to life! 🚀
