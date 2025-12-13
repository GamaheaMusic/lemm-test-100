# DiffRhythm 2 Installation Guide

## Overview
DiffRhythm 2 is a text-to-music-with-vocals model that generates complete songs including instrumentals and vocals from text prompts and lyrics.

## Prerequisites

### 1. Install espeak-ng (Local Installation)

espeak-ng is required for text-to-phoneme conversion. The setup script will copy it to the project folder.

**First-time setup:**

1. Install espeak-ng system-wide (temporarily, for copying):

```powershell
# Using Chocolatey (recommended)
choco install espeak-ng

# OR using Scoop
scoop install espeak-ng

# OR manual download
# Visit: https://github.com/espeak-ng/espeak-ng/releases
```

2. Run the setup script to copy espeak-ng to project:

```powershell
.\setup_diffrhythm2.ps1
```

This will copy espeak-ng from the system installation to `external/espeak-ng/` in your project folder, making it portable.

**Already have espeak-ng in the project?**

The setup script will detect it automatically. No system-wide installation needed.

## Python Dependencies

### Install all requirements:
```bash
pip install -r requirements.txt
```

This will install:
- phonemizer for text-to-phoneme conversion
- muq (MuQ-MuLan) for style encoding from GitHub
- transformers, torch, torchaudio
- Audio processing libraries (pedalboard, librosa, soundfile)
- Chinese text processing (jieba, pypinyin, cn2an)

### Clone DiffRhythm 2 Repository

DiffRhythm 2 is not a pip package - clone it for direct code access:
```powershell
git clone https://github.com/ASLP-lab/DiffRhythm2.git models/diffrhythm2_source
```

Or use the automated setup script:
```powershell
.\setup_diffrhythm2.ps1
```

## Model Download

Models will be automatically downloaded from HuggingFace on first run:
- DiffRhythm 2 Repository: `ASLP-lab/DiffRhythm2` (~2-4GB)
- MuQ-MuLan Repository: `OpenMuQ/MuQ-MuLan-large` (~1-2GB)
- Location: `models/diffrhythm2/` and `models/muq-mulan/`

To pre-download:
```python
from huggingface_hub import snapshot_download

# DiffRhythm 2 model weights
snapshot_download(
    repo_id="ASLP-lab/DiffRhythm2",
    local_dir="./models/diffrhythm2"
)

# MuQ-MuLan style encoder
snapshot_download(
    repo_id="OpenMuQ/MuQ-MuLan-large",
    local_dir="./models/muq-mulan"
)
```

Or use the automated setup script:
```powershell
.\setup_diffrhythm2.ps1
```

## Troubleshooting

### espeak-ng not found
- Run setup script: `.\setup_diffrhythm2.ps1`
- Check if `external\espeak-ng\espeak-ng.exe` exists
- If not, install system-wide first, then run setup script to copy

### CUDA/GPU Issues
- DiffRhythm 2 works with CUDA (NVIDIA) or CPU
- For AMD GPUs, use CPU mode (DirectML not supported by DiffRhythm 2)
- Set `device="cpu"` in settings.py if needed

### Memory Issues
- Minimum RAM: 8GB
- Recommended: 16GB+
- GPU VRAM: 4GB+ recommended for GPU mode

## Verify Installation

Run the test script:
```bash
python test_diffrhythm2.py
```

Expected output: 30-second audio clip with vocals and instrumentals in `outputs/music/`

## Configuration

Edit `backend/config/settings.py`:
```python
DIFFRHYTHM_DEVICE = "cuda"  # or "cpu"
DIFFRHYTHM_SAMPLE_RATE = 24000
DIFFRHYTHM_CFG_SCALE = 2.0
DIFFRHYTHM_NUM_STEPS = 16
```

## Usage

See [QUICKSTART.md](QUICKSTART.md) for usage examples.
