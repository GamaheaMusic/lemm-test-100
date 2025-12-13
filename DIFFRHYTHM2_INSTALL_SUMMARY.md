# DiffRhythm 2 Installation Summary

## Installation Status: ✅ COMPLETE & PORTABLE

### What Was Done

1. **Fixed Requirements.txt**
   - Removed incorrect `git+https://github.com/ASLP-lab/DiffRhythm2.git` line
   - DiffRhythm2 is NOT a pip package - it's source code that must be cloned
   - Added correct dependencies: phonemizer, jieba, pypinyin, cn2an, onnxruntime

2. **Cloned DiffRhythm2 Repository**
   - Location: `models/diffrhythm2_source/`
   - Contains all source code: diffrhythm2/cfm.py, bigvgan/, g2p/, etc.
   - Size: ~77MB with all source files

3. **Installed Python Dependencies**
   - ✅ phonemizer>=3.2.0
   - ✅ jieba>=0.42.0 (Chinese text segmentation)
   - ✅ pypinyin>=0.50.0 (Chinese to pinyin)
   - ✅ cn2an>=0.5.0 (Chinese number conversion)
   - ✅ onnxruntime>=1.15.0 (for g2p Chinese model)

4. **Downloaded Models**
   - ✅ DiffRhythm2 config.json from HuggingFace
   - Models will auto-download on first use (~2-4GB)

5. **System Dependencies - NOW PORTABLE!**
   - ✅ espeak-ng 1.52.0 copied to `external/espeak-ng/`
   - ✅ Includes libespeak-ng.dll (phonemizer library dependency)
   - ✅ Fully portable - no system PATH modification needed
   - ✅ Environment variables set automatically by startup scripts

### Installation Complete!
```powershell
.\.venv\Scripts\python.exe test_diffrhythm2.py
```

Should show all tests passing.

### How It Works

All startup scripts automatically set these environment variables:
- `ESPEAK_DATA_PATH`: Points to local `external/espeak-ng/espeak-ng-data`
- `PATH`: Includes `external/espeak-ng` for executables
- `PHONEMIZER_ESPEAK_LIBRARY`: Points to `external/espeak-ng/libespeak-ng.dll`

No system PATH modification needed. Project is fully portable!

### Usage

```powershell
# Just run:
.\start_app.ps1
```

All dependencies are automatically configured.

### Verification

Run the test script:
```powershell
.\start_app.ps1
```

### File Changes Made

- ✅ [requirements.txt](requirements.txt) - Fixed DiffRhythm2 dependencies
- ✅ [backend/services/diffrhythm_service.py](backend/services/diffrhythm_service.py) - Updated to use cloned repo
- ✅ [setup_diffrhythm2.ps1](setup_diffrhythm2.ps1) - Automated setup script
- ✅ [test_diffrhythm2.py](test_diffrhythm2.py) - Installation verification
- ✅ [INSTALL_DIFFRHYTHM2.md](INSTALL_DIFFRHYTHM2.md) - Complete installation guide
- ✅ [CHANGES.md](CHANGES.md) - Updated with DiffRhythm 2 migration details

### Models Location

```external/
└── espeak-ng/                 # Local espeak-ng (portable) ✅
    ├── espeak-ng.exe
    └── espeak-ng-data/
models/
├── diffrhythm2/           # Model weights (auto-downloaded)
│   └── models--ASLP-lab--DiffRhythm2/
│       └── snapshots/
│           └── 9aa157.../
│               ├── config.json ✅
│               ├── model.safetensors (will download)
│               ├── decoder.bin (will download)
│               └── decoder.json (will download)
├── diffrhythm2_source/    # Source code (cloned) ✅
│   ├── diffrhythm2/
│   ├── bigvgan/
│   ├── g2p/
│   └── inference.py
└── muq-mulan/             # MuQ-MuLan style encoder (will download)
```

### Architecture Overview

**DiffRhythm 2 Pipeline:**
```
User Prompt → Prompt Analyzer → Genre/BPM/Mood
                ↓
Lyrics (optional) → g2p Tokenizer → Phoneme Tokens
                ↓
Style Prompt → MuQ-MuLan → Style Embedding
                ↓
CFM + DiT Transformer → Latent Representation
                ↓
BigVGAN Decoder → Audio Output (vocals + instrumentals)
```

### Next Actions

1. **Add espeak-ng to PATH** (see options above)
2. **Restart VS Code/PowerShell** to pick up PATH changes
3. **Run test script**: `.\.venv\Scripts\python.exe test_diffrhythm2.py`
4. **Start the app**: `.\start_app.ps1`

### Troubleshooting

**espeak-ng not found:**
- Run: `.\setup_diffrhythm2.ps1` to copy espeak-ng to project
- Check: `external\espeak-ng\espeak-ng.exe` exists
- Location is portable - no system PATH needed

**Import errors:**
- Run: `pip install -r requirements.txt` in virtual environment
- Ensure `.venv` is activated

**Model download slow:**
- First run will download ~2-4GB of models
- Progress shown in console
- Models cached for future use

### References

- [DiffRhythm 2 GitHub](https://github.com/ASLP-lab/DiffRhythm2)
- [DiffRhythm 2 Paper](https://arxiv.org/pdf/2510.22950)
- [DiffRhythm 2 Demo](https://aslp-lab.github.io/DiffRhythm2.github.io)
- [Models on HuggingFace](https://huggingface.co/ASLP-lab/DiffRhythm2)
- [MuQ-MuLan on HuggingFace](https://huggingface.co/OpenMuQ/MuQ-MuLan-large)
