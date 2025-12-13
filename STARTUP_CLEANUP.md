# Startup Cleanup & Streamlining - Dec 11, 2025

## Summary

Consolidated multiple startup scripts into a single unified launcher that handles all setup and startup tasks automatically.

## Changes Made

### 1. Removed Duplicate Scripts ✓
Deleted unnecessary startup files:
- `start_app.ps1`
- `start_app.bat`
- `start_backend.ps1`
- `start_backend.bat`
- `setup_models.ps1`
- `setup_models.bat`
- `setup_diffrhythm2.ps1`

### 2. Created Unified Launcher ✓

**launch.ps1** (Windows) - New comprehensive launcher:
- Automatic virtual environment creation
- Smart dependency installation (only when needed)
- Flag-based caching to avoid redundant installs
- Sequential startup: Requirements → Models → Backend → Frontend
- Proper health checking with configurable timeout
- Graceful shutdown on Ctrl+C
- Clear status messages and error reporting

**launch.sh** (Linux/Mac) - Already existed, compatible with new flow

### 3. Fixed Backend Issues ✓

**Problem**: Backend was crashing due to torch version conflicts
- torch-directml requires torch==2.4.1
- DiffRhythm2 requires torch>=2.4.0 (we installed 2.7.0)

**Solution**:
- Updated `requirements.txt` to use torch>=2.4.0 (CPU mode)
- Removed torch-directml dependency to avoid conflicts
- Updated `diffrhythm_service.py` device selection to use CPU by default
- Added clear notes about GPU support limitations

**Result**: Backend now starts successfully with CPU mode

### 4. Updated Requirements ✓

**requirements.txt** changes:
```diff
- torch-directml==0.2.5.dev240914
- torchaudio==2.4.1
+ torch>=2.4.0
+ torchaudio>=2.4.0
+ # Note: DiffRhythm2 requires torch>=2.4 which is incompatible with torch-directml
+ # Using CPU mode for compatibility
```

### 5. Created New Documentation ✓

**QUICKSTART.md** - Complete user guide:
- Simple one-command startup instructions
- Explanation of what launcher does
- First-run expectations
- Step-by-step usage guide
- Common troubleshooting scenarios
- Manual setup option for advanced users

## Current Startup Flow

1. **User runs**: `.\launch.ps1`

2. **Launcher checks**:
   - Python 3.11 virtual environment
   - Dependencies installed (via flag file)
   - DiffRhythm2 dependencies (via flag file)
   - AI models present (warns if missing)

3. **Launcher starts**:
   - Backend server (with health check)
   - Frontend server (blocking)

4. **User accesses**: http://localhost:8000

## Benefits

✅ **Simplified**: One command to rule them all  
✅ **Intelligent**: Only installs/updates when needed  
✅ **Reliable**: Proper error checking and reporting  
✅ **Fast**: Skip reinstalls on subsequent runs  
✅ **Clean**: Removed 7 redundant script files  
✅ **Cross-platform**: Windows (PowerShell) and Linux/Mac (Bash)  

## File Structure Now

```
Angen/
├── launch.ps1          # Windows launcher (MAIN ENTRY POINT)
├── launch.sh           # Linux/Mac launcher (MAIN ENTRY POINT)
├── QUICKSTART.md       # User guide
├── requirements.txt    # Updated dependencies
└── backend/
    └── run.py          # Backend server
```

## Testing

✅ Virtual environment creation  
✅ Dependency installation  
✅ Backend health check  
✅ Backend starts successfully  
✅ Frontend serves correctly  
✅ Graceful shutdown  

## Known Issues

- **DirectML disabled**: AMD GPU acceleration not available due to version conflicts
- **CPU mode**: Generation slower but more stable
- **First generation**: Models download automatically (~5GB)

## Future Improvements

- Consider CUDA support for NVIDIA GPUs
- Optimize CPU performance
- Add model pre-download option
- Create Windows installer (.exe)
