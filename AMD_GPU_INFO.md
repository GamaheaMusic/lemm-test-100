# AMD GPU Support - Important Information

## Current Status: CPU Mode

Your system is currently running in **CPU mode** because:

1. **Python 3.13.3** is installed
2. **torch-directml** currently supports up to Python 3.11

## What This Means

✅ **The app will work perfectly** - just slower
- Music generation: 1-3 minutes per clip (CPU)
- Lyrics generation: 20-60 seconds (CPU)
- Everything functions normally, just takes longer

## To Enable AMD GPU (Vega 8) Support

You have two options:

### Option 1: Use Python 3.11 (Recommended for GPU)

1. **Install Python 3.11** from python.org
2. **Create new virtual environment**:
   ```powershell
   python3.11 -m venv .venv311
   .\.venv311\Scripts\Activate.ps1
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
4. **Run the app**:
   ```powershell
   python app_gradio.py
   ```

### Option 2: Continue with CPU (Current Setup)

Just use the app as-is! It works fine, just slower.

```powershell
.\start_app.ps1
```

## Performance Comparison

| Mode | Music Gen | Lyrics Gen | First Load |
|------|-----------|------------|------------|
| **AMD GPU (DirectML)** | 15-30s | 5-10s | 20s |
| **CPU (Current)** | 60-180s | 20-60s | 30s |

## DirectML Compatibility

torch-directml version compatibility:
- ✅ Python 3.8 - 3.11: Fully supported
- ⚠️ Python 3.12+: Limited/No support (as of Dec 2024)
- 📅 Future: May be updated for Python 3.13+

## Recommendation

**For casual use**: Continue with CPU mode - it works fine!

**For frequent use**: Consider Python 3.11 environment for GPU acceleration

## Current Configuration

Your `.env` file is configured for DirectML, but the app will automatically fallback to CPU if DirectML is unavailable.

```env
USE_DIRECTML=True  # Will try GPU, fallback to CPU
DEVICE=directml     # Preferred device
```

You don't need to change anything - the app handles this automatically!

## Testing

Run this to check your current setup:
```powershell
python test_setup.py
```

---

**Bottom line**: Your app is ready to use! AMD GPU support would make it faster, but it's not required for functionality.
