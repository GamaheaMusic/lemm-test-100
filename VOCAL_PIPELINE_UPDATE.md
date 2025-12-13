# 🎤 Vocal Pipeline Update - Testing Guide

## Latest Update: Gradio UI Improvements (December 11, 2025)

### ✅ UI Cleanup & Fixes
1. **Removed redundant "Lyrics Style" dropdown** - Prompt analysis handles this automatically
2. **Removed "Add Vocals" checkbox** - Vocals are auto-detected or added when lyrics provided
3. **Fixed timeline not updating** - Clips now properly appear in Timeline tab after generation
4. **Optimized auto-lyrics generation** - Better performance on CPU
5. **Cleaned up interface** - Removed unnecessary textboxes and controls

### 🎯 Simplified Workflow
- Just enter a prompt and generate!
- Vocals automatically added when:
  - You provide lyrics manually, OR
  - Prompt implies vocals (detected by analysis)
- Genre, BPM, mood all auto-detected from prompt

---

## What Changed

### ✅ Removed Fish Speech
- Fish Speech service has been completely removed
- DiffRhythm now handles BOTH instrumentals and vocals in a single generation step

### ✅ Added Intelligent Prompt Analysis
- New `PromptAnalyzer` utility automatically extracts:
  - **Genre/Style** (pop, rock, jazz, etc.)
  - **BPM/Tempo** (slow, fast, specific BPM)
  - **Mood** (happy, sad, energetic, etc.)
  - **Instruments** (guitar, piano, drums, etc.)
  - **Style tags** (vintage, modern, acoustic, etc.)

### ✅ Enhanced Lyrics Generation
- LyricsMind now uses prompt analysis for better context
- Automatically adapts to detected genre and mood
- More accurate and appropriate lyrics

### ✅ Streamlined API
- Single generation call creates music with vocals
- Auto-lyrics feature: enable vocals without providing lyrics
- Returns analysis data for transparency

## How to Test

### 1. Basic Instrumental Generation
```bash
# Start the backend
.\start_backend.ps1

# In another terminal, test with curl:
curl -X POST http://localhost:5000/api/generation/generate-music \
  -H "Content-Type: application/json" \
  -d '{"prompt": "upbeat pop song with drums", "duration": 30}'
```

### 2. Music with Manual Lyrics
```bash
curl -X POST http://localhost:5000/api/generation/generate-music \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "energetic rock song at 140 BPM",
    "lyrics": "Standing tall, we never fall\nRocking out, hear the call",
    "use_vocals": true,
    "duration": 30
  }'
```

### 3. Auto-Generated Lyrics
```bash
curl -X POST http://localhost:5000/api/generation/generate-music \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "romantic ballad about summer love",
    "use_vocals": true,
    "auto_lyrics": true,
    "duration": 30
  }'
```

### 4. Test Prompt Analysis
```bash
curl -X POST http://localhost:5000/api/generation/generate-lyrics \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "dark atmospheric electronic music with synth",
    "duration": 30
  }'

# Response will include analysis data:
# {
#   "success": true,
#   "lyrics": "...",
#   "analysis": {
#     "genre": "electronic",
#     "bpm": 120,
#     "mood": "dark",
#     "instruments": ["synth"],
#     "style_tags": ["atmospheric", "electronic"]
#   }
# }
```

### 5. Using the Frontend
1. Open http://localhost:8000 in your browser
2. Enter a music prompt (e.g., "happy upbeat pop song")
3. Check "Add Vocals" checkbox
4. Click "Generate Music Clip" (lyrics will auto-generate)
5. OR click "Auto Gen Lyrics" first to preview/edit lyrics

## Expected Behavior

### Without Vocals
- Generates instrumental music based on prompt
- Analysis extracts genre, BPM, mood for better results
- Fast generation

### With Vocals (Manual Lyrics)
- Combines your lyrics with the music
- Single generation step
- DiffRhythm creates both music and vocals

### With Vocals (Auto-Lyrics)
1. Analyzes your prompt
2. Generates appropriate lyrics using detected genre/mood
3. Creates music with vocals in one step
4. Returns both the audio and the generated lyrics

## Prompt Analysis Examples

| Prompt | Detected Genre | BPM | Mood |
|--------|---------------|-----|------|
| "upbeat pop song with drums" | pop | 120-140 | energetic |
| "slow emotional ballad" | pop | 60-80 | sad |
| "heavy metal with aggressive guitar" | metal | 150-180 | angry |
| "chill ambient electronic music" | electronic | 90-110 | calm |
| "fast-paced jazz with saxophone" | jazz | 140-180 | energetic |

## Troubleshooting

### Vocals Not Present in Output
- Make sure `use_vocals: true` is set
- Provide lyrics OR set `auto_lyrics: true`
- Check backend logs for errors in DiffRhythm generation

### Poor Quality Lyrics
- Try more specific prompts with genre/mood
- Manual lyrics give you full control
- Check that prompt analysis detected correct genre (see response)

### Generation Fails
- Check that models are downloaded: `python setup_models.py`
- Verify backend is running: http://localhost:5000/api/generation/status
- Check logs in `logs/app.log`

## API Reference

### POST /api/generation/generate-music
```json
{
  "prompt": "string (required) - Music description",
  "lyrics": "string (optional) - Manual lyrics",
  "use_vocals": "boolean (default: false) - Include vocals",
  "auto_lyrics": "boolean (default: false) - Auto-generate lyrics",
  "duration": "integer (10-120, default: 30) - Seconds"
}
```

**Response:**
```json
{
  "success": true,
  "clip_id": "uuid",
  "file_path": "path/to/audio.wav",
  "duration": 30,
  "analysis": {
    "genre": "pop",
    "bpm": 120,
    "mood": "happy",
    "genres": ["pop"],
    "moods": ["happy"],
    "instruments": ["drums", "guitar"],
    "style_tags": ["upbeat"]
  },
  "generated_lyrics": "lyrics text (if auto_lyrics was true)"
}
```

### POST /api/generation/generate-lyrics
```json
{
  "prompt": "string (required) - Lyrics theme",
  "style": "string (optional) - Genre/style (auto-detected if omitted)",
  "duration": "integer (10-120, default: 30) - Seconds"
}
```

**Response:**
```json
{
  "success": true,
  "lyrics": "generated lyrics text",
  "analysis": {
    "genre": "detected genre",
    "bpm": 120,
    "mood": "detected mood"
  }
}
```

## Files Changed

### New Files
- ✨ `backend/utils/prompt_analyzer.py` - Prompt analysis utility

### Modified Files
- 📝 `backend/services/diffrhythm_service.py` - Now accepts lyrics
- 📝 `backend/services/lyricmind_service.py` - Uses prompt analysis
- 📝 `backend/routes/generation.py` - Auto-lyrics + analysis
- 📝 `backend/models/schemas.py` - Added auto_lyrics field
- 📝 `backend/config/settings.py` - Removed Fish Speech path
- 📝 `backend/services/__init__.py` - Removed Fish Speech import
- 📝 `frontend/js/app.js` - Auto-lyrics support
- 📝 `frontend/index.html` - Updated UI text

### Deprecated Files (Can be Deleted)
- ❌ `backend/services/fish_speech_service.py`
- ❌ `models/fish_speech/` directory (optional, can keep for future)

## Next Steps

1. Test the updated pipeline with various prompts
2. Verify vocals are present in generated audio
3. Experiment with prompt analysis to see detected attributes
4. Fine-tune prompts based on analysis feedback
5. Delete deprecated Fish Speech files if not needed

## Notes

- The type checking errors in service files are **false positives** (models are lazily loaded)
- DiffRhythm may still use MusicGen as fallback - both support vocals
- Analysis improves results but is not strictly required
- All old API endpoints still work (backward compatible)
