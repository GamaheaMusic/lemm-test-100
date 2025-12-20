---
title: Music Generation Studio
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎵 Music Generation Studio

Create AI-powered music with intelligent prompt analysis and context-aware generation using DiffRhythm2 and LyricMind AI.

**⚠️ Important:** 
- This Space requires ZeroGPU to run
- **You must be logged in** to HuggingFace to use GPU features
- Free users get daily ZeroGPU quota - check your usage at https://huggingface.co/settings/billing
- If you see quota errors while logged in, try duplicating this Space to your account

## Features

- **Intelligent Music Generation**: DiffRhythm2 model for high-quality music with vocals
- **Smart Lyrics Generation**: LyricMind AI for context-aware lyric creation
- **Prompt Analysis**: Automatically detects genre, BPM, and mood from your description
- **Flexible Vocal Modes**:
  - Instrumental: Pure music without vocals
  - User Lyrics: Provide your own lyrics
  - Auto Lyrics: AI-generated lyrics based on prompt
- **Timeline Management**: Build complete songs clip-by-clip
- **Export**: Download your creations in WAV, MP3, or FLAC formats

## How to Use

1. **Generate Music**:
   - Enter a descriptive prompt (e.g., "energetic rock song with electric guitar at 140 BPM")
   - Choose vocal mode (Instrumental, User Lyrics, or Auto Lyrics)
   - Set duration (10-120 seconds)
   - Click "Generate Music Clip"

2. **Manage Timeline**:
   - View all generated clips in the timeline
   - Remove specific clips or clear all
   - Clips are arranged sequentially

3. **Export**:
   - Enter a filename
   - Choose format (WAV recommended for best quality)
   - Download your complete song

## Models

- **DiffRhythm2**: Music generation with integrated vocals ([ASLP-lab/DiffRhythm2](https://huggingface.co/ASLP-lab/DiffRhythm2))
- **MuQ-MuLan**: Music style encoding ([OpenMuQ/MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large))

## Performance

⏱️ Generation time: ~2-4 minutes per 30-second clip on CPU (HuggingFace Spaces free tier)

💡 Tip: Start with shorter durations (10-20 seconds) for faster results

## Technical Details

- Built with Gradio and PyTorch
- Uses DiffRhythm2 for music generation with vocals
- Employs flow-matching techniques for high-quality audio synthesis
- Supports multiple languages for lyrics (English, Chinese, Japanese)

## Credits

- DiffRhythm2 by ASLP-lab
- MuQ-MuLan by OpenMuQ
- Application interface and integration by Music Generation App Team

## License

MIT License - See LICENSE file for details
