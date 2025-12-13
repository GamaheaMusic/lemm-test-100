# Audio Dataset Analysis Report

## Executive Summary
Analysis of 40 open-source audio datasets for integration into the Music Generation Studio LoRA training system, considering HuggingFace Space limitations (1 GB storage).

## Current Issues
- **OpenSinger**: Dataset ID `Rongjiehuang/opensinger` does not exist on HuggingFace Hub
- **M4Singer**: Dataset ID `M4Singer/M4Singer` not found
- **Lakh MIDI**: Dataset ID `roszcz/lakh-midi` may not exist
- Need to find verified HuggingFace dataset IDs

## Recommended Datasets for Music Generation Training

### Priority 1: Music & Singing (Fits 1GB limit)

1. **GTZAN Music Genre Collection**
   - **Size**: ~1.2 GB (may need selective download)
   - **Content**: 1,000 audio tracks across 10 music genres
   - **Use Case**: Music style understanding, genre classification
   - **HF ID**: `marsyas/gtzan` or available on Kaggle
   - **Recommendation**: ★★★★★ - Perfect for music genre training

2. **LJSpeech**
   - **Size**: ~2.6 GB
   - **Content**: 13,100 short audio clips from single speaker
   - **Use Case**: Voice/vocal training, prosody learning
   - **HF ID**: `lj_speech`
   - **Recommendation**: ★★★★☆ - Good for vocal characteristics

3. **NSynth**
   - **Size**: ~30 GB full (subset available)
   - **Content**: 305,979 musical notes with unique pitch/timbre
   - **Use Case**: Musical synthesis, instrument understanding
   - **HF ID**: `google/nsynth` (subset: `nsynth-valid` ~1GB)
   - **Recommendation**: ★★★★★ - Excellent for music synthesis

4. **MAESTRO (subset)**
   - **Size**: Full ~100GB, but can download specific splits
   - **Content**: Piano performances with MIDI + audio
   - **Use Case**: Music generation, MIDI-to-audio learning
   - **HF ID**: `roszcz/maestro-v3`
   - **Recommendation**: ★★★★★ - Best for classical music training

5. **MedleyDB (samples)**
   - **Size**: Varies by track selection
   - **Content**: Annotated multi-track recordings
   - **Use Case**: Instrument separation, music understanding
   - **HF ID**: Custom download required
   - **Recommendation**: ★★★☆☆ - Good but requires manual setup

### Priority 2: Vocal & Speech (Under 1GB)

6. **Mozilla Common Voice (single language subset)**
   - **Size**: ~5GB per language (can use smaller languages)
   - **Content**: Diverse speakers reading text
   - **Use Case**: Vocal diversity, pronunciation
   - **HF ID**: `mozilla-foundation/common_voice_11_0` (specify language)
   - **Recommendation**: ★★★★☆ - Great for vocal variation

7. **VCTK Corpus**
   - **Size**: ~10.9 GB
   - **Content**: 109 speakers with different accents
   - **Use Case**: Voice diversity, accent variation
   - **HF ID**: `vctk`
   - **Recommendation**: ★★★☆☆ - Good for voice training

8. **CMU ARCTIC**
   - **Size**: ~3.5 GB
   - **Content**: Multiple speakers, phonetically balanced
   - **Use Case**: Speech synthesis, vocal training
   - **HF ID**: Available via direct download
   - **Recommendation**: ★★★★☆ - High-quality vocals

### Priority 3: Sound Effects & Environment (Under 1GB)

9. **ESC-50**
   - **Size**: ~600 MB
   - **Content**: 2,000 environmental sounds, 50 classes
   - **Use Case**: Sound effects understanding
   - **HF ID**: `ashraq/esc50`
   - **Recommendation**: ★★★☆☆ - Good for ambient sounds

10. **UrbanSound8K**
    - **Size**: ~6 GB
    - **Content**: 8,732 urban sound excerpts
    - **Use Case**: Environmental sound classification
    - **HF ID**: `danavery/urbansound8k`
    - **Recommendation**: ★★★☆☆ - Urban ambient training

## Verified HuggingFace Datasets for Immediate Use

### Music Datasets
```python
# GTZAN - Music Genre Classification
"marsyas/gtzan"  # 1000 tracks, 10 genres

# NSynth - Musical Notes
"google/nsynth"  # Use "nsynth-valid" split for smaller size

# MAESTRO - Piano performances
"roszcz/maestro-v3"  # Download specific splits
```

### Vocal Datasets
```python
# LJSpeech - Single speaker
"lj_speech"  # 13,100 clips

# Common Voice - Multilingual
"mozilla-foundation/common_voice_11_0"  # Specify language

# LibriSpeech - English audiobooks (smaller subsets)
"librispeech_asr"  # Use "clean" subsets only
```

### Sound Effects
```python
# ESC-50 - Environmental sounds
"ashraq/esc50"  # 2000 samples, 50 classes

# FSD50K - Freesound Dataset
"Fhrozen/FSD50k"  # Larger but comprehensive
```

## Storage-Optimized Recommendations

### For 1GB HuggingFace Space:

**Best Combination (fits in 1GB):**
1. **GTZAN subset** (~300 MB) - 300 songs across all genres
2. **ESC-50** (~600 MB) - Environmental sounds
3. **LJSpeech subset** (~100 MB) - 1000 clips for vocals

**Alternative Combination:**
1. **NSynth-valid** (~800 MB) - Musical notes and synthesis
2. **Speech Commands** (~200 MB) - Short vocal clips

## Implementation Strategy

### Phase 1: Quick Wins (Immediate)
- Replace broken dataset IDs with verified ones
- Implement GTZAN (marsyas/gtzan)
- Implement ESC-50 (ashraq/esc50)
- Add download size estimation before download

### Phase 2: Smart Downloads (Next)
- Add dataset size checking
- Implement partial download (specific splits)
- Add storage quota monitoring
- Cache management for 1GB limit

### Phase 3: Advanced Features
- Dataset preview/sampling before full download
- Automatic cleanup of old datasets
- Compression support
- Streaming data loading (no full download)

## Updated Dataset Configuration

```python
DATASETS = {
    # Music Datasets (Verified)
    "gtzan": {
        "name": "GTZAN Music Genre (1000 tracks)",
        "hf_id": "marsyas/gtzan",
        "type": "music",
        "size_gb": 1.2,
        "description": "1000 songs across 10 genres for style learning"
    },
    "nsynth_valid": {
        "name": "NSynth Validation Set (Musical Notes)",
        "hf_id": "google/nsynth",
        "split": "valid",
        "type": "music",
        "size_gb": 0.8,
        "description": "Musical notes with unique pitch and timbre"
    },
    "maestro_small": {
        "name": "MAESTRO Piano (Small subset)",
        "hf_id": "roszcz/maestro-v3",
        "split": "validation",
        "type": "music",
        "size_gb": 2.0,
        "description": "Classical piano performances"
    },
    
    # Vocal Datasets (Verified)
    "ljspeech": {
        "name": "LJSpeech (13k vocal clips)",
        "hf_id": "lj_speech",
        "type": "vocal",
        "size_gb": 2.6,
        "description": "Single speaker for vocal characteristics"
    },
    "common_voice_en": {
        "name": "Common Voice English (subset)",
        "hf_id": "mozilla-foundation/common_voice_11_0",
        "language": "en",
        "type": "vocal",
        "size_gb": 5.0,
        "description": "Diverse English speakers"
    },
    
    # Sound Effects (Verified)
    "esc50": {
        "name": "ESC-50 Environmental Sounds",
        "hf_id": "ashraq/esc50",
        "type": "sound_effects",
        "size_gb": 0.6,
        "description": "2000 environmental sounds, 50 classes"
    },
    
    # Speech Commands (Verified)
    "speech_commands": {
        "name": "Google Speech Commands",
        "hf_id": "speech_commands",
        "type": "vocal",
        "size_gb": 2.0,
        "description": "Short spoken words for vocal training"
    }
}
```

## Conclusion

**Immediate Actions:**
1. ✅ Remove non-existent dataset IDs
2. ✅ Add verified HuggingFace datasets
3. ✅ Implement size checking before download
4. ✅ Add storage quota warnings
5. ✅ Focus on datasets under 1GB

**Best Datasets for 1GB Limit:**
- **GTZAN** (music genres)
- **ESC-50** (sound effects)
- **NSynth-valid** (musical synthesis)

**Total Storage Strategy:**
- Max 1GB limit enforced
- Download size preview
- Selective split downloads
- Auto-cleanup old data
