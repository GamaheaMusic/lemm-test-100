# Style Consistency Feature

## Overview

The Music Generation App now includes **automatic style consistency** to ensure that all generated clips maintain a cohesive musical style throughout your project. Once you generate your first clip, all subsequent clips will automatically match its musical characteristics.

## How It Works

### 1. **Audio Feature Analysis**
When generating a new clip after the first one, the system:
- Analyzes all existing clips on the timeline
- Extracts comprehensive audio features including:
  - **Spectral features**: Brightness, timbre, frequency distribution
  - **Temporal features**: Energy, dynamics, rhythm patterns
  - **Harmonic features**: Chord progressions, melodic contours
  - **Percussive features**: Drum patterns, rhythmic intensity

### 2. **Style Profile Creation**
The system builds a unified style profile by:
- Computing statistical summaries of all extracted features
- Averaging characteristics across all timeline clips
- Creating a "style signature" representing your project's sound

### 3. **Reference Audio Generation**
All existing clips are mixed together into a single reference track that:
- Captures the overall sonic character of your project
- Provides concrete audio guidance (not just text descriptions)
- Serves as a style template for new generations

### 4. **Style-Guided Generation**
When generating new clips, the system:
- **Blends style embeddings**: 70% from reference audio + 30% from your text prompt
- **Enhances prompts**: Automatically adds style descriptors like "bright", "warm", "energetic"
- **Uses MuQ-MuLan**: Leverages DiffRhythm2's audio encoder for precise style matching

## Technical Implementation

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Generates First Clip                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Clip Added to Timeline (No Style Guidance)          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              User Generates Subsequent Clips                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┴──────────────┐
            │                              │
            ▼                              ▼
┌──────────────────────┐       ┌─────────────────────────┐
│ Extract Features     │       │ Create Reference Audio  │
│ from All Clips       │       │ (Mix All Clips)         │
└──────┬───────────────┘       └──────┬──────────────────┘
       │                              │
       ▼                              ▼
┌──────────────────────┐       ┌─────────────────────────┐
│ Compute Style        │       │ Encode with MuQ-MuLan   │
│ Statistics           │       │ Audio Encoder           │
└──────┬───────────────┘       └──────┬──────────────────┘
       │                              │
       └───────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│         Enhance Prompt + Blend Style Embeddings             │
│       (Reference Audio 70% + Text Prompt 30%)               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Generate New Clip with Style Guidance            │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. **StyleConsistencyService** (`style_consistency_service.py`)
- `extract_audio_features()`: Extracts librosa features from audio files
- `compute_style_statistics()`: Summarizes features into style metrics
- `analyze_timeline_style()`: Aggregates style across multiple clips
- `create_style_reference_audio()`: Mixes clips into reference track
- `enhance_prompt_with_style()`: Adds style descriptors to user prompts

#### 2. **DiffRhythmService Integration** (`diffrhythm_service.py`)
- Accepts `reference_audio` parameter in `generate()` method
- Blends MuLan embeddings: audio style + text style
- Uses weighted combination (70/30) for balanced control

#### 3. **Generation Route** (`routes/generation.py`)
- Fetches existing timeline clips before generation
- Calls style service to prepare guidance
- Passes reference audio to DiffRhythm service
- Returns style consistency metadata in response

## Feature Extraction Details

### Spectral Features
| Feature | Description | Use Case |
|---------|-------------|----------|
| **Mel Spectrogram** | Time-frequency representation | Overall timbral character |
| **Spectral Centroid** | Frequency "center of mass" | Brightness (high = bright, low = dark) |
| **Spectral Bandwidth** | Frequency range spread | Tonal vs noisy character |
| **Spectral Contrast** | Peak-to-valley differences | Harmonic richness |
| **Spectral Rolloff** | High-frequency cutoff | Presence of high frequencies |

### Temporal Features
| Feature | Description | Use Case |
|---------|-------------|----------|
| **Zero Crossing Rate** | Signal sign changes | Percussiveness, noisiness |
| **RMS Energy** | Root mean square amplitude | Loudness, dynamics |
| **Tempo** | Beats per minute | Rhythmic consistency |

### Harmonic Features
| Feature | Description | Use Case |
|---------|-------------|----------|
| **Chroma** | Pitch class distribution | Harmonic/melodic character |
| **Harmonic Ratio** | Harmonic vs total energy | Tonality vs percussion |
| **MFCC** | Mel-frequency cepstral coefficients | Timbre signature |

## Style Matching Process

### 1. Statistical Aggregation
```python
# Example: Average spectral centroid across clips
centroids = [clip.spectral_centroid_mean for clip in clips]
aggregate_centroid = np.mean(centroids)

# Classify brightness
if aggregate_centroid > 3000:
    style_descriptor = "bright"
elif aggregate_centroid < 1500:
    style_descriptor = "warm"
```

### 2. Prompt Enhancement
```python
# Original prompt
"upbeat pop song"

# Enhanced with style analysis
"upbeat pop song, consistent with existing style: bright, energetic, rhythmic"
```

### 3. Embedding Blend
```python
# Load reference audio
ref_audio = load_audio(reference_path)

# Encode both sources
audio_embed = mulan.encode_audio(ref_audio)     # Concrete style
text_embed = mulan.encode_text(prompt)          # User intent

# Weighted combination
final_embed = 0.7 * audio_embed + 0.3 * text_embed
```

## Usage Examples

### Example 1: Building a Consistent Track
```javascript
// First generation - establishes style
generateMusic("chill lofi hip hop beat");
// → Generated with no style guidance

// Second generation - matches first
generateMusic("smooth jazz piano");
// → Automatically matches tempo, brightness, dynamics of first clip
// → Result: Jazz-influenced but maintains lofi character

// Third generation - maintains consistency
generateMusic("ambient pad with strings");
// → Matches both previous clips
// → Result: Ambient texture fits the established lofi aesthetic
```

### Example 2: Genre Consistency
```javascript
// Clip 1: "dark techno bassline" 
// → Sets dark, electronic, bass-heavy style

// Clip 2: "melodic synth lead"
// → Enhanced to: "melodic synth lead, consistent with: dark, rhythmic, electronic"
// → Result: Synth matches techno aesthetic

// Clip 3: "percussion break"
// → Enhanced to match tempo, spectral character
// → Result: Percussion fits seamlessly
```

## Benefits

### 1. **Cohesive Projects**
- All clips sound like they belong together
- Consistent production quality and character
- Professional-sounding continuity

### 2. **Reduced Manual Tweaking**
- No need to specify style details repeatedly
- Automatic genre/mood matching
- Fewer failed generations

### 3. **Creative Workflow**
- Focus on musical ideas, not technical consistency
- Build complex arrangements confidently
- Experiment while maintaining coherence

### 4. **Intelligent Adaptation**
- Respects user prompts (30% weight)
- Balances consistency with variety
- Allows intentional style evolution

## Technical Details

### MuQ-MuLan Audio Encoder
MuLan (Music Language Model) is DiffRhythm2's multimodal encoder that:
- Encodes both audio and text into shared embedding space
- Trained on music-text pairs for semantic understanding
- Enables audio-to-audio style transfer
- 24kHz input sampling rate
- 512-dimensional embedding output

### Blending Ratio (70/30)
The 70% audio / 30% text split was chosen because:
- **70% audio**: Strong enough to maintain consistency
- **30% text**: Allows creative variation per user request
- Prevents both complete copying and total deviation
- Empirically balanced for music generation

### Reference Audio Creation
Mixing strategy:
- Loads all timeline clips
- Pads to maximum length
- Averages waveforms (not concatenation)
- Normalizes output
- Preserves spectral characteristics better than random sampling

## Limitations

### 1. **First Clip Sets Style**
- Initial generation has outsized influence
- Consider generating a "style seed" clip first
- Can restart project by clearing timeline

### 2. **Gradual Style Drift**
- Each new clip slightly shifts the aggregate style
- May diverge over many generations
- Solution: Periodically remove outlier clips

### 3. **Genre Boundaries**
- Strong style consistency may resist major genre changes
- For intentional genre shifts, consider clearing earlier clips
- Or use stronger text prompts with specific descriptors

### 4. **Reference Audio Mixing**
- Simple averaging may blur distinct elements
- Works best with harmonically compatible clips
- Better for similar durations

## Future Enhancements

### Planned Features
1. **Style Lock Toggle**: Option to disable auto-consistency for specific clips
2. **Manual Style Selection**: Choose which clips to use as reference
3. **Style Intensity Control**: Adjustable blend ratio (e.g., 50/50, 80/20)
4. **Multi-Style Projects**: Support separate style profiles per timeline section
5. **Style Presets**: Save/load style profiles for reuse

### Advanced Possibilities
- Real-time style preview before generation
- Style similarity visualization
- Automatic style checkpoint/tagging
- Style interpolation between clips
- Per-track style profiles in multi-track projects

## Dependencies

```python
# Required packages (already in requirements.txt)
librosa>=0.10.0        # Audio feature extraction
soundfile>=0.12.0      # Audio I/O
numpy>=1.24.0          # Numerical operations
torch>=2.0.0           # Neural network operations
torchaudio>=2.0.0      # Audio processing
```

## Performance Considerations

### Computational Cost
- **Feature extraction**: ~2-3 seconds per clip (CPU)
- **Reference audio creation**: ~1 second for 5 clips (CPU)
- **MuLan encoding**: ~0.5 seconds (GPU) / ~3 seconds (CPU)
- **Total overhead**: ~5-10 seconds before generation starts

### Optimization Strategies
- Features cached per clip (future enhancement)
- Reference audio reused if timeline unchanged
- Parallel feature extraction for multiple clips
- GPU acceleration for MuLan encoding

### Memory Usage
- Feature storage: ~50KB per clip
- Reference audio: ~5MB for 30-second mix
- MuLan embeddings: ~2KB (512 floats)
- Total: Negligible for typical projects (<100MB)

## Troubleshooting

### Style Consistency Not Working
1. **Check logs**: Look for "Style guidance ready" message
2. **Verify clip paths**: Ensure audio files exist
3. **Confirm MuLan loaded**: Check DiffRhythm2 initialization

### Unwanted Style Matching
1. Clear timeline to reset style
2. Generate first clip with very specific prompt
3. Consider reducing blend ratio in future update

### Inconsistent Results
1. Check reference audio quality (may have artifacts)
2. Verify all clips use same sample rate
3. Ensure clips have similar durations for better averaging

## API Reference

### REST Endpoint Changes

#### POST `/api/generation/generate-music`
**Response (Enhanced)**:
```json
{
    "success": true,
    "clip_id": "abc123",
    "file_path": "/outputs/music/abc123.wav",
    "duration": 30,
    "analysis": {...},
    "style_consistent": true,      // NEW
    "num_reference_clips": 3        // NEW
}
```

### Python API

#### StyleConsistencyService
```python
from services.style_consistency_service import StyleConsistencyService

style_service = StyleConsistencyService()

# Analyze timeline
reference_audio, style_profile = style_service.get_style_guidance_for_generation(timeline_clips)

# Enhance prompt
enhanced = style_service.enhance_prompt_with_style("upbeat pop", style_profile)
```

#### DiffRhythmService
```python
from services.diffrhythm_service import DiffRhythmService

service = DiffRhythmService(model_path="models/diffrhythm2")

# Generate with style reference
output = service.generate(
    prompt="smooth jazz",
    duration=30,
    reference_audio="outputs/style_reference/timeline_reference.wav"  # NEW parameter
)
```

## Conclusion

The style consistency feature transforms music generation from isolated clips into cohesive musical projects. By analyzing and matching the sonic characteristics of existing clips, it ensures every new generation fits naturally into your evolving composition.

**Key Takeaway**: Generate your first clip to establish a style foundation, then let the system automatically maintain consistency as you build out your project.
