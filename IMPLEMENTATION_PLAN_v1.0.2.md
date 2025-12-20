# LEMM v1.0.2 Implementation Plan
## Million Song Dataset & Symbolic Music Integration

**Branch**: `v1.0.2` (GitHub: `v1.0.2-msd-hf`, HuggingFace: `v1.0.2`)  
**Status**: Planning Phase  
**Target Completion**: 12-16 weeks

---

## Phase 1: Foundation & Quick Wins (Weeks 1-2)

### Objectives
- Set up Million Song Dataset (MSD) infrastructure
- Create database system for metadata storage
- Implement basic genre-aware parameter suggestions
- Add validation system for musical parameters

### Implementation Tasks

#### 1.1 Database Setup
```python
# backend/services/msd_database_service.py
class MSDDatabaseService:
    """Manages MSD metadata storage and querying"""
    
    def __init__(self, db_path='data/msd_metadata.db'):
        self.db_path = db_path
        self.conn = None
        
    def initialize_database(self):
        """Create tables for MSD metadata"""
        # Tables: songs, genre_profiles, instrument_cooccurrence,
        #         section_templates, key_tempo_patterns
        
    def import_msd_subset(self, msd_path, max_songs=10000):
        """Import MSD subset into database"""
        # Parse MSD H5 files
        # Extract: tempo, key, mode, time_signature, loudness
        # Store in SQLite database
```

#### 1.2 Genre Profile Extraction
```python
# backend/services/genre_profiler.py
class GenreProfiler:
    """Extract and store genre characteristics"""
    
    def analyze_genre_patterns(self, genre):
        """Extract typical patterns for a genre"""
        return {
            'typical_bpm': float,
            'bpm_std': float,
            'common_keys': [(key, probability)],
            'common_modes': [(mode, probability)],
            'tempo_range': (min, max)
        }
```

#### 1.3 UI Integration - Parameter Suggestions
```python
# In app.py - add to music generation UI
def suggest_parameters_from_genre(text_prompt):
    """Suggest BPM and key based on detected genre"""
    genre = extract_genre_from_prompt(text_prompt)
    profile = msd_service.get_genre_profile(genre)
    
    return {
        'suggested_bpm': profile.typical_bpm,
        'suggested_key': profile.most_common_key,
        'info_message': f"💡 {genre}: Typical BPM {profile.typical_bpm}, Common in {profile.most_common_key}"
    }

# Add UI components
with gr.Row():
    suggest_btn = gr.Button("💡 Suggest Parameters")
    suggestion_info = gr.Textbox(label="Suggestions", interactive=False)

suggest_btn.click(
    fn=suggest_parameters_from_genre,
    inputs=[prompt_input],
    outputs=[bpm_input, key_input, suggestion_info]
)
```

### Deliverables
- ✅ `msd_database_service.py` - Database management
- ✅ `genre_profiler.py` - Genre pattern extraction  
- ✅ 10K song MSD subset imported
- ✅ Basic parameter suggestion UI
- ✅ Documentation: MSD setup guide

---

## Phase 2: Music Theory Intelligence (Weeks 3-8)

### Objectives
- Integrate Lakh MIDI dataset
- Extract harmonic and melodic patterns
- Implement music theory constraint system
- Build chord progression models

### Implementation Tasks

#### 2.1 Lakh MIDI Integration
```python
# backend/services/lakh_midi_service.py
class LakhMIDIService:
    """Process and analyze MIDI files from Lakh dataset"""
    
    def extract_chord_progressions(self, midi_file):
        """Extract chord progression from MIDI"""
        # Use music21 or pretty_midi
        # Return: [(chord_name, duration, beat), ...]
        
    def analyze_melodic_patterns(self, midi_file):
        """Extract melodic contours and intervals"""
        # Return: interval patterns, range, rhythmic patterns
        
    def build_genre_theory_model(self, genre):
        """Build music theory model for genre"""
        # Markov models for chord progressions
        # Common interval patterns
        # Typical voice leading
```

#### 2.2 Theory-Aware Generation
```python
# backend/services/theory_constraint_service.py
class TheoryConstraintService:
    """Apply music theory constraints to generation"""
    
    def get_harmonic_constraints(self, key, genre):
        """Get likely chord progressions for key/genre"""
        # Return probability distribution over chord progressions
        
    def validate_harmonic_coherence(self, generated_audio):
        """Check if generated audio follows theory"""
        # Extract chroma features
        # Compare to expected progressions
        # Return coherence score
```

#### 2.3 Enhanced Conditioning
```python
# Integrate into DiffRhythm2 generation
def generate_with_theory_constraints(prompt, genre, key, bpm):
    """Generate music with theory-aware conditioning"""
    
    # Get theory constraints
    constraints = theory_service.get_constraints(genre, key)
    
    # Build enhanced conditioning
    conditioning = {
        'text': prompt,
        'genre_embedding': genre_profiler.get_embedding(genre),
        'target_chroma': constraints.target_chroma_profile,
        'structure_template': structure_service.get_template(genre, duration),
        'bpm': bpm,
        'key': key
    }
    
    # Generate with constraints
    audio = diffrhythm2_model.generate(conditioning)
    
    return audio
```

### Deliverables
- ✅ `lakh_midi_service.py` - MIDI processing
- ✅ `theory_constraint_service.py` - Theory constraints
- ✅ 45K matched MIDI files processed
- ✅ Chord progression models per genre
- ✅ Theory-aware conditioning system
- ✅ Documentation: Music theory integration guide

---

## Phase 3: Structural Intelligence (Weeks 9-12)

### Objectives
- Implement section-aware generation
- Add dynamic contour modeling
- Build structural templates per genre
- Enable multi-section composition

### Implementation Tasks

#### 3.1 Section Analysis
```python
# backend/services/structure_analyzer.py
class StructureAnalyzer:
    """Analyze song structure from MSD section data"""
    
    def extract_section_templates(self, genre, duration):
        """Get typical section arrangement for genre/duration"""
        # Query MSD section analysis
        # Return: [('intro', 4s), ('verse', 16s), ('chorus', 12s), ...]
        
    def analyze_energy_curves(self, genre):
        """Extract typical loudness/energy progression"""
        # Return: energy curve template per section
```

#### 3.2 Multi-Section Generation
```python
# In generation service
def generate_structured_composition(prompt, genre, duration):
    """Generate multi-section composition"""
    
    # Get structure template
    template = structure_analyzer.get_template(genre, duration)
    
    sections = []
    for section_name, section_duration in template:
        # Generate each section with appropriate characteristics
        section_prompt = f"{prompt}, {section_name} section"
        
        section_audio = generate_music_clip(
            prompt=section_prompt,
            duration=section_duration,
            energy_target=template.energy_for_section(section_name)
        )
        
        sections.append(section_audio)
    
    # Apply crossfades between sections
    composed = crossfade_sections(sections, template.transitions)
    
    return composed
```

#### 3.3 UI for Structured Generation
```python
# Add to app.py
with gr.Tab("🎼 Structured Composition"):
    gr.Markdown("Generate multi-section songs with proper structure")
    
    with gr.Row():
        comp_prompt = gr.Textbox(label="Style Description")
        comp_genre = gr.Dropdown(label="Genre", choices=genres)
        comp_duration = gr.Slider(60, 300, value=120, label="Duration (seconds)")
    
    structure_display = gr.Textbox(label="Generated Structure", interactive=False)
    generate_comp_btn = gr.Button("🎵 Generate Structured Song")
    
    generate_comp_btn.click(
        fn=generate_structured_composition,
        inputs=[comp_prompt, comp_genre, comp_duration],
        outputs=[structure_display, audio_output]
    )
```

### Deliverables
- ✅ `structure_analyzer.py` - Section analysis
- ✅ Multi-section generation system
- ✅ Energy curve modeling
- ✅ Structured composition UI
- ✅ Documentation: Structure guide

---

## Phase 4: LoRA Training Enhancement (Weeks 13-16)

### Objectives
- Train genre-specific LoRAs using MSD/Lakh data
- Implement theory-aware data augmentation
- Add FMA integration for audio training
- Create pre-trained genre LoRAs

### Implementation Tasks

#### 4.1 Genre-Specific LoRA Training
```python
# backend/services/enhanced_lora_training.py
class EnhancedLoRATraining:
    """LoRA training with symbolic music intelligence"""
    
    def prepare_training_data_with_theory(self, audio_files, genre):
        """Augment training data with theory information"""
        
        augmented_data = []
        for audio_file in audio_files:
            # Extract features
            metadata = analyze_audio(audio_file)
            
            # Enhance with MSD/Lakh patterns
            theory_target = theory_service.get_target_for_genre(genre)
            structure_target = structure_service.get_template(genre)
            
            augmented_data.append({
                'audio': audio_file,
                'metadata': metadata,
                'theory_target': theory_target,
                'structure_target': structure_target
            })
        
        return augmented_data
    
    def train_genre_lora(self, genre, audio_files):
        """Train genre-specific LoRA with theory awareness"""
        # Prepare data with augmentation
        # Train with theory constraints
        # Validate against genre characteristics
```

#### 4.2 Pre-trained Genre LoRAs
```python
# Create collection of genre LoRAs
genres_to_train = ['pop', 'rock', 'electronic', 'jazz', 'classical', 
                   'hip-hop', 'country', 'metal', 'r&b', 'folk']

for genre in genres_to_train:
    # Get genre-specific data from FMA
    audio_files = fma_service.get_genre_samples(genre, n=1000)
    
    # Train LoRA with MSD/Lakh intelligence
    lora = enhanced_training.train_genre_lora(genre, audio_files)
    
    # Upload to HuggingFace
    upload_lora(lora, name=f"lemm-{genre}-v1")
```

### Deliverables
- ✅ Enhanced LoRA training with theory awareness
- ✅ 10 pre-trained genre LoRAs
- ✅ FMA dataset integration
- ✅ Theory-aware data augmentation
- ✅ Documentation: Enhanced training guide

---

## File Structure

```
backend/services/
├── msd_database_service.py          # MSD metadata storage/query
├── genre_profiler.py                # Genre pattern extraction
├── lakh_midi_service.py             # MIDI processing
├── theory_constraint_service.py     # Music theory constraints
├── structure_analyzer.py            # Song structure analysis
├── enhanced_lora_training.py        # Theory-aware LoRA training
└── fma_service.py                   # Free Music Archive integration

data/
├── msd_metadata.db                  # SQLite database
├── lakh_midi/                       # MIDI files
├── genre_profiles.json              # Extracted profiles
└── theory_patterns.json             # Theory models

frontend/
└── (structured composition UI updates)

docs/
├── MSD_SETUP_GUIDE.md
├── MUSIC_THEORY_INTEGRATION.md
├── STRUCTURE_GUIDE.md
└── ENHANCED_TRAINING_GUIDE.md
```

---

## Dependencies to Add

```txt
# requirements.txt additions

# MIDI processing
music21>=8.1.0                 # Music theory and MIDI analysis
pretty_midi>=0.2.10            # MIDI file processing
mido>=1.3.0                    # MIDI I/O

# Database
sqlalchemy>=2.0.0              # ORM for database
alembic>=1.12.0                # Database migrations

# Music analysis
essentia>=2.1                  # Audio feature extraction (optional)
madmom>=0.16.1                 # Music information retrieval (optional)

# Data processing
h5py>=3.9.0                    # For reading MSD H5 files
tables>=3.8.0                  # PyTables for HDF5
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_msd_integration.py
def test_genre_profile_extraction():
    """Test genre profile extraction from MSD"""
    
def test_theory_constraints():
    """Test music theory constraint generation"""
    
def test_structure_templates():
    """Test section template extraction"""
```

### Integration Tests
```python
# tests/test_enhanced_generation.py
def test_theory_aware_generation():
    """Test generation with theory constraints"""
    
def test_structured_composition():
    """Test multi-section composition"""
```

### Quality Metrics
- Genre accuracy: Target +15%
- Harmonic coherence: Target +17%
- Structural consistency: Target +20%
- User satisfaction: Target +1.3 points

---

## Risk Mitigation

### Technical Risks
1. **MSD Subset Quality**: Some metadata may be noisy
   - **Mitigation**: Statistical outlier detection, cross-validation
   
2. **MIDI Processing Performance**: 45K files to process
   - **Mitigation**: Parallel processing, caching, incremental updates
   
3. **Integration Complexity**: Many new components
   - **Mitigation**: Phased rollout, comprehensive testing

### Resource Risks
1. **Storage**: 50-500GB needed
   - **Mitigation**: Start with subsets, cloud storage options
   
2. **Processing Time**: Initial analysis takes time
   - **Mitigation**: One-time batch processing, pre-computed results

---

## Success Criteria

### Phase 1 Success
- ✅ 10K MSD songs imported
- ✅ Genre profiles extracted
- ✅ Parameter suggestions working in UI
- ✅ Validation system operational

### Phase 2 Success
- ✅ 45K MIDI files processed
- ✅ Chord progression models built
- ✅ Theory-aware conditioning implemented
- ✅ Improved harmonic coherence measured

### Phase 3 Success
- ✅ Section templates extracted
- ✅ Multi-section generation working
- ✅ Energy curves implemented
- ✅ Structured composition UI complete

### Phase 4 Success
- ✅ 10 genre LoRAs trained and deployed
- ✅ FMA integration complete
- ✅ Enhanced training system operational
- ✅ Quality improvements validated

---

## Rollout Plan

### Week 1-2: Infrastructure
- Set up databases
- Import MSD subset
- Basic parameter suggestions

### Week 3-8: Core Intelligence
- MIDI processing
- Theory constraints
- Enhanced conditioning

### Week 9-12: Structural Features
- Section analysis
- Multi-section generation
- UI updates

### Week 13-16: Training & Optimization
- Genre LoRA training
- FMA integration
- Performance optimization
- Documentation completion

---

**Next Steps**: Begin Phase 1 implementation on v1.0.2 branch

**Last Updated**: December 20, 2025
