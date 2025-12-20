"""
Test script for Phase 2: Music Theory & MIDI Services
Verifies MIDI parsing, chord detection, and theory constraint functionality
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from services.lakh_midi_service import LakhMIDIService
from services.theory_constraint_service import TheoryConstraintService

def test_phase2_services():
    """Test Phase 2 services"""
    
    print("=" * 60)
    print("Phase 2: Music Theory & MIDI Services Test")
    print("=" * 60)
    
    # Initialize services
    print("\n1. Initializing services...")
    try:
        midi_service = LakhMIDIService("data/test_phase2.db")
        theory_service = TheoryConstraintService()
        print("✅ Services initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        return False
    
    # Test MIDI data generation
    print("\n2. Generating sample MIDI data...")
    try:
        samples = midi_service.generate_sample_midi_data(count=20)
        print(f"✅ Generated {len(samples)} sample MIDI files")
        print(f"   Sample: {samples[0]['midi_id']} - Tempo: {samples[0]['tempo']:.0f} BPM, Key: {samples[0]['key_signature']}")
    except Exception as e:
        print(f"❌ MIDI generation failed: {e}")
        return False
    
    # Test MIDI import
    print("\n3. Testing MIDI import...")
    try:
        result = midi_service.import_sample_data(count=20)
        print(f"✅ Import results: {result['imported']} successful, {result['failed']} failed")
    except Exception as e:
        print(f"❌ MIDI import failed: {e}")
        return False
    
    # Test chord detection
    print("\n4. Testing chord detection...")
    try:
        # Test with C major triad
        notes = [60, 64, 67]  # C, E, G
        chords = midi_service.detect_chords_from_notes(notes)
        print(f"✅ Detected chords from [C, E, G]: {chords}")
        
        # Test with more complex chord
        notes = [60, 64, 67, 71]  # C, E, G, B (C major 7)
        chords = midi_service.detect_chords_from_notes(notes)
        print(f"✅ Detected chords from [C, E, G, B]: {chords}")
    except Exception as e:
        print(f"❌ Chord detection failed: {e}")
        return False
    
    # Test chord progression analysis
    print("\n5. Testing chord progression analysis...")
    try:
        test_chords = ['C major', 'F major', 'G major', 'C major']
        analysis = midi_service.analyze_chord_progression(test_chords, key=0, mode=1)
        print(f"✅ Analyzed progression: {test_chords}")
        print(f"   Num chords: {analysis['num_chords']}, Unique: {analysis['unique_chords']}")
        print(f"   Sequence: {analysis['chord_sequence']}")
    except Exception as e:
        print(f"❌ Progression analysis failed: {e}")
        return False
    
    # Test music theory - scale notes
    print("\n6. Testing scale generation...")
    try:
        # C major scale
        c_major = theory_service.get_scale_notes(0, 'major')
        print(f"✅ C major scale: {c_major}")
        
        # A minor scale
        a_minor = theory_service.get_scale_notes(9, 'minor')
        print(f"✅ A minor scale: {a_minor}")
    except Exception as e:
        print(f"❌ Scale generation failed: {e}")
        return False
    
    # Test key compatibility
    print("\n7. Testing key/tempo compatibility...")
    try:
        validation = theory_service.validate_key_tempo_compatibility(
            key=0, mode=1, tempo=120
        )
        print(f"✅ Validation for C major at 120 BPM:")
        print(f"   Valid: {validation['valid']}")
        print(f"   Key: {validation['key']} {validation['mode']}")
        if validation['suggestions']:
            print(f"   Suggestions: {validation['suggestions'][0]}")
    except Exception as e:
        print(f"❌ Compatibility check failed: {e}")
        return False
    
    # Test compatible keys
    print("\n8. Testing compatible key suggestions...")
    try:
        compatible = theory_service.suggest_compatible_keys(current_key=0, mode=1)
        print(f"✅ Keys compatible with C major:")
        for k in compatible[:3]:
            print(f"   - {k['key_name']} {k['mode_name']}: {k['relationship']} ({k['strength']})")
    except Exception as e:
        print(f"❌ Compatible keys failed: {e}")
        return False
    
    # Test chord progressions for style
    print("\n9. Testing chord progressions by style...")
    try:
        for style in ['pop', 'jazz', 'rock']:
            progs = theory_service.get_chord_progressions_for_style(style, key=0, mode=1)
            if progs:
                prog = progs[0]
                print(f"✅ {style.title()}: {' → '.join(prog['progression'])}")
    except Exception as e:
        print(f"❌ Style progressions failed: {e}")
        return False
    
    # Test harmonic compatibility
    print("\n10. Testing harmonic compatibility...")
    try:
        compat = theory_service.analyze_harmonic_compatibility(
            key1=0, mode1=1,  # C major
            key2=7, mode2=1   # G major
        )
        print(f"✅ Compatibility between C major and G major:")
        print(f"   Relationship: {compat['relationship']}")
        print(f"   Score: {compat['compatibility_score']}/10")
        print(f"   Recommendation: {compat['recommendation']}")
    except Exception as e:
        print(f"❌ Harmonic compatibility failed: {e}")
        return False
    
    # Test comprehensive constraints
    print("\n11. Testing comprehensive theory constraints...")
    try:
        constraints = theory_service.suggest_generation_constraints(
            genre='rock', tempo=130, key=0, mode=1
        )
        print(f"✅ Constraints for rock at 130 BPM in C major:")
        print(f"   Scale notes: {', '.join(constraints['scale_notes'])}")
        print(f"   Tempo range: {constraints['constraints']['tempo_range']}")
        if constraints['recommended_progressions']:
            prog = constraints['recommended_progressions'][0]
            print(f"   Recommended: {' → '.join(prog['progression'])}")
    except Exception as e:
        print(f"❌ Constraint generation failed: {e}")
        return False
    
    # Test MIDI database queries
    print("\n12. Testing MIDI database queries...")
    try:
        stats = midi_service.get_midi_stats()
        print(f"✅ MIDI database stats:")
        print(f"   Total MIDI files: {stats.get('total_midi_files', 0)}")
        print(f"   Total progressions: {stats.get('total_progressions', 0)}")
        print(f"   Average tempo: {stats.get('avg_tempo', 0):.0f} BPM")
    except Exception as e:
        print(f"❌ MIDI stats failed: {e}")
        return False
    
    # Test chord progression validation
    print("\n13. Testing chord progression validation...")
    try:
        test_prog = ['I', 'V', 'vi', 'IV']
        validation = theory_service.validate_chord_progression(test_prog, key=0, mode=1)
        print(f"✅ Validation for {' → '.join(test_prog)}:")
        print(f"   Valid: {validation['valid']}")
        print(f"   Strength: {validation['strength']}")
        if validation['suggestions']:
            print(f"   Suggestions: {validation['suggestions'][0]}")
    except Exception as e:
        print(f"❌ Progression validation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL PHASE 2 TESTS PASSED!")
    print("=" * 60)
    
    # Cleanup test database
    print("\nCleaning up test database...")
    try:
        import os
        if os.path.exists("data/test_phase2.db"):
            os.remove("data/test_phase2.db")
            print("✅ Test database removed")
    except Exception as e:
        print(f"⚠️ Could not remove test database: {e}")
    
    return True

if __name__ == "__main__":
    success = test_phase2_services()
    sys.exit(0 if success else 1)
