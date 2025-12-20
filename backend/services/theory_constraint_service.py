"""
Music Theory Constraint Service
Provides music theory rules and suggestions for generation parameters
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# Music theory constants
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SCALE_INTERVALS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10]
}

# Circle of fifths (for key relationships)
CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

# Common chord progressions by function
PROGRESSIONS = {
    'pop': [
        ['I', 'V', 'vi', 'IV'],  # Most popular
        ['I', 'IV', 'V', 'I'],   # Classic
        ['vi', 'IV', 'I', 'V'],  # Sensitive
        ['I', 'vi', 'IV', 'V']   # 50s progression
    ],
    'jazz': [
        ['ii', 'V', 'I'],        # ii-V-I
        ['I', 'VI', 'ii', 'V'],  # Jazz turnaround
        ['iii', 'vi', 'ii', 'V'] # Descending sequence
    ],
    'blues': [
        ['I', 'I', 'I', 'I', 'IV', 'IV', 'I', 'I', 'V', 'IV', 'I', 'V'],  # 12-bar blues
        ['I', 'IV', 'V']         # Simple blues
    ],
    'rock': [
        ['I', 'IV', 'V'],        # Power chord progression
        ['I', 'bVII', 'IV'],     # Mixolydian rock
        ['i', 'bVII', 'bVI']     # Minor rock
    ]
}


class TheoryConstraintService:
    """Service for music theory validation and suggestions."""
    
    def __init__(self, db_path: str = "data/msd_metadata.db"):
        """
        Initialize Theory Constraint Service.
        
        Args:
            db_path: Path to database (for integration with MSD/MIDI services)
        """
        self.db_path = db_path
        logger.info("Theory Constraint Service initialized")
    
    def get_scale_notes(self, root: int, scale_type: str = 'major') -> List[int]:
        """
        Get notes in a scale.
        
        Args:
            root: Root note (0-11, C=0)
            scale_type: Type of scale (major, minor, dorian, etc.)
            
        Returns:
            List of note numbers in scale
        """
        intervals = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS['major'])
        return [(root + interval) % 12 for interval in intervals]
    
    def validate_key_tempo_compatibility(self, key: int, mode: int, 
                                          tempo: float) -> Dict:
        """
        Check if key/mode and tempo are compatible based on music theory.
        
        Args:
            key: Musical key (0-11)
            mode: Mode (0=minor, 1=major)
            tempo: Tempo in BPM
            
        Returns:
            Validation result with suggestions
        """
        mode_name = 'major' if mode == 1 else 'minor'
        key_name = NOTE_NAMES[key]
        
        # General guidelines
        warnings = []
        suggestions = []
        
        # Tempo ranges by mode
        if mode == 1:  # Major
            if tempo < 60:
                warnings.append("Very slow tempo for major key - consider minor key")
            elif tempo > 180:
                suggestions.append("Fast tempo works well with major keys")
        else:  # Minor
            if tempo > 160:
                warnings.append("Very fast tempo for minor key - consider major key")
            elif 60 <= tempo <= 90:
                suggestions.append("Moderate tempo ideal for emotional minor pieces")
        
        return {
            'valid': True,
            'key': key_name,
            'mode': mode_name,
            'tempo': tempo,
            'warnings': warnings,
            'suggestions': suggestions
        }
    
    def suggest_compatible_keys(self, current_key: int, mode: int) -> List[Dict]:
        """
        Suggest keys compatible with current key for modulation.
        
        Args:
            current_key: Current key (0-11)
            mode: Current mode (0=minor, 1=major)
            
        Returns:
            List of compatible key suggestions
        """
        suggestions = []
        
        # Relative major/minor
        if mode == 1:  # Major
            relative_minor = (current_key + 9) % 12  # 3 semitones down
            suggestions.append({
                'key': relative_minor,
                'key_name': NOTE_NAMES[relative_minor],
                'mode': 0,
                'mode_name': 'minor',
                'relationship': 'relative minor',
                'strength': 'very strong'
            })
        else:  # Minor
            relative_major = (current_key + 3) % 12  # 3 semitones up
            suggestions.append({
                'key': relative_major,
                'key_name': NOTE_NAMES[relative_major],
                'mode': 1,
                'mode_name': 'major',
                'relationship': 'relative major',
                'strength': 'very strong'
            })
        
        # Parallel major/minor
        suggestions.append({
            'key': current_key,
            'key_name': NOTE_NAMES[current_key],
            'mode': 1 - mode,  # Flip mode
            'mode_name': 'major' if mode == 0 else 'minor',
            'relationship': 'parallel',
            'strength': 'strong'
        })
        
        # Dominant (up a fifth)
        dominant = (current_key + 7) % 12
        suggestions.append({
            'key': dominant,
            'key_name': NOTE_NAMES[dominant],
            'mode': mode,
            'mode_name': 'major' if mode == 1 else 'minor',
            'relationship': 'dominant',
            'strength': 'strong'
        })
        
        # Subdominant (down a fifth / up a fourth)
        subdominant = (current_key + 5) % 12
        suggestions.append({
            'key': subdominant,
            'key_name': NOTE_NAMES[subdominant],
            'mode': mode,
            'mode_name': 'major' if mode == 1 else 'minor',
            'relationship': 'subdominant',
            'strength': 'moderate'
        })
        
        return suggestions
    
    def get_chord_progressions_for_style(self, style: str, key: int, 
                                          mode: int) -> List[Dict]:
        """
        Get appropriate chord progressions for a musical style.
        
        Args:
            style: Musical style (pop, jazz, blues, rock)
            key: Root key (0-11)
            mode: Mode (0=minor, 1=major)
            
        Returns:
            List of chord progression suggestions
        """
        style = style.lower()
        progressions = PROGRESSIONS.get(style, PROGRESSIONS['pop'])
        
        results = []
        for prog in progressions:
            results.append({
                'style': style,
                'root_key': NOTE_NAMES[key],
                'mode': 'major' if mode == 1 else 'minor',
                'progression': prog,
                'description': self._get_progression_description(prog, style)
            })
        
        return results
    
    def _get_progression_description(self, progression: List[str], 
                                      style: str) -> str:
        """Get human-readable description of a chord progression."""
        prog_str = '-'.join(progression)
        
        descriptions = {
            'I-V-vi-IV': 'The most popular pop progression (emotional and uplifting)',
            'I-IV-V-I': 'Classic rock/pop progression (strong and resolved)',
            'vi-IV-I-V': 'Sensitive and melancholic (popular in ballads)',
            'ii-V-I': 'The fundamental jazz progression (smooth resolution)',
            'I-IV-V': 'Simple and powerful (rock/blues staple)'
        }
        
        return descriptions.get(prog_str, f'Common {style} progression')
    
    def analyze_harmonic_compatibility(self, key1: int, mode1: int, 
                                        key2: int, mode2: int) -> Dict:
        """
        Analyze compatibility between two keys for transitions/mashups.
        
        Args:
            key1, mode1: First key and mode
            key2, mode2: Second key and mode
            
        Returns:
            Compatibility analysis
        """
        # Calculate interval between keys
        interval = (key2 - key1) % 12
        
        # Determine relationship
        relationships = {
            0: 'same key',
            3: 'minor third',
            4: 'major third',
            5: 'perfect fourth',
            7: 'perfect fifth',
            9: 'major sixth'
        }
        
        relationship = relationships.get(interval, f'{interval} semitones')
        
        # Compatibility score (0-10)
        compatibility_scores = {
            0: 10,  # Same key
            7: 9,   # Perfect fifth
            5: 8,   # Perfect fourth
            3: 7,   # Minor third
            4: 7,   # Major third
            9: 6,   # Major sixth
        }
        
        score = compatibility_scores.get(interval, 4)
        
        # Same mode bonus
        if mode1 == mode2:
            score += 1
        
        return {
            'key1': NOTE_NAMES[key1],
            'mode1': 'major' if mode1 == 1 else 'minor',
            'key2': NOTE_NAMES[key2],
            'mode2': 'major' if mode2 == 1 else 'minor',
            'interval': interval,
            'relationship': relationship,
            'compatibility_score': min(score, 10),
            'recommendation': 'Excellent' if score >= 8 else 'Good' if score >= 6 else 'Moderate'
        }
    
    def suggest_generation_constraints(self, genre: str, tempo: float, 
                                        key: int, mode: int) -> Dict:
        """
        Generate comprehensive theory-based constraints for music generation.
        
        Args:
            genre: Music genre
            tempo: Tempo in BPM
            key: Root key (0-11)
            mode: Mode (0=minor, 1=major)
            
        Returns:
            Constraint suggestions dictionary
        """
        # Get scale notes
        scale_type = 'major' if mode == 1 else 'minor'
        scale_notes = self.get_scale_notes(key, scale_type)
        
        # Get chord progressions
        # Map genre to style
        style_map = {
            'rock': 'rock',
            'pop': 'pop',
            'jazz': 'jazz',
            'blues': 'blues',
            'metal': 'rock',
            'electronic': 'pop',
            'hip-hop': 'pop',
            'classical': 'pop'  # Simplified
        }
        style = style_map.get(genre.lower(), 'pop')
        progressions = self.get_chord_progressions_for_style(style, key, mode)
        
        # Compatible keys for modulation
        compatible_keys = self.suggest_compatible_keys(key, mode)
        
        # Tempo validation
        validation = self.validate_key_tempo_compatibility(key, mode, tempo)
        
        return {
            'key': NOTE_NAMES[key],
            'mode': 'major' if mode == 1 else 'minor',
            'scale_notes': [NOTE_NAMES[n] for n in scale_notes],
            'tempo': tempo,
            'genre': genre,
            'recommended_progressions': progressions[:3],
            'compatible_keys': compatible_keys[:3],
            'validation': validation,
            'constraints': {
                'allowed_notes': scale_notes,
                'tempo_range': (max(60, tempo - 20), min(200, tempo + 20)),
                'time_signature': '4/4',  # Most common
                'preferred_chords': progressions[0]['progression'] if progressions else []
            }
        }
    
    def validate_chord_progression(self, chords: List[str], key: int, 
                                     mode: int) -> Dict:
        """
        Validate if a chord progression is theoretically sound.
        
        Args:
            chords: List of chord names/numerals
            key: Root key (0-11)
            mode: Mode (0=minor, 1=major)
            
        Returns:
            Validation result with suggestions
        """
        if not chords:
            return {'valid': False, 'message': 'No chords provided'}
        
        issues = []
        suggestions = []
        
        # Check for common patterns
        chord_str = '-'.join(chords)
        
        # Check for resolution
        if chords[-1] != 'I' and chords[-1] != 'i':
            suggestions.append("Consider ending on the tonic (I) for stronger resolution")
        
        # Check for variety
        if len(set(chords)) < len(chords) * 0.5:
            issues.append("Progression may be repetitive - consider more chord variety")
        
        return {
            'valid': len(issues) == 0,
            'chords': chords,
            'key': NOTE_NAMES[key],
            'mode': 'major' if mode == 1 else 'minor',
            'issues': issues,
            'suggestions': suggestions,
            'strength': 'strong' if len(issues) == 0 else 'moderate' if len(issues) < 2 else 'weak'
        }
