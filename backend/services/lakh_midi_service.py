"""
Lakh MIDI Dataset Service
Parses MIDI files and extracts music theory information (chords, melodies, progressions)
"""

import logging
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# Musical constants
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_TEMPLATES = {
    'major': [0, 4, 7],
    'minor': [0, 3, 7],
    'diminished': [0, 3, 6],
    'augmented': [0, 4, 8],
    'major7': [0, 4, 7, 11],
    'minor7': [0, 3, 7, 10],
    'dominant7': [0, 4, 7, 10],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7]
}


class LakhMIDIService:
    """Service for parsing and analyzing MIDI files."""
    
    def __init__(self, db_path: str = "data/msd_metadata.db"):
        """
        Initialize Lakh MIDI Service.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_data_directory()
        self._initialize_midi_tables()
        logger.info(f"Lakh MIDI Service initialized: {db_path}")
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_midi_tables(self):
        """Create MIDI-related database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # MIDI files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS midi_files (
                    midi_id TEXT PRIMARY KEY,
                    track_id TEXT,
                    file_path TEXT,
                    tempo REAL,
                    time_signature TEXT,
                    key_signature INTEGER,
                    duration REAL,
                    num_tracks INTEGER,
                    num_notes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_id) REFERENCES songs(track_id)
                )
            """)
            
            # Chord progressions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chord_progressions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    midi_id TEXT,
                    chord_sequence TEXT,
                    duration REAL,
                    key INTEGER,
                    mode INTEGER,
                    occurrence_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (midi_id) REFERENCES midi_files(midi_id)
                )
            """)
            
            # Melodic patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS melodic_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    midi_id TEXT,
                    pattern TEXT,
                    interval_sequence TEXT,
                    length INTEGER,
                    occurrence_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (midi_id) REFERENCES midi_files(midi_id)
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_midi_track_id 
                ON midi_files(track_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chord_midi_id 
                ON chord_progressions(midi_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chord_key_mode 
                ON chord_progressions(key, mode)
            """)
            
            conn.commit()
            logger.info("MIDI database tables initialized")
    
    def parse_midi_simple(self, midi_data: Dict) -> Dict:
        """
        Parse MIDI data structure (simplified for demo without real MIDI library).
        
        Args:
            midi_data: Dictionary with MIDI information
            
        Returns:
            Parsed MIDI metadata
        """
        try:
            return {
                'midi_id': midi_data.get('midi_id'),
                'track_id': midi_data.get('track_id'),
                'file_path': midi_data.get('file_path'),
                'tempo': midi_data.get('tempo', 120.0),
                'time_signature': midi_data.get('time_signature', '4/4'),
                'key_signature': midi_data.get('key_signature', 0),
                'duration': midi_data.get('duration', 180.0),
                'num_tracks': midi_data.get('num_tracks', 1),
                'num_notes': midi_data.get('num_notes', 100)
            }
        except Exception as e:
            logger.error(f"Error parsing MIDI: {e}")
            return None
    
    def detect_chords_from_notes(self, notes: List[int]) -> List[str]:
        """
        Detect chords from a list of MIDI note numbers.
        
        Args:
            notes: List of MIDI note numbers (0-127)
            
        Returns:
            List of detected chord names
        """
        if not notes:
            return []
        
        # Normalize to pitch classes (0-11)
        pitch_classes = sorted(set(note % 12 for note in notes))
        
        if len(pitch_classes) < 3:
            return []  # Need at least 3 notes for a chord
        
        detected_chords = []
        
        # Try each note as potential root
        for root in pitch_classes:
            # Normalize intervals from this root
            intervals = sorted((pc - root) % 12 for pc in pitch_classes)
            
            # Check against chord templates
            for chord_type, template in CHORD_TEMPLATES.items():
                if intervals[:len(template)] == template:
                    chord_name = f"{NOTE_NAMES[root]}{chord_type}"
                    detected_chords.append(chord_name)
        
        return detected_chords if detected_chords else ['Unknown']
    
    def extract_chord_progression(self, midi_data: Dict) -> List[str]:
        """
        Extract chord progression from MIDI data.
        
        Args:
            midi_data: Dictionary with MIDI track information
            
        Returns:
            List of chord names in sequence
        """
        try:
            # Simplified: Get chords from provided data
            if 'chords' in midi_data:
                return midi_data['chords']
            
            # Or detect from notes
            if 'notes' in midi_data:
                notes = midi_data['notes']
                return self.detect_chords_from_notes(notes)
            
            return []
            
        except Exception as e:
            logger.error(f"Error extracting chord progression: {e}")
            return []
    
    def analyze_chord_progression(self, chords: List[str], key: int = 0, 
                                   mode: int = 1) -> Dict:
        """
        Analyze a chord progression for patterns and characteristics.
        
        Args:
            chords: List of chord names
            key: Musical key (0-11)
            mode: Mode (0=minor, 1=major)
            
        Returns:
            Analysis results dictionary
        """
        if not chords:
            return {}
        
        # Count chord frequencies
        chord_counts = Counter(chords)
        
        # Identify most common chords
        common_chords = chord_counts.most_common(5)
        
        # Detect common progressions (simplified)
        progression_patterns = self._detect_progression_patterns(chords)
        
        return {
            'num_chords': len(chords),
            'unique_chords': len(chord_counts),
            'most_common': common_chords,
            'progression_patterns': progression_patterns,
            'key': key,
            'mode': mode,
            'chord_sequence': ' -> '.join(chords[:10])  # First 10 chords
        }
    
    def _detect_progression_patterns(self, chords: List[str]) -> List[str]:
        """Detect common chord progression patterns."""
        patterns = []
        
        # Look for common patterns
        chord_str = ' '.join(chords)
        
        # Common progressions (simplified detection)
        common = {
            'I-IV-V': ['C major', 'F major', 'G major'],
            'I-V-vi-IV': ['C major', 'G major', 'A minor', 'F major'],
            'ii-V-I': ['D minor', 'G major', 'C major'],
            'I-vi-IV-V': ['C major', 'A minor', 'F major', 'G major']
        }
        
        for pattern_name, pattern_chords in common.items():
            pattern_str = ' '.join(pattern_chords)
            if pattern_str in chord_str:
                patterns.append(pattern_name)
        
        return patterns
    
    def import_midi_file(self, midi_data: Dict) -> bool:
        """
        Import MIDI file metadata and analysis to database.
        
        Args:
            midi_data: Dictionary with MIDI information
            
        Returns:
            True if successful
        """
        try:
            parsed = self.parse_midi_simple(midi_data)
            
            if not parsed:
                return False
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert MIDI file metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO midi_files (
                        midi_id, track_id, file_path, tempo, time_signature,
                        key_signature, duration, num_tracks, num_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    parsed['midi_id'],
                    parsed.get('track_id'),
                    parsed.get('file_path'),
                    parsed['tempo'],
                    parsed['time_signature'],
                    parsed['key_signature'],
                    parsed['duration'],
                    parsed['num_tracks'],
                    parsed['num_notes']
                ))
                
                # Extract and store chord progression
                chords = self.extract_chord_progression(midi_data)
                if chords:
                    chord_seq = json.dumps(chords)
                    cursor.execute("""
                        INSERT INTO chord_progressions (
                            midi_id, chord_sequence, duration, key, mode
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        parsed['midi_id'],
                        chord_seq,
                        parsed['duration'],
                        midi_data.get('key', 0),
                        midi_data.get('mode', 1)
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error importing MIDI file: {e}")
            return False
    
    def get_chord_progressions_by_key(self, key: int, mode: int, 
                                       limit: int = 10) -> List[Dict]:
        """
        Get chord progressions for a specific key/mode.
        
        Args:
            key: Musical key (0-11)
            mode: Mode (0=minor, 1=major)
            limit: Maximum number to return
            
        Returns:
            List of chord progression dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM chord_progressions
                    WHERE key = ? AND mode = ?
                    ORDER BY occurrence_count DESC
                    LIMIT ?
                """, (key, mode, limit))
                
                progressions = []
                for row in cursor.fetchall():
                    prog = dict(row)
                    prog['chord_sequence'] = json.loads(prog['chord_sequence'])
                    progressions.append(prog)
                
                return progressions
                
        except Exception as e:
            logger.error(f"Error getting chord progressions: {e}")
            return []
    
    def get_common_progressions(self, limit: int = 20) -> List[Dict]:
        """
        Get most common chord progressions across all keys.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of progression dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        chord_sequence,
                        key,
                        mode,
                        COUNT(*) as count
                    FROM chord_progressions
                    GROUP BY chord_sequence, key, mode
                    ORDER BY count DESC
                    LIMIT ?
                """, (limit,))
                
                progressions = []
                for row in cursor.fetchall():
                    prog = dict(row)
                    prog['chord_sequence'] = json.loads(prog['chord_sequence'])
                    progressions.append(prog)
                
                return progressions
                
        except Exception as e:
            logger.error(f"Error getting common progressions: {e}")
            return []
    
    def generate_sample_midi_data(self, count: int = 100) -> List[Dict]:
        """
        Generate sample MIDI data for testing (without real MIDI files).
        
        Args:
            count: Number of sample MIDI files to generate
            
        Returns:
            List of sample MIDI data dictionaries
        """
        import random
        
        # Common chord progressions
        progressions = [
            ['C major', 'F major', 'G major', 'C major'],  # I-IV-V-I
            ['C major', 'A minor', 'F major', 'G major'],  # I-vi-IV-V
            ['A minor', 'F major', 'C major', 'G major'],  # vi-IV-I-V
            ['C major', 'G major', 'A minor', 'F major'],  # I-V-vi-IV
            ['D minor', 'G major', 'C major', 'F major'],  # ii-V-I-IV
            ['C major', 'E minor', 'A minor', 'F major'],  # I-iii-vi-IV
        ]
        
        samples = []
        for i in range(count):
            key = random.randint(0, 11)
            mode = random.choice([0, 1])
            tempo = random.uniform(80, 160)
            
            # Select and possibly transpose progression
            base_prog = random.choice(progressions)
            
            sample = {
                'midi_id': f'MIDI{i:06d}',
                'track_id': f'TEST{i:06d}',
                'file_path': f'/data/midi/song_{i}.mid',
                'tempo': tempo,
                'time_signature': random.choice(['4/4', '3/4', '6/8']),
                'key_signature': key,
                'duration': random.uniform(120, 300),
                'num_tracks': random.randint(3, 12),
                'num_notes': random.randint(500, 5000),
                'chords': base_prog,
                'key': key,
                'mode': mode
            }
            
            samples.append(sample)
        
        return samples
    
    def import_sample_data(self, count: int = 100) -> Dict:
        """
        Generate and import sample MIDI data.
        
        Args:
            count: Number of samples to generate
            
        Returns:
            Import statistics
        """
        logger.info(f"Generating {count} sample MIDI files...")
        samples = self.generate_sample_midi_data(count)
        
        successful = 0
        failed = 0
        
        for sample in samples:
            if self.import_midi_file(sample):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"MIDI import complete: {successful} successful, {failed} failed")
        
        return {
            'imported': successful,
            'failed': failed,
            'total': len(samples)
        }
    
    def get_midi_stats(self) -> Dict:
        """Get MIDI database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total MIDI files
                cursor.execute("SELECT COUNT(*) FROM midi_files")
                total_midi = cursor.fetchone()[0]
                
                # Total progressions
                cursor.execute("SELECT COUNT(*) FROM chord_progressions")
                total_progressions = cursor.fetchone()[0]
                
                # Average tempo
                cursor.execute("SELECT AVG(tempo) FROM midi_files WHERE tempo IS NOT NULL")
                avg_tempo = cursor.fetchone()[0]
                
                return {
                    'total_midi_files': total_midi,
                    'total_progressions': total_progressions,
                    'avg_tempo': round(avg_tempo, 2) if avg_tempo else None
                }
                
        except Exception as e:
            logger.error(f"Error getting MIDI stats: {e}")
            return {}
