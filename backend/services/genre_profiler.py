"""
Genre Profiler Service
Analyzes MSD data to extract genre-specific patterns for parameter suggestions.
"""

import sqlite3
import logging
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

# Musical key names for display
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODE_NAMES = {0: 'minor', 1: 'major'}


class GenreProfiler:
    """Service for analyzing and profiling genre patterns from MSD."""
    
    def __init__(self, db_path: str = "data/msd_metadata.db"):
        """
        Initialize Genre Profiler.
        
        Args:
            db_path: Path to MSD SQLite database
        """
        self.db_path = db_path
        logger.info(f"Genre Profiler initialized with database: {db_path}")
    
    def analyze_genre(self, genre: str) -> Optional[Dict]:
        """
        Analyze a genre to extract common patterns.
        
        Args:
            genre: Genre name to analyze
            
        Returns:
            Dictionary with genre profile data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get all songs for this genre
                cursor.execute("""
                    SELECT tempo, key, mode, time_signature, energy, 
                           danceability, loudness
                    FROM songs 
                    WHERE genre = ? AND tempo IS NOT NULL
                """, (genre,))
                
                songs = cursor.fetchall()
                
                if not songs:
                    logger.warning(f"No songs found for genre: {genre}")
                    return None
                
                # Collect data for analysis
                tempos = []
                keys = defaultdict(int)
                modes = defaultdict(int)
                time_signatures = defaultdict(int)
                energies = []
                danceabilities = []
                loudnesses = []
                
                for song in songs:
                    if song['tempo']:
                        tempos.append(song['tempo'])
                    if song['key'] is not None:
                        keys[song['key']] += 1
                    if song['mode'] is not None:
                        modes[song['mode']] += 1
                    if song['time_signature']:
                        time_signatures[song['time_signature']] += 1
                    if song['energy'] is not None:
                        energies.append(song['energy'])
                    if song['danceability'] is not None:
                        danceabilities.append(song['danceability'])
                    if song['loudness'] is not None:
                        loudnesses.append(song['loudness'])
                
                # Calculate statistics
                profile = {
                    'genre': genre,
                    'song_count': len(songs),
                    'tempo': self._analyze_tempo(tempos),
                    'keys': self._get_common_keys(keys, top_n=5),
                    'modes': self._get_common_modes(modes),
                    'time_signatures': self._get_common_time_signatures(time_signatures),
                    'energy': self._calculate_avg(energies),
                    'danceability': self._calculate_avg(danceabilities),
                    'loudness': self._calculate_avg(loudnesses)
                }
                
                return profile
                
        except Exception as e:
            logger.error(f"Error analyzing genre {genre}: {str(e)}")
            return None
    
    def _analyze_tempo(self, tempos: List[float]) -> Dict:
        """Analyze tempo distribution."""
        if not tempos:
            return {}
        
        return {
            'avg': round(statistics.mean(tempos), 2),
            'std': round(statistics.stdev(tempos), 2) if len(tempos) > 1 else 0,
            'min': round(min(tempos), 2),
            'max': round(max(tempos), 2),
            'median': round(statistics.median(tempos), 2)
        }
    
    def _get_common_keys(self, keys: Dict[int, int], top_n: int = 5) -> List[Dict]:
        """Get most common musical keys."""
        sorted_keys = sorted(keys.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {
                'key': key,
                'key_name': KEY_NAMES[key] if 0 <= key < 12 else 'Unknown',
                'count': count,
                'percentage': round((count / sum(keys.values())) * 100, 1)
            }
            for key, count in sorted_keys
        ]
    
    def _get_common_modes(self, modes: Dict[int, int]) -> List[Dict]:
        """Get mode distribution."""
        total = sum(modes.values())
        if total == 0:
            return []
        
        return [
            {
                'mode': mode,
                'mode_name': MODE_NAMES.get(mode, 'Unknown'),
                'count': count,
                'percentage': round((count / total) * 100, 1)
            }
            for mode, count in sorted(modes.items(), key=lambda x: x[1], reverse=True)
        ]
    
    def _get_common_time_signatures(self, time_sigs: Dict[int, int]) -> List[Dict]:
        """Get common time signatures."""
        total = sum(time_sigs.values())
        if total == 0:
            return []
        
        return [
            {
                'time_signature': ts,
                'count': count,
                'percentage': round((count / total) * 100, 1)
            }
            for ts, count in sorted(time_sigs.items(), key=lambda x: x[1], reverse=True)
        ]
    
    def _calculate_avg(self, values: List[float]) -> Optional[float]:
        """Calculate average of values."""
        if not values:
            return None
        return round(statistics.mean(values), 3)
    
    def save_genre_profile(self, profile: Dict) -> bool:
        """
        Save genre profile to database.
        
        Args:
            profile: Genre profile dictionary
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert lists to JSON strings
                common_keys = json.dumps(profile.get('keys', []))
                common_modes = json.dumps(profile.get('modes', []))
                common_time_sigs = json.dumps(profile.get('time_signatures', []))
                
                tempo_data = profile.get('tempo', {})
                
                cursor.execute("""
                    INSERT OR REPLACE INTO genre_profiles (
                        genre, avg_tempo, tempo_std, common_keys, common_modes,
                        avg_energy, avg_danceability, avg_loudness,
                        common_time_signatures, song_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile['genre'],
                    tempo_data.get('avg'),
                    tempo_data.get('std'),
                    common_keys,
                    common_modes,
                    profile.get('energy'),
                    profile.get('danceability'),
                    profile.get('loudness'),
                    common_time_sigs,
                    profile.get('song_count')
                ))
                
                conn.commit()
                logger.info(f"Saved genre profile for: {profile['genre']}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving genre profile: {str(e)}")
            return False
    
    def get_genre_profile(self, genre: str) -> Optional[Dict]:
        """
        Retrieve genre profile from database.
        
        Args:
            genre: Genre name
            
        Returns:
            Genre profile dictionary or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM genre_profiles WHERE genre = ?
                """, (genre,))
                
                row = cursor.fetchone()
                
                if row:
                    profile = dict(row)
                    # Parse JSON fields
                    profile['common_keys'] = json.loads(profile['common_keys'])
                    profile['common_modes'] = json.loads(profile['common_modes'])
                    profile['common_time_signatures'] = json.loads(profile['common_time_signatures'])
                    return profile
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving genre profile: {str(e)}")
            return None
    
    def get_all_genre_names(self) -> List[str]:
        """
        Get list of all available genre profiles.
        
        Returns:
            List of genre names
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT genre FROM genre_profiles ORDER BY genre")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error retrieving genre names: {str(e)}")
            return []
    
    def suggest_parameters_for_genre(self, genre: str) -> Optional[Dict]:
        """
        Suggest generation parameters based on genre profile.
        
        Args:
            genre: Genre name
            
        Returns:
            Dictionary with suggested parameters
        """
        profile = self.get_genre_profile(genre)
        
        if not profile:
            # Try to analyze and save if not in database
            profile = self.analyze_genre(genre)
            if profile:
                self.save_genre_profile(profile)
            else:
                return None
        
        # Build suggestions
        suggestions = {
            'genre': genre,
            'tempo_bpm': profile.get('avg_tempo'),
            'tempo_range': {
                'min': max(60, profile.get('avg_tempo', 120) - profile.get('tempo_std', 20)),
                'max': min(200, profile.get('avg_tempo', 120) + profile.get('tempo_std', 20))
            },
            'recommended_keys': [],
            'recommended_mode': None,
            'energy_level': profile.get('avg_energy'),
            'danceability': profile.get('avg_danceability')
        }
        
        # Get top recommended keys
        common_keys = profile.get('common_keys', [])
        if common_keys:
            suggestions['recommended_keys'] = [
                f"{k['key_name']} ({k['percentage']}%)" 
                for k in common_keys[:3]
            ]
        
        # Get most common mode
        common_modes = profile.get('common_modes', [])
        if common_modes:
            top_mode = common_modes[0]
            suggestions['recommended_mode'] = f"{top_mode['mode_name']} ({top_mode['percentage']}%)"
        
        return suggestions
    
    def analyze_all_genres(self) -> int:
        """
        Analyze all genres in the database and save profiles.
        
        Returns:
            Number of genres analyzed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all unique genres
                cursor.execute("""
                    SELECT DISTINCT genre 
                    FROM songs 
                    WHERE genre IS NOT NULL 
                    ORDER BY genre
                """)
                
                genres = [row[0] for row in cursor.fetchall()]
                
                logger.info(f"Analyzing {len(genres)} genres...")
                
                analyzed = 0
                for genre in genres:
                    profile = self.analyze_genre(genre)
                    if profile and self.save_genre_profile(profile):
                        analyzed += 1
                
                logger.info(f"Successfully analyzed {analyzed} genres")
                return analyzed
                
        except Exception as e:
            logger.error(f"Error analyzing all genres: {str(e)}")
            return 0
    
    def analyze_key_tempo_patterns(self) -> List[Dict]:
        """
        Analyze correlations between musical keys and tempo ranges.
        
        Returns:
            List of key-tempo pattern dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Group songs by key and mode
                cursor.execute("""
                    SELECT key, mode, 
                           MIN(tempo) as tempo_min,
                           MAX(tempo) as tempo_max,
                           AVG(tempo) as tempo_avg,
                           COUNT(*) as count,
                           GROUP_CONCAT(DISTINCT genre) as genres
                    FROM songs
                    WHERE key IS NOT NULL 
                      AND mode IS NOT NULL 
                      AND tempo IS NOT NULL
                    GROUP BY key, mode
                    HAVING count > 10
                    ORDER BY key, mode
                """)
                
                patterns = []
                for row in cursor.fetchall():
                    patterns.append({
                        'key': row[0],
                        'key_name': KEY_NAMES[row[0]] if 0 <= row[0] < 12 else 'Unknown',
                        'mode': row[1],
                        'mode_name': MODE_NAMES.get(row[1], 'Unknown'),
                        'tempo_min': round(row[2], 2),
                        'tempo_max': round(row[3], 2),
                        'tempo_avg': round(row[4], 2),
                        'count': row[5],
                        'genres': row[6].split(',') if row[6] else []
                    })
                
                logger.info(f"Found {len(patterns)} key-tempo patterns")
                return patterns
                
        except Exception as e:
            logger.error(f"Error analyzing key-tempo patterns: {str(e)}")
            return []
    
    def save_key_tempo_patterns(self, patterns: List[Dict]) -> int:
        """
        Save key-tempo patterns to database.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            Number of patterns saved
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing patterns
                cursor.execute("DELETE FROM key_tempo_patterns")
                
                saved = 0
                for pattern in patterns:
                    genres_json = json.dumps(pattern.get('genres', []))
                    
                    cursor.execute("""
                        INSERT INTO key_tempo_patterns (
                            key, mode, tempo_min, tempo_max, tempo_avg,
                            occurrence_count, genres
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern['key'],
                        pattern['mode'],
                        pattern['tempo_min'],
                        pattern['tempo_max'],
                        pattern['tempo_avg'],
                        pattern['count'],
                        genres_json
                    ))
                    saved += 1
                
                conn.commit()
                logger.info(f"Saved {saved} key-tempo patterns")
                return saved
                
        except Exception as e:
            logger.error(f"Error saving key-tempo patterns: {str(e)}")
            return 0
