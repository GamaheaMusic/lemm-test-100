"""
Million Song Dataset Database Service
Handles SQLite database operations for MSD metadata storage and retrieval.
"""

import sqlite3
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class MSDDatabaseService:
    """Service for managing Million Song Dataset metadata in SQLite."""
    
    def __init__(self, db_path: str = "data/msd_metadata.db"):
        """
        Initialize MSD Database Service.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_data_directory()
        self._initialize_database()
        logger.info(f"MSD Database Service initialized: {db_path}")
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Songs table - core MSD metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS songs (
                    track_id TEXT PRIMARY KEY,
                    title TEXT,
                    artist_name TEXT,
                    release TEXT,
                    year INTEGER,
                    duration REAL,
                    tempo REAL,
                    key INTEGER,
                    mode INTEGER,
                    time_signature INTEGER,
                    loudness REAL,
                    energy REAL,
                    danceability REAL,
                    artist_familiarity REAL,
                    artist_hotttnesss REAL,
                    song_hotttnesss REAL,
                    tags TEXT,
                    genre TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Genre profiles table - aggregated patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS genre_profiles (
                    genre TEXT PRIMARY KEY,
                    avg_tempo REAL,
                    tempo_std REAL,
                    common_keys TEXT,
                    common_modes TEXT,
                    avg_energy REAL,
                    avg_danceability REAL,
                    avg_loudness REAL,
                    common_time_signatures TEXT,
                    song_count INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Key-tempo patterns table - music theory correlations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS key_tempo_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key INTEGER,
                    mode INTEGER,
                    tempo_min REAL,
                    tempo_max REAL,
                    tempo_avg REAL,
                    occurrence_count INTEGER,
                    genres TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_songs_genre 
                ON songs(genre)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_songs_tempo 
                ON songs(tempo)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_songs_key_mode 
                ON songs(key, mode)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_key_tempo_patterns 
                ON key_tempo_patterns(key, mode)
            """)
            
            conn.commit()
            logger.info("Database tables initialized successfully")
    
    def import_song(self, song_data: Dict) -> bool:
        """
        Import a single song into the database.
        
        Args:
            song_data: Dictionary containing song metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert lists/dicts to JSON strings
                tags = json.dumps(song_data.get('tags', [])) if 'tags' in song_data else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO songs (
                        track_id, title, artist_name, release, year,
                        duration, tempo, key, mode, time_signature,
                        loudness, energy, danceability,
                        artist_familiarity, artist_hotttnesss, song_hotttnesss,
                        tags, genre
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    song_data.get('track_id'),
                    song_data.get('title'),
                    song_data.get('artist_name'),
                    song_data.get('release'),
                    song_data.get('year'),
                    song_data.get('duration'),
                    song_data.get('tempo'),
                    song_data.get('key'),
                    song_data.get('mode'),
                    song_data.get('time_signature'),
                    song_data.get('loudness'),
                    song_data.get('energy'),
                    song_data.get('danceability'),
                    song_data.get('artist_familiarity'),
                    song_data.get('artist_hotttnesss'),
                    song_data.get('song_hotttnesss'),
                    tags,
                    song_data.get('genre')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error importing song {song_data.get('track_id')}: {str(e)}")
            return False
    
    def import_songs_batch(self, songs: List[Dict]) -> Tuple[int, int]:
        """
        Import multiple songs in a batch.
        
        Args:
            songs: List of song metadata dictionaries
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        for song in songs:
            if self.import_song(song):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Batch import complete: {successful} successful, {failed} failed")
        return successful, failed
    
    def get_song(self, track_id: str) -> Optional[Dict]:
        """
        Retrieve a song by track ID.
        
        Args:
            track_id: MSD track ID
            
        Returns:
            Song metadata dictionary or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM songs WHERE track_id = ?", (track_id,))
                row = cursor.fetchone()
                
                if row:
                    song = dict(row)
                    # Parse JSON fields
                    if song.get('tags'):
                        song['tags'] = json.loads(song['tags'])
                    return song
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving song {track_id}: {str(e)}")
            return None
    
    def get_songs_by_genre(self, genre: str, limit: int = 100) -> List[Dict]:
        """
        Retrieve songs by genre.
        
        Args:
            genre: Genre name
            limit: Maximum number of songs to return
            
        Returns:
            List of song metadata dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM songs 
                    WHERE genre = ? 
                    LIMIT ?
                """, (genre, limit))
                
                songs = [dict(row) for row in cursor.fetchall()]
                
                # Parse JSON fields
                for song in songs:
                    if song.get('tags'):
                        song['tags'] = json.loads(song['tags'])
                
                return songs
                
        except Exception as e:
            logger.error(f"Error retrieving songs by genre {genre}: {str(e)}")
            return []
    
    def get_songs_by_tempo_range(self, min_tempo: float, max_tempo: float, 
                                   limit: int = 100) -> List[Dict]:
        """
        Retrieve songs within a tempo range.
        
        Args:
            min_tempo: Minimum tempo (BPM)
            max_tempo: Maximum tempo (BPM)
            limit: Maximum number of songs to return
            
        Returns:
            List of song metadata dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM songs 
                    WHERE tempo BETWEEN ? AND ? 
                    ORDER BY RANDOM()
                    LIMIT ?
                """, (min_tempo, max_tempo, limit))
                
                songs = [dict(row) for row in cursor.fetchall()]
                
                # Parse JSON fields
                for song in songs:
                    if song.get('tags'):
                        song['tags'] = json.loads(song['tags'])
                
                return songs
                
        except Exception as e:
            logger.error(f"Error retrieving songs by tempo range: {str(e)}")
            return []
    
    def get_songs_by_key_mode(self, key: int, mode: int, 
                               limit: int = 100) -> List[Dict]:
        """
        Retrieve songs by musical key and mode.
        
        Args:
            key: Musical key (0-11, C=0)
            mode: Mode (0=minor, 1=major)
            limit: Maximum number of songs to return
            
        Returns:
            List of song metadata dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM songs 
                    WHERE key = ? AND mode = ? 
                    ORDER BY RANDOM()
                    LIMIT ?
                """, (key, mode, limit))
                
                songs = [dict(row) for row in cursor.fetchall()]
                
                # Parse JSON fields
                for song in songs:
                    if song.get('tags'):
                        song['tags'] = json.loads(song['tags'])
                
                return songs
                
        except Exception as e:
            logger.error(f"Error retrieving songs by key/mode: {str(e)}")
            return []
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with stats (total_songs, genres_count, etc.)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total songs
                cursor.execute("SELECT COUNT(*) FROM songs")
                total_songs = cursor.fetchone()[0]
                
                # Unique genres
                cursor.execute("SELECT COUNT(DISTINCT genre) FROM songs WHERE genre IS NOT NULL")
                genres_count = cursor.fetchone()[0]
                
                # Genre distribution
                cursor.execute("""
                    SELECT genre, COUNT(*) as count 
                    FROM songs 
                    WHERE genre IS NOT NULL
                    GROUP BY genre 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                top_genres = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Tempo stats
                cursor.execute("""
                    SELECT AVG(tempo), MIN(tempo), MAX(tempo) 
                    FROM songs 
                    WHERE tempo IS NOT NULL
                """)
                tempo_stats = cursor.fetchone()
                
                return {
                    'total_songs': total_songs,
                    'genres_count': genres_count,
                    'top_genres': top_genres,
                    'tempo_avg': round(tempo_stats[0], 2) if tempo_stats[0] else None,
                    'tempo_min': tempo_stats[1],
                    'tempo_max': tempo_stats[2]
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    def clear_database(self):
        """Clear all data from database tables (for testing/reset)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM songs")
                cursor.execute("DELETE FROM genre_profiles")
                cursor.execute("DELETE FROM key_tempo_patterns")
                conn.commit()
                logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
