"""
MSD Subset Importer
Downloads and imports Million Song Dataset subset for testing and development.
"""

import logging
import os
import json
import h5py
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from backend.services.msd_database_service import MSDDatabaseService
from backend.services.genre_profiler import GenreProfiler

logger = logging.getLogger(__name__)


class MSDSubsetImporter:
    """Handles downloading and importing MSD subset data."""
    
    # MSD subset download URLs
    MSD_SUBSET_URL = "http://static.echonest.com/millionsongsubset_full.tar.gz"
    
    def __init__(self, data_dir: str = "data/msd"):
        """
        Initialize MSD Subset Importer.
        
        Args:
            data_dir: Directory to store MSD data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_service = MSDDatabaseService()
        self.profiler = GenreProfiler()
        
        logger.info(f"MSD Subset Importer initialized: {data_dir}")
    
    def parse_h5_file(self, h5_path: str) -> Optional[Dict]:
        """
        Parse a single HDF5 file from MSD.
        
        Args:
            h5_path: Path to HDF5 file
            
        Returns:
            Dictionary with song metadata
        """
        try:
            with h5py.File(h5_path, 'r') as h5:
                # Extract metadata
                song_data = {
                    'track_id': h5['metadata']['songs']['track_id'][0].decode('utf-8'),
                    'title': h5['metadata']['songs']['title'][0].decode('utf-8'),
                    'artist_name': h5['metadata']['songs']['artist_name'][0].decode('utf-8'),
                    'release': h5['metadata']['songs']['release'][0].decode('utf-8'),
                    'year': int(h5['musicbrainz']['songs']['year'][0]) if h5['musicbrainz']['songs']['year'][0] > 0 else None,
                    'duration': float(h5['analysis']['songs']['duration'][0]),
                    'tempo': float(h5['analysis']['songs']['tempo'][0]),
                    'key': int(h5['analysis']['songs']['key'][0]),
                    'mode': int(h5['analysis']['songs']['mode'][0]),
                    'time_signature': int(h5['analysis']['songs']['time_signature'][0]),
                    'loudness': float(h5['analysis']['songs']['loudness'][0]),
                    'energy': float(h5['analysis']['songs']['energy'][0]) if 'energy' in h5['analysis']['songs'].dtype.names else None,
                    'danceability': float(h5['analysis']['songs']['danceability'][0]) if 'danceability' in h5['analysis']['songs'].dtype.names else None,
                    'artist_familiarity': float(h5['metadata']['songs']['artist_familiarity'][0]),
                    'artist_hotttnesss': float(h5['metadata']['songs']['artist_hotttnesss'][0]),
                    'song_hotttnesss': float(h5['metadata']['songs']['song_hotttnesss'][0]) if h5['metadata']['songs']['song_hotttnesss'][0] > 0 else None,
                }
                
                # Extract tags if available
                if 'artist_mbtags' in h5['metadata']:
                    tags = [tag.decode('utf-8') for tag in h5['metadata']['artist_mbtags'][:10]]
                    song_data['tags'] = tags
                
                return song_data
                
        except Exception as e:
            logger.error(f"Error parsing H5 file {h5_path}: {str(e)}")
            return None
    
    def import_from_directory(self, directory: str, max_files: int = 10000) -> Dict:
        """
        Import songs from a directory of HDF5 files.
        
        Args:
            directory: Directory containing HDF5 files
            max_files: Maximum number of files to import
            
        Returns:
            Dictionary with import statistics
        """
        h5_files = list(Path(directory).rglob("*.h5"))[:max_files]
        
        if not h5_files:
            logger.warning(f"No HDF5 files found in {directory}")
            return {'imported': 0, 'failed': 0}
        
        logger.info(f"Found {len(h5_files)} HDF5 files to import")
        
        imported = 0
        failed = 0
        
        for h5_path in tqdm(h5_files, desc="Importing songs"):
            song_data = self.parse_h5_file(str(h5_path))
            
            if song_data and self.db_service.import_song(song_data):
                imported += 1
            else:
                failed += 1
        
        logger.info(f"Import complete: {imported} imported, {failed} failed")
        
        return {
            'imported': imported,
            'failed': failed,
            'total': len(h5_files)
        }
    
    def import_from_json(self, json_path: str) -> Dict:
        """
        Import songs from a JSON file (for testing without full MSD).
        
        Args:
            json_path: Path to JSON file with song data
            
        Returns:
            Dictionary with import statistics
        """
        try:
            with open(json_path, 'r') as f:
                songs = json.load(f)
            
            if not isinstance(songs, list):
                songs = [songs]
            
            successful, failed = self.db_service.import_songs_batch(songs)
            
            return {
                'imported': successful,
                'failed': failed,
                'total': len(songs)
            }
            
        except Exception as e:
            logger.error(f"Error importing from JSON: {str(e)}")
            return {'imported': 0, 'failed': 0, 'total': 0}
    
    def generate_sample_data(self, count: int = 100) -> List[Dict]:
        """
        Generate sample MSD-like data for testing (without actual MSD files).
        
        Args:
            count: Number of sample songs to generate
            
        Returns:
            List of sample song dictionaries
        """
        import random
        
        genres = ['rock', 'pop', 'jazz', 'electronic', 'hip-hop', 'classical', 
                  'country', 'blues', 'metal', 'folk']
        
        artists = ['The Beatles', 'Miles Davis', 'Daft Punk', 'Bob Dylan', 
                   'Metallica', 'Bach', 'Johnny Cash', 'Led Zeppelin']
        
        samples = []
        
        for i in range(count):
            genre = random.choice(genres)
            
            # Genre-specific parameter ranges
            if genre == 'jazz':
                tempo_range = (80, 140)
                energy_range = (0.3, 0.7)
            elif genre == 'electronic':
                tempo_range = (120, 140)
                energy_range = (0.7, 0.95)
            elif genre == 'metal':
                tempo_range = (140, 180)
                energy_range = (0.8, 0.95)
            elif genre == 'classical':
                tempo_range = (60, 120)
                energy_range = (0.2, 0.6)
            else:
                tempo_range = (90, 130)
                energy_range = (0.5, 0.8)
            
            song = {
                'track_id': f'TEST{i:06d}',
                'title': f'Sample Song {i+1}',
                'artist_name': random.choice(artists),
                'release': f'Album {i//10 + 1}',
                'year': random.randint(1960, 2023),
                'duration': random.uniform(120, 300),
                'tempo': random.uniform(*tempo_range),
                'key': random.randint(0, 11),
                'mode': random.randint(0, 1),
                'time_signature': random.choice([3, 4, 5, 7]),
                'loudness': random.uniform(-15, -3),
                'energy': random.uniform(*energy_range),
                'danceability': random.uniform(0.3, 0.9),
                'artist_familiarity': random.uniform(0.4, 0.9),
                'artist_hotttnesss': random.uniform(0.3, 0.8),
                'song_hotttnesss': random.uniform(0.3, 0.8),
                'tags': random.sample(['rock', 'pop', 'dance', 'electronic', 'indie'], 
                                     k=random.randint(1, 3)),
                'genre': genre
            }
            
            samples.append(song)
        
        return samples
    
    def import_sample_data(self, count: int = 1000) -> Dict:
        """
        Generate and import sample data for testing.
        
        Args:
            count: Number of sample songs to generate and import
            
        Returns:
            Dictionary with import statistics
        """
        logger.info(f"Generating {count} sample songs...")
        samples = self.generate_sample_data(count)
        
        logger.info(f"Importing {len(samples)} sample songs...")
        successful, failed = self.db_service.import_songs_batch(samples)
        
        logger.info("Analyzing genres...")
        genres_analyzed = self.profiler.analyze_all_genres()
        
        logger.info("Analyzing key-tempo patterns...")
        patterns = self.profiler.analyze_key_tempo_patterns()
        patterns_saved = self.profiler.save_key_tempo_patterns(patterns)
        
        return {
            'imported': successful,
            'failed': failed,
            'total': len(samples),
            'genres_analyzed': genres_analyzed,
            'patterns_saved': patterns_saved
        }
    
    def get_import_status(self) -> Dict:
        """
        Get current import status from database.
        
        Returns:
            Dictionary with database statistics
        """
        stats = self.db_service.get_database_stats()
        
        genres = self.profiler.get_all_genre_names()
        
        return {
            **stats,
            'genres_with_profiles': len(genres),
            'available_genres': genres
        }
