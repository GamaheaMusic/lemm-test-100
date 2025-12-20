"""
Audio Analysis Service
Analyzes uploaded audio to automatically generate metadata for training.
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
import torch

logger = logging.getLogger(__name__)

# Try to import mutagen for metadata extraction
try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3
    from mutagen.easyid3 import EasyID3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logger.warning("mutagen not available - embedded metadata extraction disabled")


class AudioAnalysisService:
    """Service for analyzing audio files and generating metadata"""
    
    def __init__(self):
        """Initialize audio analysis service"""
        self.sample_rate = 44100
        
        # Genre classification mapping (simple heuristic-based)
        self.genre_classifiers = {
            'classical': {'tempo_range': (60, 140), 'spectral_centroid_mean': (1000, 3000)},
            'pop': {'tempo_range': (100, 130), 'spectral_centroid_mean': (2000, 4000)},
            'rock': {'tempo_range': (110, 150), 'spectral_centroid_mean': (2500, 5000)},
            'jazz': {'tempo_range': (80, 180), 'spectral_centroid_mean': (1500, 3500)},
            'electronic': {'tempo_range': (120, 140), 'spectral_centroid_mean': (3000, 6000)},
            'folk': {'tempo_range': (80, 120), 'spectral_centroid_mean': (1500, 3000)},
        }
        
        logger.info("AudioAnalysisService initialized")
    
    def extract_embedded_metadata(self, audio_path: str) -> Dict:
        """
        Extract embedded metadata from audio file tags (ID3, etc.)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with extracted metadata (empty dict if not available)
        """
        if not MUTAGEN_AVAILABLE:
            return {}
        
        try:
            audio_file = MutagenFile(audio_path, easy=True)
            if audio_file is None:
                return {}
            
            metadata = {}
            
            # Extract common tags
            if 'genre' in audio_file:
                genre = audio_file['genre'][0] if isinstance(audio_file['genre'], list) else audio_file['genre']
                metadata['genre'] = str(genre).lower()
                logger.info(f"Found embedded genre: {metadata['genre']}")
            
            if 'bpm' in audio_file:
                bpm = audio_file['bpm'][0] if isinstance(audio_file['bpm'], list) else audio_file['bpm']
                try:
                    metadata['bpm'] = int(float(bpm))
                    logger.info(f"Found embedded BPM: {metadata['bpm']}")
                except (ValueError, TypeError):
                    pass
            
            # Try to extract key/mood from comments or initialkey tag
            if 'initialkey' in audio_file:
                key = audio_file['initialkey'][0] if isinstance(audio_file['initialkey'], list) else audio_file['initialkey']
                metadata['key'] = str(key)
                logger.info(f"Found embedded key: {metadata['key']}")
            
            if 'comment' in audio_file:
                comment = audio_file['comment'][0] if isinstance(audio_file['comment'], list) else audio_file['comment']
                metadata['description'] = str(comment)
                logger.info(f"Found embedded comment/description")
            
            # Extract artist and title for additional context
            if 'artist' in audio_file:
                artist = audio_file['artist'][0] if isinstance(audio_file['artist'], list) else audio_file['artist']
                metadata['artist'] = str(artist)
            
            if 'title' in audio_file:
                title = audio_file['title'][0] if isinstance(audio_file['title'], list) else audio_file['title']
                metadata['title'] = str(title)
            
            if metadata:
                logger.info(f"Extracted {len(metadata)} metadata field(s) from file tags")
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to extract embedded metadata: {str(e)}")
            return {}
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Analyze audio file and generate comprehensive metadata.
        Prioritizes embedded metadata when available, only analyzing what's missing.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing:
                - bpm: Detected tempo (from tags or analysis)
                - key: Detected musical key (from tags or analysis)
                - genre: Predicted genre (from tags or analysis)
                - duration: Audio duration in seconds
                - energy: Overall energy level
                - spectral_features: Various spectral characteristics
                - segments: Suggested clip boundaries for training
        """
        try:
            logger.info(f"Analyzing audio: {audio_path}")
            
            # First, try to extract embedded metadata
            embedded_metadata = self.extract_embedded_metadata(audio_path)
            
            # Load audio for analysis
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Use embedded metadata when available, otherwise analyze
            if 'bpm' in embedded_metadata and embedded_metadata['bpm']:
                bpm = embedded_metadata['bpm']
                logger.info(f"Using embedded BPM: {bpm}")
            else:
                bpm = self._detect_tempo(y, sr)
                logger.info(f"Analyzed BPM: {bpm}")
            
            if 'key' in embedded_metadata and embedded_metadata['key']:
                key = embedded_metadata['key']
                logger.info(f"Using embedded key: {key}")
            else:
                key = self._detect_key(y, sr)
                logger.info(f"Analyzed key: {key}")
            
            if 'genre' in embedded_metadata and embedded_metadata['genre']:
                genre = embedded_metadata['genre']
                logger.info(f"Using embedded genre: {genre}")
            else:
                genre = self._predict_genre(y, sr)
                logger.info(f"Analyzed genre: {genre}")
            
            # Always analyze these features
            energy = self._calculate_energy(y)
            spectral_features = self._extract_spectral_features(y, sr)
            segments = self._suggest_segments(y, sr, duration)
            
            metadata = {
                'bpm': int(bpm) if not isinstance(bpm, int) else bpm,
                'key': key,
                'genre': genre,
                'duration': round(duration, 2),
                'energy': energy,
                'spectral_features': spectral_features,
                'segments': segments,
                'sample_rate': sr,
                'channels': 1 if y.ndim == 1 else y.shape[0],
                'has_embedded_metadata': bool(embedded_metadata)
            }
            
            # Add any additional embedded metadata
            if 'description' in embedded_metadata:
                metadata['description'] = embedded_metadata['description']
            if 'artist' in embedded_metadata:
                metadata['artist'] = embedded_metadata['artist']
            if 'title' in embedded_metadata:
                metadata['title'] = embedded_metadata['title']
            
            logger.info(f"Analysis complete: BPM={bpm}, Key={key}, Genre={genre} (embedded={bool(embedded_metadata)})")
            return metadata
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {str(e)}")
            raise
    
    def _detect_tempo(self, y: np.ndarray, sr: int) -> float:
        """Detect tempo (BPM) using librosa"""
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            # Handle array or scalar return
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0] if len(tempo) > 0 else 120.0
            return float(tempo)
        except Exception as e:
            logger.warning(f"Tempo detection failed: {str(e)}, defaulting to 120 BPM")
            return 120.0
    
    def _detect_key(self, y: np.ndarray, sr: int) -> str:
        """Detect musical key using chroma features"""
        try:
            # Compute chroma features
            chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_vals = chromagram.mean(axis=1)
            
            # Find dominant pitch class
            key_idx = np.argmax(chroma_vals)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Simple major/minor detection based on interval relationships
            # This is a simplified heuristic
            major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            # Rotate templates to match detected key
            major_rolled = np.roll(major_template, key_idx)
            minor_rolled = np.roll(minor_template, key_idx)
            
            # Correlate with actual chroma
            major_corr = np.corrcoef(chroma_vals, major_rolled)[0, 1]
            minor_corr = np.corrcoef(chroma_vals, minor_rolled)[0, 1]
            
            mode = "major" if major_corr > minor_corr else "minor"
            key = f"{keys[key_idx]} {mode}"
            
            return key
            
        except Exception as e:
            logger.warning(f"Key detection failed: {str(e)}, defaulting to C major")
            return "C major"
    
    def _predict_genre(self, y: np.ndarray, sr: int) -> str:
        """Predict genre using simple heuristic classification"""
        try:
            # Extract features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0]
            
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            sc_mean = np.mean(spectral_centroids)
            
            # Simple heuristic matching
            best_genre = 'unknown'
            best_score = -1
            
            for genre, criteria in self.genre_classifiers.items():
                tempo_min, tempo_max = criteria['tempo_range']
                sc_min, sc_max = criteria['spectral_centroid_mean']
                
                # Score based on how well it matches criteria
                tempo_score = 1.0 if tempo_min <= tempo <= tempo_max else 0.5
                sc_score = 1.0 if sc_min <= sc_mean <= sc_max else 0.5
                
                total_score = tempo_score * sc_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_genre = genre
            
            return best_genre
            
        except Exception as e:
            logger.warning(f"Genre prediction failed: {str(e)}, defaulting to unknown")
            return "unknown"
    
    def _calculate_energy(self, y: np.ndarray) -> str:
        """Calculate overall energy level (low/medium/high)"""
        try:
            rms = librosa.feature.rms(y=y)
            mean_rms = np.mean(rms)
            
            if mean_rms < 0.05:
                return "low"
            elif mean_rms < 0.15:
                return "medium"
            else:
                return "high"
                
        except Exception as e:
            logger.warning(f"Energy calculation failed: {str(e)}")
            return "medium"
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract various spectral features"""
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate))
            }
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {str(e)}")
            return {}
    
    def _suggest_segments(self, y: np.ndarray, sr: int, duration: float) -> list:
        """
        Suggest clip boundaries for training
        Splits audio into 10-30 second segments at natural boundaries
        """
        try:
            # Target clip length: 10-30 seconds
            min_clip_length = 10.0  # seconds
            max_clip_length = 30.0
            
            # Detect onset events (musical boundaries)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            segments = []
            current_start = 0.0
            
            for onset_time in onset_times:
                segment_length = onset_time - current_start
                
                # If segment is within acceptable range, add it
                if min_clip_length <= segment_length <= max_clip_length:
                    segments.append({
                        'start': round(current_start, 2),
                        'end': round(onset_time, 2),
                        'duration': round(segment_length, 2)
                    })
                    current_start = onset_time
                    
                # If segment is too long, force split at max_clip_length
                elif segment_length > max_clip_length:
                    while current_start + max_clip_length < onset_time:
                        segments.append({
                            'start': round(current_start, 2),
                            'end': round(current_start + max_clip_length, 2),
                            'duration': max_clip_length
                        })
                        current_start += max_clip_length
            
            # Add final segment
            if duration - current_start >= min_clip_length:
                segments.append({
                    'start': round(current_start, 2),
                    'end': round(duration, 2),
                    'duration': round(duration - current_start, 2)
                })
            
            # If no segments found, split into equal chunks
            if not segments:
                num_clips = int(np.ceil(duration / max_clip_length))
                clip_length = duration / num_clips
                
                for i in range(num_clips):
                    start = i * clip_length
                    end = min((i + 1) * clip_length, duration)
                    segments.append({
                        'start': round(start, 2),
                        'end': round(end, 2),
                        'duration': round(end - start, 2)
                    })
            
            logger.info(f"Suggested {len(segments)} training segments")
            return segments
            
        except Exception as e:
            logger.error(f"Segment suggestion failed: {str(e)}")
            # Fallback: simple equal splits
            num_clips = int(np.ceil(duration / 20.0))
            clip_length = duration / num_clips
            return [
                {
                    'start': round(i * clip_length, 2),
                    'end': round(min((i + 1) * clip_length, duration), 2),
                    'duration': round(clip_length, 2)
                }
                for i in range(num_clips)
            ]
    
    def split_audio_to_clips(
        self, 
        audio_path: str, 
        output_dir: str, 
        segments: Optional[list] = None,
        metadata: Optional[Dict] = None
    ) -> list:
        """
        Split audio file into training clips based on suggested segments
        
        Args:
            audio_path: Path to source audio file
            output_dir: Directory to save clips
            segments: Optional segment list (if None, will auto-detect)
            metadata: Optional metadata to include in filenames
            
        Returns:
            List of paths to created clip files
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Get segments if not provided
            if segments is None:
                segments = self._suggest_segments(y, sr, duration)
            
            # Generate base filename
            base_name = Path(audio_path).stem
            if metadata and 'genre' in metadata:
                base_name = f"{metadata['genre']}_{base_name}"
            
            clip_paths = []
            
            for i, segment in enumerate(segments):
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                
                clip_audio = y[start_sample:end_sample]
                
                # Create filename
                clip_filename = f"{base_name}_clip{i+1:03d}.wav"
                clip_path = output_path / clip_filename
                
                # Save clip
                sf.write(clip_path, clip_audio, sr)
                clip_paths.append(str(clip_path))
                
                logger.info(f"Created clip {i+1}/{len(segments)}: {clip_filename}")
            
            logger.info(f"Split audio into {len(clip_paths)} clips")
            return clip_paths
            
        except Exception as e:
            logger.error(f"Audio splitting failed: {str(e)}")
            raise
    
    def separate_vocal_stems(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """
        Separate audio into vocal and instrumental stems
        Uses Demucs for separation
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save stems
            
        Returns:
            Dictionary with paths to separated stems
        """
        try:
            from backend.services.stem_enhancement_service import StemEnhancementService
            
            logger.info(f"Separating stems from: {audio_path}")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Initialize stem separator
            stem_service = StemEnhancementService()
            
            # Separate stems (without enhancement processing)
            temp_input = Path("temp_stem_input.wav")
            sf.write(temp_input, y, sr)
            
            # Use Demucs to separate
            # Note: This reuses the stem enhancement service's Demucs model
            # but we won't apply the enhancement processing
            separated = stem_service._separate_stems(str(temp_input))
            
            # Clean up temp file
            temp_input.unlink()
            
            # Save stems
            base_name = Path(audio_path).stem
            stem_paths = {}
            
            for stem_name, stem_audio in separated.items():
                stem_filename = f"{base_name}_{stem_name}.wav"
                stem_path = output_path / stem_filename
                sf.write(stem_path, stem_audio.T, sr)
                stem_paths[stem_name] = str(stem_path)
                logger.info(f"Saved {stem_name} stem: {stem_filename}")
            
            return stem_paths
            
        except Exception as e:
            logger.error(f"Stem separation failed: {str(e)}")
            # Return original audio as fallback
            return {'full_mix': audio_path}
