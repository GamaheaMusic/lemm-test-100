"""
Style Consistency Service
Uses audio feature extraction and style embeddings to ensure consistent generation
"""
import os
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import torch

logger = logging.getLogger(__name__)

class StyleConsistencyService:
    """
    Ensures style consistency across generated clips by analyzing existing audio
    and providing style guidance for new generations
    """
    
    def __init__(self):
        self.sample_rate = 44100
        logger.info("Style Consistency Service initialized")
    
    def extract_audio_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive audio features for style analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            features = {}
            
            # Spectral features
            features['mel_spectrogram'] = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512
            )
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=audio, sr=sr, n_fft=2048, hop_length=512
            )
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
                y=audio, sr=sr, n_fft=2048, hop_length=512
            )
            features['spectral_contrast'] = librosa.feature.spectral_contrast(
                y=audio, sr=sr, n_fft=2048, hop_length=512, n_bands=6
            )
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
                y=audio, sr=sr, n_fft=2048, hop_length=512
            )
            
            # Temporal features
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
                audio, frame_length=2048, hop_length=512
            )
            features['rms'] = librosa.feature.rms(
                y=audio, frame_length=2048, hop_length=512
            )
            
            # Harmonic/percussive
            harmonic, percussive = librosa.effects.hpss(audio)
            features['harmonic_ratio'] = np.mean(np.abs(harmonic)) / (np.mean(np.abs(audio)) + 1e-10)
            features['percussive_ratio'] = np.mean(np.abs(percussive)) / (np.mean(np.abs(audio)) + 1e-10)
            
            # Chroma features
            features['chroma'] = librosa.feature.chroma_stft(
                y=audio, sr=sr, n_chroma=12, n_fft=2048, hop_length=512
            )
            
            # MFCC
            features['mfcc'] = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=20
            )
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = tempo
            features['beat_frames'] = beats
            
            logger.info(f"Extracted features from {audio_path}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features from {audio_path}: {e}")
            return {}
    
    def compute_style_statistics(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute statistical summaries of audio features for style matching
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary of style statistics
        """
        stats = {}
        
        # Compute mean/std for spectral features
        for key in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
                    'zero_crossing_rate', 'rms']:
            if key in features:
                stats[f'{key}_mean'] = float(np.mean(features[key]))
                stats[f'{key}_std'] = float(np.std(features[key]))
        
        # Spectral contrast summary
        if 'spectral_contrast' in features:
            stats['spectral_contrast_mean'] = float(np.mean(features['spectral_contrast']))
            stats['spectral_contrast_std'] = float(np.std(features['spectral_contrast']))
        
        # Harmonic/percussive balance
        stats['harmonic_ratio'] = float(features.get('harmonic_ratio', 0.5))
        stats['percussive_ratio'] = float(features.get('percussive_ratio', 0.5))
        
        # Tempo
        stats['tempo'] = float(features.get('tempo', 120.0))
        
        # Chroma energy distribution
        if 'chroma' in features:
            chroma_mean = np.mean(features['chroma'], axis=1)
            stats['chroma_energy'] = chroma_mean.tolist()
        
        # MFCC summary (timbre)
        if 'mfcc' in features:
            mfcc_mean = np.mean(features['mfcc'], axis=1)
            stats['timbre_signature'] = mfcc_mean[:13].tolist()  # First 13 MFCCs
        
        return stats
    
    def analyze_timeline_style(self, clip_paths: List[str]) -> Dict[str, any]:
        """
        Analyze style characteristics of all clips on timeline
        
        Args:
            clip_paths: List of audio file paths from timeline
            
        Returns:
            Aggregate style profile
        """
        if not clip_paths:
            return {}
        
        all_features = []
        all_stats = []
        
        for path in clip_paths:
            if os.path.exists(path):
                features = self.extract_audio_features(path)
                if features:
                    stats = self.compute_style_statistics(features)
                    all_features.append(features)
                    all_stats.append(stats)
        
        if not all_stats:
            return {}
        
        # Aggregate statistics across all clips
        aggregate_style = {}
        
        # Average numerical features
        numeric_keys = [k for k in all_stats[0].keys() if isinstance(all_stats[0][k], (int, float))]
        for key in numeric_keys:
            values = [stats[key] for stats in all_stats if key in stats]
            aggregate_style[key] = float(np.mean(values))
        
        # Average chroma and timbre
        if 'chroma_energy' in all_stats[0]:
            chroma_arrays = [np.array(stats['chroma_energy']) for stats in all_stats if 'chroma_energy' in stats]
            if chroma_arrays:
                aggregate_style['chroma_energy'] = np.mean(chroma_arrays, axis=0).tolist()
        
        if 'timbre_signature' in all_stats[0]:
            timbre_arrays = [np.array(stats['timbre_signature']) for stats in all_stats if 'timbre_signature' in stats]
            if timbre_arrays:
                aggregate_style['timbre_signature'] = np.mean(timbre_arrays, axis=0).tolist()
        
        logger.info(f"Analyzed style from {len(clip_paths)} clips")
        return aggregate_style
    
    def create_style_reference_audio(self, clip_paths: List[str], output_path: str) -> str:
        """
        Mix all timeline clips into a single reference audio for style guidance
        
        Args:
            clip_paths: List of audio file paths
            output_path: Where to save the reference audio
            
        Returns:
            Path to created reference audio
        """
        if not clip_paths:
            raise ValueError("No clips provided for style reference")
        
        try:
            # Load all clips and find max duration
            clips_audio = []
            max_length = 0
            
            for path in clip_paths:
                if os.path.exists(path):
                    audio, sr = librosa.load(path, sr=self.sample_rate)
                    clips_audio.append(audio)
                    max_length = max(max_length, len(audio))
            
            if not clips_audio:
                raise ValueError("No valid audio files found")
            
            # Pad all clips to same length
            padded_clips = []
            for audio in clips_audio:
                if len(audio) < max_length:
                    audio = np.pad(audio, (0, max_length - len(audio)))
                padded_clips.append(audio)
            
            # Mix clips (average them)
            mixed_audio = np.mean(padded_clips, axis=0)
            
            # Normalize
            mixed_audio = librosa.util.normalize(mixed_audio)
            
            # Save reference audio
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, mixed_audio, self.sample_rate)
            
            logger.info(f"Created style reference audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create style reference: {e}")
            raise
    
    def enhance_prompt_with_style(
        self, 
        base_prompt: str, 
        style_profile: Dict[str, any]
    ) -> str:
        """
        Enhance generation prompt with style characteristics
        
        Args:
            base_prompt: User's original prompt
            style_profile: Style analysis from timeline
            
        Returns:
            Enhanced prompt
        """
        if not style_profile:
            return base_prompt
        
        style_descriptors = []
        
        # Tempo descriptor
        tempo = style_profile.get('tempo', 120)
        if tempo < 90:
            style_descriptors.append("slow tempo")
        elif tempo > 140:
            style_descriptors.append("fast tempo")
        
        # Energy/dynamics descriptor
        rms_mean = style_profile.get('rms_mean', 0.1)
        if rms_mean > 0.15:
            style_descriptors.append("energetic")
        elif rms_mean < 0.08:
            style_descriptors.append("gentle")
        
        # Harmonic/percussive balance
        harmonic_ratio = style_profile.get('harmonic_ratio', 0.5)
        percussive_ratio = style_profile.get('percussive_ratio', 0.5)
        
        if harmonic_ratio > percussive_ratio * 1.3:
            style_descriptors.append("melodic")
        elif percussive_ratio > harmonic_ratio * 1.3:
            style_descriptors.append("rhythmic")
        
        # Spectral brightness
        centroid_mean = style_profile.get('spectral_centroid_mean', 2000)
        if centroid_mean > 3000:
            style_descriptors.append("bright")
        elif centroid_mean < 1500:
            style_descriptors.append("warm")
        
        # Combine with base prompt
        if style_descriptors:
            enhanced = f"{base_prompt}, consistent with existing style: {', '.join(style_descriptors)}"
            logger.info(f"Enhanced prompt: {enhanced}")
            return enhanced
        
        return base_prompt
    
    def get_style_guidance_for_generation(
        self, 
        timeline_clips: List[Dict]
    ) -> Tuple[Optional[str], Dict[str, any]]:
        """
        Prepare style guidance for new generation
        
        Args:
            timeline_clips: List of clip dictionaries from timeline
            
        Returns:
            Tuple of (reference_audio_path, style_profile)
        """
        if not timeline_clips:
            logger.info("No existing clips - no style guidance available")
            return None, {}
        
        # Get audio paths from clips
        clip_paths = []
        for clip in timeline_clips:
            audio_path = clip.get('music_path') or clip.get('mixed_path') or clip.get('file_path')
            if audio_path and os.path.exists(audio_path):
                clip_paths.append(audio_path)
        
        if not clip_paths:
            return None, {}
        
        # Analyze timeline style
        style_profile = self.analyze_timeline_style(clip_paths)
        
        # Create reference audio (mix of all clips)
        try:
            ref_dir = os.path.join('outputs', 'style_reference')
            os.makedirs(ref_dir, exist_ok=True)
            ref_path = os.path.join(ref_dir, 'timeline_reference.wav')
            
            reference_audio = self.create_style_reference_audio(clip_paths, ref_path)
            logger.info(f"Style guidance ready: {len(clip_paths)} clips analyzed")
            return reference_audio, style_profile
            
        except Exception as e:
            logger.error(f"Failed to create reference audio: {e}")
            return None, style_profile
