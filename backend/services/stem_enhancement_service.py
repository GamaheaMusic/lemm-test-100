"""
Stem Enhancement Service
Separates audio into stems (vocals, drums, bass, other) and enhances each independently
"""
import os
import logging
import numpy as np
import soundfile as sf
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class StemEnhancementService:
    """Service for stem separation and per-stem enhancement"""
    
    def __init__(self):
        """Initialize stem enhancement service"""
        self.separator = None
        self.model_loaded = False
        logger.info("Stem enhancement service initialized")
    
    def _load_model(self):
        """Lazy load Demucs model (1.3GB download on first use)"""
        if self.model_loaded:
            return
        
        try:
            logger.info("Loading Demucs model (htdemucs_ft)...")
            import demucs.api
            
            # Use htdemucs_ft - best quality for music
            self.separator = demucs.api.Separator(model="htdemucs_ft")
            self.model_loaded = True
            logger.info("Demucs model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}", exc_info=True)
            raise
    
    def enhance_clip(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        enhancement_level: str = "balanced"
    ) -> str:
        """
        Enhance audio quality through stem separation and processing
        
        Args:
            audio_path: Input audio file path
            output_path: Output audio file path (optional)
            enhancement_level: 'fast', 'balanced', or 'maximum'
            
        Returns:
            Path to enhanced audio file
        """
        try:
            logger.info(f"Starting stem enhancement: {audio_path} (level: {enhancement_level})")
            
            # Load model if not already loaded
            self._load_model()
            
            # Generate output path if not provided
            if output_path is None:
                base, ext = os.path.splitext(audio_path)
                output_path = f"{base}_enhanced{ext}"
            
            # Load audio
            logger.info(f"Loading audio from: {audio_path}")
            audio, sr = sf.read(audio_path)
            
            # Ensure audio is in correct format for Demucs (2D array, stereo)
            if audio.ndim == 1:
                # Mono to stereo
                audio = np.stack([audio, audio], axis=1)
            
            # Separate stems
            logger.info("Separating stems with Demucs...")
            origin, stems = self.separator.separate_audio_file(audio_path)
            
            # stems is a dict: {'vocals': ndarray, 'drums': ndarray, 'bass': ndarray, 'other': ndarray}
            logger.info(f"Stems separated: {list(stems.keys())}")
            
            # Process each stem based on enhancement level
            if enhancement_level == "fast":
                # Fast mode: minimal processing, just denoise vocals
                vocals_enhanced = self._denoise_stem(stems['vocals'], sr, intensity=0.5)
                drums_enhanced = stems['drums']
                bass_enhanced = stems['bass']
                other_enhanced = stems['other']
                
            elif enhancement_level == "balanced":
                # Balanced mode: denoise + basic processing
                vocals_enhanced = self._enhance_vocals(stems['vocals'], sr, aggressive=False)
                drums_enhanced = self._enhance_drums(stems['drums'], sr, aggressive=False)
                bass_enhanced = self._enhance_bass(stems['bass'], sr, aggressive=False)
                other_enhanced = self._denoise_stem(stems['other'], sr, intensity=0.5)
                
            else:  # maximum
                # Maximum mode: full processing
                vocals_enhanced = self._enhance_vocals(stems['vocals'], sr, aggressive=True)
                drums_enhanced = self._enhance_drums(stems['drums'], sr, aggressive=True)
                bass_enhanced = self._enhance_bass(stems['bass'], sr, aggressive=True)
                other_enhanced = self._enhance_other(stems['other'], sr)
            
            # Reassemble stems
            logger.info("Reassembling enhanced stems...")
            enhanced_audio = (
                vocals_enhanced + 
                drums_enhanced + 
                bass_enhanced + 
                other_enhanced
            )
            
            # Normalize to prevent clipping
            max_val = np.abs(enhanced_audio).max()
            if max_val > 0:
                enhanced_audio = enhanced_audio / max_val * 0.95
            
            # Save enhanced audio
            logger.info(f"Saving enhanced audio to: {output_path}")
            sf.write(output_path, enhanced_audio, sr)
            
            logger.info(f"Stem enhancement complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Stem enhancement failed: {e}", exc_info=True)
            # Return original if enhancement fails
            return audio_path
    
    def _denoise_stem(self, stem: np.ndarray, sr: int, intensity: float = 1.0) -> np.ndarray:
        """
        Apply noise reduction to a stem
        
        Args:
            stem: Audio stem (ndarray)
            sr: Sample rate
            intensity: Denoising intensity (0-1)
            
        Returns:
            Denoised stem
        """
        try:
            import noisereduce as nr
            
            # Handle stereo
            if stem.ndim == 2:
                # Process each channel
                denoised = np.zeros_like(stem)
                for ch in range(stem.shape[1]):
                    denoised[:, ch] = nr.reduce_noise(
                        y=stem[:, ch],
                        sr=sr,
                        stationary=True,
                        prop_decrease=intensity,
                        freq_mask_smooth_hz=500,
                        time_mask_smooth_ms=50
                    )
                return denoised
            else:
                # Mono
                return nr.reduce_noise(
                    y=stem,
                    sr=sr,
                    stationary=True,
                    prop_decrease=intensity
                )
                
        except Exception as e:
            logger.warning(f"Denoising failed: {e}, returning original stem")
            return stem
    
    def _enhance_vocals(self, vocals: np.ndarray, sr: int, aggressive: bool = False) -> np.ndarray:
        """
        Enhance vocal stem (critical for LyricMind AI vocals)
        
        Args:
            vocals: Vocal stem
            sr: Sample rate
            aggressive: Use more aggressive processing
            
        Returns:
            Enhanced vocals
        """
        try:
            # 1. Denoise (reduce AI artifacts)
            intensity = 1.0 if aggressive else 0.7
            vocals_clean = self._denoise_stem(vocals, sr, intensity=intensity)
            
            # 2. Apply subtle compression and EQ with Pedalboard
            try:
                from pedalboard import Pedalboard, Compressor, HighShelfFilter, LowpassFilter
                
                board = Pedalboard([
                    # Remove very high frequencies (often artifacts)
                    LowpassFilter(cutoff_frequency_hz=16000),
                    # Subtle compression for consistency
                    Compressor(
                        threshold_db=-20,
                        ratio=3 if aggressive else 2,
                        attack_ms=5,
                        release_ms=50
                    ),
                    # Add air
                    HighShelfFilter(cutoff_frequency_hz=8000, gain_db=2 if aggressive else 1)
                ])
                
                vocals_processed = board(vocals_clean, sr)
                logger.info(f"Vocals enhanced (aggressive={aggressive})")
                return vocals_processed
                
            except Exception as e:
                logger.warning(f"Pedalboard processing failed: {e}, using denoised only")
                return vocals_clean
                
        except Exception as e:
            logger.error(f"Vocal enhancement failed: {e}", exc_info=True)
            return vocals
    
    def _enhance_drums(self, drums: np.ndarray, sr: int, aggressive: bool = False) -> np.ndarray:
        """
        Enhance drum stem
        
        Args:
            drums: Drum stem
            sr: Sample rate
            aggressive: Use more aggressive processing
            
        Returns:
            Enhanced drums
        """
        try:
            from pedalboard import Pedalboard, NoiseGate, Compressor
            
            board = Pedalboard([
                # Gate to clean up between hits
                NoiseGate(
                    threshold_db=-40 if aggressive else -35,
                    ratio=10,
                    attack_ms=1,
                    release_ms=100
                ),
                # Compression for punch
                Compressor(
                    threshold_db=-15,
                    ratio=4 if aggressive else 3,
                    attack_ms=10,
                    release_ms=100
                )
            ])
            
            drums_processed = board(drums, sr)
            logger.info(f"Drums enhanced (aggressive={aggressive})")
            return drums_processed
            
        except Exception as e:
            logger.warning(f"Drum enhancement failed: {e}, returning original")
            return drums
    
    def _enhance_bass(self, bass: np.ndarray, sr: int, aggressive: bool = False) -> np.ndarray:
        """
        Enhance bass stem
        
        Args:
            bass: Bass stem
            sr: Sample rate
            aggressive: Use more aggressive processing
            
        Returns:
            Enhanced bass
        """
        try:
            from pedalboard import Pedalboard, HighpassFilter, Compressor
            
            board = Pedalboard([
                # Remove sub-bass rumble
                HighpassFilter(cutoff_frequency_hz=30),
                # Compression for consistency
                Compressor(
                    threshold_db=-18,
                    ratio=3 if aggressive else 2.5,
                    attack_ms=30,
                    release_ms=200
                )
            ])
            
            bass_processed = board(bass, sr)
            logger.info(f"Bass enhanced (aggressive={aggressive})")
            return bass_processed
            
        except Exception as e:
            logger.warning(f"Bass enhancement failed: {e}, returning original")
            return bass
    
    def _enhance_other(self, other: np.ndarray, sr: int) -> np.ndarray:
        """
        Enhance other instruments stem
        
        Args:
            other: Other instruments stem
            sr: Sample rate
            
        Returns:
            Enhanced other stem
        """
        try:
            # Spectral cleanup with moderate denoising
            other_clean = self._denoise_stem(other, sr, intensity=0.5)
            logger.info("Other instruments enhanced")
            return other_clean
            
        except Exception as e:
            logger.warning(f"Other enhancement failed: {e}, returning original")
            return other
