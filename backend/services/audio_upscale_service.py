"""
Audio Upscale Service
Uses AudioSR for neural upsampling to 48kHz
"""
import os
import logging
import numpy as np
import soundfile as sf
from typing import Optional

logger = logging.getLogger(__name__)

class AudioUpscaleService:
    """Service for upscaling audio to 48kHz using AudioSR"""
    
    def __init__(self):
        """Initialize audio upscale service"""
        self.model = None
        self.model_loaded = False
        logger.info("Audio upscale service initialized")
    
    def _load_model(self):
        """Lazy load AudioSR model"""
        if self.model_loaded:
            return
        
        try:
            logger.info("Loading AudioSR model...")
            from audiosr import build_model, super_resolution
            
            # Build AudioSR model (will download on first use)
            self.model = build_model(model_name="basic", device="auto")
            self.super_resolution = super_resolution
            self.model_loaded = True
            logger.info("AudioSR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load AudioSR model: {e}", exc_info=True)
            raise
    
    def upscale_audio(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        target_sr: int = 48000
    ) -> str:
        """
        Upscale audio to higher sample rate using neural super-resolution
        
        Args:
            audio_path: Input audio file path
            output_path: Output audio file path (optional)
            target_sr: Target sample rate (default: 48000)
            
        Returns:
            Path to upscaled audio file
        """
        try:
            logger.info(f"Starting audio upscaling: {audio_path} -> {target_sr}Hz")
            
            # Load model if not already loaded
            self._load_model()
            
            # Generate output path if not provided
            if output_path is None:
                base, ext = os.path.splitext(audio_path)
                output_path = f"{base}_48kHz{ext}"
            
            # Load audio
            logger.info(f"Loading audio from: {audio_path}")
            audio, sr = sf.read(audio_path)
            
            # Check if upscaling is needed
            if sr >= target_sr:
                logger.warning(f"Audio already at {sr}Hz, >= target {target_sr}Hz. Skipping upscale.")
                return audio_path
            
            logger.info(f"Original sample rate: {sr}Hz, upscaling to {target_sr}Hz")
            
            # AudioSR expects specific format
            # Handle stereo by processing each channel separately
            if audio.ndim == 2:
                logger.info("Processing stereo audio (2 channels)")
                upscaled_channels = []
                
                for ch_idx in range(audio.shape[1]):
                    logger.info(f"Upscaling channel {ch_idx + 1}/2...")
                    channel_audio = audio[:, ch_idx]
                    
                    # AudioSR super resolution
                    upscaled_channel = self.super_resolution(
                        self.model,
                        channel_audio,
                        sr,
                        guidance_scale=3.5,  # Balance between quality and fidelity
                        ddim_steps=50  # Quality vs speed trade-off
                    )
                    
                    upscaled_channels.append(upscaled_channel)
                
                # Combine channels
                upscaled_audio = np.stack(upscaled_channels, axis=1)
                
            else:
                logger.info("Processing mono audio")
                # Mono audio
                upscaled_audio = self.super_resolution(
                    self.model,
                    audio,
                    sr,
                    guidance_scale=3.5,
                    ddim_steps=50
                )
            
            # Save upscaled audio
            logger.info(f"Saving upscaled audio to: {output_path}")
            sf.write(output_path, upscaled_audio, target_sr)
            
            logger.info(f"Audio upscaling complete: {output_path} ({target_sr}Hz)")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio upscaling failed: {e}", exc_info=True)
            # Return original if upscaling fails
            return audio_path
    
    def quick_upscale(
        self,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Quick upscale with default settings
        
        Args:
            audio_path: Input audio file
            output_path: Output audio file (optional)
            
        Returns:
            Path to upscaled audio
        """
        try:
            logger.info(f"Quick upscale: {audio_path}")
            
            # For quick mode, use simple resampling instead of neural upscaling
            # This is faster and good enough for many use cases
            import librosa
            
            if output_path is None:
                base, ext = os.path.splitext(audio_path)
                output_path = f"{base}_48kHz{ext}"
            
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Check if upscaling is needed
            target_sr = 48000
            if sr >= target_sr:
                logger.info(f"Audio already at {sr}Hz, no upscaling needed")
                return audio_path
            
            logger.info(f"Resampling from {sr}Hz to {target_sr}Hz")
            
            # Resample with high-quality filter
            if audio.ndim == 2:
                # Stereo
                upscaled = np.zeros((int(len(audio) * target_sr / sr), audio.shape[1]))
                for ch in range(audio.shape[1]):
                    upscaled[:, ch] = librosa.resample(
                        audio[:, ch],
                        orig_sr=sr,
                        target_sr=target_sr,
                        res_type='kaiser_best'
                    )
            else:
                # Mono
                upscaled = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=target_sr,
                    res_type='kaiser_best'
                )
            
            # Save
            sf.write(output_path, upscaled, target_sr)
            logger.info(f"Quick upscale complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Quick upscale failed: {e}", exc_info=True)
            return audio_path
