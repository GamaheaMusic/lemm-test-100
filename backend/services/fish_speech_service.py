"""
Fish Speech TTS/vocals service
"""
import os
import logging
import uuid
import torch
from typing import Optional
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

class FishSpeechService:
    """Service for Fish Speech TTS and vocal synthesis"""
    
    def __init__(self, model_path: str):
        """
        Initialize Fish Speech service
        
        Args:
            model_path: Path to Fish Speech model files
        """
        self.model_path = model_path
        self.model = None
        self.vocoder = None
        self.is_initialized = False
        self.device = self._get_device()
        logger.info(f"Fish Speech service created with model path: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self):
        """Get compute device (AMD GPU via DirectML or CPU)"""
        try:
            from utils.amd_gpu import DEFAULT_DEVICE
            return DEFAULT_DEVICE
        except:
            return torch.device("cpu")
    
    def _initialize_model(self):
        """Lazy load the model when first needed"""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing Fish Speech model...")
            # TODO: Load actual Fish Speech model
            # from fish_speech import FishSpeechModel
            # self.model = FishSpeechModel.load(self.model_path)
            
            self.is_initialized = True
            logger.info("Fish Speech model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Fish Speech model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Could not load Fish Speech model: {str(e)}")
    
    def synthesize_vocals(
        self,
        lyrics: str,
        duration: int = 30,
        sample_rate: int = 44100
    ) -> str:
        """
        Synthesize vocals from lyrics
        
        Args:
            lyrics: Lyrics text to sing
            duration: Target duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Path to generated vocals file
        """
        try:
            self._initialize_model()
            
            logger.info(f"Synthesizing vocals: {len(lyrics)} characters")
            
            # TODO: Replace with actual Fish Speech synthesis
            # vocals = self.model.synthesize(lyrics, duration=duration, sample_rate=sample_rate)
            
            # Placeholder: Generate silence
            vocals = np.zeros(int(duration * sample_rate), dtype=np.float32)
            
            # Save to file
            output_dir = os.path.join('outputs', 'vocals')
            os.makedirs(output_dir, exist_ok=True)
            
            vocals_id = str(uuid.uuid4())
            output_path = os.path.join(output_dir, f"{vocals_id}.wav")
            
            sf.write(output_path, vocals, sample_rate)
            logger.info(f"Vocals synthesized: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Vocal synthesis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to synthesize vocals: {str(e)}")
    
    def add_vocals(
        self,
        music_path: str,
        lyrics: str,
        duration: int = 30
    ) -> str:
        """
        Add synthesized vocals to music track
        
        Args:
            music_path: Path to music audio file
            lyrics: Lyrics to sing
            duration: Duration in seconds
            
        Returns:
            Path to mixed audio file
        """
        try:
            logger.info(f"Adding vocals to music: {music_path}")
            
            # Load music
            music_audio, sr = sf.read(music_path)
            
            # Synthesize vocals
            vocals_path = self.synthesize_vocals(lyrics, duration, sr)
            vocals_audio, _ = sf.read(vocals_path)
            
            # Mix vocals with music
            # Ensure same length
            min_len = min(len(music_audio), len(vocals_audio))
            mixed = music_audio[:min_len] * 0.7 + vocals_audio[:min_len] * 0.3
            
            # Save mixed audio
            output_dir = os.path.join('outputs', 'mixed')
            os.makedirs(output_dir, exist_ok=True)
            
            mixed_id = str(uuid.uuid4())
            output_path = os.path.join(output_dir, f"{mixed_id}.wav")
            
            sf.write(output_path, mixed, sr)
            logger.info(f"Vocals added successfully: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Adding vocals failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to add vocals: {str(e)}")
