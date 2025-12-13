"""
Export and merge service
"""
import os
import logging
from typing import Optional, List
import numpy as np
import soundfile as sf
from services.timeline_service import TimelineService

logger = logging.getLogger(__name__)

class ExportService:
    """Service for exporting and merging audio"""
    
    def __init__(self):
        """Initialize export service"""
        self.timeline_service = TimelineService()
        logger.info("Export service initialized")
    
    def merge_clips(
        self,
        filename: str = "output",
        export_format: str = "wav"
    ) -> Optional[str]:
        """
        Merge all timeline clips into a single file
        
        Args:
            filename: Output filename (without extension)
            export_format: Output format (wav, mp3, flac)
            
        Returns:
            Path to merged file, or None if no clips
        """
        try:
            clips = self.timeline_service.get_all_clips()
            
            if not clips:
                logger.warning("No clips to merge")
                return None
            
            logger.info(f"Merging {len(clips)} clips")
            
            # Load all clips
            audio_data = []
            sample_rate = None
            
            for clip in clips:
                audio, sr = sf.read(clip['file_path'])
                
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    logger.warning(f"Sample rate mismatch: {sr} vs {sample_rate}")
                    # Could resample here if needed
                
                audio_data.append(audio)
            
            # Apply crossfading between clips (2 second overlap)
            crossfade_duration = 2.0  # seconds
            crossfade_samples = int(crossfade_duration * sample_rate)
            
            if len(audio_data) == 1:
                # Single clip, no crossfading needed
                merged_audio = audio_data[0]
            else:
                # Start with first clip
                merged_audio = audio_data[0].copy()
                
                # Crossfade each subsequent clip
                for i in range(1, len(audio_data)):
                    current_clip = audio_data[i]
                    
                    # Calculate overlap region
                    overlap_samples = min(crossfade_samples, len(merged_audio), len(current_clip))
                    
                    if overlap_samples > 0:
                        # Create fade out curve for end of previous audio
                        fade_out = np.linspace(1.0, 0.0, overlap_samples)
                        # Create fade in curve for start of current clip
                        fade_in = np.linspace(0.0, 1.0, overlap_samples)
                        
                        # Handle stereo vs mono
                        if merged_audio.ndim == 2:
                            fade_out = fade_out[:, np.newaxis]
                            fade_in = fade_in[:, np.newaxis]
                        
                        # Apply crossfade
                        merged_audio[-overlap_samples:] = (
                            merged_audio[-overlap_samples:] * fade_out +
                            current_clip[:overlap_samples] * fade_in
                        )
                        
                        # Append the rest of the current clip
                        if len(current_clip) > overlap_samples:
                            merged_audio = np.concatenate([
                                merged_audio,
                                current_clip[overlap_samples:]
                            ])
                        
                        logger.info(f"Applied {crossfade_duration}s crossfade between clips {i-1} and {i}")
                    else:
                        # No overlap possible, just concatenate
                        merged_audio = np.concatenate([merged_audio, current_clip])
            
            # Normalize
            max_val = np.abs(merged_audio).max()
            if max_val > 0:
                merged_audio = merged_audio / max_val * 0.95
            
            # Save merged file
            output_dir = 'outputs'
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"{filename}.{export_format}")
            
            sf.write(output_path, merged_audio, sample_rate)
            
            logger.info(f"Clips merged successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to merge clips: {str(e)}", exc_info=True)
            raise
    
    def export_clip(
        self,
        clip_id: str,
        export_format: str = "wav"
    ) -> Optional[str]:
        """
        Export a single clip
        
        Args:
            clip_id: ID of clip to export
            export_format: Output format
            
        Returns:
            Path to exported file, or None if clip not found
        """
        try:
            clip = self.timeline_service.get_clip(clip_id)
            
            if not clip:
                logger.warning(f"Clip not found: {clip_id}")
                return None
            
            logger.info(f"Exporting clip: {clip_id}")
            
            # Load clip
            audio, sr = sf.read(clip.file_path)
            
            # Export with requested format
            output_dir = 'outputs'
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"{clip_id}.{export_format}")
            
            sf.write(output_path, audio, sr)
            
            logger.info(f"Clip exported: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export clip: {str(e)}", exc_info=True)
            raise
