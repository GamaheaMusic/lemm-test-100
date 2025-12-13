"""
Timeline management service
"""
import logging
from typing import List, Dict, Optional
from models.schemas import ClipPosition, TimelineClip

logger = logging.getLogger(__name__)

class TimelineService:
    """Service for managing timeline clips"""
    
    def __init__(self):
        """Initialize timeline service"""
        self.clips: List[TimelineClip] = []
        logger.info("Timeline service initialized")
    
    def add_clip(
        self,
        clip_id: str,
        file_path: str,
        duration: float,
        position: ClipPosition
    ) -> Dict:
        """
        Add a clip to the timeline
        
        Args:
            clip_id: Unique clip identifier
            file_path: Path to audio file
            duration: Clip duration in seconds
            position: Where to place the clip
            
        Returns:
            Clip information with timeline position
        """
        try:
            logger.info(f"[TIMELINE] Adding clip '{clip_id}' with position={position}, duration={duration}")
            logger.info(f"[TIMELINE] Current clips before add: {len(self.clips)}")
            
            # Calculate timeline position based on requested position
            if position == ClipPosition.INTRO:
                timeline_position = 0
                start_time = 0.0
                # Shift all existing clips
                for clip in self.clips:
                    clip.timeline_position += 1
                    clip.start_time += duration
                    
            elif position == ClipPosition.PREVIOUS:
                if len(self.clips) == 0:
                    timeline_position = 0
                    start_time = 0.0
                else:
                    # Add after the second-to-last clip (before the last one)
                    # If only one clip exists, add before it
                    if len(self.clips) == 1:
                        timeline_position = 0
                        start_time = 0.0
                        # Shift the existing clip
                        self.clips[0].timeline_position += 1
                        self.clips[0].start_time += duration
                    else:
                        # Insert before last clip
                        timeline_position = len(self.clips) - 1
                        start_time = self.clips[-2].start_time + self.clips[-2].duration
                        # Shift last clip
                        self.clips[-1].timeline_position += 1
                        self.clips[-1].start_time += duration
                    
            elif position == ClipPosition.NEXT:
                timeline_position = len(self.clips)
                start_time = self.get_total_duration()
                
            else:  # OUTRO
                timeline_position = len(self.clips)
                start_time = self.get_total_duration()
            
            # Create clip
            clip = TimelineClip(
                clip_id=clip_id,
                file_path=file_path,
                duration=duration,
                timeline_position=timeline_position,
                start_time=start_time,
                music_path=file_path  # Store as music_path for consistent access
            )
            
            # Insert clip at correct position
            self.clips.insert(timeline_position, clip)
            
            logger.info(f"Clip added: {clip_id} at position {timeline_position}")
            logger.info(f"[TIMELINE] Total clips after add: {len(self.clips)}")
            logger.info(f"[TIMELINE] All clip IDs: {[c.clip_id for c in self.clips]}")
            
            return {
                'clip_id': clip_id,
                'timeline_position': timeline_position,
                'start_time': start_time,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Failed to add clip: {str(e)}", exc_info=True)
            raise
    
    def remove_clip(self, clip_id: str):
        """
        Remove a clip from timeline
        
        Args:
            clip_id: Clip to remove
        """
        try:
            # Find and remove clip
            clip_index = None
            for i, clip in enumerate(self.clips):
                if clip.clip_id == clip_id:
                    clip_index = i
                    break
            
            if clip_index is None:
                raise ValueError(f"Clip not found: {clip_id}")
            
            removed_clip = self.clips.pop(clip_index)
            
            # Recalculate positions
            self._recalculate_positions()
            
            logger.info(f"Clip removed: {clip_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove clip: {str(e)}", exc_info=True)
            raise
    
    def reorder_clips(self, clip_ids: List[str]):
        """
        Reorder clips on timeline
        
        Args:
            clip_ids: New order of clip IDs
        """
        try:
            # Validate all clip IDs exist
            existing_ids = {clip.clip_id for clip in self.clips}
            requested_ids = set(clip_ids)
            
            if existing_ids != requested_ids:
                raise ValueError("Clip IDs don't match existing clips")
            
            # Create new order
            clip_dict = {clip.clip_id: clip for clip in self.clips}
            self.clips = [clip_dict[cid] for cid in clip_ids]
            
            # Recalculate positions
            self._recalculate_positions()
            
            logger.info("Clips reordered")
            
        except Exception as e:
            logger.error(f"Failed to reorder clips: {str(e)}", exc_info=True)
            raise
    
    def get_all_clips(self) -> List[Dict]:
        """Get all clips with their information"""
        logger.info(f"[TIMELINE] get_all_clips called, returning {len(self.clips)} clips")
        result = [
            {
                'clip_id': clip.clip_id,
                'file_path': clip.file_path,
                'duration': clip.duration,
                'timeline_position': clip.timeline_position,
                'start_time': clip.start_time
            }
            for clip in self.clips
        ]
        logger.info(f"[TIMELINE] Clips data: {result}")
        return result
    
    def get_clip(self, clip_id: str) -> Optional[TimelineClip]:
        """Get a specific clip"""
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        return None
    
    def get_total_duration(self) -> float:
        """Get total duration of all clips"""
        if not self.clips:
            return 0.0
        return sum(clip.duration for clip in self.clips)
    
    def clear(self):
        """Clear all clips from timeline"""
        self.clips = []
        logger.info("Timeline cleared")
    
    def _recalculate_positions(self):
        """Recalculate all clip positions and start times"""
        current_time = 0.0
        for i, clip in enumerate(self.clips):
            clip.timeline_position = i
            clip.start_time = current_time
            current_time += clip.duration
