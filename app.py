"""
Music Generation Studio - HuggingFace Spaces Deployment
Main application file for Gradio interface
"""
import os
import sys
import gradio as gr
import logging
from pathlib import Path
import shutil
import subprocess
import json
import time

# Import spaces for ZeroGPU support
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    # Create a dummy decorator for local development
    class spaces:
        @staticmethod
        def GPU(func):
            return func

# Run DiffRhythm2 source setup if needed
setup_script = Path(__file__).parent / "setup_diffrhythm2_src.sh"
if setup_script.exists():
    try:
        subprocess.run(["bash", str(setup_script)], check=True)
    except Exception as e:
        print(f"Warning: Failed to run setup script: {e}")

# Configure environment for HuggingFace Spaces (espeak-ng paths, etc.)
import hf_config

# Setup paths for HuggingFace Spaces
SPACE_DIR = Path(__file__).parent
sys.path.insert(0, str(SPACE_DIR / 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import services
try:
    from services.diffrhythm_service import DiffRhythmService
    from services.lyricmind_service import LyricMindService
    from services.timeline_service import TimelineService
    from services.export_service import ExportService
    from config.settings import Config
    from utils.prompt_analyzer import PromptAnalyzer
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

# Initialize configuration
config = Config()

# Create necessary directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/music", exist_ok=True)
os.makedirs("outputs/mixed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Initialize services - these persist at module level
timeline_service = TimelineService()
export_service = ExportService()

# Lazy-load AI services (heavy models)
diffrhythm_service = None
lyricmind_service = None

def get_diffrhythm_service():
    """Lazy load DiffRhythm service"""
    global diffrhythm_service
    if diffrhythm_service is None:
        logger.info("Loading DiffRhythm2 model...")
        diffrhythm_service = DiffRhythmService(model_path=config.DIFFRHYTHM_MODEL_PATH)
        logger.info("DiffRhythm2 model loaded")
    return diffrhythm_service

def get_lyricmind_service():
    """Lazy load LyricMind service"""
    global lyricmind_service
    if lyricmind_service is None:
        logger.info("Loading LyricMind model...")
        lyricmind_service = LyricMindService(model_path=config.LYRICMIND_MODEL_PATH)
        logger.info("LyricMind model loaded")
    return lyricmind_service

@spaces.GPU
def generate_lyrics(prompt: str, progress=gr.Progress()):
    """Generate lyrics from prompt using analysis"""
    try:
        if not prompt or not prompt.strip():
            return "❌ Please enter a prompt"
        
        # Fixed duration for all clips
        duration = 32
        
        progress(0, desc="🔍 Analyzing prompt...")
        logger.info(f"Generating lyrics for: {prompt}")
        
        # Analyze prompt
        analysis = PromptAnalyzer.analyze(prompt)
        genre = analysis.get('genres', ['general'])[0] if analysis.get('genres') else 'general'
        mood = analysis.get('mood', 'unknown')
        
        logger.info(f"Analysis - Genre: {genre}, Mood: {mood}")
        
        progress(0.3, desc=f"✍️ Generating {genre} lyrics...")
        
        service = get_lyricmind_service()
        lyrics = service.generate(
            prompt=prompt,
            duration=duration,
            prompt_analysis=analysis
        )
        
        progress(1.0, desc="✅ Lyrics generated!")
        return lyrics
        
    except Exception as e:
        logger.error(f"Error generating lyrics: {e}", exc_info=True)
        return f"❌ Error: {str(e)}"

@spaces.GPU
def generate_music(prompt: str, lyrics: str, lyrics_mode: str, position: str, context_length: int, timeline_state: dict, progress=gr.Progress()):
    """Generate music clip and add to timeline"""
    try:
        # Restore timeline from state
        if timeline_state and 'clips' in timeline_state:
            timeline_service.clips = []
            for clip_data in timeline_state['clips']:
                from models.schemas import TimelineClip
                clip = TimelineClip(**clip_data)
                timeline_service.clips.append(clip)
            logger.info(f"[STATE] Restored {len(timeline_service.clips)} clips from state")
        
        if not prompt or not prompt.strip():
            return "❌ Please enter a music prompt", get_timeline_display(), None, timeline_state
        
        # Fixed duration for all clips
        duration = 32
        
        # Estimate time (CPU on HF Spaces)
        est_time = int(duration * 4)  # Conservative estimate for CPU
        
        progress(0, desc=f"🔍 Analyzing prompt... (Est. {est_time}s)")
        logger.info(f"Generating music: {prompt}, mode={lyrics_mode}, duration={duration}s")
        
        # Analyze prompt
        analysis = PromptAnalyzer.analyze(prompt)
        genre = analysis.get('genres', ['general'])[0] if analysis.get('genres') else 'general'
        bpm = analysis.get('bpm', 120)
        mood = analysis.get('mood', 'neutral')
        
        logger.info(f"Analysis - Genre: {genre}, BPM: {bpm}, Mood: {mood}")
        
        # Apply style consistency from previous clips within context window
        # Auto-disable context if this is the first clip
        clips = timeline_service.get_all_clips()
        effective_context_length = 0 if len(clips) == 0 else context_length
        
        if effective_context_length > 0 and clips:
            # Calculate which clips fall within the context window
            total_duration = timeline_service.get_total_duration()
            context_start = max(0, total_duration - effective_context_length)
            
            context_clips = [c for c in clips if c['start_time'] >= context_start]
            
            if context_clips:
                logger.info(f"Using {len(context_clips)} clips for style consistency (context: {effective_context_length}s)")
                # Enhance prompt with style consistency guidance
                prompt = f"{prompt} (maintaining consistent {genre} style at {bpm} BPM with {mood} mood)"
            else:
                logger.info("No clips in context window")
        else:
            if len(clips) == 0:
                logger.info("First clip - style consistency disabled")
            else:
                logger.info("Style consistency disabled (context length: 0)")
        
        # Determine lyrics based on mode
        lyrics_to_use = None
        
        if lyrics_mode == "Instrumental":
            logger.info("Generating instrumental (no vocals)")
            progress(0.1, desc=f"🎹 Preparing instrumental generation... ({est_time}s)")
            
        elif lyrics_mode == "User Lyrics":
            if not lyrics or not lyrics.strip():
                return "❌ Please enter lyrics or switch mode", get_timeline_display(), None
            lyrics_to_use = lyrics.strip()
            logger.info(f"Using user-provided lyrics (length: {len(lyrics_to_use)} chars)")
            logger.info(f"First 100 chars: {lyrics_to_use[:100]}")
            progress(0.1, desc=f"🎤 Preparing vocal generation... ({est_time}s)")
            
        elif lyrics_mode == "Auto Lyrics":
            if lyrics and lyrics.strip():
                lyrics_to_use = lyrics.strip()
                logger.info("Using existing lyrics from textbox")
                progress(0.1, desc=f"🎤 Using provided lyrics... ({est_time}s)")
            else:
                progress(0.1, desc="✍️ Generating lyrics...")
                logger.info("Auto-generating lyrics...")
                lyric_service = get_lyricmind_service()
                lyrics_to_use = lyric_service.generate(
                    prompt=prompt,
                    duration=duration,
                    prompt_analysis=analysis
                )
                logger.info(f"Generated {len(lyrics_to_use)} characters of lyrics")
                progress(0.25, desc=f"🎵 Lyrics ready, generating music... ({est_time}s)")
        
        # Generate music
        progress(0.3, desc=f"🎼 Generating {genre} at {bpm} BPM... ({est_time}s)")
        service = get_diffrhythm_service()
        
        final_path = service.generate(
            prompt=prompt,
            duration=duration,
            lyrics=lyrics_to_use
        )
        
        # Add to timeline
        progress(0.9, desc="📊 Adding to timeline...")
        clip_id = os.path.basename(final_path).split('.')[0]
        
        logger.info(f"[GENERATE] About to add clip: {clip_id}, position: {position}")
        logger.info(f"[GENERATE] Timeline service ID: {id(timeline_service)}")
        logger.info(f"[GENERATE] Clips before add: {len(timeline_service.clips)}")
        
        from models.schemas import ClipPosition
        clip_info = timeline_service.add_clip(
            clip_id=clip_id,
            file_path=final_path,
            duration=float(duration),
            position=ClipPosition(position)
        )
        
        logger.info(f"Music added to timeline at position {clip_info['timeline_position']}")
        logger.info(f"[GENERATE] Clips after add: {len(timeline_service.clips)}")
        
        # Build status message
        progress(1.0, desc="✅ Complete!")
        status_msg = f"✅ Music generated successfully!\n"
        status_msg += f"🎸 Genre: {genre} | 🥁 BPM: {bpm} | 🎭 Mood: {mood}\n"
        status_msg += f"🎤 Mode: {lyrics_mode} | 📍 Position: {position}\n"
        
        if lyrics_mode == "Auto Lyrics" and lyrics_to_use and not lyrics:
            status_msg += "✍️ (Lyrics auto-generated)"
        
        # Save timeline to state
        new_state = {
            'clips': [{
                'clip_id': c.clip_id,
                'file_path': c.file_path,
                'duration': c.duration,
                'timeline_position': c.timeline_position,
                'start_time': c.start_time,
                'music_path': c.music_path
            } for c in timeline_service.clips]
        }
        logger.info(f"[STATE] Saved {len(new_state['clips'])} clips to state")
        
        return status_msg, get_timeline_display(), final_path, new_state
        
    except Exception as e:
        logger.error(f"Error generating music: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", get_timeline_display(), None, timeline_state

def get_timeline_display():
    """Get timeline clips as HTML visualization with waveform-style display"""
    clips = timeline_service.get_all_clips()
    
    if not clips:
        return "<div style='text-align:center; padding:40px; color:#888;'>📭 Timeline is empty. Generate clips to get started!</div>"
    
    total_duration = timeline_service.get_total_duration()
    
    # Build HTML timeline
    html = f"""
    <div style="font-family: Arial, sans-serif; background: #1a1a1a; padding: 20px; border-radius: 8px; color: white;">
        <div style="margin-bottom: 15px; font-size: 14px; color: #aaa;">
            <strong>📊 Timeline:</strong> {len(clips)} clips | Total: {format_duration(total_duration)}
        </div>
        <div style="background: #2a2a2a; border-radius: 6px; padding: 15px; position: relative; min-height: 80px;">
            <div style="position: absolute; top: 10px; left: 15px; right: 15px; height: 60px; background: #333; border-radius: 4px; overflow: hidden;">
    """
    
    # Calculate pixel width (scale to fit)
    if total_duration > 0:
        pixels_per_second = 800 / total_duration  # 800px total width
    else:
        pixels_per_second = 10
    
    # Add clip blocks
    colors = ['#8b5cf6', '#ec4899', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']
    for i, clip in enumerate(clips):
        start_px = clip['start_time'] * pixels_per_second
        width_px = clip['duration'] * pixels_per_second
        color = colors[i % len(colors)]
        
        # Create waveform-style bars
        bars = ''.join([
            f'<div style="display:inline-block; width:2px; height:{20 + (i*7 % 30)}px; background:rgba(255,255,255,0.3); margin:0 1px; vertical-align:bottom;"></div>'
            for i in range(min(int(width_px / 4), 50))
        ])
        
        html += f"""
                <div style="position: absolute; left: {start_px}px; width: {width_px}px; height: 60px; 
                     background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                     border-radius: 4px; border: 1px solid rgba(255,255,255,0.2);
                     display: flex; align-items: center; justify-content: center;
                     overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                    <div style="position: absolute; bottom: 5px; left: 0; right: 0; height: 40px; display: flex; align-items: flex-end; justify-content: space-evenly; padding: 0 5px;">
                        {bars}
                    </div>
                    <div style="position: relative; z-index: 1; font-size: 11px; font-weight: bold; 
                         text-shadow: 0 1px 2px rgba(0,0,0,0.5); text-align: center; padding: 0 5px;">
                        Clip {i+1}<br>{format_duration(clip['duration'])}
                    </div>
                </div>
        """
    
    html += """
            </div>
            <div style="margin-top: 75px; font-size: 11px; color: #888;">
                <div style="display: flex; justify-content: space-between;">
                    <span>0:00</span>
                    <span>{}</span>
                </div>
            </div>
        </div>
    </div>
    """.format(format_duration(total_duration))
    
    return html

def remove_clip(clip_number: int, timeline_state: dict):
    """Remove a clip from timeline"""
    try:
        # Restore timeline from state
        if timeline_state and 'clips' in timeline_state:
            timeline_service.clips = []
            for clip_data in timeline_state['clips']:
                from models.schemas import TimelineClip
                clip = TimelineClip(**clip_data)
                timeline_service.clips.append(clip)
        
        clips = timeline_service.get_all_clips()
        
        if not clips:
            return "📭 Timeline is empty", get_timeline_display(), timeline_state
        
        if clip_number < 1 or clip_number > len(clips):
            return f"❌ Invalid clip number. Choose 1-{len(clips)}", get_timeline_display(), timeline_state
        
        clip_id = clips[clip_number - 1]['clip_id']
        timeline_service.remove_clip(clip_id)
        
        # Save updated state
        new_state = {
            'clips': [{
                'clip_id': c.clip_id,
                'file_path': c.file_path,
                'duration': c.duration,
                'timeline_position': c.timeline_position,
                'start_time': c.start_time,
                'music_path': c.music_path
            } for c in timeline_service.clips]
        }
        
        return f"✅ Clip {clip_number} removed", get_timeline_display(), new_state
        
    except Exception as e:
        logger.error(f"Error removing clip: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", get_timeline_display(), timeline_state

def clear_timeline(timeline_state: dict):
    """Clear all clips from timeline"""
    try:
        timeline_service.clear()
        new_state = {'clips': []}
        return "✅ Timeline cleared", get_timeline_display(), new_state
    except Exception as e:
        logger.error(f"Error clearing timeline: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", get_timeline_display(), timeline_state

def export_timeline(filename: str, export_format: str, timeline_state: dict, progress=gr.Progress()):
    """Export timeline to audio file"""
    try:
        # Restore timeline from state
        if timeline_state and 'clips' in timeline_state:
            timeline_service.clips = []
            for clip_data in timeline_state['clips']:
                from models.schemas import TimelineClip
                clip = TimelineClip(**clip_data)
                timeline_service.clips.append(clip)
            logger.info(f"[STATE] Restored {len(timeline_service.clips)} clips for export")
        
        clips = timeline_service.get_all_clips()
        
        if not clips:
            return "❌ No clips to export", None, timeline_state
        
        if not filename or not filename.strip():
            filename = "output"
        
        progress(0, desc="🔄 Merging clips...")
        logger.info(f"Exporting timeline: {filename}.{export_format}")
        
        export_service.timeline_service = timeline_service
        
        progress(0.5, desc="💾 Encoding audio...")
        output_path = export_service.merge_clips(
            filename=filename,
            export_format=export_format
        )
        
        if output_path:
            progress(1.0, desc="✅ Export complete!")
            return f"✅ Exported: {os.path.basename(output_path)}", output_path, timeline_state
        else:
            return "❌ Export failed", None, timeline_state
            
    except Exception as e:
        logger.error(f"Error exporting: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", None, timeline_state

def get_timeline_playback(timeline_state: dict):
    """Get merged timeline audio for playback"""
    try:
        # Restore timeline from state
        if timeline_state and 'clips' in timeline_state:
            timeline_service.clips = []
            for clip_data in timeline_state['clips']:
                from models.schemas import TimelineClip
                clip = TimelineClip(**clip_data)
                timeline_service.clips.append(clip)
            logger.info(f"[STATE] Restored {len(timeline_service.clips)} clips for playback")
        
        clips = timeline_service.get_all_clips()
        
        if not clips:
            return None
        
        # Use export service to merge clips
        export_service.timeline_service = timeline_service
        output_path = export_service.merge_clips(
            filename="timeline_preview",
            export_format="wav"
        )
        
        logger.info(f"Timeline playback ready: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating playback: {e}", exc_info=True)
        return None

def preview_mastering_preset(preset_name: str, timeline_state: dict):
    """Preview mastering preset on the most recent clip"""
    try:
        # Restore timeline from state
        if timeline_state and 'clips' in timeline_state:
            timeline_service.clips = []
            for clip_data in timeline_state['clips']:
                from models.schemas import TimelineClip
                clip = TimelineClip(**clip_data)
                timeline_service.clips.append(clip)
        
        clips = timeline_service.get_all_clips()
        if not clips:
            return None, "❌ No clips in timeline to preview"
        
        # Use the most recent clip for preview
        latest_clip = clips[-1]
        clip_path = latest_clip['file_path']
        
        if not os.path.exists(clip_path):
            return None, f"❌ Clip file not found: {clip_path}"
        
        # Extract preset name
        preset_key = preset_name.split(" - ")[0].lower().replace(" ", "_")
        
        # Create temporary preview file
        import tempfile
        preview_path = os.path.join(tempfile.gettempdir(), f"preview_{latest_clip['clip_id']}.wav")
        
        from services.mastering_service import MasteringService
        mastering_service = MasteringService()
        
        # Apply preset to preview file
        mastering_service.apply_preset(
            audio_path=clip_path,
            preset_name=preset_key,
            output_path=preview_path
        )
        
        logger.info(f"Created mastering preview: {preview_path}")
        return preview_path, f"✅ Preview ready: {preset_name.split(' - ')[0]} applied to latest clip"
        
    except Exception as e:
        logger.error(f"Error creating preview: {e}", exc_info=True)
        return None, f"❌ Preview error: {str(e)}"

def apply_mastering_preset(preset_name: str, timeline_state: dict):
    """Apply mastering preset to all clips in timeline"""
    try:
        logger.info(f"[STATE DEBUG] apply_mastering_preset called")
        logger.info(f"[STATE DEBUG] timeline_state type: {type(timeline_state)}")
        logger.info(f"[STATE DEBUG] timeline_state value: {timeline_state}")
        
        # Restore timeline from state
        if timeline_state and 'clips' in timeline_state:
            timeline_service.clips = []
            for clip_data in timeline_state['clips']:
                from models.schemas import TimelineClip
                clip = TimelineClip(**clip_data)
                timeline_service.clips.append(clip)
            logger.info(f"[STATE] Restored {len(timeline_service.clips)} clips for mastering")
        else:
            logger.warning(f"[STATE DEBUG] State restoration failed - timeline_state is None or missing 'clips' key")
        
        clips = timeline_service.get_all_clips()
        logger.info(f"[MASTERING DEBUG] Retrieved {len(clips)} clips from timeline")
        
        if not clips:
            logger.warning("[MASTERING DEBUG] No clips found in timeline")
            return "❌ No clips in timeline", timeline_state
        
        # Log clip details for debugging
        for i, clip in enumerate(clips):
            logger.info(f"[MASTERING DEBUG] Clip {i+1}: {clip}")
        
        # Extract preset name from dropdown value
        preset_key = preset_name.split(" - ")[0].lower().replace(" ", "_")
        
        logger.info(f"Applying preset '{preset_key}' to {len(clips)} clip(s)")
        
        # Import mastering service
        from services.mastering_service import MasteringService
        mastering_service = MasteringService()
        
        # Apply preset to all clips
        for clip in clips:
            clip_path = clip['file_path']
            
            if not os.path.exists(clip_path):
                logger.warning(f"Audio file not found: {clip_path}")
                continue
            
            # Apply preset
            mastering_service.apply_preset(
                audio_path=clip_path,
                preset_name=preset_key,
                output_path=clip_path  # Overwrite original
            )
            logger.info(f"Applied preset to: {clip['clip_id']}")
        
        return f"✅ Applied '{preset_name.split(' - ')[0]}' to {len(clips)} clip(s)", timeline_state
        
    except Exception as e:
        logger.error(f"Error applying preset: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", timeline_state

def preview_custom_eq(low_shelf, low_mid, mid, high_mid, high_shelf, timeline_state: dict):
    """Preview custom EQ on the most recent clip"""
    try:
        # Restore timeline from state
        if timeline_state and 'clips' in timeline_state:
            timeline_service.clips = []
            for clip_data in timeline_state['clips']:
                from models.schemas import TimelineClip
                clip = TimelineClip(**clip_data)
                timeline_service.clips.append(clip)
        
        clips = timeline_service.get_all_clips()
        if not clips:
            return None, "❌ No clips in timeline to preview"
        
        # Use the most recent clip for preview
        latest_clip = clips[-1]
        clip_path = latest_clip['file_path']
        
        if not os.path.exists(clip_path):
            return None, f"❌ Clip file not found: {clip_path}"
        
        # Create temporary preview file
        import tempfile
        preview_path = os.path.join(tempfile.gettempdir(), f"eq_preview_{latest_clip['clip_id']}.wav")
        
        from services.mastering_service import MasteringService
        mastering_service = MasteringService()
        
        # Format EQ bands
        eq_bands = [
            {'type': 'lowshelf', 'frequency': 100, 'gain': low_shelf, 'q': 0.7},
            {'type': 'peak', 'frequency': 500, 'gain': low_mid, 'q': 1.0},
            {'type': 'peak', 'frequency': 2000, 'gain': mid, 'q': 1.0},
            {'type': 'peak', 'frequency': 5000, 'gain': high_mid, 'q': 1.0},
            {'type': 'highshelf', 'frequency': 10000, 'gain': high_shelf, 'q': 0.7}
        ]
        
        # Apply EQ to preview file
        mastering_service.apply_custom_eq(
            audio_path=clip_path,
            eq_bands=eq_bands,
            output_path=preview_path
        )
        
        logger.info(f"Created EQ preview: {preview_path}")
        return preview_path, f"✅ Preview ready: Custom EQ applied to latest clip"
        
    except Exception as e:
        logger.error(f"Error creating EQ preview: {e}", exc_info=True)
        return None, f"❌ Preview error: {str(e)}"

def apply_custom_eq(low_shelf, low_mid, mid, high_mid, high_shelf, timeline_state: dict):
    """Apply custom EQ to all clips in timeline"""
    try:
        logger.info(f"[STATE DEBUG] apply_custom_eq called")
        logger.info(f"[STATE DEBUG] timeline_state type: {type(timeline_state)}")
        logger.info(f"[STATE DEBUG] timeline_state value: {timeline_state}")
        
        # Restore timeline from state
        if timeline_state and 'clips' in timeline_state:
            timeline_service.clips = []
            for clip_data in timeline_state['clips']:
                from models.schemas import TimelineClip
                clip = TimelineClip(**clip_data)
                timeline_service.clips.append(clip)
            logger.info(f"[STATE] Restored {len(timeline_service.clips)} clips for EQ")
        else:
            logger.warning(f"[STATE DEBUG] State restoration failed - timeline_state is None or missing 'clips' key")
        
        clips = timeline_service.get_all_clips()
        logger.info(f"[EQ DEBUG] Retrieved {len(clips)} clips from timeline")
        
        if not clips:
            logger.warning("[EQ DEBUG] No clips found in timeline")
            return "❌ No clips in timeline", timeline_state
        
        # Log clip details for debugging
        for i, clip in enumerate(clips):
            logger.info(f"[EQ DEBUG] Clip {i+1}: {clip}")
        
        logger.info(f"Applying custom EQ to {len(clips)} clip(s)")
        
        # Import mastering service
        from services.mastering_service import MasteringService
        mastering_service = MasteringService()
        
        # Apply custom EQ - format eq_bands as expected by the service
        eq_bands = [
            {'type': 'lowshelf', 'frequency': 100, 'gain': low_shelf, 'q': 0.7},
            {'type': 'peak', 'frequency': 500, 'gain': low_mid, 'q': 1.0},
            {'type': 'peak', 'frequency': 2000, 'gain': mid, 'q': 1.0},
            {'type': 'peak', 'frequency': 5000, 'gain': high_mid, 'q': 1.0},
            {'type': 'highshelf', 'frequency': 10000, 'gain': high_shelf, 'q': 0.7}
        ]
        
        # Apply to all clips
        for clip in clips:
            clip_path = clip['file_path']
            
            if not os.path.exists(clip_path):
                logger.warning(f"Audio file not found: {clip_path}")
                continue
            
            mastering_service.apply_custom_eq(
                audio_path=clip_path,
                eq_bands=eq_bands,
                output_path=clip_path  # Overwrite original
            )
            logger.info(f"Applied EQ to: {clip['clip_id']}")
        
        return f"✅ Applied custom EQ to {len(clips)} clip(s)", timeline_state
        
    except Exception as e:
        logger.error(f"Error applying EQ: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", timeline_state

def enhance_timeline_clips(enhancement_level: str, timeline_state: dict):
    """Enhance all clips in timeline using stem separation"""
    try:
        logger.info(f"[ENHANCEMENT] Starting enhancement: level={enhancement_level}")
        
        # Restore timeline from state
        if timeline_state and 'clips' in timeline_state:
            timeline_service.clips = []
            for clip_data in timeline_state['clips']:
                from models.schemas import TimelineClip
                clip = TimelineClip(**clip_data)
                timeline_service.clips.append(clip)
            logger.info(f"[STATE] Restored {len(timeline_service.clips)} clips for enhancement")
        
        clips = timeline_service.get_all_clips()
        
        if not clips:
            return "❌ No clips in timeline", timeline_state
        
        # Import stem enhancement service
        from services.stem_enhancement_service import StemEnhancementService
        enhancer = StemEnhancementService()
        
        # Convert enhancement level to service format
        level_map = {
            "Fast": "fast",
            "Balanced": "balanced",
            "Maximum": "maximum"
        }
        service_level = level_map.get(enhancement_level, "balanced")
        
        # Enhance each clip
        enhanced_count = 0
        for clip in clips:
            clip_path = clip['file_path']
            
            if not os.path.exists(clip_path):
                logger.warning(f"Clip file not found: {clip_path}")
                continue
            
            logger.info(f"Enhancing clip: {clip['clip_id']} ({service_level})")
            
            # Enhance in-place (overwrites original)
            enhancer.enhance_clip(
                audio_path=clip_path,
                output_path=clip_path,
                enhancement_level=service_level
            )
            
            enhanced_count += 1
            logger.info(f"Enhanced {enhanced_count}/{len(clips)} clips")
        
        return f"✅ Enhanced {enhanced_count} clip(s) ({enhancement_level} quality)", timeline_state
        
    except Exception as e:
        logger.error(f"Enhancement failed: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", timeline_state

def upscale_timeline_clips(upscale_mode: str, timeline_state: dict):
    """Upscale all clips in timeline to 48kHz"""
    try:
        logger.info(f"[UPSCALE] Starting upscale: mode={upscale_mode}")
        
        # Restore timeline from state
        if timeline_state and 'clips' in timeline_state:
            timeline_service.clips = []
            for clip_data in timeline_state['clips']:
                from models.schemas import TimelineClip
                clip = TimelineClip(**clip_data)
                timeline_service.clips.append(clip)
            logger.info(f"[STATE] Restored {len(timeline_service.clips)} clips for upscale")
        
        clips = timeline_service.get_all_clips()
        
        if not clips:
            return "❌ No clips in timeline", timeline_state
        
        # Import upscale service
        from services.audio_upscale_service import AudioUpscaleService
        upscaler = AudioUpscaleService()
        
        # Upscale each clip
        upscaled_count = 0
        for clip in clips:
            clip_path = clip['file_path']
            
            if not os.path.exists(clip_path):
                logger.warning(f"Clip file not found: {clip_path}")
                continue
            
            logger.info(f"Upscaling clip: {clip['clip_id']} ({upscale_mode})")
            
            # Choose upscale method
            if upscale_mode == "Quick (Resample)":
                upscaler.quick_upscale(
                    audio_path=clip_path,
                    output_path=clip_path
                )
            else:  # Neural (AudioSR)
                upscaler.upscale_audio(
                    audio_path=clip_path,
                    output_path=clip_path,
                    target_sr=48000
                )
            
            upscaled_count += 1
            logger.info(f"Upscaled {upscaled_count}/{len(clips)} clips")
        
        return f"✅ Upscaled {upscaled_count} clip(s) to 48kHz ({upscale_mode})", timeline_state
        
    except Exception as e:
        logger.error(f"Upscale failed: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", timeline_state

def format_duration(seconds: float) -> str:
    """Format duration as MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

# LoRA Training Functions
def analyze_user_audio(audio_files, split_clips, separate_stems):
    """Analyze uploaded audio files and generate metadata"""
    try:
        if not audio_files:
            return "❌ No audio files uploaded", None
        
        from backend.services.audio_analysis_service import AudioAnalysisService
        analyzer = AudioAnalysisService()
        
        results = []
        for audio_file in audio_files:
            # Analyze audio
            metadata = analyzer.analyze_audio(audio_file.name)
            
            # Add to results
            results.append([
                Path(audio_file.name).name,
                metadata.get('genre', 'unknown'),
                metadata.get('bpm', 120),
                metadata.get('key', 'C major'),
                metadata.get('energy', 'medium'),
                '',  # Instruments (user fills in)
                ''   # Description (user fills in)
            ])
        
        status = f"✅ Analyzed {len(results)} file(s)"
        return status, results
        
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        return f"❌ Error: {str(e)}", None

def ai_generate_all_metadata(metadata_table):
    """AI generate metadata for all files in table"""
    try:
        if not metadata_table:
            return "❌ No files in metadata table"
        
        # This is a placeholder - would use actual AI model
        # For now, return sample metadata
        updated_table = []
        for row in metadata_table:
            if row and row[0]:  # If filename exists
                updated_table.append([
                    row[0],  # Filename
                    row[1] if row[1] else "pop",  # Genre
                    row[2] if row[2] else 120,     # BPM
                    row[3] if row[3] else "C major",  # Key
                    row[4] if row[4] else "energetic",  # Mood
                    "synth, drums, bass",  # Instruments
                    f"AI-generated music in {row[1] if row[1] else 'unknown'} style"  # Description
                ])
        
        return f"✅ Generated metadata for {len(updated_table)} file(s)"
        
    except Exception as e:
        logger.error(f"Metadata generation failed: {e}")
        return f"❌ Error: {str(e)}"

def check_downloaded_datasets():
    """Check and display status of already downloaded datasets"""
    try:
        from backend.services.dataset_service import DatasetService
        
        dataset_service = DatasetService()
        downloaded = dataset_service.get_downloaded_datasets()
        
        if not downloaded:
            return "📁 No datasets downloaded yet.\n\nSelect datasets above and click 'Download Datasets' to get started."
        
        status_messages = []
        status_messages.append(f"📊 Downloaded Datasets Status\n")
        status_messages.append(f"{'='*60}\n")
        
        for dataset_key, info in downloaded.items():
            status_messages.append(f"✅ {info.get('name', dataset_key)}")
            status_messages.append(f"   Type: {info.get('type', 'unknown')}")
            status_messages.append(f"   Size: ~{info.get('size_gb', 0):.1f} GB")
            
            if info.get('prepared'):
                status_messages.append(f"   Status: ✅ Prepared for training")
                status_messages.append(f"   Train samples: {info.get('num_train_samples', 0)}")
                status_messages.append(f"   Val samples: {info.get('num_val_samples', 0)}")
            else:
                status_messages.append(f"   Status: ⏳ Downloaded, needs preparation")
                
            status_messages.append("")
        
        status_messages.append(f"{'='*60}")
        status_messages.append(f"Total downloaded: {len(downloaded)} dataset(s)")
        
        return "\n".join(status_messages)
        
    except Exception as e:
        logger.error(f"Error checking datasets: {e}", exc_info=True)
        return f"❌ Error checking datasets: {str(e)}"

def get_dataset_choices_with_status():
    """Get dataset choices with download/preparation status indicators"""
    try:
        from backend.services.dataset_service import DatasetService
        
        dataset_service = DatasetService()
        downloaded = dataset_service.get_downloaded_datasets()
        
        # Dataset display mappings
        dataset_display_map = {
            "gtzan": "GTZAN Music Genre (1000 tracks, 10 genres)",
            "msd": "Million Song Dataset (10K subset)",
            "jamendo": "MTG-Jamendo (55k tracks, music tagging)",
            "musiccaps": "MusicCaps (5.5k clips with descriptions)",
            "fleurs": "FLEURS English Speech (multi-speaker)",
            "librispeech": "LibriSpeech ASR (speech recognition)",
            "libritts": "LibriTTS (audiobooks for TTS)",
            "audioset_strong": "AudioSet Strong (labeled audio events)",
            "esc50": "ESC-50 Environmental Sounds",
            "urbansound8k": "UrbanSound8K (urban sounds)"
        }
        
        music_choices = []
        vocal_choices = []
        prepare_choices = []
        
        music_keys = ["gtzan", "msd", "jamendo", "musiccaps"]
        vocal_keys = ["fleurs", "librispeech", "libritts", "audioset_strong", "esc50", "urbansound8k"]
        
        for key in music_keys:
            display_name = dataset_display_map.get(key, key)
            if key in downloaded:
                info = downloaded[key]
                if info.get('prepared'):
                    music_choices.append(f"✅ {display_name} [Downloaded & Prepared]")
                    prepare_choices.append(f"✅ {key} [Already Prepared]")
                else:
                    music_choices.append(f"📥 {display_name} [Downloaded]")
                    prepare_choices.append(key)
            else:
                music_choices.append(display_name)
        
        for key in vocal_keys:
            display_name = dataset_display_map.get(key, key)
            if key in downloaded:
                info = downloaded[key]
                if info.get('prepared'):
                    vocal_choices.append(f"✅ {display_name} [Downloaded & Prepared]")
                else:
                    vocal_choices.append(f"📥 {display_name} [Downloaded]")
            else:
                vocal_choices.append(display_name)
        
        return music_choices, vocal_choices, prepare_choices
        
    except Exception as e:
        logger.error(f"Error getting dataset status: {e}", exc_info=True)
        # Return default choices on error
        return [], [], []

def download_prepare_datasets(vocal_datasets, symbolic_datasets):
    """Download and prepare curated datasets for training"""
    try:
        from backend.services.dataset_service import DatasetService
        
        selected_datasets = []
        if vocal_datasets:
            selected_datasets.extend(vocal_datasets)
        if symbolic_datasets:
            selected_datasets.extend(symbolic_datasets)
        
        if not selected_datasets:
            return "❌ No datasets selected. Please check at least one dataset."
        
        dataset_service = DatasetService()
        
        # Map display names to dataset keys (handle status indicators)
        dataset_map = {
            # Music datasets
            "GTZAN Music Genre (1000 tracks, 10 genres)": "gtzan",
            "Million Song Dataset (10K subset)": "msd",
            "MTG-Jamendo (55k tracks, music tagging)": "jamendo",
            "MusicCaps (5.5k clips with descriptions)": "musiccaps",
            # Vocal & Sound datasets
            "FLEURS English Speech (multi-speaker)": "fleurs",
            "LibriSpeech ASR (speech recognition)": "librispeech",
            "LibriTTS (audiobooks for TTS)": "libritts",
            "AudioSet Strong (labeled audio events)": "audioset_strong",
            "ESC-50 Environmental Sounds": "esc50",
            "UrbanSound8K (urban sounds)": "urbansound8k"
        }
        
        # Extract dataset keys from selected items (strip status indicators)
        dataset_keys = []
        for item in selected_datasets:
            # Remove status prefixes and suffixes
            clean_item = item.replace("✅ ", "").replace("📥 ", "")
            clean_item = clean_item.split(" [")[0]  # Remove [Downloaded] or [Prepared] suffix
            
            if clean_item in dataset_map:
                dataset_keys.append(dataset_map[clean_item])
            else:
                # Direct key match
                if clean_item in ["gtzan", "msd", "jamendo", "musiccaps",
                                   "fleurs", "librispeech", "libritts", "audioset_strong", "esc50", "urbansound8k"]:
                    dataset_keys.append(clean_item)
        
        if not dataset_keys:
            return "❌ No valid datasets selected."
        
        status_messages = []
        status_messages.append(f"📥 Starting download for {len(dataset_keys)} dataset(s)...\n")
        
        success_count = 0
        already_downloaded_count = 0
        manual_count = 0
        error_count = 0
        
        for dataset_key in dataset_keys:
            # Get display name for this key
            dataset_config = dataset_service.DATASETS.get(dataset_key, {})
            dataset_display_name = dataset_config.get('name', dataset_key)
            
            status_messages.append(f"\n{'='*60}")
            status_messages.append(f"📦 Processing: {dataset_display_name}")
            status_messages.append(f"{'='*60}\n")
            
            # Progress callback
            progress_msgs = []
            def progress_callback(msg):
                progress_msgs.append(msg)
            
            # Download dataset
            result = dataset_service.download_dataset(dataset_key, progress_callback)
            
            # Add progress messages
            for msg in progress_msgs:
                status_messages.append(f"   {msg}")
            
            if result.get('success'):
                if result.get('already_downloaded'):
                    already_downloaded_count += 1
                else:
                    success_count += 1
                    info = result.get('info', {})
                    status_messages.append(f"\n   ✅ Successfully downloaded!")
                    status_messages.append(f"   📊 Examples: {info.get('num_examples', 'N/A')}")
                    status_messages.append(f"   💾 Path: {info.get('path', 'N/A')}\n")
            elif result.get('manual_download_required'):
                manual_count += 1
                status_messages.append(f"\n   ℹ️ Manual download required")
                status_messages.append(f"   🔗 URL: {result.get('url', 'N/A')}\n")
            else:
                error_count += 1
                status_messages.append(f"\n   ❌ Error: {result.get('error', 'Unknown error')}\n")
        
        # Summary
        status_messages.append(f"\n{'='*60}")
        status_messages.append("📊 DOWNLOAD SUMMARY")
        status_messages.append(f"{'='*60}")
        status_messages.append(f"✅ Newly downloaded: {success_count}")
        status_messages.append(f"♻️  Already downloaded: {already_downloaded_count}")
        status_messages.append(f"ℹ️  Manual required: {manual_count}")
        status_messages.append(f"❌ Errors: {error_count}")
        status_messages.append(f"📁 Total processed: {len(selected_datasets)}")
        
        if success_count > 0 or already_downloaded_count > 0:
            status_messages.append(f"\n✅ Datasets are ready!")
            status_messages.append(f"💡 Use the 'Training Configuration' tab to start training")
        
        return "\n".join(status_messages)
        
    except Exception as e:
        logger.error(f"Dataset download failed: {e}", exc_info=True)
        return f"❌ Error: {str(e)}\n\nStacktrace logged to console."

def prepare_datasets_for_training(selected_datasets, max_samples_per_dataset):
    """Prepare downloaded datasets for LoRA training by extracting audio files"""
    try:
        from backend.services.dataset_service import DatasetService
        
        if not selected_datasets:
            return "❌ No datasets selected. Please check at least one downloaded dataset."
        
        dataset_service = DatasetService()
        
        # Extract dataset keys (remove status indicators)
        dataset_keys = []
        for item in selected_datasets:
            # Remove status prefix if present
            clean_item = item.replace("✅ ", "").replace("📥 ", "").split(" [")[0]
            dataset_keys.append(clean_item)
        
        status_messages = []
        status_messages.append(f"🔧 Preparing {len(dataset_keys)} dataset(s) for training...\n")
        
        success_count = 0
        already_prepared_count = 0
        error_count = 0
        
        max_samples = int(max_samples_per_dataset) if max_samples_per_dataset > 0 else None
        
        for dataset_key in dataset_keys:
            status_messages.append(f"\n{'='*60}")
            status_messages.append(f"🔧 Preparing: {dataset_key}")
            status_messages.append(f"{'='*60}\n")
            
            # Progress callback
            progress_msgs = []
            def progress_callback(msg):
                progress_msgs.append(msg)
            
            # Prepare dataset
            result = dataset_service.prepare_dataset_for_training(
                dataset_key,
                train_val_split=0.8,
                max_samples=max_samples,
                progress_callback=progress_callback
            )
            
            # Add progress messages
            for msg in progress_msgs:
                status_messages.append(f"   {msg}")
            
            if result.get('success'):
                if result.get('already_prepared'):
                    already_prepared_count += 1
                    status_messages.append(f"\n   ℹ️ Dataset was already prepared")
                else:
                    success_count += 1
                    status_messages.append(f"\n   ✅ Preparation complete!")
                    status_messages.append(f"   📊 Training samples: {result.get('num_train', 0)}")
                    status_messages.append(f"   📊 Validation samples: {result.get('num_val', 0)}")
            else:
                error_count += 1
                status_messages.append(f"\n   ❌ Error: {result.get('error', 'Unknown error')}\n")
        
        # Summary
        status_messages.append(f"\n{'='*60}")
        status_messages.append(f"Summary:")
        status_messages.append(f"  ✅ Successfully prepared: {success_count}")
        status_messages.append(f"  ℹ️ Already prepared: {already_prepared_count}")
        status_messages.append(f"  ❌ Errors: {error_count}")
        status_messages.append(f"{'='*60}")
        
        if success_count > 0:
            status_messages.append(f"\n✅ Datasets are now ready for LoRA training!")
            status_messages.append(f"💡 Go to 'Training Configuration' tab to start training")
        
        return "\n".join(status_messages)
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}", exc_info=True)
        return f"❌ Error: {str(e)}"

def refresh_dataset_status():
    """Refresh dataset status and return updated choices"""
    music_choices, vocal_choices, prepare_choices = get_dataset_choices_with_status()
    return (
        gr.update(choices=music_choices if music_choices else [
            "GTZAN Music Genre (1000 tracks, 10 genres)",
            "Million Song Dataset (10K subset)",
            "MTG-Jamendo (55k tracks, music tagging)",
            "MusicCaps (5.5k clips with descriptions)"
        ]),
        gr.update(choices=vocal_choices if vocal_choices else [
            "FLEURS English Speech (multi-speaker)",
            "LibriSpeech ASR (speech recognition)",
            "LibriTTS (audiobooks for TTS)",
            "AudioSet Strong (labeled audio events)",
            "ESC-50 Environmental Sounds",
            "UrbanSound8K (urban sounds)"
        ]),
        gr.update(choices=prepare_choices if prepare_choices else [
            "gtzan", "msd", "jamendo", "musiccaps",
            "fleurs", "librispeech", "libritts", "audioset_strong", "esc50", "urbansound8k"
        ])
    )

def prepare_user_training_dataset(audio_files, metadata_table, split_clips, separate_stems):
    """Prepare user audio dataset for training"""
    try:
        if not audio_files:
            return "❌ No audio files uploaded"
        
        from backend.services.audio_analysis_service import AudioAnalysisService
        from backend.services.lora_training_service import LoRATrainingService
        
        analyzer = AudioAnalysisService()
        lora_service = LoRATrainingService()
        
        # Process audio files
        processed_files = []
        processed_metadata = []
        
        for i, audio_file in enumerate(audio_files):
            # Get metadata from table
            if metadata_table and i < len(metadata_table):
                file_metadata = {
                    'genre': metadata_table[i][1],
                    'bpm': int(metadata_table[i][2]) if metadata_table[i][2] else 120,
                    'key': metadata_table[i][3],
                    'mood': metadata_table[i][4],
                    'instrumentation': metadata_table[i][5],
                    'description': metadata_table[i][6]
                }
            else:
                # Analyze if no metadata
                file_metadata = analyzer.analyze_audio(audio_file.name)
            
            # Split into clips if requested
            if split_clips:
                clip_paths = analyzer.split_audio_to_clips(
                    audio_file.name,
                    "training_data/user_uploads/clips",
                    metadata=file_metadata
                )
                processed_files.extend(clip_paths)
                processed_metadata.extend([file_metadata] * len(clip_paths))
            else:
                processed_files.append(audio_file.name)
                processed_metadata.append(file_metadata)
            
            # Separate stems if requested
            if separate_stems:
                stem_paths = analyzer.separate_vocal_stems(
                    audio_file.name,
                    "training_data/user_uploads/stems"
                )
                # Use vocals only for vocal training
                if 'vocals' in stem_paths:
                    processed_files.append(stem_paths['vocals'])
                    processed_metadata.append({**file_metadata, 'type': 'vocal'})
        
        # Prepare dataset
        dataset_name = f"user_dataset_{int(time.time())}"
        dataset_info = lora_service.prepare_dataset(
            dataset_name,
            processed_files,
            processed_metadata
        )
        
        return f"✅ Prepared dataset '{dataset_name}' with {dataset_info['num_samples']} samples ({dataset_info['num_train']} train, {dataset_info['num_val']} val)"
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return f"❌ Error: {str(e)}"

def refresh_dataset_list():
    """Refresh list of available datasets"""
    try:
        from backend.services.lora_training_service import LoRATrainingService
        lora_service = LoRATrainingService()
        
        datasets = lora_service.list_datasets()
        return gr.Dropdown(choices=datasets)
        
    except Exception as e:
        logger.error(f"Failed to refresh datasets: {e}")
        return gr.Dropdown(choices=[])

def start_lora_training(lora_name, dataset, batch_size, learning_rate, num_epochs, lora_rank, lora_alpha):
    """Start LoRA training"""
    try:
        if not lora_name:
            return "❌ Please enter LoRA adapter name", ""
        
        if not dataset:
            return "❌ Please select a dataset", ""
        
        from backend.services.lora_training_service import LoRATrainingService
        lora_service = LoRATrainingService()
        
        # Training config
        config = {
            'batch_size': int(batch_size),
            'learning_rate': float(learning_rate),
            'num_epochs': int(num_epochs),
            'lora_rank': int(lora_rank),
            'lora_alpha': int(lora_alpha)
        }
        
        # Progress callback
        progress_log = []
        def progress_callback(status):
            progress_log.append(
                f"Epoch {status['epoch']} | Step {status['step']} | Loss: {status['loss']:.4f} | Progress: {status['progress']:.1f}%"
            )
            return "\n".join(progress_log[-20:])  # Last 20 lines
        
        # Start training
        progress = f"🚀 Starting training: {lora_name}\nDataset: {dataset}\nConfig: {config}\n\n"
        log = "Training started...\n"
        
        # Note: In production, this should run in a background thread
        # For now, this is a simplified synchronous version
        results = lora_service.train_lora(
            dataset,
            lora_name,
            training_type="vocal",
            config=config,
            progress_callback=progress_callback
        )
        
        progress += f"\n✅ Training complete!\nFinal validation loss: {results['final_val_loss']:.4f}"
        log += f"\n\nTraining Results:\n{json.dumps(results, indent=2)}"
        
        return progress, log
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return f"❌ Error: {str(e)}", str(e)

def stop_lora_training():
    """Stop current training"""
    try:
        from backend.services.lora_training_service import LoRATrainingService
        lora_service = LoRATrainingService()
        
        lora_service.stop_training()
        return "⏹️ Training stopped"
        
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        return f"❌ Error: {str(e)}"

def refresh_lora_list():
    """Refresh list of LoRA adapters"""
    try:
        from backend.services.lora_training_service import LoRATrainingService
        lora_service = LoRATrainingService()
        
        adapters = lora_service.list_lora_adapters()
        
        # Format as table
        table_data = []
        lora_names = []
        
        for adapter in adapters:
            table_data.append([
                adapter.get('name', ''),
                adapter.get('saved_at', ''),
                adapter.get('training_steps', 0),
                adapter.get('training_type', 'unknown')
            ])
            lora_names.append(adapter.get('name', ''))
        
        return table_data, gr.Dropdown(choices=lora_names)
        
    except Exception as e:
        logger.error(f"Failed to refresh LoRA list: {e}")
        return [], gr.Dropdown(choices=[])

def delete_lora(lora_name):
    """Delete selected LoRA adapter"""
    try:
        if not lora_name:
            return "❌ No LoRA selected"
        
        from backend.services.lora_training_service import LoRATrainingService
        lora_service = LoRATrainingService()
        
        success = lora_service.delete_lora_adapter(lora_name)
        
        if success:
            return f"✅ Deleted LoRA adapter: {lora_name}"
        else:
            return f"❌ Failed to delete: {lora_name}"
        
    except Exception as e:
        logger.error(f"Failed to delete LoRA: {e}")
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    title="🎵 Music Generation Studio",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="pink")
) as app:
    
    gr.Markdown(
        """
        # 🎵 Music Generation Studio
        
        Create AI-powered music with DiffRhythm2 and LyricMind AI
        
        💡 **Tip**: Start with 10-20 second clips for faster generation with ZeroGPU
        """
    )
    
    # Timeline state - persists across GPU context switches
    timeline_state = gr.State(value={'clips': []})
    
    # Generation Section
    gr.Markdown("### 🎼 Music Generation")
    
    prompt_input = gr.Textbox(
        label="🎯 Music Prompt",
        placeholder="energetic rock song with electric guitar at 140 BPM",
        lines=3,
        info="Describe the music style, instruments, tempo, and mood"
    )
    
    lyrics_mode = gr.Radio(
        choices=["Instrumental", "User Lyrics", "Auto Lyrics"],
        value="Instrumental",
        label="🎤 Vocal Mode",
        info="Instrumental: no vocals | User: provide lyrics | Auto: AI-generated"
    )
    
    with gr.Row():
        auto_gen_btn = gr.Button("✍️ Generate Lyrics", size="sm")
    
    lyrics_input = gr.Textbox(
        label="📝 Lyrics",
        placeholder="Enter lyrics or click 'Generate Lyrics'...",
        lines=6
    )
    
    with gr.Row():
        context_length_input = gr.Slider(
            minimum=0,
            maximum=240,
            value=0,
            step=10,
            label="🎨 Style Context (seconds)",
            info="How far back to analyze for style consistency (0 = disabled, auto-disabled for first clip)",
            interactive=True
        )
        position_input = gr.Radio(
            choices=["intro", "previous", "next", "outro"],
            value="next",
            label="📍 Position",
            info="Where to add clip on timeline"
        )
    
    gr.Markdown("*All clips are generated at 32 seconds*")
    
    with gr.Row():
        generate_btn = gr.Button(
            "✨ Generate Music Clip",
            variant="primary",
            size="lg"
        )
    
    gen_status = gr.Textbox(label="📊 Status", lines=2, interactive=False)
    audio_output = gr.Audio(
        label="🎧 Preview", 
        type="filepath",
        waveform_options=gr.WaveformOptions(
            waveform_color="#9333ea",
            waveform_progress_color="#c084fc"
        )
    )
    
    # Timeline Section
    gr.Markdown("---")
    gr.Markdown("### 📊 Timeline")
    
    timeline_display = gr.HTML(
        value=get_timeline_display()
    )
    
    # Playback controls
    timeline_playback = gr.Audio(
        label="🎵 Timeline Playback",
        type="filepath",
        interactive=False,
        autoplay=False,
        waveform_options=gr.WaveformOptions(
            waveform_color="#06b6d4",
            waveform_progress_color="#22d3ee",
            show_controls=True
        )
    )
    
    with gr.Row():
        play_timeline_btn = gr.Button("▶️ Load Timeline for Playback", variant="secondary", scale=2)
        clip_number_input = gr.Number(
            label="Clip #",
            precision=0,
            minimum=1,
            scale=1
        )
        remove_btn = gr.Button("🗑️ Remove Clip", size="sm", scale=1)
        clear_btn = gr.Button("🗑️ Clear All", variant="stop", scale=1)
    
    timeline_status = gr.Textbox(label="Timeline Status", lines=1, interactive=False)
    
    # Advanced Controls
    with gr.Accordion("⚙️ Advanced Audio Mastering", open=False):
        gr.Markdown("### Professional Mastering & EQ")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Mastering Presets**")
                preset_select = gr.Dropdown(
                    choices=[
                        "Clean Master - Transparent mastering",
                        "Subtle Warmth - Gentle low-end enhancement",
                        "Modern Pop - Radio-ready pop sound",
                        "Radio Ready - Maximum loudness",
                        "Punchy Commercial - Aggressive punch",
                        "Rock Master - Guitar-focused mastering",
                        "Metal Aggressive - Heavy metal mastering",
                        "Indie Rock - Lo-fi indie character",
                        "EDM Club - Electronic dance music",
                        "House Groovy - House music vibe",
                        "Techno Dark - Dark techno atmosphere",
                        "Dubstep Heavy - Heavy bass dubstep",
                        "HipHop Modern - Modern hip-hop mix",
                        "Trap 808 - Trap with 808 bass",
                        "RnB Smooth - Smooth R&B sound",
                        "Acoustic Natural - Natural acoustic tone",
                        "Folk Warm - Warm folk sound",
                        "Jazz Vintage - Vintage jazz character",
                        "Orchestral Wide - Wide orchestral space",
                        "Classical Concert - Concert hall sound",
                        "Ambient Spacious - Spacious atmospheric",
                        "Harmonic Enhance - Adds brightness and warmth"
                    ],
                    value="Clean Master - Transparent mastering",
                    label="Select Preset"
                )
                
                preset_description = gr.Textbox(
                    label="Description",
                    value="Transparent mastering with gentle compression",
                    lines=2,
                    interactive=False
                )
                
                with gr.Row():
                    preview_preset_btn = gr.Button("🔊 Preview Preset", variant="secondary")
                    apply_preset_btn = gr.Button("✨ Apply to Timeline", variant="primary")
                
                preset_preview_audio = gr.Audio(
                    label="🎵 Preset Preview (Latest Clip)",
                    type="filepath",
                    interactive=False,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#9333ea",
                        waveform_progress_color="#c084fc"
                    )
                )
                preset_status = gr.Textbox(label="Status", lines=1, interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("**Custom EQ**")
                gr.Markdown("*5-band parametric EQ. Adjust gain for each frequency band (-12 to +12 dB).*")
                
                # DAW-style vertical sliders in columns
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("<center>**Low**<br>100 Hz</center>")
                        low_shelf_gain = gr.Slider(
                            -12, 12, 0, step=0.5,
                            label="Low (100 Hz)"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("<center>**Low-Mid**<br>500 Hz</center>")
                        low_mid_gain = gr.Slider(
                            -12, 12, 0, step=0.5,
                            label="Low-Mid (500 Hz)"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("<center>**Mid**<br>2000 Hz</center>")
                        mid_gain = gr.Slider(
                            -12, 12, 0, step=0.5,
                            label="Mid (2000 Hz)"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("<center>**High-Mid**<br>5000 Hz</center>")
                        high_mid_gain = gr.Slider(
                            -12, 12, 0, step=0.5,
                            label="High-Mid (5000 Hz)"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("<center>**High**<br>10k Hz</center>")
                        high_shelf_gain = gr.Slider(
                            -12, 12, 0, step=0.5,
                            label="High (10k Hz)"
                        )
                
                with gr.Row():
                    preview_eq_btn = gr.Button("🔊 Preview EQ", variant="secondary")
                    apply_custom_eq_btn = gr.Button("🎹 Apply to Timeline", variant="primary")
                
                eq_preview_audio = gr.Audio(
                    label="🎵 EQ Preview (Latest Clip)",
                    type="filepath",
                    interactive=False,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#ec4899",
                        waveform_progress_color="#f9a8d4"
                    )
                )
                eq_status = gr.Textbox(label="Status", lines=1, interactive=False)
        
        # Audio Enhancement Section
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**🎛️ Stem Enhancement**")
                gr.Markdown("*Separate and enhance vocals, drums, bass independently (improves AI audio quality)*")
                
                enhancement_level = gr.Radio(
                    choices=["Fast", "Balanced", "Maximum"],
                    value="Balanced",
                    label="Enhancement Level",
                    info="Fast: Quick denoise | Balanced: Best quality/speed | Maximum: Full processing"
                )
                
                enhance_timeline_btn = gr.Button("✨ Enhance All Clips", variant="primary")
                enhancement_status = gr.Textbox(label="Status", lines=2, interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("**🔊 Audio Upscaling**")
                gr.Markdown("*Neural upsampling to 48kHz for enhanced high-frequency detail*")
                
                upscale_mode = gr.Radio(
                    choices=["Quick (Resample)", "Neural (AudioSR)"],
                    value="Quick (Resample)",
                    label="Upscale Method",
                    info="Quick: Fast resampling | Neural: AI-powered super resolution"
                )
                
                upscale_timeline_btn = gr.Button("⬆️ Upscale All Clips", variant="primary")
                upscale_status = gr.Textbox(label="Status", lines=2, interactive=False)
    
    # Export Section
    gr.Markdown("---")
    gr.Markdown("### 💾 Export")
    
    with gr.Row():
        export_filename = gr.Textbox(
            label="Filename",
            value="my_song",
            scale=2
        )
        export_format = gr.Dropdown(
            choices=["wav", "mp3"],
            value="wav",
            label="Format",
            scale=1
        )
        export_btn = gr.Button("💾 Export", variant="primary", scale=1)
    
    export_status = gr.Textbox(label="Status", lines=1, interactive=False)
    export_audio = gr.Audio(
        label="📥 Download", 
        type="filepath",
        waveform_options=gr.WaveformOptions(
            waveform_color="#10b981",
            waveform_progress_color="#34d399"
        )
    )
    
    # Event handlers
    auto_gen_btn.click(
        fn=generate_lyrics,
        inputs=[prompt_input],
        outputs=lyrics_input
    )
    
    generate_btn.click(
        fn=generate_music,
        inputs=[prompt_input, lyrics_input, lyrics_mode, position_input, context_length_input, timeline_state],
        outputs=[gen_status, timeline_display, audio_output, timeline_state]
    )
    
    remove_btn.click(
        fn=remove_clip,
        inputs=[clip_number_input, timeline_state],
        outputs=[timeline_status, timeline_display, timeline_state]
    )
    
    clear_btn.click(
        fn=clear_timeline,
        inputs=[timeline_state],
        outputs=[timeline_status, timeline_display, timeline_state]
    )
    
    play_timeline_btn.click(
        fn=get_timeline_playback,
        inputs=[timeline_state],
        outputs=[timeline_playback]
    )
    
    export_btn.click(
        fn=export_timeline,
        inputs=[export_filename, export_format, timeline_state],
        outputs=[export_status, export_audio, timeline_state]
    )
    
    # Mastering event handlers
    preview_preset_btn.click(
        fn=preview_mastering_preset,
        inputs=[preset_select, timeline_state],
        outputs=[preset_preview_audio, preset_status]
    )
    
    apply_preset_btn.click(
        fn=apply_mastering_preset,
        inputs=[preset_select, timeline_state],
        outputs=[preset_status, timeline_state]
    ).then(
        fn=get_timeline_playback,
        inputs=[timeline_state],
        outputs=[timeline_playback]
    )
    
    preview_eq_btn.click(
        fn=preview_custom_eq,
        inputs=[low_shelf_gain, low_mid_gain, mid_gain, high_mid_gain, high_shelf_gain, timeline_state],
        outputs=[eq_preview_audio, eq_status]
    )
    
    apply_custom_eq_btn.click(
        fn=apply_custom_eq,
        inputs=[low_shelf_gain, low_mid_gain, mid_gain, high_mid_gain, high_shelf_gain, timeline_state],
        outputs=[eq_status, timeline_state]
    ).then(
        fn=get_timeline_playback,
        inputs=[timeline_state],
        outputs=[timeline_playback]
    )
    
    # Enhancement event handlers
    enhance_timeline_btn.click(
        fn=enhance_timeline_clips,
        inputs=[enhancement_level, timeline_state],
        outputs=[enhancement_status, timeline_state]
    ).then(
        fn=get_timeline_playback,
        inputs=[timeline_state],
        outputs=[timeline_playback]
    )
    
    upscale_timeline_btn.click(
        fn=upscale_timeline_clips,
        inputs=[upscale_mode, timeline_state],
        outputs=[upscale_status, timeline_state]
    ).then(
        fn=get_timeline_playback,
        inputs=[timeline_state],
        outputs=[timeline_playback]
    )
    
    # LoRA Training Section
    gr.Markdown("---")
    with gr.Accordion("🎓 LoRA Training (Advanced)", open=False):
        gr.Markdown(
            """
            # 🧠 Train Custom LoRA Adapters
            
            Fine-tune DiffRhythm2 with your own audio or curated datasets to create specialized music generation models.
            
            **Training is permanent** - LoRA adapters are saved to disk and persist across sessions.
            """
        )
        
        with gr.Tabs():
            # Tab 1: Dataset Training
            with gr.Tab("📚 Dataset Training"):
                gr.Markdown("### Pre-curated Dataset Training")
                gr.Markdown("Select datasets from the categories below. All datasets can be used for music generation training.")
                gr.Markdown("**Datasets persist across sessions** - once downloaded/prepared, they remain available.")
                
                refresh_datasets_status_btn = gr.Button("🔄 Refresh Dataset Status", size="sm", variant="secondary")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Music Datasets**")
                        vocal_datasets = gr.CheckboxGroup(
                            choices=[
                                "GTZAN Music Genre (1000 tracks, 10 genres)",
                                "Million Song Dataset (10K subset)",
                                "MTG-Jamendo (55k tracks, music tagging)",
                                "MusicCaps (5.5k clips with descriptions)"
                            ],
                            label="Select Music Datasets",
                            info="Verified working music datasets (11TB storage available)"
                        )
                    
                    with gr.Column():
                        gr.Markdown("**Vocal & Sound Datasets**")
                        symbolic_datasets = gr.CheckboxGroup(
                            choices=[
                                "FLEURS English Speech (multi-speaker)",
                                "LibriSpeech ASR (speech recognition)",
                                "LibriTTS (audiobooks for TTS)",
                                "AudioSet Strong (labeled audio events)",
                                "ESC-50 Environmental Sounds",
                                "UrbanSound8K (urban sounds)"
                            ],
                            label="Select Vocal/Sound Datasets",
                            info="Verified working vocal and sound datasets"
                        )
                
                with gr.Row():
                    dataset_download_btn = gr.Button("📥 Download Datasets", variant="secondary")
                    check_status_btn = gr.Button("📊 Check Downloaded Datasets", variant="secondary")
                
                dataset_status = gr.Textbox(
                    label="Download Status", 
                    lines=15, 
                    max_lines=25,
                    interactive=False,
                    show_copy_button=True
                )
                
                gr.Markdown("---")
                gr.Markdown("### 🔧 Prepare Downloaded Datasets for Training")
                gr.Markdown("After downloading, prepare datasets by extracting audio files and creating train/val splits.")
                
                prepare_datasets_selector = gr.CheckboxGroup(
                    choices=["gtzan", "msd", "jamendo", "musiccaps",
                             "fleurs", "librispeech", "libritts", "audioset_strong", "esc50", "urbansound8k"],
                    label="Select Downloaded Datasets to Prepare",
                    info="Only select datasets you've already downloaded above"
                )
                
                max_samples_slider = gr.Slider(
                    minimum=0,
                    maximum=10000,
                    value=1000,
                    step=100,
                    label="Max Samples per Dataset (0 = all)",
                    info="Limit samples to speed up preparation. 0 means process all samples."
                )
                
                prepare_btn = gr.Button("🔧 Prepare Datasets for Training", variant="primary")
                hf_prepare_status = gr.Textbox(
                    label="Preparation Status",
                    lines=15,
                    max_lines=25,
                    interactive=False,
                    show_copy_button=True
                )
            
            # Tab 2: User Audio Training
            with gr.Tab("🎵 User Audio Training"):
                gr.Markdown("### Train on Your Own Audio")
                
                user_audio_upload = gr.File(
                    label="Upload Audio Files (.wav)",
                    file_count="multiple",
                    file_types=[".wav"]
                )
                
                gr.Markdown("#### Audio Processing Options")
                
                with gr.Row():
                    split_to_clips = gr.Checkbox(
                        label="Auto-split into clips",
                        value=True,
                        info="Split long audio into 10-30s training clips"
                    )
                    separate_stems = gr.Checkbox(
                        label="Separate vocal stems",
                        value=False,
                        info="Extract vocals for vocal-only training (slower)"
                    )
                
                analyze_audio_btn = gr.Button("🔍 Analyze & Generate Metadata", variant="secondary")
                analysis_status = gr.Textbox(label="Analysis Status", lines=2, interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("#### Metadata Editor")
                
                metadata_table = gr.Dataframe(
                    headers=["File", "Genre", "BPM", "Key", "Mood", "Instruments", "Description"],
                    datatype=["str", "str", "number", "str", "str", "str", "str"],
                    row_count=5,
                    col_count=(7, "fixed"),
                    label="Audio Metadata",
                    interactive=True
                )
                
                with gr.Row():
                    ai_generate_metadata_btn = gr.Button("✨ AI Generate All Metadata", size="sm")
                    save_metadata_btn = gr.Button("💾 Save Metadata", variant="primary", size="sm")
                
                metadata_status = gr.Textbox(label="Metadata Status", lines=1, interactive=False)
                
                prepare_user_dataset_btn = gr.Button("📦 Prepare Training Dataset", variant="primary")
                user_prepare_status = gr.Textbox(label="Preparation Status", lines=2, interactive=False)
            
            # Tab 3: Training Configuration
            with gr.Tab("⚙️ Training Configuration"):
                gr.Markdown("### LoRA Training Settings")
                
                lora_name_input = gr.Textbox(
                    label="LoRA Adapter Name",
                    placeholder="my_custom_lora_v1",
                    info="Unique name for this LoRA adapter"
                )
                
                selected_dataset = gr.Dropdown(
                    choices=[],
                    label="Training Dataset",
                    info="Select prepared dataset to train on"
                )
                
                refresh_datasets_btn = gr.Button("🔄 Refresh Datasets", size="sm")
                
                gr.Markdown("#### Hyperparameters")
                
                with gr.Row():
                    with gr.Column():
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=16,
                            value=4,
                            step=1,
                            label="Batch Size",
                            info="Larger = faster but more GPU memory"
                        )
                        
                        learning_rate = gr.Slider(
                            minimum=1e-5,
                            maximum=1e-3,
                            value=3e-4,
                            step=1e-5,
                            label="Learning Rate",
                            info="Lower = more stable, higher = faster convergence"
                        )
                        
                        num_epochs = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1,
                            label="Number of Epochs",
                            info="How many times to iterate over dataset"
                        )
                    
                    with gr.Column():
                        lora_rank = gr.Slider(
                            minimum=4,
                            maximum=64,
                            value=16,
                            step=4,
                            label="LoRA Rank",
                            info="Higher = more capacity but slower"
                        )
                        
                        lora_alpha = gr.Slider(
                            minimum=8,
                            maximum=128,
                            value=32,
                            step=8,
                            label="LoRA Alpha",
                            info="Scaling factor for LoRA weights"
                        )
                
                gr.Markdown("---")
                
                start_training_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                stop_training_btn = gr.Button("⏹️ Stop Training", variant="stop", size="sm")
                
                training_progress = gr.Textbox(
                    label="Training Progress",
                    lines=5,
                    interactive=False
                )
                
                training_log = gr.Textbox(
                    label="Training Log",
                    lines=10,
                    interactive=False
                )
            
            # Tab 4: Manage LoRA Adapters
            with gr.Tab("📂 Manage LoRA Adapters"):
                gr.Markdown("### Installed LoRA Adapters")
                
                lora_list = gr.Dataframe(
                    headers=["Name", "Created", "Training Steps", "Type"],
                    datatype=["str", "str", "number", "str"],
                    row_count=10,
                    label="Available LoRA Adapters"
                )
                
                with gr.Row():
                    refresh_lora_btn = gr.Button("🔄 Refresh List", size="sm")
                    selected_lora = gr.Dropdown(
                        choices=[],
                        label="Select LoRA",
                        scale=2
                    )
                    delete_lora_btn = gr.Button("🗑️ Delete LoRA", variant="stop", size="sm")
                
                lora_management_status = gr.Textbox(label="Status", lines=1, interactive=False)
                
                gr.Markdown("---")
                gr.Markdown(
                    """
                    ### 💡 Training Tips
                    
                    **Dataset Size:**
                    - Vocal: 20-50 hours minimum for good results
                    - Symbolic: 1000+ MIDI files recommended
                    - User audio: 30+ minutes minimum (more is better)
                    
                    **Training Time Estimates:**
                    - Small dataset (< 1 hour): 2-4 hours training
                    - Medium dataset (1-10 hours): 4-12 hours training
                    - Large dataset (> 10 hours): 12-48 hours training
                    
                    **GPU Requirements:**
                    - Minimum: 16GB VRAM (LoRA training)
                    - Recommended: 24GB+ VRAM
                    - CPU training: 10-50x slower (not recommended)
                    
                    **Best Practices:**
                    1. Start with small learning rate (3e-4)
                    2. Use batch size 4-8 for best results
                    3. Monitor validation loss to prevent overfitting
                    4. Save checkpoints every 500 steps
                    5. Test generated samples during training
                    
                    **Audio Preprocessing:**
                    - Split long files into 10-30s clips for diversity
                    - Separate vocal stems for vocal-specific training
                    - Use AI metadata generation for consistent labels
                    - Ensure audio quality (44.1kHz, no compression artifacts)
                    """
                )
    
    # LoRA Training Event Handlers
    refresh_datasets_status_btn.click(
        fn=refresh_dataset_status,
        inputs=[],
        outputs=[vocal_datasets, symbolic_datasets, prepare_datasets_selector]
    )
    
    dataset_download_btn.click(
        fn=download_prepare_datasets,
        inputs=[vocal_datasets, symbolic_datasets],
        outputs=[dataset_status]
    ).then(
        fn=refresh_dataset_status,
        inputs=[],
        outputs=[vocal_datasets, symbolic_datasets, prepare_datasets_selector]
    )
    
    check_status_btn.click(
        fn=check_downloaded_datasets,
        inputs=[],
        outputs=[dataset_status]
    )
    
    prepare_btn.click(
        fn=prepare_datasets_for_training,
        inputs=[prepare_datasets_selector, max_samples_slider],
        outputs=[hf_prepare_status]
    ).then(
        fn=refresh_dataset_status,
        inputs=[],
        outputs=[vocal_datasets, symbolic_datasets, prepare_datasets_selector]
    )
    
    analyze_audio_btn.click(
        fn=analyze_user_audio,
        inputs=[user_audio_upload, split_to_clips, separate_stems],
        outputs=[analysis_status, metadata_table]
    )
    
    ai_generate_metadata_btn.click(
        fn=ai_generate_all_metadata,
        inputs=[metadata_table],
        outputs=[metadata_status]
    )
    
    prepare_user_dataset_btn.click(
        fn=prepare_user_training_dataset,
        inputs=[user_audio_upload, metadata_table, split_to_clips, separate_stems],
        outputs=[user_prepare_status]
    )
    
    refresh_datasets_btn.click(
        fn=refresh_dataset_list,
        inputs=[],
        outputs=[selected_dataset]
    )
    
    start_training_btn.click(
        fn=start_lora_training,
        inputs=[lora_name_input, selected_dataset, batch_size, learning_rate, num_epochs, lora_rank, lora_alpha],
        outputs=[training_progress, training_log]
    )
    
    stop_training_btn.click(
        fn=stop_lora_training,
        inputs=[],
        outputs=[training_progress]
    )
    
    refresh_lora_btn.click(
        fn=refresh_lora_list,
        inputs=[],
        outputs=[lora_list, selected_lora]
    )
    
    delete_lora_btn.click(
        fn=delete_lora,
        inputs=[selected_lora],
        outputs=[lora_management_status]
    ).then(
        fn=refresh_lora_list,
        inputs=[],
        outputs=[lora_list, selected_lora]
    )
    
    # Help section
    with gr.Accordion("ℹ️ Help & Tips", open=False):
        gr.Markdown(
            """
            ## 🚀 Quick Start
            
            1. **Enter a prompt**: "upbeat pop song with synth at 128 BPM"
            2. **Choose mode**: Instrumental (fastest) or with vocals
            3. **Set duration**: Start with 10-20s for quick results
            4. **Generate**: Click the button and wait ~2-4 minutes
            5. **Export**: Download your complete song
            
            ## ⚡ Performance Tips
            
            - **Shorter clips = faster**: 10-20s clips generate in ~1-2 minutes
            - **Instrumental mode**: ~30% faster than with vocals
            - **HF Spaces uses CPU**: Expect 2-4 minutes per 30s clip
            - **Build incrementally**: Generate short clips, then combine
            
            ## 🎯 Prompt Tips
            
            - **Be specific**: "energetic rock with distorted guitar" > "rock song"
            - **Include BPM**: "at 140 BPM" helps set tempo
            - **Mention instruments**: "with piano and drums"
            - **Describe mood**: "melancholic", "upbeat", "aggressive"
            
            ## 🎤 Vocal Modes
            
            - **Instrumental**: Pure music, no vocals (fastest)
            - **User Lyrics**: Provide your own lyrics
            - **Auto Lyrics**: AI generates lyrics based on prompt
            
            ## 📊 Timeline
            
            - Clips are arranged sequentially
            - Remove or clear clips as needed
            - Export combines all clips into one file
            
            ## 🎛️ Audio Enhancement (Advanced)
            
            - **Stem Enhancement**: Separates vocals, drums, bass for individual processing
              - *Fast*: Quick denoise (~2-3s per clip)
              - *Balanced*: Best quality/speed (~5-7s per clip)
              - *Maximum*: Full processing (~10-15s per clip)
            - **Audio Upscaling**: Increase sample rate to 48kHz
              - *Quick*: Fast resampling (~1s per clip)
              - *Neural*: AI super-resolution (~10-20s per clip, better quality)
            
            **Note**: Apply enhancements AFTER all clips are generated and BEFORE export
            
            ---
            
            ⏱️ **Average Generation Time**: 2-4 minutes per 30-second clip on CPU
            
            🎵 **Models**: DiffRhythm2 + MuQ-MuLan + LyricMind AI
            """
        )

# Configure and launch
if __name__ == "__main__":
    logger.info("🎵 Starting Music Generation Studio on HuggingFace Spaces...")
    
    app.queue(
        default_concurrency_limit=1,
        max_size=5
    )
    
    app.launch()
