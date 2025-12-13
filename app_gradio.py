"""
Gradio-based GUI for Music Generation App
"""
import os
import sys
import gradio as gr
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.diffrhythm_service import DiffRhythmService
from services.lyricmind_service import LyricMindService
from services.timeline_service import TimelineService
from services.export_service import ExportService
from config.settings import Config
from utils.logger import setup_logger
from utils.prompt_analyzer import PromptAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
config = Config()
timeline_service = TimelineService()
export_service = ExportService()

# Lazy-load AI services
diffrhythm_service = None
lyricmind_service = None

def get_diffrhythm_service():
    global diffrhythm_service
    if diffrhythm_service is None:
        diffrhythm_service = DiffRhythmService(model_path=config.DIFFRHYTHM_MODEL_PATH)
    return diffrhythm_service

def get_lyricmind_service():
    global lyricmind_service
    if lyricmind_service is None:
        lyricmind_service = LyricMindService(model_path=config.LYRICMIND_MODEL_PATH)
    return lyricmind_service

def generate_lyrics(prompt: str, duration: int, progress=gr.Progress()):
    """Generate lyrics from prompt using analysis"""
    try:
        if not prompt or not prompt.strip():
            return "Error: Please enter a prompt"
        
        # Estimate time: ~5-10 seconds
        progress(0, desc="Analyzing prompt...")
        logger.info(f"Generating lyrics: {prompt}")
        
        # Analyze prompt for context
        analysis = PromptAnalyzer.analyze(prompt)
        logger.info(f"Detected: {analysis['genre']} at {analysis['bpm']} BPM, {analysis['mood']} mood")
        
        progress(0.3, desc=f"Generating {analysis['genre']} lyrics... (Est. 5-10s)")
        
        service = get_lyricmind_service()
        lyrics = service.generate(prompt=prompt, duration=duration, prompt_analysis=analysis)
        
        progress(1.0, desc="✅ Lyrics generated!")
        return lyrics
    except Exception as e:
        logger.error(f"Error generating lyrics: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def generate_music(prompt: str, lyrics: str, lyrics_mode: str, duration: int, position: str, progress=gr.Progress()):
    """Generate music clip with optional vocals and add to timeline"""
    try:
        if not prompt or not prompt.strip():
            return "Error: Please enter a music prompt", get_timeline_display(), None
        
        # Calculate estimated time based on duration and mode
        # CPU: ~1.5-2x duration, GPU: ~0.3-0.5x duration
        est_time_cpu = int(duration * 1.5)
        est_time_gpu = int(duration * 0.4)
        
        progress(0, desc=f"Analyzing prompt... (Est. {est_time_cpu}s CPU / {est_time_gpu}s GPU)")
        logger.info(f"Generating music: {prompt}, lyrics_mode={lyrics_mode}")
        
        # Analyze prompt
        analysis = PromptAnalyzer.analyze(prompt)
        logger.info(f"Analysis: {analysis['genre']} at {analysis['bpm']} BPM, {analysis['mood']} mood")
        
        # Determine lyrics based on mode
        lyrics_to_use = None
        
        if lyrics_mode == "Instrumental":
            # No vocals
            lyrics_to_use = None
            logger.info("Generating instrumental music (no vocals)")
            progress(0.1, desc=f"Preparing instrumental generation... ({est_time_cpu}s)")
            
        elif lyrics_mode == "User Lyrics":
            # Use provided lyrics
            if not lyrics or not lyrics.strip():
                return "Error: Please enter lyrics or switch to Instrumental/Auto Lyrics mode", get_timeline_display(), None
            lyrics_to_use = lyrics.strip()
            logger.info("Using user-provided lyrics")
            progress(0.1, desc=f"Preparing vocal generation... ({est_time_cpu}s)")
            
        elif lyrics_mode == "Auto Lyrics":
            # Auto-generate lyrics
            if lyrics and lyrics.strip():
                # Use existing lyrics if provided
                lyrics_to_use = lyrics.strip()
                logger.info("Using existing lyrics in textbox")
                progress(0.1, desc=f"Using existing lyrics... ({est_time_cpu}s)")
            else:
                # Generate new lyrics
                progress(0.1, desc="Generating lyrics... (10s)")
                logger.info("Auto-generating lyrics...")
                lyric_service = get_lyricmind_service()
                lyrics_to_use = lyric_service.generate(
                    prompt=prompt,
                    duration=duration,
                    prompt_analysis=analysis
                )
                logger.info(f"Generated {len(lyrics_to_use)} characters of lyrics")
                progress(0.25, desc=f"Lyrics ready, generating music... ({est_time_cpu}s)")
        
        # Generate music with DiffRhythm (includes vocals if lyrics provided)
        progress(0.3, desc=f"Generating {analysis['genre']} music at {analysis['bpm']} BPM... ({est_time_cpu}s)")
        service = get_diffrhythm_service()
        final_path = service.generate(
            prompt=prompt,
            duration=duration,
            lyrics=lyrics_to_use
        )
        
        # Add to timeline
        progress(0.9, desc="Adding to timeline...")
        clip_id = os.path.basename(final_path).split('.')[0]
        from models.schemas import ClipPosition
        
        clip_info = timeline_service.add_clip(
            clip_id=clip_id,
            file_path=final_path,
            duration=float(duration),
            position=ClipPosition(position)
        )
        
        logger.info(f"Music generated and added to timeline: {clip_id} at position {clip_info['timeline_position']}")
        
        # Build status message
        progress(1.0, desc="✅ Generation complete!")
        status_msg = f"✅ Music clip generated successfully!\n"
        status_msg += f"Genre: {analysis['genre']} | BPM: {analysis['bpm']} | Mood: {analysis['mood']}\n"
        status_msg += f"Mode: {lyrics_mode} | Position: {position}\n"
        
        if lyrics_mode == "Auto Lyrics" and lyrics_to_use and not lyrics:
            status_msg += "(Lyrics auto-generated)"
        
        return (
            status_msg,
            get_timeline_display(),
            final_path
        )
        
    except Exception as e:
        logger.error(f"Error generating music: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", get_timeline_display(), None

def get_timeline_display():
    """Get timeline clips as formatted text"""
    clips = timeline_service.get_all_clips()
    
    if not clips:
        return "Timeline is empty. Generate clips to get started!"
    
    total_duration = timeline_service.get_total_duration()
    
    display = f"**Timeline ({len(clips)} clips, {format_duration(total_duration)} total)**\n\n"
    
    for i, clip in enumerate(clips, 1):
        display += f"{i}. Clip {clip['clip_id'][:8]}... | "
        display += f"Duration: {format_duration(clip['duration'])} | "
        display += f"Start: {format_duration(clip['start_time'])}\n"
    
    return display

def remove_clip(clip_number: int):
    """Remove a clip from timeline by its number"""
    try:
        clips = timeline_service.get_all_clips()
        
        if not clips:
            return "Timeline is empty", get_timeline_display()
        
        if clip_number < 1 or clip_number > len(clips):
            return f"Error: Invalid clip number. Choose between 1 and {len(clips)}", get_timeline_display()
        
        clip_id = clips[clip_number - 1]['clip_id']
        timeline_service.remove_clip(clip_id)
        
        return f"✅ Clip {clip_number} removed", get_timeline_display()
        
    except Exception as e:
        logger.error(f"Error removing clip: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", get_timeline_display()

def clear_timeline():
    """Clear all clips from timeline"""
    try:
        timeline_service.clear()
        return "✅ Timeline cleared", get_timeline_display()
    except Exception as e:
        logger.error(f"Error clearing timeline: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", get_timeline_display()

def export_timeline(filename: str, export_format: str):
    """Export timeline to audio file"""
    try:
        clips = timeline_service.get_all_clips()
        
        if not clips:
            return "Error: No clips to export", None
        
        if not filename or not filename.strip():
            filename = "output"
        
        logger.info(f"Exporting timeline: {filename}")
        
        # Update export service to use the same timeline instance
        export_service.timeline_service = timeline_service
        
        output_path = export_service.merge_clips(
            filename=filename,
            export_format=export_format
        )
        
        if output_path:
            return f"✅ Exported successfully: {os.path.basename(output_path)}", output_path
        else:
            return "Error: Export failed", None
            
    except Exception as e:
        logger.error(f"Error exporting: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", None

def format_duration(seconds: float) -> str:
    """Format duration in seconds to MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

def create_gradio_interface():
    """Create and configure Gradio interface"""
    
    with gr.Blocks(title="🎵 Music Generation Studio") as app:
        gr.Markdown(
            """
            # 🎵 Music Generation Studio
            Create AI-powered music with intelligent prompt analysis and context-aware generation
            """
        )
        
        with gr.Row():
            # Left Column - Generation Controls
            with gr.Column(scale=2):
                gr.Markdown("### 🎼 Music Generation")
                
                prompt_input = gr.Textbox(
                    label="Music Prompt",
                    placeholder="Describe the music you want (e.g., upbeat pop song with drums and synth)",
                    lines=3
                )
                
                # Lyrics Mode Radio Buttons
                lyrics_mode = gr.Radio(
                    choices=["Instrumental", "User Lyrics", "Auto Lyrics"],
                    value="Instrumental",
                    label="Vocal Mode",
                    info="Instrumental: no vocals | User Lyrics: provide your own | Auto Lyrics: AI-generated"
                )
                
                with gr.Row():
                    auto_gen_btn = gr.Button("🎤 Generate Lyrics", size="sm", visible=True)
                
                lyrics_input = gr.Textbox(
                    label="Lyrics",
                    placeholder="Enter lyrics here or click 'Generate Lyrics' button above...",
                    lines=6,
                    visible=True
                )
                
                with gr.Row():
                    duration_input = gr.Slider(
                        minimum=10,
                        maximum=120,
                        value=30,
                        step=1,
                        label="Duration (seconds)"
                    )
                    position_input = gr.Radio(
                        choices=["intro", "previous", "next", "outro"],
                        value="next",
                        label="Timeline Position"
                    )
                
                generate_btn = gr.Button(
                    "✨ Generate Music Clip", 
                    variant="primary", 
                    size="lg"
                )
                
                gen_status = gr.Textbox(label="Status", lines=3, interactive=False)
                audio_output = gr.Audio(label="Generated Clip Preview", type="filepath")
            
            # Right Column - Timeline
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Timeline")
                
                timeline_display = gr.Textbox(
                    label="Clips",
                    value=get_timeline_display(),
                    lines=15,
                    interactive=False
                )
                
                with gr.Row():
                    clip_number_input = gr.Number(
                        label="Clip #",
                        precision=0,
                        minimum=1,
                        scale=1
                    )
                    remove_btn = gr.Button("🗑️ Remove", size="sm", scale=1)
                
                clear_btn = gr.Button("🗑️ Clear All", variant="stop")
                
                timeline_status = gr.Textbox(label="Timeline Status", lines=1, interactive=False)
        
        # Export Section (below main area)
        gr.Markdown("---")
        gr.Markdown("### 💾 Export Timeline")
        
        with gr.Row():
            export_filename = gr.Textbox(
                label="Filename (without extension)",
                value="my_song",
                placeholder="my_song",
                scale=2
            )
            export_format = gr.Dropdown(
                choices=["wav", "mp3", "flac"],
                value="wav",
                label="Format",
                scale=1
            )
            export_btn = gr.Button("💾 Export", variant="primary", size="lg", scale=1)
        
        export_status = gr.Textbox(label="Export Status", lines=1, interactive=False)
        export_audio = gr.Audio(label="Exported Audio", type="filepath")
        
        # Event handlers
        
        # Auto-generate lyrics event
        auto_gen_btn.click(
            fn=generate_lyrics,
            inputs=[prompt_input, duration_input],
            outputs=lyrics_input
        )
        
        # Generate music event with progress tracking
        generate_btn.click(
            fn=generate_music,
            inputs=[prompt_input, lyrics_input, lyrics_mode, duration_input, position_input],
            outputs=[gen_status, timeline_display, audio_output]
        )
        
        # Remove clip event
        remove_btn.click(
            fn=remove_clip,
            inputs=clip_number_input,
            outputs=[timeline_status, timeline_display]
        )
        
        # Clear timeline event
        clear_btn.click(
            fn=clear_timeline,
            inputs=None,
            outputs=[timeline_status, timeline_display]
        )
        
        # Export event
        export_btn.click(
            fn=export_timeline,
            inputs=[export_filename, export_format],
            outputs=[export_status, export_audio]
        )
        
        # Help Section
        with gr.Accordion("ℹ️ Help & Instructions", open=False):
            gr.Markdown(
                """
                ## How to Use
                
                ### 1. Generate Music
                1. **Enter a music prompt** describing what you want
                2. **Choose Vocal Mode:**
                   - **Instrumental**: Pure music, no vocals
                   - **User Lyrics**: Provide your own lyrics in the textbox
                   - **Auto Lyrics**: Click "Generate Lyrics" for AI-generated lyrics
                3. Set **duration** (10-120 seconds)
                4. Select **timeline position** where clip will be added
                5. Click **"Generate Music Clip"**
                
                ### 2. Manage Timeline
                - View all clips in the Timeline panel (right side)
                - **Remove** specific clips by number
                - **Clear All** to start fresh
                
                ### 3. Export
                - Enter a filename
                - Choose format (WAV, MP3, FLAC)
                - Click "Export" to download your complete song
                
                ## Features
                - **Intelligent Prompt Analysis**: Auto-detects genre, BPM, mood
                - **Flexible Vocals**: Choose instrumental, manual, or AI lyrics
                - **GPU Support**: AMD GPU via DirectML (or CPU fallback)
                - **Local Processing**: All models run on your hardware
                
                ## Models
                - **DiffRhythm**: Music generation with vocals
                - **LyricsMindAI**: Context-aware lyrics generation
                
                ## Tips
                - Be specific in prompts: "energetic rock song at 140 BPM with electric guitar"
                - For vocals, try Auto Lyrics mode for genre-appropriate lyrics
                - Build songs clip-by-clip using timeline positions
                
                ---
                *Estimated generation time: ~30-60 seconds per clip on CPU, ~10-20 seconds on GPU*
                """
            )
    
    return app

if __name__ == "__main__":
    logger.info("Starting Music Generation Studio...")
    
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create and launch Gradio app
    app = create_gradio_interface()
    
    # Enable queue for progress tracking
    app.queue(
        default_concurrency_limit=1,
        max_size=10
    )
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
