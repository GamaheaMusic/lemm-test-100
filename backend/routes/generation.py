"""
Routes for music generation
"""
import os
import logging
from flask import Blueprint, request, jsonify, current_app
from services.diffrhythm_service import DiffRhythmService
from services.lyricmind_service import LyricMindService
from services.style_consistency_service import StyleConsistencyService
from services.timeline_service import TimelineService
from models.schemas import GenerationRequest, LyricsRequest
from utils.validators import validate_generation_params
from utils.prompt_analyzer import PromptAnalyzer

logger = logging.getLogger(__name__)

generation_bp = Blueprint('generation', __name__)

# Initialize services (lazy loading)
diffrhythm_service = None
lyricmind_service = None
style_service = None
timeline_service = None

def get_diffrhythm_service():
    """Get or create DiffRhythm service instance"""
    global diffrhythm_service
    if diffrhythm_service is None:
        diffrhythm_service = DiffRhythmService(
            model_path=current_app.config['DIFFRHYTHM_MODEL_PATH']
        )
    return diffrhythm_service

def get_lyricmind_service():
    """Get or create LyricMind service instance"""
    global lyricmind_service
    if lyricmind_service is None:
        lyricmind_service = LyricMindService(
            model_path=current_app.config['LYRICMIND_MODEL_PATH']
        )
    return lyricmind_service

def get_style_service():
    """Get or create Style Consistency service instance"""
    global style_service
    if style_service is None:
        style_service = StyleConsistencyService()
    return style_service

def get_timeline_service():
    """Get or create Timeline service instance"""
    global timeline_service
    if timeline_service is None:
        timeline_service = TimelineService()
    return timeline_service

@generation_bp.route('/generate-lyrics', methods=['POST'])
def generate_lyrics():
    """Generate lyrics from prompt using LyricMind AI with prompt analysis"""
    try:
        data = LyricsRequest(**request.json)
        
        # Analyze prompt for better context
        logger.info(f"Analyzing prompt for lyrics generation: {data.prompt}")
        prompt_analysis = PromptAnalyzer.analyze(data.prompt)
        logger.info(f"Prompt analysis: {prompt_analysis}")
        
        # Get lyrics service
        lyrics_service = get_lyricmind_service()
        
        # Generate lyrics with analysis context
        style = data.style or (prompt_analysis.get('genres', [''])[0] if prompt_analysis.get('genres') else None)
        logger.info(f"Generating lyrics with style: {style}")
        
        lyrics = lyrics_service.generate(
            prompt=data.prompt,
            style=style,
            duration=data.duration,
            prompt_analysis=prompt_analysis
        )
        
        return jsonify({
            'lyrics': lyrics,
            'analysis': prompt_analysis
        })
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating lyrics: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to generate lyrics: {str(e)}'}), 500

@generation_bp.route('/generate-music', methods=['POST'])
def generate_music():
    """
    Generate music clip from prompt with optional vocals
    
    Request body:
    {
        "prompt": "upbeat pop song with drums",
        "lyrics": "optional lyrics text",
        "duration": 30
    }
    """
    try:
        data = request.get_json()
        logger.info(f"Received music generation request: {data.get('prompt', 'No prompt')}")
        
        # Validate request
        validation_error = validate_generation_params(data)
        if validation_error:
            return jsonify({'error': validation_error}), 400
        
        # Parse request
        gen_request = GenerationRequest(**data)
        
        # Analyze prompt for musical attributes
        prompt_analysis = PromptAnalyzer.analyze(gen_request.prompt)
        logger.info(f"Prompt analysis: {prompt_analysis['analysis_text']}")
        
        # Get timeline clips for style consistency
        timeline_svc = get_timeline_service()
        existing_clips = timeline_svc.get_all_clips()
        
        # Prepare style guidance if clips exist
        reference_audio = None
        style_profile = {}
        enhanced_prompt = gen_request.prompt
        
        if existing_clips:
            logger.info(f"Found {len(existing_clips)} existing clips - applying style consistency")
            style_svc = get_style_service()
            reference_audio, style_profile = style_svc.get_style_guidance_for_generation(existing_clips)
            
            # Enhance prompt with style characteristics
            enhanced_prompt = style_svc.enhance_prompt_with_style(gen_request.prompt, style_profile)
            logger.info(f"Enhanced prompt for style consistency: {enhanced_prompt}")
        else:
            logger.info("No existing clips - generating without style guidance")
        
        # Generate music with DiffRhythm2 (includes vocals if lyrics provided)
        service = get_diffrhythm_service()
        lyrics_to_use = gen_request.lyrics if gen_request.lyrics else None
        
        final_path = service.generate(
            prompt=enhanced_prompt,
            duration=gen_request.duration,
            lyrics=lyrics_to_use,
            reference_audio=reference_audio
        )
        
        logger.info(f"Music generation successful: {final_path}")
        
        # Convert filesystem path to URL path (forward slashes, relative to outputs)
        relative_path = os.path.relpath(final_path, 'outputs')
        url_path = f"/outputs/{relative_path.replace(os.sep, '/')}"
        
        return jsonify({
            'success': True,
            'clip_id': os.path.basename(final_path).split('.')[0],
            'file_path': url_path,
            'duration': gen_request.duration,
            'analysis': prompt_analysis,
            'style_consistent': len(existing_clips) > 0,
            'num_reference_clips': len(existing_clips)
        })
        
    except Exception as e:
        logger.error(f"Error generating music: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to generate music',
            'details': str(e)
        }), 500

@generation_bp.route('/status', methods=['GET'])
def get_status():
    """Check if generation services are available"""
    try:
        status = {
            'diffrhythm': diffrhythm_service is not None
        }
        
        return jsonify({
            'services': status,
            'ready': status['diffrhythm']
        })
        
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return jsonify({'error': str(e)}), 500
