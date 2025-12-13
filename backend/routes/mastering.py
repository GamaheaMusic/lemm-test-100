"""
Routes for audio mastering and EQ
"""
import os
import logging
from flask import Blueprint, request, jsonify, current_app, send_file
from services.mastering_service import MasteringService
from pathlib import Path

logger = logging.getLogger(__name__)

mastering_bp = Blueprint('mastering', __name__)

# Initialize service
mastering_service = None

def get_mastering_service():
    """Get or create mastering service instance"""
    global mastering_service
    if mastering_service is None:
        mastering_service = MasteringService()
    return mastering_service

@mastering_bp.route('/presets', methods=['GET'])
def get_presets():
    """Get list of all available mastering presets"""
    try:
        service = get_mastering_service()
        presets = service.get_preset_list()
        return jsonify({'presets': presets})
    except Exception as e:
        logger.error(f"Error getting presets: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get presets'}), 500

@mastering_bp.route('/apply-preset', methods=['POST'])
def apply_preset():
    """Apply mastering preset to audio clip"""
    try:
        data = request.json
        clip_id = data.get('clip_id')
        preset_name = data.get('preset')
        audio_path = data.get('audio_path')
        
        if not all([clip_id, preset_name, audio_path]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Verify audio file exists
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        # Generate output path
        output_dir = Path(current_app.config['OUTPUT_FOLDER']) / 'mastered'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = Path(audio_path).stem
        output_path = output_dir / f"{filename}_mastered_{preset_name}.wav"
        
        # Apply preset
        service = get_mastering_service()
        processed_path = service.apply_preset(audio_path, preset_name, str(output_path))
        
        # Return URL to processed file
        relative_path = os.path.relpath(processed_path, current_app.config['OUTPUT_FOLDER'])
        file_url = f"/outputs/{relative_path.replace(os.sep, '/')}"
        
        return jsonify({
            'success': True,
            'processed_path': file_url,
            'clip_id': clip_id,
            'preset': preset_name
        })
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error applying preset: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to apply preset: {str(e)}'}), 500

@mastering_bp.route('/apply-custom-eq', methods=['POST'])
def apply_custom_eq():
    """Apply custom EQ settings to audio clip"""
    try:
        data = request.json
        clip_id = data.get('clip_id')
        audio_path = data.get('audio_path')
        eq_bands = data.get('eq_bands', [])
        compression = data.get('compression')
        limiting = data.get('limiting')
        
        if not all([clip_id, audio_path]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Verify audio file exists
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        # Generate output path
        output_dir = Path(current_app.config['OUTPUT_FOLDER']) / 'mastered'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = Path(audio_path).stem
        output_path = output_dir / f"{filename}_custom_eq.wav"
        
        # Apply custom EQ
        service = get_mastering_service()
        processed_path = service.apply_custom_eq(
            audio_path,
            str(output_path),
            eq_bands,
            compression,
            limiting
        )
        
        # Return URL to processed file
        relative_path = os.path.relpath(processed_path, current_app.config['OUTPUT_FOLDER'])
        file_url = f"/outputs/{relative_path.replace(os.sep, '/')}"
        
        return jsonify({
            'success': True,
            'processed_path': file_url,
            'clip_id': clip_id
        })
        
    except Exception as e:
        logger.error(f"Error applying custom EQ: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to apply custom EQ: {str(e)}'}), 500

@mastering_bp.route('/preview', methods=['POST'])
def preview_mastering():
    """Preview mastering effect (non-destructive)"""
    try:
        data = request.json
        clip_id = data.get('clip_id')
        audio_path = data.get('audio_path')
        preset_name = data.get('preset')
        eq_bands = data.get('eq_bands')
        
        if not all([clip_id, audio_path]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Verify audio file exists
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        # Generate temp output path for preview
        output_dir = Path(current_app.config['OUTPUT_FOLDER']) / 'preview'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = Path(audio_path).stem
        output_path = output_dir / f"{filename}_preview.wav"
        
        service = get_mastering_service()
        
        if preset_name:
            # Apply preset for preview
            processed_path = service.apply_preset(audio_path, preset_name, str(output_path))
        elif eq_bands:
            # Apply custom EQ for preview
            compression = data.get('compression')
            limiting = data.get('limiting')
            processed_path = service.apply_custom_eq(
                audio_path,
                str(output_path),
                eq_bands,
                compression,
                limiting
            )
        else:
            return jsonify({'error': 'No preset or EQ settings provided'}), 400
        
        # Return URL to preview file  
        # Use absolute path from project root for frontend to access
        relative_path = os.path.relpath(processed_path, 'outputs')
        file_url = f"http://localhost:7860/outputs/{relative_path.replace(os.sep, '/')}"
        
        return jsonify({
            'success': True,
            'preview_path': file_url,
            'clip_id': clip_id
        })
        
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to generate preview: {str(e)}'}), 500
