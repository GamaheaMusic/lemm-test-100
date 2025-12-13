"""
Training API Routes
Endpoints for LoRA training, dataset management, and audio analysis.
"""

from flask import Blueprint, request, jsonify
from backend.services.lora_training_service import LoRATrainingService
from backend.services.audio_analysis_service import AudioAnalysisService
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

training_bp = Blueprint('training', __name__, url_prefix='/api/training')

# Initialize services
lora_service = LoRATrainingService()
audio_analysis_service = AudioAnalysisService()


@training_bp.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    """Analyze uploaded audio and generate metadata"""
    try:
        data = request.json
        audio_path = data.get('audio_path')
        
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': 'Invalid audio path'}), 400
        
        # Analyze audio
        metadata = audio_analysis_service.analyze_audio(audio_path)
        
        return jsonify({
            'success': True,
            'metadata': metadata
        })
        
    except Exception as e:
        logger.error(f"Audio analysis failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/split-audio', methods=['POST'])
def split_audio():
    """Split audio into training clips"""
    try:
        data = request.json
        audio_path = data.get('audio_path')
        output_dir = data.get('output_dir', 'training_data/user_uploads/clips')
        segments = data.get('segments')  # Optional
        metadata = data.get('metadata')  # Optional
        
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': 'Invalid audio path'}), 400
        
        # Split audio
        clip_paths = audio_analysis_service.split_audio_to_clips(
            audio_path,
            output_dir,
            segments,
            metadata
        )
        
        return jsonify({
            'success': True,
            'num_clips': len(clip_paths),
            'clip_paths': clip_paths
        })
        
    except Exception as e:
        logger.error(f"Audio splitting failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/separate-stems', methods=['POST'])
def separate_stems():
    """Separate audio into vocal/instrumental stems"""
    try:
        data = request.json
        audio_path = data.get('audio_path')
        output_dir = data.get('output_dir', 'training_data/user_uploads/stems')
        
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': 'Invalid audio path'}), 400
        
        # Separate stems
        stem_paths = audio_analysis_service.separate_vocal_stems(
            audio_path,
            output_dir
        )
        
        return jsonify({
            'success': True,
            'stems': stem_paths
        })
        
    except Exception as e:
        logger.error(f"Stem separation failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/prepare-dataset', methods=['POST'])
def prepare_dataset():
    """Prepare training dataset from audio files"""
    try:
        data = request.json
        dataset_name = data.get('dataset_name')
        audio_files = data.get('audio_files', [])
        metadata_list = data.get('metadata_list', [])
        split_ratio = data.get('split_ratio', 0.9)
        
        if not dataset_name:
            return jsonify({'error': 'Dataset name required'}), 400
        
        if not audio_files:
            return jsonify({'error': 'No audio files provided'}), 400
        
        # Prepare dataset
        dataset_info = lora_service.prepare_dataset(
            dataset_name,
            audio_files,
            metadata_list,
            split_ratio
        )
        
        return jsonify({
            'success': True,
            'dataset_info': dataset_info
        })
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/datasets', methods=['GET'])
def list_datasets():
    """List available datasets"""
    try:
        datasets = lora_service.list_datasets()
        
        # Get detailed info for each dataset
        dataset_details = []
        for dataset_name in datasets:
            info = lora_service.load_dataset(dataset_name)
            if info:
                dataset_details.append(info)
        
        return jsonify({
            'success': True,
            'datasets': dataset_details
        })
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/train-lora', methods=['POST'])
def train_lora():
    """Start LoRA training"""
    try:
        data = request.json
        dataset_name = data.get('dataset_name')
        lora_name = data.get('lora_name')
        training_type = data.get('training_type', 'vocal')
        config = data.get('config', {})
        
        if not dataset_name:
            return jsonify({'error': 'Dataset name required'}), 400
        
        if not lora_name:
            return jsonify({'error': 'LoRA name required'}), 400
        
        # Start training (in background thread in production)
        # For now, this is synchronous
        results = lora_service.train_lora(
            dataset_name,
            lora_name,
            training_type,
            config
        )
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/training-status', methods=['GET'])
def training_status():
    """Get current training status"""
    try:
        status = lora_service.get_training_status()
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/stop-training', methods=['POST'])
def stop_training():
    """Stop current training"""
    try:
        lora_service.stop_training()
        
        return jsonify({
            'success': True,
            'message': 'Training stopped'
        })
        
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/lora-adapters', methods=['GET'])
def list_lora_adapters():
    """List available LoRA adapters"""
    try:
        adapters = lora_service.list_lora_adapters()
        
        return jsonify({
            'success': True,
            'adapters': adapters
        })
        
    except Exception as e:
        logger.error(f"Failed to list LoRA adapters: {str(e)}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/lora-adapters/<lora_name>', methods=['DELETE'])
def delete_lora_adapter(lora_name):
    """Delete a LoRA adapter"""
    try:
        success = lora_service.delete_lora_adapter(lora_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'LoRA adapter {lora_name} deleted'
            })
        else:
            return jsonify({'error': 'LoRA adapter not found'}), 404
        
    except Exception as e:
        logger.error(f"Failed to delete LoRA adapter: {str(e)}")
        return jsonify({'error': str(e)}), 500


def register_training_routes(app):
    """Register training routes with Flask app"""
    app.register_blueprint(training_bp)
    logger.info("Training routes registered")
