"""
Routes for exporting/downloading music
"""
import logging
import os
from flask import Blueprint, request, jsonify, send_file, current_app
from services.export_service import ExportService
from models.schemas import ExportFormat

logger = logging.getLogger(__name__)

export_bp = Blueprint('export', __name__)
export_service = ExportService()

@export_bp.route('/merge', methods=['POST'])
def merge_timeline():
    """
    Merge all clips in the timeline into a single file
    
    Request body:
    {
        "format": "wav",  // wav, mp3, flac
        "filename": "my_song"  // optional
    }
    """
    try:
        data = request.get_json() or {}
        logger.info("Merging timeline clips")
        
        # Validate format
        export_format = data.get('format', 'wav')
        try:
            ExportFormat(export_format)
        except ValueError:
            return jsonify({
                'error': f"Invalid format. Must be one of: {', '.join([f.value for f in ExportFormat])}"
            }), 400
        
        filename = data.get('filename', 'merged_output')
        
        # Merge clips
        output_path = export_service.merge_clips(
            filename=filename,
            export_format=export_format
        )
        
        if not output_path:
            return jsonify({
                'error': 'No clips to merge. Add clips to timeline first.'
            }), 400
        
        logger.info(f"Timeline merged successfully: {output_path}")
        
        return jsonify({
            'success': True,
            'file_path': output_path,
            'filename': os.path.basename(output_path)
        })
        
    except Exception as e:
        logger.error(f"Error merging timeline: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to merge timeline',
            'details': str(e)
        }), 500

@export_bp.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download an exported file"""
    try:
        output_folder = current_app.config['OUTPUT_FOLDER']
        file_path = os.path.join(output_folder, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Security check: ensure file is in output folder
        if not os.path.abspath(file_path).startswith(os.path.abspath(output_folder)):
            return jsonify({'error': 'Invalid file path'}), 403
        
        logger.info(f"Downloading file: {filename}")
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@export_bp.route('/export-clip/<clip_id>', methods=['GET'])
def export_single_clip(clip_id):
    """Export a single clip"""
    try:
        export_format = request.args.get('format', 'wav')
        
        try:
            ExportFormat(export_format)
        except ValueError:
            return jsonify({
                'error': f"Invalid format. Must be one of: {', '.join([f.value for f in ExportFormat])}"
            }), 400
        
        logger.info(f"Exporting single clip: {clip_id}")
        
        output_path = export_service.export_clip(
            clip_id=clip_id,
            export_format=export_format
        )
        
        if not output_path:
            return jsonify({'error': 'Clip not found'}), 404
        
        return jsonify({
            'success': True,
            'file_path': output_path,
            'filename': os.path.basename(output_path)
        })
        
    except Exception as e:
        logger.error(f"Error exporting clip: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
