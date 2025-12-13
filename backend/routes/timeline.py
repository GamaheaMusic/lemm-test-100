"""
Routes for timeline management
"""
import logging
from flask import Blueprint, request, jsonify
from services.timeline_service import TimelineService
from models.schemas import ClipPosition

logger = logging.getLogger(__name__)

timeline_bp = Blueprint('timeline', __name__)
timeline_service = TimelineService()

@timeline_bp.route('/clips', methods=['GET'])
def get_clips():
    """Get all clips in the timeline"""
    try:
        clips = timeline_service.get_all_clips()
        return jsonify({
            'success': True,
            'clips': clips,
            'total_duration': timeline_service.get_total_duration()
        })
    except Exception as e:
        logger.error(f"Error fetching clips: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@timeline_bp.route('/clips', methods=['POST'])
def add_clip():
    """
    Add a clip to the timeline
    
    Request body:
    {
        "clip_id": "unique_id",
        "file_path": "/path/to/clip.wav",
        "duration": 30,
        "position": "next"  // intro, previous, next, outro
    }
    """
    try:
        data = request.get_json()
        logger.info(f"Adding clip to timeline: {data.get('clip_id')}")
        
        # Validate required fields
        required_fields = ['clip_id', 'file_path', 'duration', 'position']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate position
        try:
            position = ClipPosition(data['position'])
        except ValueError:
            return jsonify({
                'error': f"Invalid position. Must be one of: {', '.join([p.value for p in ClipPosition])}"
            }), 400
        
        # Add clip to timeline
        result = timeline_service.add_clip(
            clip_id=data['clip_id'],
            file_path=data['file_path'],
            duration=data['duration'],
            position=position
        )
        
        logger.info(f"Clip added successfully at position: {result['timeline_position']}")
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        logger.error(f"Error adding clip: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@timeline_bp.route('/clips/<clip_id>', methods=['DELETE'])
def remove_clip(clip_id):
    """Remove a clip from the timeline"""
    try:
        logger.info(f"Removing clip: {clip_id}")
        timeline_service.remove_clip(clip_id)
        
        return jsonify({
            'success': True,
            'message': f'Clip {clip_id} removed'
        })
        
    except Exception as e:
        logger.error(f"Error removing clip: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@timeline_bp.route('/clips/reorder', methods=['POST'])
def reorder_clips():
    """
    Reorder clips in the timeline
    
    Request body:
    {
        "clip_ids": ["id1", "id2", "id3"]
    }
    """
    try:
        data = request.get_json()
        clip_ids = data.get('clip_ids', [])
        
        if not clip_ids:
            return jsonify({'error': 'clip_ids array is required'}), 400
        
        logger.info(f"Reordering clips: {clip_ids}")
        timeline_service.reorder_clips(clip_ids)
        
        return jsonify({
            'success': True,
            'message': 'Clips reordered successfully'
        })
        
    except Exception as e:
        logger.error(f"Error reordering clips: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@timeline_bp.route('/clear', methods=['POST'])
def clear_timeline():
    """Clear all clips from the timeline"""
    try:
        logger.info("Clearing timeline")
        timeline_service.clear()
        
        return jsonify({
            'success': True,
            'message': 'Timeline cleared'
        })
        
    except Exception as e:
        logger.error(f"Error clearing timeline: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
