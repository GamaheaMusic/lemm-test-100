"""
Request validation utilities
"""
from typing import Dict, Optional

def validate_generation_params(data: Dict) -> Optional[str]:
    """
    Validate music generation parameters
    
    Args:
        data: Request data dictionary
        
    Returns:
        Error message if validation fails, None otherwise
    """
    if not data:
        return "Request body is required"
    
    if 'prompt' not in data:
        return "Missing required field: prompt"
    
    if not data['prompt'] or not data['prompt'].strip():
        return "Prompt cannot be empty"
    
    if 'duration' in data:
        duration = data['duration']
        if not isinstance(duration, (int, float)):
            return "Duration must be a number"
        if duration < 10 or duration > 120:
            return "Duration must be between 10 and 120 seconds"
    
    if 'use_vocals' in data:
        if not isinstance(data['use_vocals'], bool):
            return "use_vocals must be a boolean"
        
        if data['use_vocals'] and not data.get('lyrics'):
            return "Lyrics are required when use_vocals is true"
    
    return None

def validate_clip_data(data: Dict) -> Optional[str]:
    """
    Validate timeline clip data
    
    Args:
        data: Clip data dictionary
        
    Returns:
        Error message if validation fails, None otherwise
    """
    required_fields = ['clip_id', 'file_path', 'duration', 'position']
    
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"
    
    if not isinstance(data['duration'], (int, float)) or data['duration'] <= 0:
        return "Duration must be a positive number"
    
    valid_positions = ['intro', 'previous', 'next', 'outro']
    if data['position'] not in valid_positions:
        return f"Invalid position. Must be one of: {', '.join(valid_positions)}"
    
    return None
