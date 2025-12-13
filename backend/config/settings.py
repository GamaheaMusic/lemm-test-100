"""
Application configuration settings
"""
import os
from pathlib import Path

class Config:
    """Base configuration"""
    
    # Base directory
    BASE_DIR = Path(__file__).parent.parent.parent
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False') == 'True'
    
    # File upload settings
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', str(BASE_DIR / 'uploads'))
    OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', str(BASE_DIR / 'outputs'))
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    
    # Model paths
    MODELS_DIR = BASE_DIR / 'models'
    DIFFRHYTHM_MODEL_PATH = os.getenv('DIFFRHYTHM_MODEL_PATH', str(MODELS_DIR / 'diffrhythm2'))
    FISH_SPEECH_MODEL_PATH = os.getenv('FISH_SPEECH_MODEL_PATH', str(MODELS_DIR / 'fish_speech'))
    LYRICMIND_MODEL_PATH = os.getenv('LYRICMIND_MODEL_PATH', str(MODELS_DIR / 'lyricmind'))
    
    # Generation settings
    DEFAULT_CLIP_DURATION = int(os.getenv('DEFAULT_CLIP_DURATION', 30))
    SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 44100))
    BIT_DEPTH = int(os.getenv('BIT_DEPTH', 16))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', str(BASE_DIR / 'logs' / 'app.log'))

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
