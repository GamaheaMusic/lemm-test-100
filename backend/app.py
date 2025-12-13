"""
Main Flask application for the Music Generation App
"""
import os
import logging
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from config.settings import Config
from routes.generation import generation_bp
from routes.timeline import timeline_bp
from routes.export import export_bp
from routes.mastering import mastering_bp
from utils.logger import setup_logger

# Load environment variables
load_dotenv()

def create_app(config_class=Config):
    """Application factory pattern"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Setup logging
    setup_logger(app)
    
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_DIR'], exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(generation_bp, url_prefix='/api/generation')
    app.register_blueprint(timeline_bp, url_prefix='/api/timeline')
    app.register_blueprint(export_bp, url_prefix='/api/export')
    app.register_blueprint(mastering_bp, url_prefix='/api/mastering')
    
    # Serve static files from outputs directory with proper MIME types
    @app.route('/outputs/<path:filename>')
    def serve_output(filename):
        response = send_from_directory(app.config['OUTPUT_FOLDER'], filename)
        # Ensure WAV files have correct MIME type
        if filename.lower().endswith('.wav'):
            response.headers['Content-Type'] = 'audio/wav'
        elif filename.lower().endswith('.mp3'):
            response.headers['Content-Type'] = 'audio/mpeg'
        return response
    
    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0'
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'Internal server error: {str(error)}')
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    app.logger.info(f'Starting server on {host}:{port}')
    app.run(host=host, port=port, debug=os.getenv('FLASK_DEBUG', 'False') == 'True')
