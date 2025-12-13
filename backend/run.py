"""
Startup script for the Music Generation App
"""
import sys
import os
import signal

# Add backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print('\n\n[INFO] Shutting down server...')
    sys.exit(0)

if __name__ == '__main__':
    try:
        app = create_app()
        port = int(os.getenv('PORT', 7860))  # Default to 7860 to match frontend expectations
        host = os.getenv('HOST', '0.0.0.0')
        
        print(f"""
    ================================================================
       Music Generation App Server Starting...
    ================================================================
    
    Server running at: http://{host}:{port}
    API endpoints: http://{host}:{port}/api
    Health check: http://{host}:{port}/api/health
    
    Press Ctrl+C to stop the server
    ================================================================
    """)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Use waitress for production-ready server
        from waitress import serve
        print('[INFO] Server is ready!')
        serve(app, host=host, port=port, threads=4)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to start server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
