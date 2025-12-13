"""
Wrapper script to start the backend with required environment variables.
This is used by the PowerShell launcher to ensure environment variables are set.
"""
import os
import sys
import subprocess
from pathlib import Path

# Get project root (parent of backend directory)
project_root = Path(__file__).parent.parent

# Set required environment variables
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(project_root / 'external' / 'espeak-ng' / 'libespeak-ng.dll')
os.environ['PHONEMIZER_ESPEAK_PATH'] = str(project_root / 'external' / 'espeak-ng')

# Run the backend run.py script
backend_script = project_root / 'backend' / 'run.py'

# Execute run.py in the same interpreter
exec(open(backend_script).read())
