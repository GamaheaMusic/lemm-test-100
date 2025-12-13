"""
Configuration for HuggingFace Spaces deployment
Handles espeak-ng and model paths for cloud environment
"""
import os
from pathlib import Path

# Detect if running on HuggingFace Spaces
IS_SPACES = os.getenv("SPACE_ID") is not None

# Configure espeak-ng for HuggingFace Spaces
if IS_SPACES:
    # On Spaces, espeak-ng is installed via packages.txt
    # It's available system-wide
    if os.path.exists("/usr/bin/espeak-ng"):
        os.environ["PHONEMIZER_ESPEAK_PATH"] = "/usr/bin/espeak-ng"
        if os.path.exists("/usr/lib/x86_64-linux-gnu/libespeak-ng.so"):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"
        elif os.path.exists("/usr/lib/libespeak-ng.so"):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/usr/lib/libespeak-ng.so"
else:
    # Local development - use bundled espeak-ng
    espeak_path = Path(__file__).parent.parent / "external" / "espeak-ng"
    if espeak_path.exists():
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_path / "libespeak-ng.dll")
        os.environ["PHONEMIZER_ESPEAK_PATH"] = str(espeak_path)

print(f"🔧 Environment: {'HuggingFace Spaces' if IS_SPACES else 'Local'}")
print(f"🔊 PHONEMIZER_ESPEAK_PATH: {os.getenv('PHONEMIZER_ESPEAK_PATH', 'Not set')}")
print(f"📚 PHONEMIZER_ESPEAK_LIBRARY: {os.getenv('PHONEMIZER_ESPEAK_LIBRARY', 'Not set')}")
