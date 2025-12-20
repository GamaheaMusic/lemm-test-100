"""
LEMM Version Information
"""

__version__ = "1.0.2"
__version_info__ = (1, 0, 2)

# Version history
VERSION_HISTORY = {
    "1.0.2": {
        "date": "2025-12-20",
        "features": [
            "Million Song Dataset integration for music theory intelligence",
            "Lakh MIDI dataset integration for symbolic music understanding",
            "Genre-aware parameter suggestions and validation",
            "Music theory constraint system",
            "Structural intelligence for section-based generation",
            "Enhanced instrumentation analysis",
            "Database system for symbolic music data"
        ],
        "improvements": [
            "Better genre accuracy through learned patterns",
            "Improved harmonic coherence",
            "Enhanced structural consistency",
            "Smarter parameter defaults based on genre"
        ]
    },
    "1.0.1": {
        "date": "2025-12-19",
        "features": [
            "Fixed pandas DataFrame indexing in dataset preparation",
            "Added embedded audio metadata extraction (ID3 tags)",
            "Fixed LoRA dropdown population",
            "Implemented equal-power crossfade",
            "Enhanced ZeroGPU authentication"
        ]
    },
    "1.0.0": {
        "date": "2025-12-15",
        "features": [
            "Initial release with DiffRhythm2 integration",
            "LoRA training system",
            "Timeline-based composition",
            "Professional mastering tools"
        ]
    }
}

def get_version():
    """Return current version string"""
    return __version__

def get_version_info():
    """Return version info tuple"""
    return __version_info__

def get_version_history():
    """Return version history dictionary"""
    return VERSION_HISTORY
