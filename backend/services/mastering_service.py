"""
Audio mastering service with industry-standard presets using Pedalboard
"""
import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import soundfile as sf
from pedalboard import (
    Pedalboard, 
    Compressor, 
    Limiter, 
    Gain,
    HighpassFilter,
    LowpassFilter,
    PeakFilter,
    LowShelfFilter,
    HighShelfFilter,
    Reverb,
    Chorus,
    Delay
)

logger = logging.getLogger(__name__)

class MasteringPreset:
    """Mastering preset configuration"""
    
    def __init__(self, name: str, description: str, chain: List):
        self.name = name
        self.description = description
        self.chain = chain

class MasteringService:
    """Audio mastering and EQ service"""
    
    # Industry-standard mastering presets
    PRESETS = {
        # Clean/Transparent Presets
        "clean_master": MasteringPreset(
            "Clean Master",
            "Transparent mastering with gentle compression",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                PeakFilter(cutoff_frequency_hz=100, gain_db=-1, q=0.7),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=0.5, q=1.0),
                PeakFilter(cutoff_frequency_hz=10000, gain_db=1.0, q=0.7),
                Compressor(threshold_db=-12, ratio=2.0, attack_ms=5, release_ms=100),
                Limiter(threshold_db=-1.0, release_ms=100)
            ]
        ),
        
        "subtle_warmth": MasteringPreset(
            "Subtle Warmth",
            "Gentle low-end enhancement with smooth highs",
            [
                HighpassFilter(cutoff_frequency_hz=25),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=200, gain_db=0.8, q=0.5),
                PeakFilter(cutoff_frequency_hz=8000, gain_db=-0.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=12000, gain_db=1.0, q=0.7),
                Compressor(threshold_db=-15, ratio=2.5, attack_ms=10, release_ms=150),
                Limiter(threshold_db=-0.5, release_ms=100)
            ]
        ),
        
        # Pop/Commercial Presets
        "modern_pop": MasteringPreset(
            "Modern Pop",
            "Radio-ready pop sound with punchy compression",
            [
                HighpassFilter(cutoff_frequency_hz=35),
                PeakFilter(cutoff_frequency_hz=80, gain_db=-1.5, q=0.8),
                LowShelfFilter(cutoff_frequency_hz=120, gain_db=2.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=2500, gain_db=1.5, q=1.2),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.5, q=0.7),
                Compressor(threshold_db=-10, ratio=4.0, attack_ms=3, release_ms=80),
                Limiter(threshold_db=-0.3, release_ms=50)
            ]
        ),
        
        "radio_ready": MasteringPreset(
            "Radio Ready",
            "Maximum loudness for commercial radio",
            [
                HighpassFilter(cutoff_frequency_hz=40),
                PeakFilter(cutoff_frequency_hz=60, gain_db=-2.0, q=1.0),
                LowShelfFilter(cutoff_frequency_hz=150, gain_db=1.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=2.0, q=1.5),
                PeakFilter(cutoff_frequency_hz=8000, gain_db=1.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=12000, gain_db=3.0, q=0.7),
                Compressor(threshold_db=-8, ratio=6.0, attack_ms=2, release_ms=60),
                Limiter(threshold_db=-0.1, release_ms=30)
            ]
        ),
        
        "punchy_commercial": MasteringPreset(
            "Punchy Commercial",
            "Aggressive punch for mainstream appeal",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                PeakFilter(cutoff_frequency_hz=100, gain_db=-2.0, q=1.2),
                LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=1000, gain_db=-1.0, q=0.8),
                PeakFilter(cutoff_frequency_hz=4000, gain_db=2.5, q=1.5),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0, q=0.8),
                Compressor(threshold_db=-9, ratio=5.0, attack_ms=1, release_ms=50),
                Limiter(threshold_db=-0.2, release_ms=40)
            ]
        ),
        
        # Rock/Alternative Presets
        "rock_master": MasteringPreset(
            "Rock Master",
            "Powerful rock sound with emphasis on mids",
            [
                HighpassFilter(cutoff_frequency_hz=35),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=400, gain_db=1.5, q=1.0),
                PeakFilter(cutoff_frequency_hz=2000, gain_db=2.0, q=1.2),
                PeakFilter(cutoff_frequency_hz=5000, gain_db=1.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=8000, gain_db=1.0, q=0.8),
                Compressor(threshold_db=-12, ratio=3.5, attack_ms=5, release_ms=120),
                Limiter(threshold_db=-0.5, release_ms=80)
            ]
        ),
        
        "metal_aggressive": MasteringPreset(
            "Metal Aggressive",
            "Heavy, aggressive metal mastering",
            [
                HighpassFilter(cutoff_frequency_hz=40),
                PeakFilter(cutoff_frequency_hz=80, gain_db=-1.5, q=1.0),
                LowShelfFilter(cutoff_frequency_hz=150, gain_db=2.0, q=0.8),
                PeakFilter(cutoff_frequency_hz=800, gain_db=-1.5, q=1.2),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=3.0, q=1.5),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.5, q=0.7),
                Compressor(threshold_db=-8, ratio=6.0, attack_ms=1, release_ms=50),
                Limiter(threshold_db=-0.1, release_ms=30)
            ]
        ),
        
        "indie_rock": MasteringPreset(
            "Indie Rock",
            "Lo-fi character with mid presence",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                LowShelfFilter(cutoff_frequency_hz=120, gain_db=0.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=500, gain_db=1.5, q=1.0),
                PeakFilter(cutoff_frequency_hz=2500, gain_db=2.0, q=1.2),
                PeakFilter(cutoff_frequency_hz=7000, gain_db=-0.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=0.5, q=0.8),
                Compressor(threshold_db=-14, ratio=3.0, attack_ms=8, release_ms=150),
                Limiter(threshold_db=-0.8, release_ms=100)
            ]
        ),
        
        # Electronic/EDM Presets
        "edm_club": MasteringPreset(
            "EDM Club",
            "Powerful club sound with deep bass",
            [
                HighpassFilter(cutoff_frequency_hz=25),
                LowShelfFilter(cutoff_frequency_hz=80, gain_db=3.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=150, gain_db=2.0, q=0.8),
                PeakFilter(cutoff_frequency_hz=1000, gain_db=-1.5, q=1.0),
                PeakFilter(cutoff_frequency_hz=5000, gain_db=2.0, q=1.2),
                HighShelfFilter(cutoff_frequency_hz=12000, gain_db=3.0, q=0.7),
                Compressor(threshold_db=-6, ratio=8.0, attack_ms=0.5, release_ms=40),
                Limiter(threshold_db=0.0, release_ms=20)
            ]
        ),
        
        "house_groovy": MasteringPreset(
            "House Groovy",
            "Smooth house music with rolling bass",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=2.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=250, gain_db=1.0, q=0.8),
                PeakFilter(cutoff_frequency_hz=2000, gain_db=0.5, q=1.0),
                PeakFilter(cutoff_frequency_hz=8000, gain_db=1.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0, q=0.7),
                Compressor(threshold_db=-10, ratio=4.0, attack_ms=2, release_ms=60),
                Limiter(threshold_db=-0.2, release_ms=40)
            ]
        ),
        
        "techno_dark": MasteringPreset(
            "Techno Dark",
            "Dark, pounding techno master",
            [
                HighpassFilter(cutoff_frequency_hz=35),
                PeakFilter(cutoff_frequency_hz=60, gain_db=2.0, q=1.0),
                LowShelfFilter(cutoff_frequency_hz=120, gain_db=1.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=800, gain_db=-2.0, q=1.5),
                PeakFilter(cutoff_frequency_hz=4000, gain_db=1.0, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=-0.5, q=0.8),
                Compressor(threshold_db=-8, ratio=6.0, attack_ms=1, release_ms=50),
                Limiter(threshold_db=-0.1, release_ms=30)
            ]
        ),
        
        "dubstep_heavy": MasteringPreset(
            "Dubstep Heavy",
            "Sub-bass focused with crispy highs",
            [
                HighpassFilter(cutoff_frequency_hz=20),
                PeakFilter(cutoff_frequency_hz=50, gain_db=3.5, q=1.2),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=2.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=500, gain_db=-2.0, q=1.5),
                PeakFilter(cutoff_frequency_hz=6000, gain_db=2.5, q=1.2),
                HighShelfFilter(cutoff_frequency_hz=12000, gain_db=3.5, q=0.7),
                Compressor(threshold_db=-6, ratio=10.0, attack_ms=0.3, release_ms=30),
                Limiter(threshold_db=0.0, release_ms=20)
            ]
        ),
        
        # Hip-Hop/R&B Presets
        "hiphop_modern": MasteringPreset(
            "Hip-Hop Modern",
            "Contemporary hip-hop with deep bass",
            [
                HighpassFilter(cutoff_frequency_hz=25),
                LowShelfFilter(cutoff_frequency_hz=80, gain_db=2.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=150, gain_db=1.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=1000, gain_db=-1.0, q=1.0),
                PeakFilter(cutoff_frequency_hz=3500, gain_db=2.0, q=1.2),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.5, q=0.7),
                Compressor(threshold_db=-10, ratio=4.0, attack_ms=5, release_ms=80),
                Limiter(threshold_db=-0.3, release_ms=60)
            ]
        ),
        
        "trap_808": MasteringPreset(
            "Trap 808",
            "808-focused trap mastering",
            [
                HighpassFilter(cutoff_frequency_hz=20),
                PeakFilter(cutoff_frequency_hz=50, gain_db=3.0, q=1.0),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=2.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=800, gain_db=-1.5, q=1.2),
                PeakFilter(cutoff_frequency_hz=5000, gain_db=2.5, q=1.2),
                HighShelfFilter(cutoff_frequency_hz=12000, gain_db=2.0, q=0.7),
                Compressor(threshold_db=-8, ratio=5.0, attack_ms=3, release_ms=60),
                Limiter(threshold_db=-0.2, release_ms=40)
            ]
        ),
        
        "rnb_smooth": MasteringPreset(
            "R&B Smooth",
            "Silky smooth R&B sound",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=300, gain_db=1.0, q=0.8),
                PeakFilter(cutoff_frequency_hz=2000, gain_db=0.5, q=1.0),
                PeakFilter(cutoff_frequency_hz=6000, gain_db=1.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0, q=0.7),
                Compressor(threshold_db=-12, ratio=3.0, attack_ms=8, release_ms=120),
                Limiter(threshold_db=-0.5, release_ms=80)
            ]
        ),
        
        # Acoustic/Organic Presets
        "acoustic_natural": MasteringPreset(
            "Acoustic Natural",
            "Natural, transparent acoustic sound",
            [
                HighpassFilter(cutoff_frequency_hz=25),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=0.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=500, gain_db=0.8, q=0.8),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=1.0, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=8000, gain_db=1.5, q=0.7),
                Compressor(threshold_db=-16, ratio=2.0, attack_ms=15, release_ms=200),
                Limiter(threshold_db=-1.0, release_ms=120)
            ]
        ),
        
        "folk_warm": MasteringPreset(
            "Folk Warm",
            "Warm, intimate folk sound",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                LowShelfFilter(cutoff_frequency_hz=150, gain_db=1.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=400, gain_db=1.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=2500, gain_db=1.0, q=1.0),
                PeakFilter(cutoff_frequency_hz=7000, gain_db=-0.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.0, q=0.8),
                Compressor(threshold_db=-18, ratio=2.5, attack_ms=20, release_ms=250),
                Limiter(threshold_db=-1.5, release_ms=150)
            ]
        ),
        
        "jazz_vintage": MasteringPreset(
            "Jazz Vintage",
            "Classic jazz warmth and space",
            [
                HighpassFilter(cutoff_frequency_hz=35),
                LowShelfFilter(cutoff_frequency_hz=120, gain_db=1.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=500, gain_db=1.0, q=0.8),
                PeakFilter(cutoff_frequency_hz=2000, gain_db=0.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=8000, gain_db=-1.0, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=12000, gain_db=0.5, q=0.8),
                Compressor(threshold_db=-20, ratio=2.0, attack_ms=25, release_ms=300),
                Limiter(threshold_db=-2.0, release_ms=180)
            ]
        ),
        
        # Classical/Orchestral Presets
        "orchestral_wide": MasteringPreset(
            "Orchestral Wide",
            "Wide, natural orchestral sound",
            [
                HighpassFilter(cutoff_frequency_hz=20),
                LowShelfFilter(cutoff_frequency_hz=80, gain_db=0.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=300, gain_db=0.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=4000, gain_db=0.8, q=0.8),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.0, q=0.7),
                Compressor(threshold_db=-24, ratio=1.5, attack_ms=30, release_ms=400),
                Limiter(threshold_db=-3.0, release_ms=250)
            ]
        ),
        
        "classical_concert": MasteringPreset(
            "Classical Concert",
            "Concert hall ambience and dynamics",
            [
                HighpassFilter(cutoff_frequency_hz=25),
                PeakFilter(cutoff_frequency_hz=200, gain_db=0.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=1000, gain_db=0.3, q=0.8),
                PeakFilter(cutoff_frequency_hz=6000, gain_db=0.8, q=0.8),
                HighShelfFilter(cutoff_frequency_hz=12000, gain_db=0.5, q=0.7),
                Compressor(threshold_db=-30, ratio=1.2, attack_ms=50, release_ms=500),
                Limiter(threshold_db=-4.0, release_ms=300)
            ]
        ),
        
        # Ambient/Atmospheric Presets
        "ambient_spacious": MasteringPreset(
            "Ambient Spacious",
            "Wide, spacious ambient master",
            [
                HighpassFilter(cutoff_frequency_hz=25),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=0.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=500, gain_db=-0.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=0.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=8000, gain_db=1.5, q=0.7),
                Compressor(threshold_db=-20, ratio=2.0, attack_ms=50, release_ms=400),
                Limiter(threshold_db=-2.0, release_ms=200)
            ]
        ),
        
        "cinematic_epic": MasteringPreset(
            "Cinematic Epic",
            "Big, powerful cinematic sound",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=2.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=250, gain_db=1.0, q=0.8),
                PeakFilter(cutoff_frequency_hz=2000, gain_db=1.5, q=1.0),
                PeakFilter(cutoff_frequency_hz=6000, gain_db=2.0, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.5, q=0.7),
                Compressor(threshold_db=-14, ratio=3.0, attack_ms=10, release_ms=150),
                Limiter(threshold_db=-0.5, release_ms=100)
            ]
        ),
        
        # Vintage/Lo-Fi Presets
        "lofi_chill": MasteringPreset(
            "Lo-Fi Chill",
            "Vintage lo-fi character",
            [
                HighpassFilter(cutoff_frequency_hz=50),
                LowpassFilter(cutoff_frequency_hz=10000),
                LowShelfFilter(cutoff_frequency_hz=150, gain_db=1.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=800, gain_db=-1.0, q=1.2),
                PeakFilter(cutoff_frequency_hz=4000, gain_db=-1.5, q=1.0),
                Compressor(threshold_db=-12, ratio=3.0, attack_ms=15, release_ms=180),
                Limiter(threshold_db=-1.0, release_ms=120)
            ]
        ),
        
        "vintage_vinyl": MasteringPreset(
            "Vintage Vinyl",
            "Classic vinyl record warmth",
            [
                HighpassFilter(cutoff_frequency_hz=40),
                LowpassFilter(cutoff_frequency_hz=12000),
                LowShelfFilter(cutoff_frequency_hz=120, gain_db=2.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=1000, gain_db=-0.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=5000, gain_db=-1.0, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-1.5, q=0.8),
                Compressor(threshold_db=-16, ratio=2.5, attack_ms=20, release_ms=200),
                Limiter(threshold_db=-1.5, release_ms=150)
            ]
        ),
        
        "retro_80s": MasteringPreset(
            "Retro 80s",
            "1980s-inspired mix with character",
            [
                HighpassFilter(cutoff_frequency_hz=45),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=800, gain_db=1.0, q=1.2),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=1.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0, q=0.7),
                Compressor(threshold_db=-14, ratio=3.5, attack_ms=5, release_ms=100),
                Limiter(threshold_db=-0.8, release_ms=80)
            ]
        ),
        
        # Enhancement Presets (Phase 2)
        "harmonic_enhance": MasteringPreset(
            "Harmonic Enhance",
            "Adds subtle harmonic overtones for brightness and warmth",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                # Subtle low-end warmth
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.0, q=0.7),
                # Presence boost
                PeakFilter(cutoff_frequency_hz=3000, gain_db=1.5, q=1.0),
                # Air and clarity
                HighShelfFilter(cutoff_frequency_hz=8000, gain_db=2.0, q=0.7),
                # Gentle saturation effect through compression
                Compressor(threshold_db=-18, ratio=2.5, attack_ms=10, release_ms=120),
                # Final limiting
                Limiter(threshold_db=-0.5, release_ms=100),
                # Note: Additional harmonic generation would require Distortion plugin
                # which adds subtle harmonic overtones
            ]
        ),
    }
    
    def __init__(self):
        """Initialize mastering service"""
        logger.info("Mastering service initialized")
            "Retro 80s",
            "80s digital warmth and punch",
            [
                HighpassFilter(cutoff_frequency_hz=35),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=800, gain_db=1.0, q=1.0),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=2.0, q=1.2),
                PeakFilter(cutoff_frequency_hz=8000, gain_db=1.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.0, q=0.8),
                Compressor(threshold_db=-10, ratio=4.0, attack_ms=5, release_ms=100),
                Limiter(threshold_db=-0.5, release_ms=80)
            ]
        ),
        
        # Specialized Presets
        "vocal_focused": MasteringPreset(
            "Vocal Focused",
            "Emphasizes vocal clarity and presence",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                PeakFilter(cutoff_frequency_hz=200, gain_db=-1.0, q=0.8),
                PeakFilter(cutoff_frequency_hz=1000, gain_db=1.0, q=1.0),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=2.5, q=1.2),
                PeakFilter(cutoff_frequency_hz=5000, gain_db=1.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.0, q=0.7),
                Compressor(threshold_db=-12, ratio=3.0, attack_ms=5, release_ms=100),
                Limiter(threshold_db=-0.5, release_ms=80)
            ]
        ),
        
        "bass_heavy": MasteringPreset(
            "Bass Heavy",
            "Maximum low-end power",
            [
                HighpassFilter(cutoff_frequency_hz=20),
                LowShelfFilter(cutoff_frequency_hz=60, gain_db=4.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=100, gain_db=2.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=500, gain_db=-1.5, q=1.0),
                PeakFilter(cutoff_frequency_hz=4000, gain_db=1.0, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.5, q=0.7),
                Compressor(threshold_db=-10, ratio=4.0, attack_ms=10, release_ms=100),
                Limiter(threshold_db=-0.3, release_ms=60)
            ]
        ),
        
        "bright_airy": MasteringPreset(
            "Bright & Airy",
            "Crystal clear highs with airiness",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=-0.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=500, gain_db=-1.0, q=0.8),
                PeakFilter(cutoff_frequency_hz=5000, gain_db=2.0, q=1.0),
                PeakFilter(cutoff_frequency_hz=10000, gain_db=2.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=12000, gain_db=3.0, q=0.7),
                Compressor(threshold_db=-14, ratio=2.5, attack_ms=8, release_ms=120),
                Limiter(threshold_db=-0.8, release_ms=100)
            ]
        ),
        
        "midrange_punch": MasteringPreset(
            "Midrange Punch",
            "Powerful mids for presence",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=0.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=500, gain_db=2.0, q=1.0),
                PeakFilter(cutoff_frequency_hz=1500, gain_db=2.5, q=1.2),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=2.0, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=8000, gain_db=0.5, q=0.7),
                Compressor(threshold_db=-11, ratio=3.5, attack_ms=5, release_ms=90),
                Limiter(threshold_db=-0.5, release_ms=70)
            ]
        ),
        
        "dynamic_range": MasteringPreset(
            "Dynamic Range",
            "Preserves maximum dynamics",
            [
                HighpassFilter(cutoff_frequency_hz=25),
                PeakFilter(cutoff_frequency_hz=100, gain_db=-0.5, q=0.7),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=0.5, q=0.8),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.0, q=0.7),
                Compressor(threshold_db=-20, ratio=1.5, attack_ms=20, release_ms=250),
                Limiter(threshold_db=-2.0, release_ms=200)
            ]
        ),
        
        "streaming_optimized": MasteringPreset(
            "Streaming Optimized",
            "Optimized for streaming platforms (Spotify, Apple Music)",
            [
                HighpassFilter(cutoff_frequency_hz=30),
                LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=500, gain_db=0.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=1.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.5, q=0.7),
                Compressor(threshold_db=-14, ratio=3.0, attack_ms=5, release_ms=100),
                Limiter(threshold_db=-1.0, release_ms=100)
            ]
        )
    }
    
    def __init__(self):
        """Initialize mastering service"""
        logger.info("Mastering service initialized with 32 presets")
    
    def apply_preset(self, audio_path: str, preset_name: str, output_path: str) -> str:
        """
        Apply mastering preset to audio file
        
        Args:
            audio_path: Path to input audio file
            preset_name: Name of preset to apply
            output_path: Path to save processed audio
            
        Returns:
            Path to processed audio file
        """
        try:
            if preset_name not in self.PRESETS:
                raise ValueError(f"Unknown preset: {preset_name}")
            
            preset = self.PRESETS[preset_name]
            logger.info(f"Applying preset '{preset.name}' to {audio_path}")
            
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Ensure stereo
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio], axis=1)
            
            # Create pedalboard with preset chain
            board = Pedalboard(preset.chain)
            
            # Process audio
            processed = board(audio.T, sr)
            
            # Save processed audio
            sf.write(output_path, processed.T, sr)
            logger.info(f"Saved mastered audio to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying preset: {str(e)}", exc_info=True)
            raise
    
    def apply_custom_eq(
        self,
        audio_path: str,
        output_path: str,
        eq_bands: List[Dict],
        compression: Optional[Dict] = None,
        limiting: Optional[Dict] = None
    ) -> str:
        """
        Apply custom EQ settings to audio file
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save processed audio
            eq_bands: List of EQ band settings
            compression: Compression settings (optional)
            limiting: Limiter settings (optional)
            
        Returns:
            Path to processed audio file
        """
        try:
            logger.info(f"Applying custom EQ to {audio_path}")
            
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Ensure stereo
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio], axis=1)
            
            # Build processing chain
            chain = []
            
            # Add EQ bands
            for band in eq_bands:
                band_type = band.get('type', 'peak')
                freq = band.get('frequency', 1000)
                gain = band.get('gain', 0)
                q = band.get('q', 1.0)
                
                if band_type == 'highpass':
                    chain.append(HighpassFilter(cutoff_frequency_hz=freq))
                elif band_type == 'lowpass':
                    chain.append(LowpassFilter(cutoff_frequency_hz=freq))
                elif band_type == 'lowshelf':
                    chain.append(LowShelfFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))
                elif band_type == 'highshelf':
                    chain.append(HighShelfFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))
                else:  # peak
                    chain.append(PeakFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))
            
            # Add compression if specified
            if compression:
                chain.append(Compressor(
                    threshold_db=compression.get('threshold', -12),
                    ratio=compression.get('ratio', 2.0),
                    attack_ms=compression.get('attack', 5),
                    release_ms=compression.get('release', 100)
                ))
            
            # Add limiting if specified
            if limiting:
                chain.append(Limiter(
                    threshold_db=limiting.get('threshold', -1.0),
                    release_ms=limiting.get('release', 100)
                ))
            
            # Create and apply pedalboard
            board = Pedalboard(chain)
            processed = board(audio.T, sr)
            
            # Save processed audio
            sf.write(output_path, processed.T, sr)
            logger.info(f"Saved custom EQ audio to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying custom EQ: {str(e)}", exc_info=True)
            raise
    
    def get_preset_list(self) -> List[Dict]:
        """Get list of available presets with descriptions"""
        return [
            {
                'id': key,
                'name': preset.name,
                'description': preset.description
            }
            for key, preset in self.PRESETS.items()
        ]
