"""
Prompt analysis utility for extracting music attributes
Analyzes user prompts to extract genre, style, BPM, mood, and other musical attributes
"""
import re
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

class PromptAnalyzer:
    """Analyzes music prompts to extract musical attributes"""
    
    # Genre/style keywords
    GENRES = {
        'pop': ['pop', 'mainstream', 'catchy', 'radio-friendly'],
        'rock': ['rock', 'guitar', 'electric', 'distortion', 'power chords'],
        'hip-hop': ['hip-hop', 'rap', 'trap', 'beats', 'rhymes', 'flow'],
        'electronic': ['edm', 'electronic', 'synth', 'techno', 'house', 'trance'],
        'jazz': ['jazz', 'swing', 'bebop', 'saxophone', 'improvisation'],
        'classical': ['classical', 'orchestra', 'symphony', 'piano', 'strings'],
        'country': ['country', 'folk', 'acoustic', 'banjo', 'bluegrass'],
        'r&b': ['r&b', 'soul', 'rnb', 'rhythm and blues', 'groove'],
        'metal': ['metal', 'heavy', 'headbanging', 'aggressive', 'brutal'],
        'indie': ['indie', 'alternative', 'underground', 'experimental'],
        'reggae': ['reggae', 'ska', 'dub', 'jamaican', 'offbeat'],
        'blues': ['blues', 'twelve bar', 'soulful', 'melancholic']
    }
    
    # BPM keywords and ranges
    BPM_KEYWORDS = {
        'slow': (60, 80),
        'ballad': (60, 80),
        'moderate': (80, 120),
        'medium': (90, 110),
        'upbeat': (120, 140),
        'fast': (140, 180),
        'energetic': (130, 150),
        'intense': (150, 180)
    }
    
    # Mood/emotion keywords
    MOODS = {
        'happy': ['happy', 'joyful', 'cheerful', 'uplifting', 'bright'],
        'sad': ['sad', 'melancholic', 'sorrowful', 'emotional', 'tearful'],
        'energetic': ['energetic', 'powerful', 'dynamic', 'intense', 'vigorous'],
        'calm': ['calm', 'peaceful', 'relaxing', 'soothing', 'tranquil'],
        'dark': ['dark', 'ominous', 'mysterious', 'sinister', 'haunting'],
        'romantic': ['romantic', 'love', 'passionate', 'tender', 'intimate'],
        'angry': ['angry', 'aggressive', 'fierce', 'furious', 'rage'],
        'nostalgic': ['nostalgic', 'reminiscent', 'wistful', 'longing']
    }
    
    # Instrumental keywords
    INSTRUMENTS = [
        'guitar', 'piano', 'drums', 'bass', 'synth', 'violin', 'saxophone',
        'trumpet', 'flute', 'organ', 'keyboard', 'strings', 'brass', 'percussion'
    ]
    
    @classmethod
    def analyze(cls, prompt: str) -> Dict[str, Any]:
        """
        Analyze a music prompt to extract attributes
        
        Args:
            prompt: User's music description
            
        Returns:
            Dictionary containing:
            - genre: Detected genre(s)
            - bpm: Estimated BPM or range
            - mood: Detected mood(s)
            - instruments: Mentioned instruments
            - style_tags: Additional style descriptors
            - analysis_text: Formatted analysis for AI models
        """
        if not prompt:
            return cls._get_default_analysis()
        
        prompt_lower = prompt.lower()
        
        # Detect genre
        detected_genres = cls._detect_genres(prompt_lower)
        
        # Detect BPM
        bpm_info = cls._detect_bpm(prompt_lower)
        
        # Detect mood
        detected_moods = cls._detect_moods(prompt_lower)
        
        # Detect instruments
        detected_instruments = cls._detect_instruments(prompt_lower)
        
        # Extract additional style tags
        style_tags = cls._extract_style_tags(prompt_lower)
        
        # Build structured analysis
        analysis = {
            'genre': detected_genres[0] if detected_genres else 'pop',
            'genres': detected_genres,
            'bpm': bpm_info['bpm'],
            'bpm_range': bpm_info['range'],
            'mood': detected_moods[0] if detected_moods else 'neutral',
            'moods': detected_moods,
            'instruments': detected_instruments,
            'style_tags': style_tags,
            'has_vocals': cls._should_have_vocals(prompt_lower),
            'analysis_text': cls._format_analysis_text(
                detected_genres, bpm_info, detected_moods, detected_instruments
            )
        }
        
        logger.info(f"Prompt analysis: genre={analysis['genre']}, bpm={analysis['bpm']}, mood={analysis['mood']}")
        
        return analysis
    
    @classmethod
    def _detect_genres(cls, prompt: str) -> List[str]:
        """Detect genres from prompt"""
        detected = []
        for genre, keywords in cls.GENRES.items():
            if any(keyword in prompt for keyword in keywords):
                detected.append(genre)
        return detected[:3]  # Top 3 genres
    
    @classmethod
    def _detect_bpm(cls, prompt: str) -> Dict[str, Any]:
        """Detect BPM or BPM range from prompt"""
        # Check for explicit BPM numbers
        bpm_match = re.search(r'\b(\d{2,3})\s*bpm\b', prompt)
        if bpm_match:
            bpm_value = int(bpm_match.group(1))
            return {
                'bpm': bpm_value,
                'range': (bpm_value - 5, bpm_value + 5)
            }
        
        # Check for BPM keywords
        for keyword, (min_bpm, max_bpm) in cls.BPM_KEYWORDS.items():
            if keyword in prompt:
                return {
                    'bpm': (min_bpm + max_bpm) // 2,
                    'range': (min_bpm, max_bpm)
                }
        
        # Default: moderate tempo
        return {'bpm': 120, 'range': (100, 140)}
    
    @classmethod
    def _detect_moods(cls, prompt: str) -> List[str]:
        """Detect moods from prompt"""
        detected = []
        for mood, keywords in cls.MOODS.items():
            if any(keyword in prompt for keyword in keywords):
                detected.append(mood)
        return detected[:2]  # Top 2 moods
    
    @classmethod
    def _detect_instruments(cls, prompt: str) -> List[str]:
        """Detect mentioned instruments"""
        detected = []
        for instrument in cls.INSTRUMENTS:
            if instrument in prompt:
                detected.append(instrument)
        return detected
    
    @classmethod
    def _extract_style_tags(cls, prompt: str) -> List[str]:
        """Extract additional style descriptors"""
        tags = []
        style_keywords = [
            'vintage', 'modern', 'retro', 'futuristic', 'minimal', 'complex',
            'acoustic', 'electric', 'orchestral', 'ambient', 'rhythmic',
            'melodic', 'harmonic', 'atmospheric', 'driving', 'groovy'
        ]
        
        for tag in style_keywords:
            if tag in prompt:
                tags.append(tag)
        
        return tags
    
    @classmethod
    def _should_have_vocals(cls, prompt: str) -> bool:
        """Determine if music should have vocals"""
        vocal_keywords = ['vocal', 'singing', 'voice', 'lyrics', 'song', 'sung']
        instrumental_keywords = ['instrumental', 'no vocals', 'no voice', 'without vocals']
        
        has_vocal_mention = any(keyword in prompt for keyword in vocal_keywords)
        has_instrumental_mention = any(keyword in prompt for keyword in instrumental_keywords)
        
        # Default to vocals unless explicitly instrumental
        if has_instrumental_mention:
            return False
        
        return True  # Default to vocals
    
    @classmethod
    def _format_analysis_text(
        cls,
        genres: List[str],
        bpm_info: Dict,
        moods: List[str],
        instruments: List[str]
    ) -> str:
        """Format analysis into text for AI model context"""
        parts = []
        
        if genres:
            parts.append(f"Genre: {', '.join(genres)}")
        
        if bpm_info.get('bpm'):
            parts.append(f"BPM: {bpm_info['bpm']}")
        
        if moods:
            parts.append(f"Mood: {', '.join(moods)}")
        
        if instruments:
            parts.append(f"Instruments: {', '.join(instruments)}")
        
        return '; '.join(parts) if parts else "General music"
    
    @classmethod
    def _get_default_analysis(cls) -> Dict[str, Any]:
        """Return default analysis when prompt is empty"""
        return {
            'genre': 'pop',
            'genres': ['pop'],
            'bpm': 120,
            'bpm_range': (100, 140),
            'mood': 'neutral',
            'moods': [],
            'instruments': [],
            'style_tags': [],
            'has_vocals': True,
            'analysis_text': 'General pop music at moderate tempo'
        }
    
    @classmethod
    def format_for_diffrhythm(cls, prompt: str, lyrics: Optional[str] = None, analysis: Optional[Dict] = None) -> str:
        """
        Format prompt for DiffRhythm model
        
        Args:
            prompt: Original user prompt
            lyrics: Optional lyrics
            analysis: Optional pre-computed analysis
            
        Returns:
            Formatted prompt for DiffRhythm
        """
        if analysis is None:
            analysis = cls.analyze(prompt)
        
        parts = [prompt]
        
        # Add analysis context
        if analysis.get('analysis_text'):
            parts.append(f"[{analysis['analysis_text']}]")
        
        # Add lyrics if provided
        if lyrics:
            parts.append(f"Lyrics: {lyrics}")
        
        return ' '.join(parts)
    
    @classmethod
    def format_for_lyrics_generation(cls, prompt: str, analysis: Optional[Dict] = None) -> str:
        """
        Format prompt for lyrics generation
        
        Args:
            prompt: Original user prompt
            analysis: Optional pre-computed analysis
            
        Returns:
            Formatted prompt for LyricsMind
        """
        if analysis is None:
            analysis = cls.analyze(prompt)
        
        genre = analysis.get('genre', 'pop')
        mood = analysis.get('mood', 'neutral')
        
        formatted = f"Write {genre} song lyrics with a {mood} mood about: {prompt}"
        
        # Add additional context
        if analysis.get('style_tags'):
            formatted += f" (Style: {', '.join(analysis['style_tags'][:2])})"
        
        return formatted
