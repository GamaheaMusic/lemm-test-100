"""
LyricMind AI lyrics generation service
"""
import os
import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

class LyricMindService:
    """Service for LyricMind AI lyrics generation"""
    
    def __init__(self, model_path: str):
        """
        Initialize LyricMind service
        
        Args:
            model_path: Path to LyricMind model files
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.device = self._get_device()
        logger.info(f"LyricMind service created with model path: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self):
        """Get compute device (AMD GPU via DirectML or CPU)"""
        try:
            from utils.amd_gpu import DEFAULT_DEVICE
            return DEFAULT_DEVICE
        except:
            return torch.device("cpu")
    
    def _initialize_model(self):
        """Lazy load the model when first needed"""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing LyricMind model...")
            
            # Try to load text generation model as fallback
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                fallback_path = os.path.join(os.path.dirname(self.model_path), "text_generator")
                
                if os.path.exists(fallback_path):
                    logger.info(f"Loading text generation model from {fallback_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_path, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        fallback_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float32  # Use FP32 for AMD GPU compatibility
                    )
                    self.model.to(self.device)
                    logger.info("✅ Text generation model loaded successfully")
                else:
                    logger.warning("Text generation model not found, using placeholder")
                    
            except Exception as e:
                logger.warning(f"Could not load text model: {str(e)}")
            
            self.is_initialized = True
            logger.info("LyricMind service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize LyricMind model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Could not load LyricMind model: {str(e)}")
    
    def generate(
        self,
        prompt: str,
        style: Optional[str] = None,
        duration: int = 30,
        prompt_analysis: Optional[dict] = None
    ) -> str:
        """
        Generate lyrics from prompt using analysis context
        
        Args:
            prompt: Description of desired lyrics theme
            style: Music style (optional, will be detected if not provided)
            duration: Target song duration (affects lyrics length)
            prompt_analysis: Pre-computed prompt analysis (optional)
            
        Returns:
            Generated lyrics text
        """
        try:
            self._initialize_model()
            
            # Use prompt analysis for better context
            from utils.prompt_analyzer import PromptAnalyzer
            
            if prompt_analysis is None:
                analysis = PromptAnalyzer.analyze(prompt)
            else:
                analysis = prompt_analysis
            
            # Use detected genre/style if not explicitly provided
            effective_style = style or analysis.get('genre', 'pop')
            mood = analysis.get('mood', 'neutral')
            
            logger.info(f"Generating lyrics: prompt='{prompt}', style={effective_style}, mood={mood}")
            
            # Try to generate with text model
            if self.model is not None and self.tokenizer is not None:
                lyrics = self._generate_with_model(prompt, effective_style, duration, analysis)
            else:
                # Fallback: placeholder lyrics
                lyrics = self._generate_placeholder(prompt, effective_style, duration)
            
            logger.info("Lyrics generated successfully")
            return lyrics
            
        except Exception as e:
            logger.error(f"Lyrics generation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate lyrics: {str(e)}")
    
    def _generate_with_model(self, prompt: str, style: str, duration: int, analysis: dict) -> str:
        """
        Generate lyrics using text generation model with analysis context
        
        Args:
            prompt: Theme prompt
            style: Music style
            duration: Duration in seconds
            analysis: Prompt analysis with genre, mood, etc.
            
        Returns:
            Generated lyrics
        """
        try:
            logger.info("Generating lyrics with AI model...")
            
            # Create structured prompt with analysis context
            mood = analysis.get('mood', 'neutral')
            bpm = analysis.get('bpm', 120)
            
            # Simpler prompt to avoid metadata in output
            full_prompt = f"Write {style} song lyrics about {prompt}. Make it {mood} with a {bpm} BPM feel.\n\n[Verse 1]\n"
            
            # Tokenize
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Calculate max length based on duration
            max_length = min(200 + inputs["input_ids"].shape[1], 512)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.9,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract lyrics - find the actual song structure
            # Remove the instruction part
            lyrics = generated_text
            if "[Verse 1]" in lyrics:
                lyrics = "[Verse 1]" + lyrics.split("[Verse 1]")[-1]
            
            lyrics = lyrics.strip()
            
            logger.info("✅ AI lyrics generation successful")
            return lyrics if lyrics else self._generate_placeholder(prompt, style, duration)
            
        except Exception as e:
            logger.error(f"Model generation failed: {str(e)}")
            return self._generate_placeholder(prompt, style, duration)
    
    def _generate_placeholder(
        self,
        prompt: str,
        style: str,
        duration: int
    ) -> str:
        """
        Generate placeholder lyrics for testing
        
        Args:
            prompt: Theme prompt
            style: Music style
            duration: Duration in seconds
            
        Returns:
            Placeholder lyrics
        """
        logger.warning("Using placeholder lyrics - LyricMind model not loaded")
        
        # Estimate number of lines based on duration
        lines_per_30s = 8
        num_lines = int((duration / 30) * lines_per_30s)
        
        lyrics_lines = [
            f"[Verse 1]",
            f"Theme: {prompt}",
            f"Style: {style}",
            "",
            "[Chorus]",
            "This is a placeholder",
            "Generated by LyricMind AI",
            "Replace with actual model output",
        ]
        
        # Pad to desired length
        while len(lyrics_lines) < num_lines:
            lyrics_lines.append("La la la...")
        
        return "\n".join(lyrics_lines[:num_lines])
