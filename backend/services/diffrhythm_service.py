"""
DiffRhythm 2 music generation service
Integrates with the DiffRhythm 2 model for music generation with vocals
"""
import os
import sys
import logging
import uuid
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
import torch
import torchaudio
import json

# Configure espeak-ng path for phonemizer (required by g2p module)
# Note: Environment configuration handled by hf_config.py for HuggingFace Spaces
# or by launch scripts for local development
if "PHONEMIZER_ESPEAK_PATH" not in os.environ:
    # Fallback for local development without launcher
    espeak_path = Path(__file__).parent.parent.parent / "external" / "espeak-ng"
    if espeak_path.exists():
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_path / "libespeak-ng.dll")
        os.environ["PHONEMIZER_ESPEAK_PATH"] = str(espeak_path)

# Add DiffRhythm2 source code to path (cloned repo, not pip package)
diffrhythm2_src = Path(__file__).parent.parent.parent / "models" / "diffrhythm2_source"
sys.path.insert(0, str(diffrhythm2_src))

logger = logging.getLogger(__name__)

class DiffRhythmService:
    """Service for DiffRhythm 2 music generation"""
    
    def __init__(self, model_path: str):
        """
        Initialize DiffRhythm 2 service
        
        Args:
            model_path: Path to DiffRhythm 2 model files
        """
        self.model_path = model_path
        self.model = None
        self.mulan = None
        self.lrc_tokenizer = None
        self.decoder = None
        self.is_initialized = False
        self.device = self._get_device()
        logger.info(f"DiffRhythm 2 service created with model path: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self):
        """Get compute device (CUDA or CPU)"""
        # Try CUDA first (NVIDIA)
        if torch.cuda.is_available():
            logger.info("Using CUDA (NVIDIA GPU)")
            return torch.device("cuda")
        
        # Note: DirectML support disabled due to version conflicts with DiffRhythm2
        # DiffRhythm2 requires torch>=2.4, but torch-directml requires torch==2.4.1
        # For AMD GPU acceleration, consider using ROCm with compatible PyTorch build
        
        # Fallback to CPU
        logger.info("Using CPU (no GPU acceleration)")
        return torch.device("cpu")
    
    def _initialize_model(self):
        """Lazy load the DiffRhythm 2 model when first needed"""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing DiffRhythm 2 model...")
            
            from diffrhythm2.cfm import CFM
            from diffrhythm2.backbones.dit import DiT
            from bigvgan.model import Generator
            from muq import MuQMuLan
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            
            # Load DiffRhythm 2 model
            repo_id = "ASLP-lab/DiffRhythm2"
            
            # Download model files
            model_ckpt = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                local_dir=self.model_path,
                local_files_only=False,
            )
            model_config_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                local_dir=self.model_path,
                local_files_only=False,
            )
            
            # Load config
            with open(model_config_path) as f:
                model_config = json.load(f)
            
            model_config['use_flex_attn'] = False
            
            # Create model
            self.model = CFM(
                transformer=DiT(**model_config),
                num_channels=model_config['mel_dim'],
                block_size=model_config['block_size'],
            )
            
            # Load weights
            ckpt = load_file(model_ckpt)
            self.model.load_state_dict(ckpt)
            self.model = self.model.to(self.device)
            
            # Load MuLan for style encoding
            self.mulan = MuQMuLan.from_pretrained(
                "OpenMuQ/MuQ-MuLan-large",
                cache_dir=os.path.join(self.model_path, "mulan")
            ).to(self.device)
            
            # Load tokenizer
            from g2p.g2p_generation import chn_eng_g2p
            
            # Look for vocab.json in the cloned DiffRhythm2 source
            diffrhythm2_src = Path(__file__).parent.parent.parent / "models" / "diffrhythm2_source"
            vocab_path = diffrhythm2_src / "g2p" / "g2p" / "vocab.json"
            
            if not vocab_path.exists():
                # Fallback: try downloading from HF Hub
                logger.warning(f"vocab.json not found at {vocab_path}, trying to download from HF Hub")
                vocab_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="vocab.json",
                    local_dir=self.model_path,
                    local_files_only=False,
                )
            
            logger.info(f"Loading vocab from: {vocab_path}")
            with open(vocab_path, 'r') as f:
                phone2id = json.load(f)['vocab']
            
            self.lrc_tokenizer = {
                'phone2id': phone2id,
                'g2p': chn_eng_g2p
            }
            
            # Load decoder (BigVGAN vocoder)
            decoder_ckpt = hf_hub_download(
                repo_id=repo_id,
                filename="decoder.bin",
                local_dir=self.model_path,
                local_files_only=False,
            )
            decoder_config = hf_hub_download(
                repo_id=repo_id,
                filename="decoder.json",
                local_dir=self.model_path,
                local_files_only=False,
            )
            
            self.decoder = Generator(decoder_config, decoder_ckpt)
            self.decoder = self.decoder.to(self.device)
            
            logger.info("✅ DiffRhythm 2 model loaded successfully")
            
            self.is_initialized = True
            logger.info("DiffRhythm 2 service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize DiffRhythm 2: {str(e)}", exc_info=True)
            raise RuntimeError(f"Could not load DiffRhythm 2 model: {str(e)}")
    
    def generate(
        self,
        prompt: str,
        duration: int = 30,
        sample_rate: int = 44100,
        lyrics: Optional[str] = None,
        reference_audio: Optional[str] = None
    ) -> str:
        """
        Generate music from text prompt with optional vocals/lyrics and style reference
        
        Args:
            prompt: Text description of desired music
            duration: Length in seconds
            sample_rate: Audio sample rate
            lyrics: Optional lyrics for vocals
            reference_audio: Optional path to reference audio for style consistency
            
        Returns:
            Path to generated audio file
        """
        try:
            self._initialize_model()
            
            if lyrics:
                logger.info(f"Generating music with vocals: prompt='{prompt}', lyrics_length={len(lyrics)}")
            else:
                logger.info(f"Generating instrumental music: prompt='{prompt}'")
            
            if reference_audio and os.path.exists(reference_audio):
                logger.info(f"Using style reference: {reference_audio}")
            
            logger.info(f"Duration={duration}s")
            
            # Try to generate with DiffRhythm 2
            if self.model is not None:
                audio = self._generate_with_diffrhythm2(prompt, lyrics, duration, sample_rate, reference_audio)
            else:
                # Fallback: Generate placeholder
                logger.warning("Using placeholder audio generation")
                audio = self._generate_placeholder(duration, sample_rate)
            
            # Save to file
            output_dir = os.path.join('outputs', 'music')
            os.makedirs(output_dir, exist_ok=True)
            
            clip_id = str(uuid.uuid4())
            output_path = os.path.join(output_dir, f"{clip_id}.wav")
            
            # Ensure audio is in correct format (channels, samples) for soundfile
            # If audio is 1D (mono), keep it as is. If 2D, ensure it's (samples, channels)
            if audio.ndim == 1:
                # Mono audio - soundfile expects (samples,) shape
                sf.write(output_path, audio, sample_rate)
            else:
                # Stereo/multi-channel - soundfile expects (samples, channels)
                sf.write(output_path, audio, sample_rate)
            
            logger.info(f"Music generated successfully: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Music generation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate music: {str(e)}")
    
    def _generate_with_diffrhythm2(
        self, 
        prompt: str, 
        lyrics: Optional[str], 
        duration: int, 
        sample_rate: int,
        reference_audio: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate music using DiffRhythm 2 model with optional style reference
        
        Args:
            prompt: Music description (used as style prompt)
            lyrics: Lyrics for vocals (required for vocal generation)
            duration: Duration in seconds
            sample_rate: Sample rate
            reference_audio: Optional path to reference audio for style guidance
            
        Returns:
            Audio array
        """
        try:
            logger.info("Generating with DiffRhythm 2 model...")
            
            # Prepare lyrics tokens
            if lyrics:
                lyrics_token = self._tokenize_lyrics(lyrics)
            else:
                # For instrumental, use empty structure
                lyrics_token = torch.tensor([500, 511], dtype=torch.long, device=self.device)  # [start][stop]
            
            # Encode style prompt with optional reference audio blending
            with torch.no_grad():
                if reference_audio and os.path.exists(reference_audio):
                    try:
                        import torchaudio
                        # Load reference audio
                        ref_waveform, ref_sr = torchaudio.load(reference_audio)
                        if ref_sr != 24000:  # MuLan expects 24kHz
                            ref_waveform = torchaudio.functional.resample(ref_waveform, ref_sr, 24000)
                        
                        # Encode reference audio with MuLan
                        ref_waveform = ref_waveform.to(self.device)
                        audio_style_embed = self.mulan(audios=ref_waveform.unsqueeze(0))
                        text_style_embed = self.mulan(texts=[prompt])
                        
                        # Blend reference audio style with text prompt (70% audio, 30% text)
                        style_prompt_embed = 0.7 * audio_style_embed + 0.3 * text_style_embed
                        logger.info("Using blended style: 70% reference audio + 30% text prompt")
                    except Exception as e:
                        logger.warning(f"Failed to use reference audio, using text prompt only: {e}")
                        style_prompt_embed = self.mulan(texts=[prompt])
                else:
                    style_prompt_embed = self.mulan(texts=[prompt])
                    
            style_prompt_embed = style_prompt_embed.to(self.device).squeeze(0)
            
            # Use FP16 if on GPU
            if self.device.type != 'cpu':
                self.model = self.model.half()
                self.decoder = self.decoder.half()
                style_prompt_embed = style_prompt_embed.half()
            
            # Generate latent representation
            with torch.inference_mode():
                latent = self.model.sample_block_cache(
                    text=lyrics_token.unsqueeze(0),
                    duration=int(duration * 5),  # DiffRhythm uses 5 frames per second
                    style_prompt=style_prompt_embed.unsqueeze(0),
                    steps=16,  # Sampling steps
                    cfg_strength=2.0,  # Classifier-free guidance
                    process_bar=False
                )
                
                # Decode to audio
                latent = latent.transpose(1, 2)
                audio = self.decoder.decode_audio(latent, overlap=5, chunk_size=20)
            
            # Convert to numpy
            audio = audio.float().cpu().numpy().squeeze()
            
            # Ensure correct length
            target_length = int(duration * sample_rate)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            # Resample if needed
            if sample_rate != 24000:  # DiffRhythm 2 native sample rate
                import scipy.signal as signal
                audio = signal.resample(audio, target_length)
            
            logger.info("✅ DiffRhythm 2 generation successful")
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"DiffRhythm 2 generation failed: {str(e)}")
            return self._generate_placeholder(duration, sample_rate)
    
    def _tokenize_lyrics(self, lyrics: str) -> torch.Tensor:
        """
        Tokenize lyrics for DiffRhythm 2
        
        Args:
            lyrics: Lyrics text
            
        Returns:
            Tokenized lyrics tensor
        """
        try:
            # Structure tags
            STRUCT_INFO = {
                "[start]": 500,
                "[end]": 501,
                "[intro]": 502,
                "[verse]": 503,
                "[chorus]": 504,
                "[outro]": 505,
                "[inst]": 506,
                "[solo]": 507,
                "[bridge]": 508,
                "[hook]": 509,
                "[break]": 510,
                "[stop]": 511,
                "[space]": 512
            }
            
            # Convert lyrics to phonemes and tokens
            phone, tokens = self.lrc_tokenizer['g2p'](lyrics)
            tokens = [x + 1 for x in tokens]  # Offset by 1
            
            # Add structure: [start] + lyrics + [stop]
            lyrics_tokens = [STRUCT_INFO['[start]']] + tokens + [STRUCT_INFO['[stop]']]
            
            return torch.tensor(lyrics_tokens, dtype=torch.long, device=self.device)
            
        except Exception as e:
            logger.error(f"Lyrics tokenization failed: {str(e)}")
            # Return minimal structure
            return torch.tensor([500, 511], dtype=torch.long, device=self.device)
    
    def _generate_placeholder(self, duration: int, sample_rate: int) -> np.ndarray:
        """
        Generate placeholder audio (for testing without actual model)
        
        Args:
            duration: Length in seconds
            sample_rate: Sample rate
            
        Returns:
            Audio array
        """
        logger.warning("Using placeholder audio - DiffRhythm 2 model not loaded")
        
        # Generate simple sine wave as placeholder
        t = np.linspace(0, duration, int(duration * sample_rate))
        frequency = 440  # A4 note
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        return audio.astype(np.float32)

