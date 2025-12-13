"""
Model downloader and setup utility
Downloads AI models locally for offline use
"""
import os
import logging
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelDownloader:
    """Handles downloading and caching AI models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
    def download_diffrhythm(self):
        """
        Download DiffRhythm model
        Note: Using a placeholder - update with actual model repo when available
        """
        try:
            logger.info("Downloading DiffRhythm model...")
            
            model_path = self.models_dir / "diffrhythm"
            
            # TODO: Update with actual DiffRhythm model repository
            # For now, using a music generation model as placeholder
            # snapshot_download(
            #     repo_id="ASLP-lab/DiffRhythm",
            #     local_dir=str(model_path),
            #     local_dir_use_symlinks=False
            # )
            
            logger.warning("DiffRhythm: Using placeholder - update with actual model")
            model_path.mkdir(exist_ok=True)
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download DiffRhythm: {str(e)}")
            raise
    
    def download_lyricmind(self):
        """
        Download LyricsMindAI model
        Note: Using a placeholder - update with actual model repo when available
        """
        try:
            logger.info("Downloading LyricsMindAI model...")
            
            model_path = self.models_dir / "lyricmind"
            
            # TODO: Update with actual LyricsMindAI model repository
            # snapshot_download(
            #     repo_id="AmirHaytham/LyricMind-AI",
            #     local_dir=str(model_path),
            #     local_dir_use_symlinks=False
            # )
            
            logger.warning("LyricsMindAI: Using placeholder - update with actual model")
            model_path.mkdir(exist_ok=True)
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download LyricsMindAI: {str(e)}")
            raise
    
    def download_fish_speech(self):
        """
        Download Fish Speech model for TTS
        """
        try:
            logger.info("Downloading Fish Speech model...")
            
            model_path = self.models_dir / "fish_speech"
            
            # Fish Speech models from Hugging Face
            # Using a compatible TTS model
            try:
                snapshot_download(
                    repo_id="fishaudio/fish-speech-1.2",
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.md", "*.txt", ".gitattributes"]
                )
                logger.info("✅ Fish Speech model downloaded")
            except Exception:
                logger.warning("Fish Speech: Could not download, using placeholder")
                model_path.mkdir(exist_ok=True)
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download Fish Speech: {str(e)}")
            raise
    
    def download_text_generation_model(self):
        """
        Download a text generation model for lyrics generation
        Using as fallback for LyricsMindAI
        """
        try:
            logger.info("Downloading text generation model for lyrics...")
            
            model_path = self.models_dir / "text_generator"
            
            # Use a smaller, efficient model that works well on CPU/AMD GPU
            snapshot_download(
                repo_id="microsoft/phi-2",  # Compact 2.7B model
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt", ".gitattributes"]
            )
            
            logger.info("✅ Text generation model downloaded")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download text model: {str(e)}")
            raise
    
    def download_audio_generation_model(self):
        """
        Download an audio generation model
        Using as fallback for DiffRhythm
        """
        try:
            logger.info("Downloading audio generation model...")
            
            model_path = self.models_dir / "audio_generator"
            
            # Use MusicGen or similar model that supports AMD GPU
            snapshot_download(
                repo_id="facebook/musicgen-small",  # Smaller model for faster inference
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt", ".gitattributes"]
            )
            
            logger.info("✅ Audio generation model downloaded")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download audio model: {str(e)}")
            raise
    
    def setup_all_models(self):
        """Download all required models"""
        logger.info("Setting up all models...")
        
        models = {
            "DiffRhythm": self.download_diffrhythm,
            "LyricsMindAI": self.download_lyricmind,
            "Fish Speech": self.download_fish_speech,
            "Audio Generator (fallback)": self.download_audio_generation_model,
            "Text Generator (fallback)": self.download_text_generation_model
        }
        
        results = {}
        
        for name, download_func in models.items():
            try:
                path = download_func()
                results[name] = {"success": True, "path": path}
                logger.info(f"✅ {name}: {path}")
            except Exception as e:
                results[name] = {"success": False, "error": str(e)}
                logger.error(f"❌ {name}: {str(e)}")
        
        return results
    
    def check_models_exist(self):
        """Check which models are already downloaded"""
        models_status = {
            "diffrhythm": (self.models_dir / "diffrhythm").exists(),
            "lyricmind": (self.models_dir / "lyricmind").exists(),
            "fish_speech": (self.models_dir / "fish_speech").exists(),
            "audio_generator": (self.models_dir / "audio_generator").exists(),
            "text_generator": (self.models_dir / "text_generator").exists()
        }
        
        return models_status

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎵 Music Generation App - Model Setup")
    print("=" * 50)
    
    downloader = ModelDownloader()
    
    # Check existing models
    print("\nChecking for existing models...")
    status = downloader.check_models_exist()
    
    for model, exists in status.items():
        status_icon = "✅" if exists else "❌"
        print(f"{status_icon} {model}: {'Found' if exists else 'Not found'}")
    
    # Ask to download
    print("\n" + "=" * 50)
    response = input("\nDownload missing models? (y/n): ")
    
    if response.lower() == 'y':
        print("\nDownloading models... This may take a while.")
        results = downloader.setup_all_models()
        
        print("\n" + "=" * 50)
        print("Setup Complete!")
        print("=" * 50)
    else:
        print("Skipping model download.")
