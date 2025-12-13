"""
Dataset download and preparation service
Downloads curated datasets from HuggingFace for LoRA training
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)

class DatasetService:
    """Service for downloading and preparing training datasets"""
    
    # Dataset configurations
    DATASETS = {
        # Music Datasets (Verified HuggingFace IDs)
        "gtzan": {
            "name": "GTZAN Music Genre (1000 tracks, 10 genres)",
            "hf_id": "marsyas/gtzan",
            "type": "music",
            "description": "1000 songs across 10 music genres for style learning",
            "size_gb": 1.2,
            "splits": ["train"]
        },
        "nsynth_valid": {
            "name": "NSynth Musical Notes (Validation set)",
            "hf_id": "google/nsynth",
            "split": "valid",
            "type": "music",
            "description": "Musical notes with unique pitch and timbre",
            "size_gb": 0.8,
            "splits": ["valid"]
        },
        "maestro": {
            "name": "MAESTRO Piano Performances (subset)",
            "hf_id": "roszcz/maestro-v3",
            "type": "music",
            "description": "Classical piano performances with MIDI + audio",
            "size_gb": 2.0,
            "splits": ["validation"]
        },
        
        # Vocal Datasets (Verified HuggingFace IDs)
        "ljspeech": {
            "name": "LJSpeech (13k vocal clips, single speaker)",
            "hf_id": "lj_speech",
            "type": "vocal",
            "description": "High-quality single speaker for vocal characteristics",
            "size_gb": 2.6,
            "splits": ["train"]
        },
        "common_voice_en": {
            "name": "Common Voice English (diverse speakers)",
            "hf_id": "mozilla-foundation/common_voice_11_0",
            "type": "vocal",
            "description": "Diverse English speakers with various accents",
            "size_gb": 5.0,
            "config": "en",
            "splits": ["train", "validation"]
        },
        "librispeech_clean": {
            "name": "LibriSpeech Clean (English audiobooks)",
            "hf_id": "librispeech_asr",
            "type": "vocal",
            "description": "Clean English speech from audiobooks",
            "size_gb": 6.3,
            "config": "clean",
            "splits": ["train.100"]
        },
        
        # Sound Effects (Verified HuggingFace IDs)
        "esc50": {
            "name": "ESC-50 Environmental Sounds (2000 samples)",
            "hf_id": "ashraq/esc50",
            "type": "sound_effects",
            "description": "2000 environmental sounds across 50 classes",
            "size_gb": 0.6,
            "splits": ["train", "test"]
        },
        
        # Speech Commands (Verified HuggingFace IDs)
        "speech_commands": {
            "name": "Google Speech Commands (short words)",
            "hf_id": "speech_commands",
            "type": "vocal",
            "description": "Short spoken words for vocal pattern learning",
            "size_gb": 2.0,
            "config": "v0.02",
            "splits": ["train", "validation", "test"]
        },
        
        # Note: Original datasets that were not found
        # "opensinger": Not available on HuggingFace Hub
        # "m4singer": Not available on HuggingFace Hub  
        # "lakh_midi": Not reliably available
        # "ccmixter": Requires manual download
        # "mutopia": Requires manual download
    }
    
    def __init__(self, base_dir: str = "training_data"):
        """
        Initialize dataset service
        
        Args:
            base_dir: Base directory for storing datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, dataset_key: str, progress_callback=None) -> Dict:
        """
        Download a dataset from HuggingFace
        
        Args:
            dataset_key: Key identifying the dataset (e.g., 'gtzan')
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with dataset info and status
        """
        try:
            if dataset_key not in self.DATASETS:
                raise ValueError(f"Unknown dataset: {dataset_key}")
            
            dataset_config = self.DATASETS[dataset_key]
            dataset_name = dataset_config['name']
            
            if progress_callback:
                progress_callback(f"📦 Starting download: {dataset_name}")
                
                # Show dataset size info
                size_gb = dataset_config.get('size_gb', 0)
                if size_gb > 100.0:
                    progress_callback(f"⚠️  Large dataset: {size_gb:.1f} GB")
                    progress_callback(f"   This may take significant time to download.")
                elif size_gb > 10.0:
                    progress_callback(f"ℹ️  Dataset size: ~{size_gb:.1f} GB (may take a few minutes)")
                else:
                    progress_callback(f"ℹ️  Dataset size: ~{size_gb:.1f} GB")
            
            # Check if dataset is available on HuggingFace
            if dataset_config['hf_id'] is None:
                # Custom download needed
                return self._handle_custom_dataset(dataset_key, dataset_config, progress_callback)
            
            # Download from HuggingFace
            return self._download_from_huggingface(dataset_key, dataset_config, progress_callback)
            
        except Exception as e:
            logger.error(f"Dataset download failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'dataset': dataset_key
            }
    
    def _download_from_huggingface(self, dataset_key: str, config: Dict, progress_callback=None) -> Dict:
        """Download dataset from HuggingFace Hub"""
        try:
            from datasets import load_dataset
            
            hf_id = config['hf_id']
            dataset_dir = self.base_dir / dataset_key
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            if progress_callback:
                progress_callback(f"🔍 Loading dataset from HuggingFace Hub: {hf_id}")
            
            logger.info(f"Loading dataset: {hf_id}")
            
            # Prepare load_dataset parameters
            load_params = {
                'path': hf_id,
                'cache_dir': str(dataset_dir / "cache"),
                'trust_remote_code': True  # Some datasets require custom code
            }
            
            # Add optional config/split parameters
            if 'config' in config:
                load_params['name'] = config['config']
            if 'split' in config:
                load_params['split'] = config['split']
            
            # Download dataset
            dataset = load_dataset(**load_params)
            
            # Save dataset info
            dataset_info = {
                'name': config['name'],
                'type': config['type'],
                'hf_id': hf_id,
                'description': config['description'],
                'size_gb': config.get('size_gb', 0),
                'splits': list(dataset.keys()) if hasattr(dataset, 'keys') else ['default'],
                'num_examples': {split: len(dataset[split]) for split in dataset.keys()} if hasattr(dataset, 'keys') else len(dataset),
                'features': str(dataset[list(dataset.keys())[0]].features) if hasattr(dataset, 'keys') else str(dataset.features),
                'path': str(dataset_dir)
            }
            
            # Save metadata
            metadata_path = dataset_dir / 'dataset_info.json'
            with open(metadata_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            if progress_callback:
                progress_callback(f"✅ Downloaded {config['name']}")
                if hasattr(dataset, 'keys'):
                    for split in dataset.keys():
                        progress_callback(f"   {split}: {len(dataset[split]):,} samples")
                else:
                    progress_callback(f"   Total: {len(dataset):,} samples")
            
            logger.info(f"Dataset downloaded successfully: {dataset_key}")
            
            return {
                'success': True,
                'dataset': dataset_key,
                'info': dataset_info
            }
            
        except ImportError:
            error_msg = "HuggingFace datasets library not installed. Install with: pip install datasets"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'dataset': dataset_key
            }
        except Exception as e:
            error_msg = f"Failed to download {config['name']}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Provide helpful error messages
            if progress_callback:
                progress_callback(f"❌ {error_msg}")
                if "doesn't exist" in str(e).lower() or "not found" in str(e).lower():
                    progress_callback(f"   💡 Dataset '{hf_id}' not found on HuggingFace Hub")
                    progress_callback(f"   Check: https://huggingface.co/datasets/{hf_id}")
                elif "connection" in str(e).lower() or "timeout" in str(e).lower():
                    progress_callback(f"   💡 Network issue - check your internet connection")
                elif "permission" in str(e).lower() or "access" in str(e).lower():
                    progress_callback(f"   💡 Dataset may require authentication or have access restrictions")
                progress_callback(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'dataset': dataset_key
            }
    
    def _handle_custom_dataset(self, dataset_key: str, config: Dict, progress_callback=None) -> Dict:
        """Handle datasets that require custom download"""
        if progress_callback:
            progress_callback(
                f"⚠️ {config['name']} requires manual download\n"
                f"   Visit: {config.get('custom_url', 'N/A')}\n"
                f"   Place files in: training_data/{dataset_key}/"
            )
        
        return {
            'success': False,
            'manual_download_required': True,
            'dataset': dataset_key,
            'url': config.get('custom_url'),
            'info': config
        }
    
    def list_available_datasets(self) -> Dict[str, Dict]:
        """List all available datasets and their configurations"""
        return self.DATASETS
    
    def get_downloaded_datasets(self) -> List[str]:
        """Get list of already downloaded datasets"""
        downloaded = []
        for dataset_key in self.DATASETS.keys():
            dataset_dir = self.base_dir / dataset_key
            metadata_path = dataset_dir / 'dataset_info.json'
            if metadata_path.exists():
                downloaded.append(dataset_key)
        return downloaded
    
    def prepare_for_training(self, dataset_key: str) -> Dict:
        """
        Prepare downloaded dataset for LoRA training
        
        Args:
            dataset_key: Dataset to prepare
            
        Returns:
            Dictionary with prepared dataset info
        """
        try:
            dataset_dir = self.base_dir / dataset_key
            metadata_path = dataset_dir / 'dataset_info.json'
            
            if not metadata_path.exists():
                raise ValueError(f"Dataset not downloaded: {dataset_key}")
            
            with open(metadata_path) as f:
                dataset_info = json.load(f)
            
            # Create prepared dataset directory
            prepared_dir = dataset_dir / "prepared"
            prepared_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Dataset {dataset_key} ready for training")
            
            return {
                'success': True,
                'dataset': dataset_key,
                'path': str(prepared_dir),
                'info': dataset_info
            }
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'dataset': dataset_key
            }
