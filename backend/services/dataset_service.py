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
    
    # Dataset configurations (Parquet format only - no loading scripts)
    DATASETS = {
        'gtzan': {
            'name': 'GTZAN Music Genre Dataset',
            'type': 'music',
            'hf_id': 'lewtun/music_genres_small',
            'description': 'Music genre classification dataset (GTZAN-based)',
            'size_gb': 1.2
        },
        'nsynth': {
            'name': 'NSynth Complete Dataset',
            'type': 'music',
            'hf_id': 'Loie/NSynth',
            'description': 'Musical note dataset with pitch and timbre variations',
            'size_gb': 30.0
        },
        'maestro': {
            'name': 'MAESTRO Piano Performances',
            'type': 'music',
            'hf_id': 'roszcz/maestro-sustain-v2.0.0',
            'description': 'Classical piano performances - Parquet format',
            'size_gb': 120.0
        },
        'million_song': {
            'name': 'Million Song Dataset (10k Subset)',
            'type': 'music',
            'hf_id': 'maharshipandya/spotify-tracks-dataset',
            'description': 'Large music dataset with audio features',
            'size_gb': 1.8
        },
        'fma_large': {
            'name': 'Free Music Archive - Large',
            'type': 'music',
            'hf_id': 'rudraml/fma-small',
            'description': 'Free Music Archive dataset - 8k tracks, 8 genres',
            'size_gb': 7.2
        },
        'ljspeech': {
            'name': 'LJSpeech Vocal Dataset',
            'type': 'vocal',
            'hf_id': 'keithito/lj-speech',
            'description': 'Single speaker vocal dataset with 13,100 clips - Parquet format',
            'size_gb': 2.6
        },
        'common_voice_en': {
            'name': 'Common Voice English (Complete)',
            'type': 'vocal',
            'hf_id': 'mozilla-foundation/common_voice_17_0',
            'config': 'en',
            'description': 'Large-scale multilingual speech corpus - latest version',
            'size_gb': 75.0
        },
        'librispeech': {
            'name': 'LibriSpeech Complete',
            'type': 'vocal',
            'hf_id': 'openslr/librispeech_asr',
            'description': 'Audiobooks dataset - 1,000 hours from multiple speakers',
            'size_gb': 60.0
        },
        'musiccaps': {
            'name': 'MusicCaps',
            'type': 'music',
            'hf_id': 'google/MusicCaps',
            'description': 'Music audio captioning with 5.5k clips and text descriptions',
            'size_gb': 5.5
        },
        'audioset_music': {
            'name': 'AudioSet Strong',
            'type': 'music',
            'hf_id': 'agkphysics/AudioSet',
            'description': 'High-quality labeled audio events',
            'size_gb': 12.0
        },
        'esc50': {
            'name': 'ESC-50 Environmental Sounds',
            'type': 'sound_effects',
            'hf_id': 'ashraq/esc50',
            'description': 'Environmental sound classification with 2,000 recordings',
            'size_gb': 0.6
        },
        'speech_commands': {
            'name': 'Google Speech Commands',
            'type': 'vocal',
            'hf_id': 'google/speech_commands',
            'config': 'v0.02',
            'description': 'Short spoken words for keyword detection',
            'size_gb': 2.0
        }
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
                'cache_dir': str(dataset_dir / "cache")
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
