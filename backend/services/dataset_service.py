"""
Dataset download and preparation service
Downloads curated datasets from HuggingFace for LoRA training
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
import json
from datetime import datetime

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
            
            # Save dataset info for LoRA training compatibility
            dataset_info = {
                'name': config['name'],
                'type': config['type'],
                'hf_id': hf_id,
                'description': config['description'],
                'size_gb': config.get('size_gb', 0),
                'splits': list(dataset.keys()) if hasattr(dataset, 'keys') else ['default'],
                'num_examples': {split: len(dataset[split]) for split in dataset.keys()} if hasattr(dataset, 'keys') else len(dataset),
                'features': str(dataset[list(dataset.keys())[0]].features) if hasattr(dataset, 'keys') else str(dataset.features),
                'path': str(dataset_dir),
                # Add placeholders for LoRA training service compatibility
                'train_files': [],
                'val_files': [],
                'train_metadata': [],
                'val_metadata': [],
                'prepared': False,  # Indicates dataset needs preparation before training
                'hf_dataset': True  # Flag that this is a HuggingFace dataset
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
    
    def prepare_dataset_for_training(
        self, 
        dataset_key: str, 
        train_val_split: float = 0.8,
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Prepare a downloaded HuggingFace dataset for LoRA training.
        Extracts audio files, creates metadata, and splits into train/val sets.
        
        Args:
            dataset_key: Key identifying the dataset (e.g., 'gtzan')
            train_val_split: Fraction of data to use for training (default: 0.8)
            max_samples: Maximum number of samples to prepare (None = all)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with preparation results
        """
        try:
            from datasets import load_from_disk
            import soundfile as sf
            import numpy as np
            
            if progress_callback:
                progress_callback(f"🔧 Preparing dataset: {dataset_key}")
            
            # Check if dataset exists
            if dataset_key not in self.DATASETS:
                raise ValueError(f"Unknown dataset: {dataset_key}")
            
            config = self.DATASETS[dataset_key]
            dataset_dir = self.base_dir / dataset_key
            cache_dir = dataset_dir / "cache"
            audio_dir = dataset_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Load dataset info
            metadata_path = dataset_dir / 'dataset_info.json'
            if not metadata_path.exists():
                raise ValueError(f"Dataset not downloaded yet. Please download {dataset_key} first.")
            
            with open(metadata_path, 'r') as f:
                dataset_info = json.load(f)
            
            if dataset_info.get('prepared'):
                if progress_callback:
                    progress_callback(f"✅ Dataset already prepared!")
                return {'success': True, 'dataset': dataset_key, 'already_prepared': True}
            
            # Load HuggingFace dataset from cache
            if progress_callback:
                progress_callback(f"📂 Loading dataset from cache...")
            
            from datasets import load_dataset
            hf_id = config['hf_id']
            load_params = {'path': hf_id, 'cache_dir': str(cache_dir)}
            if 'config' in config:
                load_params['name'] = config['config']
            if 'split' in config:
                load_params['split'] = config['split']
                
            dataset = load_dataset(**load_params)
            
            # Get the appropriate split
            if hasattr(dataset, 'keys'):
                # Use 'train' split if available, otherwise first available split
                split_name = 'train' if 'train' in dataset.keys() else list(dataset.keys())[0]
                data = dataset[split_name]
            else:
                data = dataset
            
            total_samples = len(data)
            if max_samples:
                total_samples = min(total_samples, max_samples)
            
            if progress_callback:
                progress_callback(f"📊 Processing {total_samples} samples...")
            
            # Determine audio column name (varies by dataset)
            audio_column = None
            for col in ['audio', 'file', 'path', 'wav']:
                if col in data.column_names:
                    audio_column = col
                    break
            
            if not audio_column:
                raise ValueError(f"Could not find audio column in dataset. Available columns: {data.column_names}")
            
            # Process samples
            train_files = []
            val_files = []
            train_metadata = []
            val_metadata = []
            
            num_train = int(total_samples * train_val_split)
            
            for idx in range(total_samples):
                try:
                    sample = data[idx]
                    
                    # Extract audio
                    audio_data = sample[audio_column]
                    
                    # Handle different audio formats
                    if isinstance(audio_data, dict):
                        # Format: {'array': ndarray, 'sampling_rate': int}
                        audio_array = audio_data['array']
                        sample_rate = audio_data['sampling_rate']
                    elif isinstance(audio_data, str):
                        # File path - load it
                        import librosa
                        audio_array, sample_rate = librosa.load(audio_data, sr=None)
                    else:
                        logger.warning(f"Unknown audio format for sample {idx}")
                        continue
                    
                    # Save audio file
                    audio_filename = f"sample_{idx:06d}.wav"
                    audio_path = audio_dir / audio_filename
                    sf.write(audio_path, audio_array, sample_rate)
                    
                    # Create metadata
                    metadata = {
                        'audio_file': str(audio_path),
                        'sample_rate': sample_rate,
                        'duration': len(audio_array) / sample_rate,
                        'dataset': dataset_key,
                        'index': idx
                    }
                    
                    # Extract additional metadata from dataset
                    for key in sample.keys():
                        if key != audio_column and not isinstance(sample[key], (dict, list)):
                            metadata[key] = sample[key]
                    
                    # Add to train or val set
                    if idx < num_train:
                        train_files.append(str(audio_path))
                        train_metadata.append(metadata)
                    else:
                        val_files.append(str(audio_path))
                        val_metadata.append(metadata)
                    
                    # Progress update
                    if progress_callback and (idx + 1) % 50 == 0:
                        progress_callback(f"   Processed {idx + 1}/{total_samples} samples...")
                        
                except Exception as e:
                    logger.warning(f"Error processing sample {idx}: {str(e)}")
                    continue
            
            # Update dataset_info.json with training-ready format
            dataset_info.update({
                'train_files': train_files,
                'val_files': val_files,
                'train_metadata': train_metadata,
                'val_metadata': val_metadata,
                'prepared': True,
                'preparation_date': datetime.now().isoformat(),
                'num_train_samples': len(train_files),
                'num_val_samples': len(val_files),
                'train_val_split': train_val_split
            })
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            if progress_callback:
                progress_callback(f"✅ Dataset prepared successfully!")
                progress_callback(f"   Training samples: {len(train_files)}")
                progress_callback(f"   Validation samples: {len(val_files)}")
                progress_callback(f"   Audio files saved to: {audio_dir}")
            
            logger.info(f"Dataset {dataset_key} prepared: {len(train_files)} train, {len(val_files)} val")
            
            return {
                'success': True,
                'dataset': dataset_key,
                'num_train': len(train_files),
                'num_val': len(val_files),
                'audio_dir': str(audio_dir)
            }
            
        except Exception as e:
            error_msg = f"Failed to prepare dataset {dataset_key}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if progress_callback:
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
