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
        # Vocal Datasets
        "opensinger": {
            "name": "OpenSinger (Multi-singer, 50+ hours)",
            "hf_id": "Rongjiehuang/opensinger",
            "type": "vocal",
            "description": "Multi-singer dataset with 50+ hours of vocals",
            "splits": ["train", "validation"]
        },
        "m4singer": {
            "name": "M4Singer (Chinese pop, 29 hours)",
            "hf_id": "M4Singer/M4Singer",
            "type": "vocal",
            "description": "Chinese pop music with vocals, 29 hours",
            "splits": ["train"]
        },
        "ccmixter": {
            "name": "CC Mixter (Creative Commons stems)",
            "hf_id": None,  # Not directly on HF, would need custom download
            "type": "vocal",
            "description": "Creative Commons licensed music stems",
            "custom_url": "https://ccmixter.org/",
            "splits": []
        },
        
        # Symbolic Datasets
        "lakh_midi": {
            "name": "Lakh MIDI (176k files, diverse genres)",
            "hf_id": "roszcz/lakh-midi",
            "type": "symbolic",
            "description": "176,000 MIDI files covering diverse genres",
            "splits": ["train"]
        },
        "mutopia": {
            "name": "Mutopia (Classical, 2000+ pieces)",
            "hf_id": None,  # Not directly on HF
            "type": "symbolic",
            "description": "Classical music in various formats",
            "custom_url": "https://www.mutopiaproject.org/",
            "splits": []
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
            dataset_key: Key identifying the dataset (e.g., 'opensinger')
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
                progress_callback(f"Starting download: {dataset_name}")
            
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
                progress_callback(f"Downloading {config['name']} from HuggingFace...")
            
            logger.info(f"Loading dataset: {hf_id}")
            
            # Download dataset
            dataset = load_dataset(
                hf_id,
                cache_dir=str(dataset_dir / "cache"),
                trust_remote_code=True  # Some datasets require custom code
            )
            
            # Save dataset info
            dataset_info = {
                'name': config['name'],
                'type': config['type'],
                'hf_id': hf_id,
                'description': config['description'],
                'splits': list(dataset.keys()),
                'num_examples': {split: len(dataset[split]) for split in dataset.keys()},
                'features': str(dataset[list(dataset.keys())[0]].features),
                'path': str(dataset_dir)
            }
            
            # Save metadata
            metadata_path = dataset_dir / 'dataset_info.json'
            with open(metadata_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            if progress_callback:
                progress_callback(f"✅ Downloaded {config['name']}: {dataset_info['num_examples']}")
            
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
