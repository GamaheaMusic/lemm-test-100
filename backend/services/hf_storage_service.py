"""
HuggingFace Dataset Repository Storage Service
Stores and retrieves training data and LoRA adapters from HF dataset repo
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import shutil

logger = logging.getLogger(__name__)

class HFStorageService:
    """Service for storing/retrieving data from HuggingFace dataset repo"""
    
    def __init__(self, repo_id: str = "Gamahea/lemm-dataset"):
        """
        Initialize HF storage service
        
        Args:
            repo_id: HuggingFace dataset repository ID
        """
        self.repo_id = repo_id
        self.local_cache = Path("hf_cache")
        self.local_cache.mkdir(exist_ok=True)
        
        logger.info(f"HF Storage initialized for repo: {repo_id}")
        
        # Try to import huggingface_hub
        try:
            from huggingface_hub import HfApi, hf_hub_download, upload_folder
            self.api = HfApi()
            self.has_hf = True
            logger.info("✅ HuggingFace Hub available")
        except ImportError:
            logger.warning("⚠️ huggingface_hub not available, using local storage only")
            self.has_hf = False
    
    def download_all_loras(self, target_dir: Path) -> List[str]:
        """
        Download all LoRA adapters from HF repo
        
        Args:
            target_dir: Local directory to download to
            
        Returns:
            List of downloaded LoRA names
        """
        if not self.has_hf:
            logger.warning("HuggingFace Hub not available")
            return []
        
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading LoRAs from {self.repo_id}/loras...")
            
            # Download loras folder
            loras_path = snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                allow_patterns="loras/*",
                local_dir=self.local_cache,
                local_dir_use_symlinks=False
            )
            
            # Copy to target directory
            source_loras = Path(loras_path) / "loras"
            if source_loras.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                
                downloaded = []
                for lora_dir in source_loras.iterdir():
                    if lora_dir.is_dir():
                        dest = target_dir / lora_dir.name
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(lora_dir, dest)
                        downloaded.append(lora_dir.name)
                        logger.info(f"Downloaded LoRA: {lora_dir.name}")
                
                return downloaded
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to download LoRAs: {e}")
            return []
    
    def download_all_datasets(self, target_dir: Path) -> List[str]:
        """
        Download all prepared datasets from HF repo
        
        Args:
            target_dir: Local directory to download to
            
        Returns:
            List of downloaded dataset keys
        """
        if not self.has_hf:
            logger.warning("HuggingFace Hub not available")
            return []
        
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading datasets from {self.repo_id}/datasets...")
            
            # Download datasets folder
            datasets_path = snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                allow_patterns="datasets/*",
                local_dir=self.local_cache,
                local_dir_use_symlinks=False
            )
            
            # Copy to target directory
            source_datasets = Path(datasets_path) / "datasets"
            if source_datasets.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                
                downloaded = []
                for dataset_dir in source_datasets.iterdir():
                    if dataset_dir.is_dir():
                        dest = target_dir / dataset_dir.name
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(dataset_dir, dest)
                        downloaded.append(dataset_dir.name)
                        logger.info(f"Downloaded dataset: {dataset_dir.name}")
                
                return downloaded
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to download datasets: {e}")
            return []
    
    def upload_lora(self, lora_dir: Path) -> bool:
        """
        Upload a LoRA adapter to HF repo
        
        Args:
            lora_dir: Local LoRA directory
            
        Returns:
            True if successful
        """
        if not self.has_hf:
            logger.warning("HuggingFace Hub not available")
            return False
        
        try:
            from huggingface_hub import upload_folder
            
            logger.info(f"Uploading LoRA {lora_dir.name} to {self.repo_id}...")
            
            upload_folder(
                repo_id=self.repo_id,
                repo_type="dataset",
                folder_path=str(lora_dir),
                path_in_repo=f"loras/{lora_dir.name}",
                commit_message=f"Add/Update LoRA: {lora_dir.name}"
            )
            
            logger.info(f"✅ Uploaded LoRA: {lora_dir.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload LoRA: {e}")
            return False
    
    def upload_dataset(self, dataset_dir: Path) -> bool:
        """
        Upload a prepared dataset to HF repo
        
        Args:
            dataset_dir: Local dataset directory
            
        Returns:
            True if successful
        """
        if not self.has_hf:
            logger.warning("HuggingFace Hub not available")
            return False
        
        try:
            from huggingface_hub import upload_folder
            
            logger.info(f"Uploading dataset {dataset_dir.name} to {self.repo_id}...")
            
            upload_folder(
                repo_id=self.repo_id,
                repo_type="dataset",
                folder_path=str(dataset_dir),
                path_in_repo=f"datasets/{dataset_dir.name}",
                commit_message=f"Add/Update dataset: {dataset_dir.name}"
            )
            
            logger.info(f"✅ Uploaded dataset: {dataset_dir.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            return False
    
    def sync_on_startup(self, loras_dir: Path, datasets_dir: Path) -> Dict[str, List[str]]:
        """
        Sync data from HF repo on app startup
        
        Args:
            loras_dir: Local LoRA directory
            datasets_dir: Local datasets directory
            
        Returns:
            Dict with 'loras' and 'datasets' lists
        """
        result = {'loras': [], 'datasets': []}
        
        logger.info("🔄 Syncing from HuggingFace repo...")
        
        # Download LoRAs
        loras = self.download_all_loras(loras_dir)
        result['loras'] = loras
        
        # Download datasets
        datasets = self.download_all_datasets(datasets_dir)
        result['datasets'] = datasets
        
        logger.info(f"✅ Sync complete: {len(loras)} LoRAs, {len(datasets)} datasets")
        
        return result
