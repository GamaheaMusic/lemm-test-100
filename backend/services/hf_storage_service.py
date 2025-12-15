"""
HuggingFace Collection Storage Service
Uploads LoRA adapters as individual models to HuggingFace Hub
Models can be added to the LEMM collection for organization
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import yaml

logger = logging.getLogger(__name__)

class HFStorageService:
    """Service for uploading LoRAs as models to HuggingFace Hub"""
    
    def __init__(self, username: str = "Gamahea", dataset_repo: str = "lemmdata"):
        """
        Initialize HF storage service
        
        Args:
            username: HuggingFace username
            dataset_repo: Dataset repository name for storing training artifacts
        """
        self.username = username
        self.dataset_repo = dataset_repo
        self.repo_id = f"{username}/{dataset_repo}"
        self.local_cache = Path("hf_cache")
        self.local_cache.mkdir(exist_ok=True)
        
        logger.info(f"HF Storage initialized for user: {username}")
        logger.info(f"Dataset Repo: https://huggingface.co/datasets/{self.repo_id}")
        
        # Get HF token from environment
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        # Try to import huggingface_hub
        try:
            from huggingface_hub import HfApi
            self.api = HfApi(token=self.token) if self.token else HfApi()
            self.has_hf = True
            if self.token:
                logger.info("✅ HuggingFace Hub available with authentication")
            else:
                logger.warning("⚠️ HuggingFace Hub available but no token found (uploads may fail)")
        except ImportError:
            logger.warning("⚠️ huggingface_hub not available, uploads will be skipped")
            self.has_hf = False
    
    def sync_on_startup(self, loras_dir: Path, datasets_dir: Path = None) -> Dict:
        """
        Sync LoRAs and datasets from HuggingFace dataset repo on startup
        Downloads missing LoRAs and datasets from the repo to local storage
        
        Args:
            loras_dir: Local directory for LoRA storage
            datasets_dir: Local directory for dataset storage (optional)
            
        Returns:
            Dict with sync results: {'loras': [...], 'datasets': [...], 'synced': count}
        """
        if not self.has_hf:
            logger.debug("HF not available, skipping sync")
            return {'loras': [], 'datasets': [], 'synced': 0}
        
        try:
            # List LoRAs in dataset repo
            collection_loras = self.list_dataset_loras()
            
            if not collection_loras:
                logger.info("No LoRAs found in dataset repo")
                return {'loras': [], 'datasets': [], 'synced': 0}
            
            logger.info(f"Found {len(collection_loras)} LoRA(s) in dataset repo")
            
            # Check which ones are missing locally
            loras_dir.mkdir(parents=True, exist_ok=True)
            existing_loras = set(d.name for d in loras_dir.iterdir() if d.is_dir())
            
            synced_count = 0
            for lora in collection_loras:
                lora_name = lora['name']
                
                # Handle name conflicts - add number suffix if needed
                final_name = lora_name
                counter = 1
                while final_name in existing_loras:
                    final_name = f"{lora_name}_{counter}"
                    counter += 1
                
                target_dir = loras_dir / final_name
                
                # Download if not present locally
                if not target_dir.exists():
                    logger.info(f"Downloading LoRA from dataset repo: {lora['path']}")
                    if self.download_lora(lora['path'], target_dir):
                        synced_count += 1
                        existing_loras.add(final_name)
                        if final_name != lora_name:
                            logger.info(f"Downloaded as '{final_name}' (name conflict resolved)")
            
            logger.info(f"Synced {synced_count} new LoRA(s) from dataset repo")
            return {'loras': collection_loras, 'datasets': [], 'synced': synced_count}
            
        except Exception as e:
            logger.error(f"Sync failed: {str(e)}", exc_info=True)
            return {'loras': [], 'datasets': [], 'synced': 0, 'error': str(e)}
    
    def list_dataset_loras(self) -> List[Dict[str, str]]:
        """
        List all LoRA ZIP files stored in the dataset repo
        
        Returns:
            List of dicts with 'name' and 'path'
        """
        if not self.has_hf:
            logger.debug("HF not available, skipping dataset list")
            return []
        
        try:
            from huggingface_hub import list_repo_files
            
            # List all files in the loras/ folder
            files = list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            
            # Extract LoRA names from ZIP files in loras/ folder
            loras = []
            for file in files:
                if file.startswith("loras/") and file.endswith(".zip"):
                    # Extract name from "loras/name.zip"
                    lora_name = file[6:-4]  # Remove "loras/" and ".zip"
                    loras.append({
                        'name': lora_name,
                        'path': f"loras/{lora_name}"
                    })
            
            logger.info(f"Found {len(loras)} LoRA(s) in dataset repo")
            return loras
            
        except Exception as e:
            logger.error(f"Failed to list dataset LoRAs: {e}")
            return []
    
    def download_lora(self, lora_path: str, target_dir: Path) -> bool:
        """
        Download a LoRA ZIP file from dataset repo and extract it
        
        Args:
            lora_path: Path within dataset repo (e.g., "loras/jazz-v1")
            target_dir: Local directory to extract to
            
        Returns:
            True if successful
        """
        if not self.has_hf:
            logger.debug("HF not available, skipping download")
            return False
        
        try:
            from huggingface_hub import hf_hub_download
            import zipfile
            import tempfile
            
            # Expect ZIP file
            lora_name = lora_path.split('/')[-1]
            zip_filename = f"loras/{lora_name}.zip"
            
            logger.info(f"Downloading LoRA ZIP from {self.repo_id}/{zip_filename}...")
            
            # Download ZIP file to temp location
            zip_path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=zip_filename,
                token=self.token
            )
            
            # Extract to target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(target_dir)
            
            logger.info(f"✅ Downloaded and extracted LoRA to {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download LoRA: {e}")
            return False
    
    def upload_lora(self, lora_dir: Path, training_config: Optional[Dict] = None) -> Optional[Dict]:
        """
        Upload a LoRA adapter as a ZIP file to HuggingFace dataset repo
        
        Args:
            lora_dir: Local LoRA directory
            training_config: Optional training configuration dict
            
        Returns:
            Dict with repo_id and url if successful, None otherwise
        """
        if not self.has_hf:
            logger.info(f"💾 LoRA saved locally: {lora_dir.name}")
            return None
        
        if not self.token:
            logger.warning("⚠️ No HuggingFace token found - cannot upload")
            logger.info("💡 To enable uploads: Log in to HuggingFace or set HF_TOKEN environment variable")
            logger.info(f"💾 LoRA saved locally: {lora_dir.name}")
            return None
        
        try:
            from huggingface_hub import upload_file
            import zipfile
            import tempfile
            
            lora_name = lora_dir.name
            
            logger.info(f"📤 Creating ZIP and uploading LoRA to dataset repo: {self.repo_id}/loras/{lora_name}.zip...")
            
            # Create README.md for the LoRA
            readme_content = self._generate_lora_readme(lora_name, training_config)
            readme_path = lora_dir / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            # Create ZIP file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.zip', delete=False) as tmp_file:
                zip_path = tmp_file.name
            
            try:
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in lora_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(lora_dir)
                            zipf.write(file_path, arcname)
                
                # Upload ZIP file to loras/ folder in dataset repo
                upload_file(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    path_or_fileobj=zip_path,
                    path_in_repo=f"loras/{lora_name}.zip",
                    commit_message=f"Upload LEMM LoRA adapter: {lora_name}",
                    token=self.token
                )
            finally:
                # Clean up temp file
                import os
                if os.path.exists(zip_path):
                    os.unlink(zip_path)
            
            logger.info(f"✅ Uploaded LoRA: {self.repo_id}/loras/{lora_name}.zip")
            logger.info(f"🔗 View at: https://huggingface.co/datasets/{self.repo_id}/blob/main/loras/{lora_name}.zip")
            
            return {
                'repo_id': f"{self.repo_id}/loras/{lora_name}.zip",
                'url': f"https://huggingface.co/datasets/{self.repo_id}/blob/main/loras/{lora_name}.zip",
                'dataset_repo': f"https://huggingface.co/datasets/{self.repo_id}"
            }
            
        except Exception as e:
            logger.error(f"Failed to upload LoRA: {e}")
            logger.info(f"💾 LoRA saved locally: {lora_dir.name}")
            return None
    
    def _generate_lora_readme(self, lora_name: str, config: Optional[Dict] = None) -> str:
        """Generate README.md content for a LoRA model"""
        
        config_info = ""
        if config:
            config_info = f"""
## Training Configuration

- **Dataset**: {config.get('dataset', 'N/A')}
- **Epochs**: {config.get('epochs', 'N/A')}
- **Learning Rate**: {config.get('learning_rate', 'N/A')}
- **Batch Size**: {config.get('batch_size', 'N/A')}
- **LoRA Rank**: {config.get('lora_rank', 'N/A')}
"""
        
        return f"""---
license: mit
tags:
- lora
- music-generation
- diffrhythm2
- lemm
library_name: diffusers
---

# LEMM LoRA: {lora_name}

This is a LoRA (Low-Rank Adaptation) adapter for DiffRhythm2 music generation, trained using LEMM (Let Everyone Make Music).

## About LEMM

LEMM is an advanced AI music generation system that allows you to:
- Generate high-quality music with built-in vocals
- Train custom LoRA adapters for specific styles
- Fine-tune models on your own datasets

🎵 **Try it**: [LEMM Space](https://huggingface.co/spaces/Gamahea/lemm-test-100)
{config_info}
## How to Use

### In LEMM Space
1. Visit [LEMM](https://huggingface.co/spaces/Gamahea/lemm-test-100)
2. Go to "LoRA Management" tab
3. Enter this model ID: `{self.username}/lemm-lora-{lora_name}`
4. Click "Download from Hub"
5. Use in generation or as base for continued training

### In Your Code
```python
from pathlib import Path
from huggingface_hub import snapshot_download

# Download LoRA
lora_path = snapshot_download(
    repo_id="{self.username}/lemm-lora-{lora_name}",
    local_dir="./loras/{lora_name}"
)

# Load and use with DiffRhythm2
# (See LEMM documentation for integration)
```

## Model Files

- `final_model.pt` - Trained LoRA weights
- `config.yaml` - Training configuration
- `README.md` - This file

## Dataset Repository

Part of the [LEMM Training Data Repository](https://huggingface.co/datasets/{self.repo_id})

## License

MIT License - Free to use and modify
"""
    
    def upload_dataset(self, dataset_dir: Path, dataset_info: Optional[Dict] = None) -> Optional[Dict]:
        """
        Upload a prepared dataset to HF dataset repo
        
        Args:
            dataset_dir: Local dataset directory
            dataset_info: Optional dataset metadata
            
        Returns:
            Dict with upload results or None if failed
        \"\"\"\n        if not self.has_hf:\n            logger.info(f\"💾 Dataset saved locally: {dataset_dir.name}\")\n            return None\n        \n        if not self.token:\n            logger.warning(\"⚠️ No HuggingFace token found - cannot upload dataset\")\n            logger.info(f\"💾 Dataset saved locally: {dataset_dir.name}\")\n            return None\n        \n        try:\n            from huggingface_hub import upload_folder\n            \n            dataset_name = dataset_dir.name\n            \n            logger.info(f\"📤 Uploading dataset to repo: {self.repo_id}/datasets/{dataset_name}...\")\n            \n            # Upload to datasets/ folder in dataset repo\n            upload_folder(\n                repo_id=self.repo_id,\n                repo_type=\"dataset\",\n                folder_path=str(dataset_dir),\n                path_in_repo=f\"datasets/{dataset_name}\",\n                commit_message=f\"Upload prepared dataset: {dataset_name}\",\n                token=self.token\n            )\n            \n            logger.info(f\"✅ Uploaded dataset: {self.repo_id}/datasets/{dataset_name}\")\n            \n            return {\n                'repo_id': f\"{self.repo_id}/datasets/{dataset_name}\",\n                'url': f\"https://huggingface.co/datasets/{self.repo_id}/tree/main/datasets/{dataset_name}\",\n                'dataset_repo': f\"https://huggingface.co/datasets/{self.repo_id}\"\n            }\n            \n        except Exception as e:\n            logger.error(f\"Failed to upload dataset: {e}\")\n            logger.info(f\"💾 Dataset saved locally: {dataset_dir.name}\")\n            return None
