"""
LoRA Training Service
Handles fine-tuning of DiffRhythm2 model using LoRA adapters for vocal and symbolic music.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Callable
import soundfile as sf
import numpy as np
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingDataset(Dataset):
    """Dataset for LoRA training"""
    
    def __init__(
        self, 
        audio_files: List[str], 
        metadata_list: List[Dict],
        sample_rate: int = 44100,
        clip_length: float = 10.0
    ):
        """
        Initialize training dataset
        
        Args:
            audio_files: List of paths to audio files
            metadata_list: List of metadata dicts for each audio file
            sample_rate: Target sample rate
            clip_length: Length of training clips in seconds
        """
        self.audio_files = audio_files
        self.metadata_list = metadata_list
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.clip_samples = int(clip_length * sample_rate)
        
        logger.info(f"Initialized dataset with {len(audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """Get training sample"""
        try:
            audio_path = self.audio_files[idx]
            metadata = self.metadata_list[idx]
            
            # Load audio
            y, sr = sf.read(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            
            # Ensure mono
            if y.ndim > 1:
                y = y.mean(axis=1)
            
            # Extract/pad to clip length
            if len(y) > self.clip_samples:
                # Random crop
                start = np.random.randint(0, len(y) - self.clip_samples)
                y = y[start:start + self.clip_samples]
            else:
                # Pad
                y = np.pad(y, (0, self.clip_samples - len(y)))
            
            # Generate prompt from metadata
            prompt = self._generate_prompt(metadata)
            
            return {
                'audio': torch.FloatTensor(y),
                'prompt': prompt,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return empty sample on error
            return {
                'audio': torch.zeros(self.clip_samples),
                'prompt': "",
                'metadata': {}
            }
    
    def _generate_prompt(self, metadata: Dict) -> str:
        """Generate text prompt from metadata"""
        parts = []
        
        if 'genre' in metadata and metadata['genre'] != 'unknown':
            parts.append(metadata['genre'])
        
        if 'instrumentation' in metadata:
            parts.append(f"with {metadata['instrumentation']}")
        
        if 'bpm' in metadata:
            parts.append(f"at {metadata['bpm']} BPM")
        
        if 'key' in metadata:
            parts.append(f"in {metadata['key']}")
        
        if 'mood' in metadata:
            parts.append(f"{metadata['mood']} mood")
        
        if 'description' in metadata:
            parts.append(metadata['description'])
        
        return " ".join(parts) if parts else "music"


class LoRATrainingService:
    """Service for training LoRA adapters for DiffRhythm2"""
    
    def __init__(self):
        """Initialize LoRA training service"""
        self.models_dir = Path("models")
        self.lora_dir = self.models_dir / "loras"
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_data_dir = Path("training_data")
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Training state
        self.is_training = False
        self.current_epoch = 0
        self.current_step = 0
        self.training_loss = []
        self.training_config = None
        
        logger.info(f"LoRATrainingService initialized on {self.device}")
    
    def prepare_dataset(
        self,
        dataset_name: str,
        audio_files: List[str],
        metadata_list: List[Dict],
        split_ratio: float = 0.9
    ) -> Dict:
        """
        Prepare and save training dataset
        
        Args:
            dataset_name: Name for this dataset
            audio_files: List of audio file paths
            metadata_list: List of metadata for each file
            split_ratio: Train/validation split ratio
            
        Returns:
            Dataset information dictionary
        """
        try:
            logger.info(f"Preparing dataset: {dataset_name}")
            
            # Create dataset directory
            dataset_dir = self.training_data_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Split into train/val
            num_samples = len(audio_files)
            num_train = int(num_samples * split_ratio)
            
            indices = np.random.permutation(num_samples)
            train_indices = indices[:num_train]
            val_indices = indices[num_train:]
            
            # Save metadata
            dataset_info = {
                'name': dataset_name,
                'created': datetime.now().isoformat(),
                'num_samples': num_samples,
                'num_train': num_train,
                'num_val': num_samples - num_train,
                'train_files': [audio_files[i] for i in train_indices],
                'train_metadata': [metadata_list[i] for i in train_indices],
                'val_files': [audio_files[i] for i in val_indices],
                'val_metadata': [metadata_list[i] for i in val_indices]
            }
            
            # Save to disk
            metadata_path = dataset_dir / "dataset_info.json"
            with open(metadata_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            logger.info(f"Dataset prepared: {num_train} train, {num_samples - num_train} val samples")
            return dataset_info
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {str(e)}")
            raise
    
    def load_dataset(self, dataset_name: str) -> Optional[Dict]:
        """Load prepared dataset information"""
        try:
            dataset_dir = self.training_data_dir / dataset_name
            metadata_path = dataset_dir / "dataset_info.json"
            
            if not metadata_path.exists():
                logger.warning(f"Dataset not found: {dataset_name}")
                return None
            
            with open(metadata_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
            return None
    
    def list_datasets(self) -> List[str]:
        """List available prepared datasets"""
        try:
            datasets = []
            for dataset_dir in self.training_data_dir.iterdir():
                if dataset_dir.is_dir() and (dataset_dir / "dataset_info.json").exists():
                    datasets.append(dataset_dir.name)
            return datasets
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
            return []
    
    def list_loras(self) -> List[str]:
        """List available LoRA adapters"""
        try:
            loras = []
            if not self.lora_dir.exists():
                return loras
            
            for lora_path in self.lora_dir.iterdir():
                if lora_path.is_dir():
                    # Check for adapter files
                    if (lora_path / "adapter_config.json").exists():
                        loras.append(lora_path.name)
                    # Also check for .safetensors or .bin files
                    elif list(lora_path.glob("*.safetensors")) or list(lora_path.glob("*.bin")):
                        loras.append(lora_path.name)
            
            return sorted(loras)
        except Exception as e:
            logger.error(f"Failed to list LoRAs: {str(e)}")
            return []
    
    def train_lora(
        self,
        dataset_name: str,
        lora_name: str,
        training_type: str = "vocal",  # "vocal" or "symbolic"
        config: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Train LoRA adapter
        
        Args:
            dataset_name: Name of prepared dataset
            lora_name: Name for the LoRA adapter
            training_type: Type of training ("vocal" or "symbolic")
            config: Training configuration (batch_size, learning_rate, etc.)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Training results dictionary
        """
        try:
            if self.is_training:
                raise RuntimeError("Training already in progress")
            
            self.is_training = True
            logger.info(f"Starting LoRA training: {lora_name} ({training_type})")
            
            # Load dataset
            dataset_info = self.load_dataset(dataset_name)
            if not dataset_info:
                raise ValueError(f"Dataset not found: {dataset_name}")
            
            # Check if dataset is from HuggingFace and needs preparation
            if dataset_info.get('hf_dataset') and not dataset_info.get('prepared'):
                raise ValueError(
                    f"Dataset '{dataset_name}' is a HuggingFace dataset that hasn't been prepared for training yet. "
                    f"Please use the 'User Audio Training' tab to upload and prepare your own audio files, "
                    f"or wait for dataset preparation features to be implemented."
                )
            
            # Validate dataset has required fields
            if 'train_files' not in dataset_info or 'val_files' not in dataset_info:
                raise ValueError(
                    f"Dataset '{dataset_name}' is missing required training files. "
                    f"Please use prepared datasets or upload your own audio in the 'User Audio Training' tab."
                )
            
            # Validate datasets are not empty
            if not dataset_info['train_files'] or len(dataset_info['train_files']) == 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' has no training samples. "
                    f"The dataset may not have been prepared correctly. "
                    f"Please re-prepare the dataset or use a different one."
                )
            
            if not dataset_info['val_files'] or len(dataset_info['val_files']) == 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' has no validation samples. "
                    f"The dataset may not have been prepared correctly. "
                    f"Please re-prepare the dataset or use a different one."
                )
            
            # Default config
            default_config = {
                'batch_size': 4,
                'learning_rate': 3e-4,
                'num_epochs': 10,
                'lora_rank': 16,
                'lora_alpha': 32,
                'warmup_steps': 100,
                'save_every': 500,
                'gradient_accumulation': 2
            }
            
            self.training_config = {**default_config, **(config or {})}
            
            # Create datasets
            train_dataset = TrainingDataset(
                dataset_info['train_files'],
                dataset_info['train_metadata']
            )
            
            val_dataset = TrainingDataset(
                dataset_info['val_files'],
                dataset_info['val_metadata']
            )
            
            # Create data loaders
            # Disable pin_memory and num_workers for compatibility with ZeroGPU and CPU
            # pin_memory requires persistent CUDA access which ZeroGPU doesn't provide at this stage
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # Initialize model (placeholder - actual implementation would load DiffRhythm2)
            # For now, we'll simulate training
            logger.info("Initializing model and LoRA layers...")
            
            # Note: Actual implementation would:
            # 1. Load DiffRhythm2 model
            # 2. Add LoRA adapters using peft library
            # 3. Freeze base model, only train LoRA parameters
            
            # Simulated training loop
            num_steps = len(train_loader) * self.training_config['num_epochs']
            logger.info(f"Training for {self.training_config['num_epochs']} epochs, {num_steps} total steps")
            
            results = self._training_loop(
                train_loader,
                val_loader,
                lora_name,
                progress_callback
            )
            
            self.is_training = False
            logger.info("Training complete!")
            
            return results
            
        except Exception as e:
            self.is_training = False
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def _training_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lora_name: str,
        progress_callback: Optional[Callable]
    ) -> Dict:
        """
        Main training loop
        
        Note: This is a simplified placeholder implementation.
        Actual implementation would require:
        1. Loading DiffRhythm2 model
        2. Setting up LoRA adapters with peft library
        3. Implementing proper loss functions
        4. Gradient accumulation and optimization
        """
        
        self.current_epoch = 0
        self.current_step = 0
        self.training_loss = []
        best_val_loss = float('inf')
        
        num_epochs = self.training_config['num_epochs']
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_loss = 0.0
            
            logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
            
            # Training phase
            for batch_idx, batch in enumerate(train_loader):
                self.current_step += 1
                
                # Simulate training step
                # Actual implementation would:
                # 1. Move batch to device
                # 2. Forward pass through model
                # 3. Calculate loss
                # 4. Backward pass
                # 5. Update weights
                
                # Simulated loss (decreasing over time)
                step_loss = 1.0 / (1.0 + self.current_step * 0.01)
                epoch_loss += step_loss
                self.training_loss.append(step_loss)
                
                # Progress update
                if progress_callback and batch_idx % 10 == 0:
                    progress_callback({
                        'epoch': self.current_epoch,
                        'step': self.current_step,
                        'loss': step_loss,
                        'progress': (self.current_step / (len(train_loader) * num_epochs)) * 100
                    })
                
                # Log every 50 steps
                if self.current_step % 50 == 0:
                    logger.info(f"Step {self.current_step}: Loss = {step_loss:.4f}")
                
                # Save checkpoint
                if self.current_step % self.training_config['save_every'] == 0:
                    self._save_checkpoint(lora_name, self.current_step)
            
            # Validation phase
            avg_train_loss = epoch_loss / len(train_loader)
            val_loss = self._validate(val_loader)
            
            logger.info(f"Epoch {self.current_epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_lora_adapter(lora_name, is_best=True)
                logger.info(f"New best model! Val Loss: {val_loss:.4f}")
        
        # Final save
        self._save_lora_adapter(lora_name, is_best=False)
        
        return {
            'lora_name': lora_name,
            'num_epochs': num_epochs,
            'total_steps': self.current_step,
            'final_train_loss': avg_train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'training_time': 'simulated'
        }
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation"""
        total_loss = 0.0
        
        for batch in val_loader:
            # Simulate validation
            # Actual implementation would run model inference
            val_loss = 1.0 / (1.0 + self.current_step * 0.01)
            total_loss += val_loss
        
        return total_loss / len(val_loader)
    
    def _save_checkpoint(self, lora_name: str, step: int):
        """Save training checkpoint"""
        checkpoint_dir = self.lora_dir / lora_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        # Actual implementation would save:
        # - LoRA weights
        # - Optimizer state
        # - Training step
        # - Config
        
        checkpoint_data = {
            'step': step,
            'epoch': self.current_epoch,
            'config': self.training_config,
            'loss_history': self.training_loss[-100:]  # Last 100 steps
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved checkpoint: step_{step}")
    
    def _save_lora_adapter(self, lora_name: str, is_best: bool = False):
        """Save final LoRA adapter"""
        lora_path = self.lora_dir / lora_name
        lora_path.mkdir(parents=True, exist_ok=True)
        
        filename = "best_model.pt" if is_best else "final_model.pt"
        save_path = lora_path / filename
        
        # Actual implementation would save:
        # - LoRA adapter weights only
        # - Configuration
        # - Training metadata
        
        adapter_data = {
            'lora_name': lora_name,
            'config': self.training_config,
            'training_steps': self.current_step,
            'saved_at': datetime.now().isoformat()
        }
        
        torch.save(adapter_data, save_path)
        logger.info(f"Saved LoRA adapter: {filename}")
        
        # Save metadata
        metadata_path = lora_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(adapter_data, f, indent=2)
    
    def list_lora_adapters(self) -> List[Dict]:
        """List available LoRA adapters"""
        try:
            adapters = []
            
            for lora_dir in self.lora_dir.iterdir():
                if lora_dir.is_dir():
                    metadata_path = lora_dir / "metadata.json"
                    
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            adapters.append({
                                'name': lora_dir.name,
                                **metadata
                            })
                    else:
                        # Basic info if no metadata
                        adapters.append({
                            'name': lora_dir.name,
                            'has_best': (lora_dir / "best_model.pt").exists(),
                            'has_final': (lora_dir / "final_model.pt").exists()
                        })
            
            return adapters
            
        except Exception as e:
            logger.error(f"Failed to list LoRA adapters: {str(e)}")
            return []
    
    def delete_lora_adapter(self, lora_name: str) -> bool:
        """Delete a LoRA adapter"""
        try:
            import shutil
            
            lora_path = self.lora_dir / lora_name
            
            if lora_path.exists():
                shutil.rmtree(lora_path)
                logger.info(f"Deleted LoRA adapter: {lora_name}")
                return True
            else:
                logger.warning(f"LoRA adapter not found: {lora_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete LoRA adapter {lora_name}: {str(e)}")
            return False
    
    def stop_training(self):
        """Stop current training"""
        if self.is_training:
            logger.info("Training stop requested")
            self.is_training = False
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'recent_loss': self.training_loss[-10:] if self.training_loss else [],
            'config': self.training_config
        }
