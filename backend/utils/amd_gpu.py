"""
AMD GPU Detection and Configuration
Enables DirectML support for AMD GPUs (including Vega 8)
Note: DirectML may not be compatible with Python 3.13+ - CPU fallback will be used
"""
import os
import logging
import torch

logger = logging.getLogger(__name__)

def setup_amd_gpu():
    """
    Configure DirectML for AMD GPU support
    Returns device to use for model inference
    """
    try:
        # Check if torch-directml is available
        try:
            import torch_directml
            
            # Get DirectML device
            if torch_directml.is_available():
                device = torch_directml.device()
                logger.info(f"✅ AMD GPU detected via DirectML")
                logger.info(f"Device: {device}")
                
                # Set default device
                torch.set_default_device(device)
                
                return device
            else:
                logger.warning("DirectML available but no compatible GPU found")
                return torch.device("cpu")
        except ImportError:
            logger.warning("torch-directml not available (may not support Python 3.13+)")
            logger.info("Using CPU mode - consider Python 3.11 for DirectML support")
            return torch.device("cpu")
            
    except Exception as e:
        logger.error(f"Error setting up AMD GPU: {str(e)}")
        return torch.device("cpu")

def get_device_info():
    """Get detailed information about available compute devices"""
    info = {
        "device": "cpu",
        "device_name": "CPU",
        "directml_available": False,
        "cuda_available": torch.cuda.is_available()
    }
    
    try:
        try:
            import torch_directml
            
            if torch_directml.is_available():
                info["directml_available"] = True
                info["device"] = "directml"
                info["device_name"] = "AMD GPU (DirectML)"
                
                # Get device
                device = torch_directml.device()
                info["device_object"] = device
        except ImportError:
            logger.info("DirectML not available - Python 3.13+ may not support torch-directml")
            logger.info("For AMD GPU support, consider using Python 3.11 with torch-directml")
            
    except Exception as e:
        logger.error(f"Error getting device info: {str(e)}")
    
    return info

def optimize_for_amd():
    """Apply optimizations for AMD GPU inference"""
    try:
        # Disable CUDA if present (prefer DirectML for AMD)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Set DirectML memory management
        os.environ["PYTORCH_DIRECTML_FORCE_FP32_OPS"] = "0"  # Allow FP16
        
        # Enable TensorFloat-32 for better performance
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        logger.info("✅ AMD GPU optimizations applied")
        
    except Exception as e:
        logger.error(f"Error applying AMD optimizations: {str(e)}")

# Auto-configure on module import
DEFAULT_DEVICE = setup_amd_gpu()
optimize_for_amd()

logger.info(f"Default compute device: {DEFAULT_DEVICE}")
