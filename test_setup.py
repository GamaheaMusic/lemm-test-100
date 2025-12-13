"""
Test script to verify AMD GPU and DirectML setup
"""
import sys

print("=" * 60)
print("  Music Generation App - System Check")
print("=" * 60)
print()

# Test 1: Python version
print("✓ Testing Python version...")
print(f"  Python {sys.version.split()[0]}")
print()

# Test 2: PyTorch
print("✓ Testing PyTorch...")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"  ❌ PyTorch not found: {e}")
    sys.exit(1)
print()

# Test 3: DirectML
print("✓ Testing DirectML (AMD GPU support)...")
try:
    import torch_directml
    print(f"  torch-directml installed: Yes")
    
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"  DirectML available: ✅ YES")
        print(f"  Device: {device}")
        
        # Test tensor creation
        test_tensor = torch.randn(3, 3).to(device)
        print(f"  GPU tensor test: ✅ PASSED")
    else:
        print(f"  DirectML available: ⚠️ NO (will use CPU)")
        print(f"  Note: App will work but slower on CPU")
except ImportError:
    print(f"  ❌ torch-directml not installed")
    print(f"  Run: pip install torch-directml")
except Exception as e:
    print(f"  ⚠️ DirectML error: {e}")
    print(f"  App will fallback to CPU")
print()

# Test 4: Gradio
print("✓ Testing Gradio...")
try:
    import gradio as gr
    print(f"  Gradio version: {gr.__version__}")
except ImportError as e:
    print(f"  ❌ Gradio not found: {e}")
    print(f"  Run: pip install gradio")
    sys.exit(1)
print()

# Test 5: Audio libraries
print("✓ Testing audio libraries...")
try:
    import soundfile
    print(f"  soundfile: ✅")
except ImportError:
    print(f"  ❌ soundfile not found")

try:
    import librosa
    print(f"  librosa: ✅")
except ImportError:
    print(f"  ⚠️ librosa not found (optional)")

try:
    import scipy
    print(f"  scipy: ✅")
except ImportError:
    print(f"  ❌ scipy not found")
print()

# Test 6: Transformers
print("✓ Testing Transformers library...")
try:
    import transformers
    print(f"  transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"  ❌ transformers not found: {e}")
print()

# Test 7: Hugging Face Hub
print("✓ Testing Hugging Face Hub...")
try:
    import huggingface_hub
    print(f"  huggingface-hub: ✅")
except ImportError:
    print(f"  ❌ huggingface-hub not found")
print()

# Test 8: Check models directory
print("✓ Checking models directory...")
import os
from pathlib import Path

models_dir = Path("models")
if models_dir.exists():
    print(f"  models/ directory: ✅ EXISTS")
    
    subdirs = ["audio_generator", "text_generator", "fish_speech"]
    for subdir in subdirs:
        path = models_dir / subdir
        if path.exists():
            print(f"  - {subdir}: ✅ FOUND")
        else:
            print(f"  - {subdir}: ⚠️ NOT FOUND (run setup_models.py)")
else:
    print(f"  models/ directory: ⚠️ NOT FOUND")
    print(f"  Will be created on first run")
print()

# Summary
print("=" * 60)
print("  Summary")
print("=" * 60)

try:
    import torch_directml
    if torch_directml.is_available():
        print("  GPU Support: ✅ AMD GPU via DirectML READY")
        print("  Performance: 🚀 FAST (GPU-accelerated)")
    else:
        print("  GPU Support: ⚠️ CPU fallback mode")
        print("  Performance: 🐢 SLOWER (CPU only)")
except:
    print("  GPU Support: ⚠️ CPU fallback mode")
    print("  Performance: 🐢 SLOWER (CPU only)")

print()
print("  Next Steps:")
print("  1. Run: setup_models.ps1 (to download AI models)")
print("  2. Run: start_app.ps1 (to launch the app)")
print()
print("=" * 60)
