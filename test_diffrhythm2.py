"""
Test script to verify DiffRhythm 2 installation and basic functionality.
"""
import os
import sys
from pathlib import Path

# Set up local espeak-ng path
project_root = Path(__file__).parent
espeak_path = project_root / "external" / "espeak-ng"
if espeak_path.exists():
    os.environ["ESPEAK_DATA_PATH"] = str(espeak_path / "espeak-ng-data")
    os.environ["PATH"] = f"{espeak_path};{os.environ.get('PATH', '')}"
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_path / "libespeak-ng.dll")

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import torchaudio
        print(f"✓ TorchAudio {torchaudio.__version__}")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        from phonemizer import phonemize
        print("✓ Phonemizer")
        
        from huggingface_hub import snapshot_download
        print("✓ HuggingFace Hub")
        
        import librosa
        print(f"✓ Librosa {librosa.__version__}")
        
        import soundfile as sf
        print("✓ SoundFile")
        
        print("\n✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        print("\nPlease run: pip install -r requirements.txt")
        return False

def test_espeak():
    """Test if espeak-ng is installed and accessible."""
    print("\nTesting espeak-ng...")
    try:
        from phonemizer import phonemize
        test_text = "Hello world"
        result = phonemize(test_text, language='en-us', backend='espeak')
        print(f"✓ espeak-ng working: '{test_text}' -> '{result}'")
        return True
    except Exception as e:
        print(f"✗ espeak-ng test failed: {e}")
        print("\nPlease install espeak-ng:")
        print("  Windows: choco install espeak-ng  OR  scoop install espeak-ng")
        print("  Linux:   sudo apt-get install espeak-ng")
        print("  macOS:   brew install espeak-ng")
        return False

def test_model_download():
    """Test if DiffRhythm 2 model can be accessed."""
    print("\nTesting model access...")
    try:
        from huggingface_hub import hf_hub_download
        
        # Try to download config.json from DiffRhythm 2 repo
        config_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm2",
            filename="config.json",
            cache_dir="./models/diffrhythm2"
        )
        print(f"✓ Model accessible at: {config_path}")
        return True
    except Exception as e:
        print(f"✗ Model download failed: {e}")
        print("\nNote: Model will auto-download on first use (~2-4GB)")
        return False

def test_basic_generation():
    """Test basic music generation (if models are available)."""
    print("\nTesting basic generation...")
    try:
        # Import the service
        sys.path.insert(0, str(Path(__file__).parent))
        from backend.services.diffrhythm_service import DiffRhythmService
        
        service = DiffRhythmService()
        
        # Try simple generation
        print("Generating 5-second test clip...")
        result = service.generate_music(
            prompt="upbeat pop music",
            lyrics="Hello world, this is a test",
            duration=5
        )
        
        if result and os.path.exists(result):
            print(f"✓ Generation successful: {result}")
            file_size = os.path.getsize(result) / 1024 / 1024
            print(f"  File size: {file_size:.2f} MB")
            return True
        else:
            print("✗ Generation returned no output")
            return False
            
    except Exception as e:
        print(f"✗ Generation test failed: {e}")
        print("\nThis is expected if models aren't downloaded yet.")
        print("Run the app to trigger automatic model download.")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DiffRhythm 2 Installation Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: espeak-ng
    results.append(("espeak-ng", test_espeak()))
    
    # Test 3: Model access
    results.append(("Model Access", test_model_download()))
    
    # Test 4: Generation (optional)
    # results.append(("Generation", test_basic_generation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed! DiffRhythm 2 is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
