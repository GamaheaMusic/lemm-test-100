# LEMM - Let Everyone Make Music

**Version 1.0.2 (Beta)**

An advanced AI music generation system with **training capabilities**, built-in vocals, professional mastering, audio enhancement, and **music theory intelligence**. Powered by DiffRhythm2 with LoRA fine-tuning support and symbolic music understanding.

🎵 **Live Demo**: [Try LEMM on HuggingFace Spaces](https://huggingface.co/spaces/Gamahea/lemm-test-100)  
📦 **LoRA Collection**: [Browse Trained Models](https://huggingface.co/collections/Gamahea/lemm-100-pre-beta)  
🏢 **Organization**: [lemm-ai on GitHub](https://github.com/lemm-ai)

---

## ✨ Key Features

### 🎵 Music Generation
- **Text-to-Music**: Generate music from style descriptions
- **Built-in Vocals**: DiffRhythm2 generates vocals directly with music (no separate TTS)
- **Style Consistency**: New clips inherit musical character from existing ones
- **Flexible Duration**: 10-120 second clips

### 🎓 LoRA Training
- **Custom Style Training**: Fine-tune on your own music datasets
- **Public Datasets**: GTZAN, MusicCaps, FMA support
- **Continued Training**: Use existing LoRAs as base models
- **Automatic Upload**: Trained LoRAs uploaded to HuggingFace Hub

### 🎚️ Professional Audio Tools
- **Advanced Mastering**: 32 professional presets (Pop, Rock, Electronic, etc.)
- **Custom EQ**: 8-band parametric equalizer
- **Dynamics**: Compression and limiting controls
- **Audio Enhancement**: 
  - Stem separation (Demucs)
  - Noise reduction
  - Super resolution (upscale to 48kHz)

### 🎛️ DAW-Style Interface
- **Horizontal Timeline**: Professional multi-track layout
- **Visual Waveforms**: See your music as you build
- **Track Management**: Add, remove, rearrange clips
- **Real-time Preview**: Play individual clips or full timeline

---

## 🚀 Quick Start

### Option 1: HuggingFace Spaces (Recommended)

Try LEMM instantly with zero setup:

👉 **[Launch LEMM Space](https://huggingface.co/spaces/Gamahea/lemm-test-100)**

- No installation required
- Free GPU access
- Pre-loaded models
- Immediate start

### Option 2: Local Installation

**Prerequisites:**
- Python 3.10 or 3.11
- 16GB+ RAM recommended
- NVIDIA GPU recommended (CUDA 12.x) or CPU

**Installation:**

```bash
# Clone the repository
git clone https://github.com/lemm-ai/LEMM-1.0.0-ALPHA.git
cd LEMM-1.0.0-ALPHA

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch LEMM
python app.py
```

**Access at**: http://localhost:7860

---

## 📖 Usage Guide

### 1️⃣ Generate Your First Track

1. **Enter Music Prompt**: Describe the style
   - Example: *"upbeat electronic dance music with heavy bass"*
2. **Add Lyrics** (optional): DiffRhythm2 will sing them
   - Leave empty for instrumental
3. **Set Duration**: 10-120 seconds (default: 30s)
4. **Generate**: Click "✨ Generate Music Clip"
5. **Preview**: Listen in the audio player

### 2️⃣ Build Your Composition

1. **Timeline Tab**: View all generated clips
2. **Waveform Preview**: Visual representation of each clip
3. **Add More**: Generate additional clips at different positions
4. **Style Consistency**: New clips automatically match existing style

### 3️⃣ Master & Export

1. **Mastering Tab**: 
   - Choose preset (Pop, Rock, EDM, etc.)
   - Or customize: EQ, compression, limiting
2. **Enhancement** (optional):
   - Stem separation
   - Noise reduction
   - Audio super resolution
3. **Export Tab**: 
   - Choose format (WAV, MP3, FLAC)
   - Download your finished track

### 4️⃣ Train Custom LoRAs

1. **Dataset Management Tab**:
   - Select public dataset (GTZAN, MusicCaps, FMA)
   - Or upload your own music
   - Download and prepare dataset
2. **Training Configuration Tab**:
   - Name your LoRA
   - Set training parameters
   - Choose base LoRA (optional - for continued training)
   - Start training
3. **Wait for Training**: Progress shown in real-time
4. **Auto-Upload**: LoRA uploaded to HuggingFace as model
5. **Reuse**: Download and use in future generations

---

## 🏗️ Architecture

### Core Technology

**DiffRhythm2** (ASLP-lab)
- State-of-the-art music generation with vocals
- Continuous Flow Matching (CFM) diffusion
- MuQ-MuLan style encoding for consistency
- Native vocal generation (no separate TTS)

**LoRA Fine-Tuning** (PEFT)
- Low-Rank Adaptation for efficient training
- Parameter-efficient fine-tuning
- Custom style specialization
- Continued training support

### System Components

```
LEMM/
├── app.py                      # Main Gradio interface
├── backend/
│   ├── services/
│   │   ├── diffrhythm_service.py       # DiffRhythm2 integration
│   │   ├── lora_training_service.py    # LoRA training
│   │   ├── dataset_service.py          # Dataset management
│   │   ├── mastering_service.py        # Audio mastering
│   │   ├── stem_enhancement_service.py # Audio enhancement
│   │   ├── audio_upscale_service.py    # Super resolution
│   │   ├── hf_storage_service.py       # HuggingFace uploads
│   │   └── ...
│   ├── routes/                 # API endpoints
│   ├── models/                 # Data schemas
│   └── config/                 # Configuration
├── models/
│   ├── diffrhythm2/           # Music generation model
│   ├── loras/                 # Trained LoRA adapters
│   └── ...
├── training_data/             # Prepared datasets
├── outputs/                   # Generated music
└── requirements.txt           # Dependencies
```

### Key Dependencies

- **torch**: 2.4.0+ (PyTorch)
- **diffusers**: Diffusion models
- **transformers**: 4.47.1 (HuggingFace)
- **peft**: LoRA training
- **gradio**: Web interface
- **pedalboard**: Audio mastering
- **demucs**: Stem separation
- **huggingface-hub**: Model uploads

---

## 🎓 Training Your Own LoRAs

### Supported Datasets

**Public Datasets:**
- **GTZAN**: Music genre classification (1,000 tracks, 10 genres)
- **MusicCaps**: Google's music captioning dataset
- **FMA (Free Music Archive)**: Large-scale music collection

**Custom Datasets:**
- Upload your own music collections
- Supports MP3, WAV, FLAC, OGG

### Training Process

1. **Prepare Dataset**:
   - Download or upload music
   - Extract audio samples
   - Split into train/validation sets

2. **Configure Training**:
   - **LoRA Rank**: 4-64 (higher = more expressive, slower)
   - **Learning Rate**: 1e-4 to 1e-3
   - **Batch Size**: 1-8 (depends on GPU memory)
   - **Epochs**: 10-100 (depends on dataset size)
   - **Base LoRA**: Optional - continue from existing model

3. **Monitor Training**:
   - Real-time loss graphs
   - Validation metrics
   - Progress percentage

4. **Upload & Share**:
   - Automatic upload to HuggingFace Hub
   - Model ID: `Gamahea/lemm-lora-{your-name}`
   - Add to [LEMM Collection](https://huggingface.co/collections/Gamahea/lemm-100-pre-beta)

### Example: Training on GTZAN

```
1. Dataset Management → Select GTZAN → Download
2. Prepare Dataset → GTZAN → Prepare (800 train, 200 val)
3. Training Configuration:
   - Name: "my_jazz_lora"
   - Dataset: gtzan
   - Epochs: 50
   - LoRA Rank: 8
   - Learning Rate: 1e-4
4. Start Training → Wait ~2-4 hours (GPU dependent)
5. ✅ Uploaded: Gamahea/lemm-lora-my-jazz-lora
6. Reuse in generation or continue training
```

---

## 🎨 LoRA Management

### Download from HuggingFace

1. Go to **LoRA Management Tab**
2. Enter model ID: `Gamahea/lemm-lora-{name}`
3. Click "Download from Hub"
4. Use immediately in generation

### Browse Collection

👉 [LEMM LoRA Collection](https://huggingface.co/collections/Gamahea/lemm-100-pre-beta)

Discover community-trained LoRAs:
- Genre specialists (jazz, rock, electronic)
- Style adaptations
- Custom fine-tuned models

### Export/Import

**Export:**
- Download trained LoRA as ZIP
- Share with others
- Backup your work

**Import:**
- Upload LoRA ZIP file
- Instantly available for use
- Continue training from checkpoint

---

## 🔧 Advanced Configuration

### GPU Acceleration

**NVIDIA (Recommended):**
```bash
# CUDA 12.x automatically detected
# No additional configuration needed
```

**CPU Mode:**
```bash
# Automatic fallback if no GPU detected
# Slower but fully functional
```

### Model Paths

Models downloaded to:
- DiffRhythm2: `models/diffrhythm2/`
- LoRAs: `models/loras/`
- Training data: `training_data/`

### Environment Variables

Create `.env` file:
```env
# HuggingFace token for uploads (optional)
HF_TOKEN=hf_xxxxxxxxxxxxx

# Gradio server port (default: 7860)
GRADIO_SERVER_PORT=7860

# Enable debug logging
DEBUG=false
```

---

## 📊 Technical Specifications

### Generation

- **Model**: DiffRhythm2 (CFM-based diffusion)
- **Sampling**: 22050 Hz (can upscale to 48kHz)
- **Duration**: 10-120 seconds per clip
- **Vocals**: Built-in (no separate TTS)
- **Style Encoding**: MuQ-MuLan

### Training

- **Method**: LoRA (Low-Rank Adaptation)
- **Rank**: 4-64 (configurable)
- **Precision**: Mixed (FP16/FP32)
- **Optimizer**: AdamW
- **Scheduler**: Cosine annealing

### Audio Enhancement

- **Stem Separation**: Demucs 4.0.1 (4-stem)
- **Noise Reduction**: Spectral subtraction
- **Super Resolution**: AudioSR (up to 48kHz)
- **Mastering**: Pedalboard (Spotify LUFS-compliant)

---

## 🤝 Contributing

We welcome contributions! Here's how:

### Report Issues

- [GitHub Issues](https://github.com/lemm-ai/LEMM-1.0.0-ALPHA/issues)
- Include: steps to reproduce, logs, system info

### Share LoRAs

1. Train custom LoRA in LEMM
2. Upload to HuggingFace (automatic)
3. Add to [Collection](https://huggingface.co/collections/Gamahea/lemm-100-pre-beta)
4. Share with community

### Development

```bash
# Fork the repository
# Clone your fork
git clone https://github.com/YOUR-USERNAME/LEMM-1.0.0-ALPHA.git

# Create feature branch
git checkout -b feature/your-feature

# Make changes and commit
git commit -am "Add your feature"

# Push and create PR
git push origin feature/your-feature
```

---

## 📄 License

**MIT License** - See [LICENSE](LICENSE) file

Free to use, modify, and distribute.

---

## 🙏 Acknowledgments

### Models & Technologies

- **DiffRhythm2**: ASLP-lab for state-of-the-art music generation
- **LoRA/PEFT**: HuggingFace for parameter-efficient fine-tuning
- **Gradio**: For the beautiful web interface
- **Demucs**: Meta AI for stem separation
- **Pedalboard**: Spotify for professional audio processing

### Datasets

- **GTZAN**: Music genre classification dataset
- **MusicCaps**: Google's music captioning dataset
- **FMA**: Free Music Archive community

---

## 📞 Support & Community

- **Documentation**: [Full Docs](https://github.com/lemm-ai/LEMM-1.0.0-ALPHA/wiki)
- **HuggingFace Space**: [Try Now](https://huggingface.co/spaces/Gamahea/lemm-test-100)
- **LoRA Collection**: [Browse Models](https://huggingface.co/collections/Gamahea/lemm-100-pre-beta)
- **Issues**: [GitHub Issues](https://github.com/lemm-ai/LEMM-1.0.0-ALPHA/issues)

---

## 🚀 What's Next

**Planned Features:**
- Multi-track composition tools
- Real-time style transfer
- Collaborative projects
- Mobile app
- VST plugin support

**Join the Journey!**

Built with ❤️ by the LEMM community

---

**LEMM - Let Everyone Make Music** 🎵
