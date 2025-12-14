# LEMM HuggingFace Collection Setup

## ✅ New Architecture (December 14, 2025)

LEMM now uses **HuggingFace Collections** to organize trained LoRA adapters. This is simpler and more powerful than the previous dataset repo approach.

## 🎯 How It Works

### **Each Trained LoRA** → **Individual Model on HF Hub**

When you train a LoRA in LEMM:
1. LoRA trained and saved locally in `models/loras/{name}/`
2. **Automatically uploaded** as a model to HuggingFace Hub
3. Model ID: `Gamahea/lemm-lora-{name}`
4. Auto-generated README with training details
5. Ready to share, download, and reuse

### **Collection** → **Curated List of LEMM LoRAs**

- **Collection URL**: https://huggingface.co/collections/Gamahea/lemm-100-pre-beta
- **Purpose**: Organize and showcase all LEMM LoRAs
- **Add Models**: Manually add uploaded LoRAs via web UI (one-click)

## 📦 LoRA Model Structure

Each uploaded LoRA is a complete model repo:

```
Gamahea/lemm-lora-{name}/
├── README.md              # Auto-generated model card
├── final_model.pt         # Trained LoRA weights
└── config.yaml            # Training configuration
```

## 🚀 Training → Upload Workflow

### When Training Completes:

```
✅ Training complete!
Final validation loss: 0.0234

📤 Uploading LoRA to HuggingFace Hub...
✅ LoRA uploaded as model: Gamahea/lemm-lora-jazz-v1
🔗 View: https://huggingface.co/Gamahea/lemm-lora-jazz-v1
💡 Add to collection: https://huggingface.co/collections/Gamahea/lemm-100-pre-beta
```

### Adding to Collection (Manual - One Time):

1. Click the collection link shown after upload
2. Click "Add to Collection" button on collection page
3. Paste model ID: `Gamahea/lemm-lora-jazz-v1`
4. Click "Add" - Done!

## 📥 Using LoRAs from Collection

### In LEMM Space

**Option 1: Download from Hub**
1. Go to "LoRA Management" tab
2. Find "Download from HuggingFace Hub" section
3. Enter model ID: `Gamahea/lemm-lora-jazz-v1`
4. Click "Download from Hub"
5. LoRA ready to use!

**Option 2: Browse Collection**
1. Visit collection: https://huggingface.co/collections/Gamahea/lemm-100-pre-beta
2. Browse available LoRAs
3. Copy model ID from any LoRA
4. Use in LEMM as above

### In Your Code

```python
from huggingface_hub import snapshot_download
from pathlib import Path

# Download specific LoRA
lora_path = snapshot_download(
    repo_id="Gamahea/lemm-lora-jazz-v1",
    local_dir="./loras/jazz-v1"
)

print(f"LoRA downloaded to: {lora_path}")
# Contains: final_model.pt, config.yaml, README.md
```

## 🔄 Continued Training

Use downloaded LoRAs as base models:

1. **Download** base LoRA from collection
2. **Training Config** → "Base LoRA Adapter" → Select downloaded LoRA
3. **Train** on new dataset
4. **Upload** result as new versioned model

Example:
- Base: `lemm-lora-jazz-v1` (trained on jazz dataset)
- New: `lemm-lora-jazz-v2` (continued training on bebop subset)

## 🎵 Model Card (Auto-Generated)

Each uploaded LoRA gets a detailed README:

```markdown
# LEMM LoRA: jazz-v1

LoRA adapter for DiffRhythm2 music generation, trained using LEMM.

## Training Configuration
- Dataset: gtzan
- Epochs: 50
- Learning Rate: 1e-4
- Batch Size: 4
- LoRA Rank: 8

## How to Use
### In LEMM Space
1. Visit LEMM
2. Go to "LoRA Management" tab
3. Enter model ID: Gamahea/lemm-lora-jazz-v1
4. Click "Download from Hub"

[...code examples...]

## Collection
Part of LEMM 1.0.0 Pre-Beta Collection
```

## 💡 Benefits Over Dataset Repo

| Feature | Dataset Repo | Collection + Models |
|---------|-------------|---------------------|
| **Authentication** | Required gating | Public models (no auth) |
| **Discoverability** | Hidden in datasets | Searchable as models |
| **Individual Sharing** | Must share whole repo | Each LoRA has unique URL |
| **Versioning** | Folder-based | Model versioning built-in |
| **README** | Manual | Auto-generated |
| **Downloads** | Bulk only | Individual models |
| **Organization** | Flat structure | Collection + tags |
| **Continued Training** | Complex | Download as base model |

## 🔐 Authentication

### For Uploads (HF Space)

**Required**: HuggingFace write token

1. Create token: https://huggingface.co/settings/tokens
   - Name: `LEMM-Upload-Token`
   - Type: **Write**
2. HF Space Settings → Repository Secrets
3. Add secret:
   - Name: `HF_TOKEN`
   - Value: `hf_xxxxxxxxxxxxx`
4. Restart Space

**Without token**: LoRAs saved locally only (still works for training)

### For Downloads

**No authentication needed** - all models are public!

## 📊 Collection Management

### View Collection
https://huggingface.co/collections/Gamahea/lemm-100-pre-beta

### Add Model to Collection
1. Go to collection page
2. Click "Add to Collection"
3. Enter model ID or URL
4. Confirm

### Remove from Collection
1. Go to collection page
2. Find model in list
3. Click "..." → "Remove"

### Collection Metadata
- **Title**: LEMM 1.0.0 (Pre-Beta) - Trained LoRA Adapters
- **Description**: Community-trained LoRA adapters for DiffRhythm2 music generation
- **Visibility**: Public
- **Owner**: Gamahea

## 🎯 Naming Convention

All LEMM LoRAs follow this pattern:
```
Gamahea/lemm-lora-{descriptive-name}
```

Examples:
- `Gamahea/lemm-lora-jazz-v1`
- `Gamahea/lemm-lora-rock-specialist`
- `Gamahea/lemm-lora-gtzan-balanced`
- `Gamahea/lemm-lora-custom-style-42`

## 🚀 Quick Start Guide

### 1. Train a LoRA
- Use LEMM Space training interface
- Choose dataset (e.g., GTZAN)
- Configure parameters
- Click "Start Training"

### 2. Upload Happens Automatically
- After training completes
- LoRA uploaded to HF Hub
- Model ID shown in logs

### 3. Add to Collection (One-Click)
- Click collection link from training output
- Add uploaded model to collection
- Now visible in collection for everyone

### 4. Reuse Anywhere
- Download in LEMM for generation
- Use as base for continued training
- Share model URL with others
- Integrate in custom code

## 📝 Example: Full Training Session

```
👤 User trains LoRA on GTZAN dataset
Name: "test_gtzan_1"

🔧 LEMM trains model
[Training progress...]
✅ Training complete!

📤 Auto-upload to HF Hub
✅ Model uploaded: Gamahea/lemm-lora-test-gtzan-1
🔗 https://huggingface.co/Gamahea/lemm-lora-test-gtzan-1

👤 User clicks collection link
Opens: https://huggingface.co/collections/Gamahea/lemm-100-pre-beta
Clicks: "Add to Collection"
Adds: Gamahea/lemm-lora-test-gtzan-1

✅ Now everyone can:
- Browse collection
- Download LoRA
- Use in generation
- Continue training
- Share with others
```

## 🎉 Current Status

- ✅ Collection created: https://huggingface.co/collections/Gamahea/lemm-100-pre-beta
- ✅ Auto-upload implemented
- ✅ README generation working
- ✅ Download functionality ready
- ✅ Continued training supported
- ✅ HF Space deployed with new system

**Ready to train and share LoRAs!** 🚀

---

**Last Updated**: December 14, 2025  
**Collection**: https://huggingface.co/collections/Gamahea/lemm-100-pre-beta  
**LEMM Space**: https://huggingface.co/spaces/Gamahea/lemm-test-100
