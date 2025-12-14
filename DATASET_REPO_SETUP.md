# LEMM Dataset Repository - Access Configuration

## ✅ Current Setup (Gated Access)

The `Gamahea/lemm-dataset` repository is configured with:

### Settings
- **Visibility**: Public
- **Gated Access**: Enabled
- **Access Requests**: Enabled with **Automatic Approval**

### Why This Configuration?

1. **Public Visibility**: Anyone can see what the repo contains
2. **Gated Access**: Requires users to request access (automatic)
3. **Automatic Approval**: No manual intervention needed
4. **HF Space Access**: The LEMM Space gets auto-approved to read/write

## 🔄 How It Works

### For the HF Space (Gamahea/lemm-test-100)
1. Space requests access on first operation
2. Access automatically granted (same account)
3. LoRAs and datasets sync automatically
4. No additional configuration needed

### For Other Users
1. Visit: https://huggingface.co/datasets/Gamahea/lemm-dataset
2. Click "Access repository" button
3. Fill in required information
4. Access automatically granted
5. Can download LoRAs and datasets

## 📊 What Gets Stored

### LoRA Adapters (`loras/`)
```
loras/
├── test_gtzan_1/
│   ├── final_model.pt       # Trained weights
│   └── config.yaml          # Training config
├── jazz_style_v1/
└── rock_specialist/
```

### Prepared Datasets (`datasets/`)
```
datasets/
├── gtzan/
│   ├── train/               # Training samples
│   ├── val/                 # Validation samples
│   └── metadata.json
├── musiccaps/
└── fma_small/
```

## 🚀 Automatic Sync Behavior

### On Space Startup
```
🔄 Syncing from HuggingFace repo...
📥 Downloading LoRAs from Gamahea/lemm-dataset/loras...
📥 Downloading datasets from Gamahea/lemm-dataset/datasets...
✅ Sync complete: 3 LoRAs, 2 datasets
```

### After Training
```
✅ Training complete!
📤 Uploading LoRA test_gtzan_1 to Gamahea/lemm-dataset...
✅ Uploaded LoRA: test_gtzan_1
```

### After Dataset Preparation
```
✅ Datasets are now ready for LoRA training!
📤 Uploading prepared datasets to HuggingFace repo...
✅ Uploaded 1 dataset(s) to repo
```

## ⚠️ Notes

### Storage Limits
- HuggingFace datasets have generous storage limits
- Each LoRA: ~50-200MB (depending on model size)
- Each dataset: varies (GTZAN ~1GB, MusicCaps ~5GB)
- Monitor usage in Settings → Storage overview

### Access Control
- Same-account Spaces get automatic access
- External users need to request (auto-approved)
- Can disable access requests anytime
- Can switch to private if needed

### Data Persistence
- Data survives Space rebuilds
- No need to re-upload models
- LoRAs persist across sessions
- Training progress saved

## 🔐 Alternative: Private with HF_TOKEN

If you want tighter control:

1. **Make repo private** (Settings → Change dataset visibility)
2. **Create HF token** with write access
3. **Add to Space secrets**: Settings → Repository secrets
   - Name: `HF_TOKEN`
   - Value: `hf_xxxxxxxxxxxxx`
4. **Restart Space**

This gives you:
- ✅ Private storage (only you can see)
- ✅ Space still works (uses token)
- ✅ Full control over access
- ❌ Others can't download your LoRAs

## 📖 Documentation

Dataset Card: https://huggingface.co/datasets/Gamahea/lemm-dataset

Includes:
- Purpose and usage
- Repository structure
- Code examples
- Related projects
- License information

## ✨ Benefits of This Setup

1. **Zero Configuration**: Works out of the box
2. **Persistent Storage**: Survives rebuilds
3. **Community Sharing**: Others can use your LoRAs
4. **Automatic Backup**: Training work never lost
5. **Version Control**: HF tracks changes
6. **Easy Distribution**: Share links to LoRAs

---

**Status**: ✅ Configured and working
**Last Updated**: December 14, 2025
