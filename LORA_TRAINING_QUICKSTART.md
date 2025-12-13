# LoRA Training Quick Start Guide

## What is LoRA Training?

LoRA (Low-Rank Adaptation) training allows you to fine-tune the DiffRhythm2 music generation model with your own audio or specialized datasets. This creates custom "adapters" that specialize the model for:

- **Specific genres** (classical, jazz, metal, etc.)
- **Vocal styles** (opera, rap, folk singing)
- **Instrumental sounds** (orchestral, synth-heavy, acoustic)
- **Your own musical style** (upload your compositions)

**Training is permanent** - once trained, LoRA adapters are saved forever and can be loaded anytime.

---

## Quick Start: Train on Your Own Music

### Step 1: Prepare Your Audio

✅ **Requirements**:
- WAV format audio files
- 44.1kHz sample rate recommended
- At least 30 minutes total audio (more is better)
- High quality recordings (no compression artifacts)

💡 **Tips**:
- 10-20 songs = good starting point
- Consistent style/genre = better results
- Instrumental or vocal-only works best

### Step 2: Upload & Analyze

1. Open **Music Generation Studio**
2. Expand **"🎓 LoRA Training (Advanced)"** accordion
3. Click **"🎵 User Audio Training"** tab
4. **Upload your WAV files**

5. **Enable processing options**:
   - ✅ **Auto-split into clips**: Automatically creates 10-30s training segments
   - ✅ **Separate vocal stems**: Extract vocals for vocal-only training (optional)

6. Click **"🔍 Analyze & Generate Metadata"**

### Step 3: Review Metadata

The AI will analyze your audio and generate:
- **Genre** (pop, rock, classical, etc.)
- **BPM** (tempo)
- **Key** (C major, D minor, etc.)
- **Mood** (energetic, melancholic, upbeat)

You'll see a table with all metadata. You can:
- ✏️ **Edit manually** - Click any cell to change values
- ✨ **AI re-generate** - Click "AI Generate All Metadata" to auto-fill
- 💾 **Save** - Click "Save Metadata" when happy

### Step 4: Prepare Dataset

Click **"📦 Prepare Training Dataset"**

This will:
- Split long audio into clips (if enabled)
- Separate vocal stems (if enabled)
- Create train/validation split (90/10)
- Save permanently to disk

You'll get a dataset name like: `user_dataset_1702987654`

### Step 5: Configure Training

1. Click **"⚙️ Training Configuration"** tab

2. **Name your LoRA**:
   - Example: `my_jazz_piano_v1`
   - Must be unique

3. **Select dataset**:
   - Click "🔄 Refresh Datasets"
   - Choose your prepared dataset

4. **Adjust hyperparameters** (optional - defaults are good):
   - **Batch Size**: 4 (higher = faster but more GPU memory)
   - **Learning Rate**: 3e-4 (lower = more stable)
   - **Epochs**: 10 (how many times to iterate)
   - **LoRA Rank**: 16 (higher = more capacity)
   - **LoRA Alpha**: 32 (scaling factor)

### Step 6: Start Training

Click **"🚀 Start Training"**

**What happens**:
- Model loads into GPU memory
- Training begins (this takes HOURS)
- Progress updates in real-time
- Checkpoints saved every 500 steps

**Training time estimates**:
- 30 min audio: ~2-4 hours
- 1 hour audio: ~4-8 hours
- 5 hours audio: ~12-24 hours
- 10+ hours: ~24-48 hours

💡 **Tip**: Start training before bed or over weekend!

### Step 7: Use Your LoRA

Once training completes:

1. Go to **"📂 Manage LoRA Adapters"** tab
2. Click **"🔄 Refresh List"**
3. See your new LoRA: `my_jazz_piano_v1`

**To use in generation**:
- Load the LoRA adapter (feature coming soon)
- Generate music - it will use your custom fine-tuning!
- Your music will have the style of your training data

---

## Advanced: Train on Curated Datasets

### Pre-Curated Datasets

Go to **"📚 Dataset Training"** tab:

**Vocal Datasets**:
- ✅ **OpenSinger**: Multi-singer, 50+ hours (diverse voices)
- ✅ **M4Singer**: Chinese pop, 29 hours (Mandarin vocals)
- ✅ **CC Mixter**: Creative Commons stems (various styles)

**Symbolic Datasets**:
- ✅ **Lakh MIDI**: 176k files, diverse genres (pop, rock, jazz)
- ✅ **Mutopia**: Classical music, 2000+ pieces

### How to Use

1. **Select datasets**: Check boxes for desired datasets
2. Click **"📥 Download & Prepare Datasets"**
3. Wait for download (can be LARGE - 10-50GB)
4. Datasets automatically prepared for training
5. Go to "Training Configuration" and select dataset

**Benefits**:
- Professional-quality data
- Large datasets = better results
- Pre-curated for specific styles

---

## Understanding Training Parameters

### Batch Size (Default: 4)
- **What it does**: How many audio clips processed together
- **Higher** (8-16): Faster training, more GPU memory
- **Lower** (2-4): Slower, less memory, sometimes better quality
- **Recommendation**: 4 for 16GB GPU, 8 for 24GB+ GPU

### Learning Rate (Default: 3e-4)
- **What it does**: How fast model learns
- **Higher** (1e-3): Faster convergence, might be unstable
- **Lower** (1e-5): Very stable, very slow
- **Recommendation**: 3e-4 is sweet spot for LoRA

### Number of Epochs (Default: 10)
- **What it does**: How many times to see all training data
- **Higher** (20-50): More training, risk overfitting
- **Lower** (5-10): Faster, might not learn enough
- **Recommendation**: 10 for small datasets, 5 for large datasets

### LoRA Rank (Default: 16)
- **What it does**: Capacity of LoRA adapter
- **Higher** (32-64): More capacity, slower, bigger files
- **Lower** (8-16): Faster, smaller, might lose detail
- **Recommendation**: 16 for most cases, 32 for complex styles

### LoRA Alpha (Default: 32)
- **What it does**: How much LoRA affects base model
- **Higher** (64): Stronger adaptation
- **Lower** (16): Subtle changes
- **Recommendation**: 2x LoRA Rank (rank 16 → alpha 32)

---

## Best Practices

### Dataset Preparation

✅ **DO**:
- Use high-quality audio (no compression)
- Keep consistent genre/style per dataset
- Include variety within genre (fast/slow, loud/quiet)
- Aim for 1+ hour total audio minimum
- Split long tracks into clips

❌ **DON'T**:
- Mix completely different genres (make separate LoRAs)
- Use low-quality MP3s (convert to WAV first)
- Train on less than 30 minutes (won't learn enough)
- Use copyrighted music without permission

### During Training

✅ **DO**:
- Monitor training log for errors
- Save checkpoints regularly (every 500 steps)
- Test generated samples periodically
- Stop if validation loss stops improving

❌ **DON'T**:
- Change parameters mid-training
- Stop training too early (< 1000 steps)
- Overtrain (validation loss increases)
- Run multiple trainings simultaneously (GPU limits)

### After Training

✅ **DO**:
- Test your LoRA on various prompts
- Compare to base model (A/B test)
- Keep best checkpoint, delete others to save space
- Document what you trained on (for reference)

❌ **DON'T**:
- Delete source dataset immediately (might retrain)
- Use only final checkpoint (best checkpoint often better)
- Expect perfection on first try (iterative process)

---

## Troubleshooting

### "❌ No audio files uploaded"
**Solution**: Upload WAV files in User Audio Training tab first

### "❌ Dataset not found"
**Solution**: Click "Prepare Training Dataset" to create it first

### "Training is too slow"
**Cause**: CPU training is 10-50x slower than GPU
**Solution**: Use GPU (NVIDIA CUDA) or cloud GPU instance

### "Out of memory error"
**Solution**: Reduce batch size (try 2 or 1)

### "Loss not decreasing"
**Possible causes**:
1. Learning rate too low → increase to 5e-4
2. Dataset too small → add more audio
3. Dataset inconsistent → curate better
4. Model already converged → check validation loss

### "Generated music doesn't sound like training data"
**Possible causes**:
1. Not enough training steps → train longer
2. LoRA rank too low → increase to 32
3. Dataset too diverse → narrow focus
4. Need more training data → add more audio

---

## Example Workflows

### Workflow 1: Jazz Piano LoRA

```
1. Collect 20 jazz piano recordings (WAV, ~1 hour total)
2. Upload to "User Audio Training"
3. Enable "Auto-split into clips" ✅
4. Analyze & edit metadata (set genre="jazz", instruments="piano")
5. Prepare dataset
6. Configure: batch_size=4, lr=3e-4, epochs=10, rank=16
7. Start training (wait ~6 hours)
8. Test: "smooth jazz piano at 120 BPM"
9. Result: Jazz piano style customized to your recordings!
```

### Workflow 2: Opera Vocals LoRA

```
1. Collect 10 opera recordings (WAV, ~2 hours total)
2. Upload to "User Audio Training"
3. Enable "Auto-split" ✅ + "Separate stems" ✅
4. Analyze (auto-detects vocals)
5. Edit metadata: genre="opera", mood="dramatic"
6. Prepare dataset (vocals extracted automatically)
7. Configure: batch_size=4, lr=3e-4, epochs=15, rank=32
8. Start training (wait ~10 hours)
9. Test: "dramatic opera aria in Italian"
10. Result: Opera vocal style matching training data!
```

### Workflow 3: Electronic Music from MIDI

```
1. Go to "Dataset Training" tab
2. Select "Lakh MIDI" dataset ✅
3. Download & Prepare (wait ~1 hour)
4. Configure: batch_size=8, lr=5e-4, epochs=5, rank=16
5. Start training on subset (10k MIDI files)
6. Wait ~12 hours
7. Test: "upbeat electronic dance music at 128 BPM"
8. Result: Electronic music generator!
```

---

## FAQ

**Q: How long does training take?**
A: 2-48 hours depending on dataset size and GPU. See estimates above.

**Q: Can I stop and resume training?**
A: Currently no, but checkpoints are saved every 500 steps.

**Q: How much audio do I need?**
A: Minimum 30 minutes, recommended 1+ hour, ideal 5+ hours.

**Q: Can I train multiple LoRAs?**
A: Yes! Each LoRA is independent. Train as many as you want.

**Q: Does training affect the base model?**
A: No! LoRA only adds adapters. Base model stays unchanged.

**Q: Can I share my LoRA?**
A: Yes! Just share the files in `models/loras/your_lora_name/`

**Q: What if I don't have a GPU?**
A: CPU training works but is 10-50x slower. Consider cloud GPU (RunPod, Lambda Labs).

**Q: Can I combine multiple LoRAs?**
A: Not yet, but LoRA merging is planned for future updates.

---

## Getting Help

If you encounter issues:

1. Check training log for error messages
2. Review this guide's Troubleshooting section
3. Ensure audio files are proper WAV format
4. Try with smaller dataset first (30 min audio)
5. Open issue on GitHub with:
   - Dataset details (size, format)
   - Training config used
   - Error message (full log)
   - GPU/CPU specs

---

## Summary

LoRA training lets you create custom music generation models:

1. **Upload** your audio (30+ min WAV files)
2. **Analyze** with AI (auto-metadata)
3. **Prepare** dataset (auto-split, stem separation)
4. **Configure** training (use defaults!)
5. **Train** (wait hours)
6. **Use** your custom LoRA forever!

**Result**: Music generation specialized to YOUR style! 🎵
