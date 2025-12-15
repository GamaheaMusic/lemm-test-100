# LEMM - Attribution & Licenses

This document provides comprehensive attribution for all models, libraries, and third-party code used in LEMM (Let Everyone Make Music).

---

## 🎵 AI Models

### DiffRhythm2
**Purpose**: Core music generation with built-in vocals  
**Source**: [ASLP-lab/DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2)  
**License**: MIT License  
**Citation**:
```bibtex
@article{diffrhythm2,
  title={DiffRhythm2: High-Quality Music Generation with Continuous Flow Matching},
  author={ASLP Lab},
  year={2024}
}
```
**Usage**: Music generation from text prompts, vocal synthesis
**Notes**: State-of-the-art music generation using Continuous Flow Matching (CFM)

### MuQ-MuLan
**Purpose**: Music style encoding for consistency  
**Source**: [muq-paper](https://arxiv.org/abs/2308.01546)  
**License**: Apache 2.0  
**Citation**:
```bibtex
@article{muq2023,
  title={MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization},
  author={Haici Huang, Weijian Yu, Yong Xu, Kun Wei, Dan Liang},
  journal={arXiv preprint arXiv:2308.01546},
  year={2023}
}
```
**Usage**: Style consistency across generated music clips  
**Notes**: Ensures new clips match the musical character of existing ones

### Demucs 4.0.1
**Purpose**: Music source separation (stem extraction)  
**Source**: [facebookresearch/demucs](https://github.com/facebookresearch/demucs)  
**License**: MIT License  
**Author**: Meta AI (Facebook Research)  
**Citation**:
```bibtex
@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```
**Usage**: Stem separation (vocals, drums, bass, other)  
**Notes**: 4-stem separation for audio enhancement

### AudioSR
**Purpose**: Audio super-resolution (upscaling)  
**Source**: [haoheliu/versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution)  
**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Author**: Haohe Liu  
**Citation**:
```bibtex
@article{liu2023audiosr,
  title={AudioSR: Versatile Audio Super-resolution at Scale},
  author={Liu, Haohe and Chen, Ke and Tian, Qiao and Wang, Wenwu and Plumbley, Mark D},
  journal={arXiv preprint arXiv:2309.07314},
  year={2023}
}
```
**Usage**: Upscaling audio from 22kHz to 48kHz  
**Notes**: Optional enhancement for higher quality output

---

## 🛠️ Core Libraries

### PyTorch
**Purpose**: Deep learning framework  
**Source**: [pytorch/pytorch](https://github.com/pytorch/pytorch)  
**License**: BSD-3-Clause License  
**Version**: 2.4.0+  
**URL**: https://pytorch.org/  
**Usage**: Model inference, training, tensor operations

### Transformers (HuggingFace)
**Purpose**: Model loading and utilities  
**Source**: [huggingface/transformers](https://github.com/huggingface/transformers)  
**License**: Apache 2.0  
**Version**: 4.47.1  
**URL**: https://huggingface.co/docs/transformers  
**Usage**: Model management, tokenization, utilities

### Diffusers (HuggingFace)
**Purpose**: Diffusion model pipelines  
**Source**: [huggingface/diffusers](https://github.com/huggingface/diffusers)  
**License**: Apache 2.0  
**Version**: 0.21.0+  
**URL**: https://huggingface.co/docs/diffusers  
**Usage**: CFM diffusion scheduling, pipeline management

### PEFT (HuggingFace)
**Purpose**: LoRA training (Parameter-Efficient Fine-Tuning)  
**Source**: [huggingface/peft](https://github.com/huggingface/peft)  
**License**: Apache 2.0  
**Version**: 0.6.0+  
**URL**: https://huggingface.co/docs/peft  
**Usage**: Low-rank adaptation for efficient model fine-tuning

### Gradio
**Purpose**: Web interface framework  
**Source**: [gradio-app/gradio](https://github.com/gradio-app/gradio)  
**License**: Apache 2.0  
**Version**: 4.0.0+  
**URL**: https://gradio.app/  
**Author**: Gradio Team  
**Usage**: Interactive web UI, audio playback, file management

### Pedalboard
**Purpose**: Audio effects and mastering  
**Source**: [spotify/pedalboard](https://github.com/spotify/pedalboard)  
**License**: GNU General Public License v3.0 (GPL-3.0)  
**Author**: Spotify  
**URL**: https://spotify.github.io/pedalboard/  
**Usage**: EQ, compression, limiting, reverb, audio processing  
**Notes**: Professional-grade audio effects from Spotify

### HuggingFace Hub
**Purpose**: Model downloading and uploading  
**Source**: [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)  
**License**: Apache 2.0  
**Version**: 0.20.0+  
**URL**: https://huggingface.co/docs/huggingface_hub  
**Usage**: LoRA model uploads, collection management

---

## 📊 Audio Processing Libraries

### librosa
**Purpose**: Audio analysis and manipulation  
**Source**: [librosa/librosa](https://github.com/librosa/librosa)  
**License**: ISC License  
**Version**: 0.10.0+  
**URL**: https://librosa.org/  
**Citation**:
```bibtex
@inproceedings{mcfee2015librosa,
  title={librosa: Audio and music signal analysis in python},
  author={McFee, Brian and Raffel, Colin and Liang, Dawen and Ellis, Daniel PW and McVicar, Matt and Battenberg, Eric and Nieto, Oriol},
  booktitle={Proceedings of the 14th python in science conference},
  volume={8},
  pages={18--25},
  year={2015}
}
```
**Usage**: Audio loading, feature extraction, analysis

### soundfile
**Purpose**: Audio file I/O  
**Source**: [bastibe/python-soundfile](https://github.com/bastibe/python-soundfile)  
**License**: BSD-3-Clause License  
**Version**: 0.12.0+  
**URL**: https://pysoundfile.readthedocs.io/  
**Usage**: Reading and writing audio files (WAV, FLAC, OGG)

### pydub
**Purpose**: Audio manipulation and conversion  
**Source**: [jiaaro/pydub](https://github.com/jiaaro/pydub)  
**License**: MIT License  
**Version**: 0.25.1+  
**URL**: http://pydub.com/  
**Usage**: Audio format conversion, basic editing

### resampy
**Purpose**: Audio resampling  
**Source**: [bmcfee/resampy](https://github.com/bmcfee/resampy)  
**License**: ISC License  
**Version**: 0.4.2+  
**URL**: https://resampy.readthedocs.io/  
**Usage**: High-quality audio resampling

### scipy
**Purpose**: Scientific computing and signal processing  
**Source**: [scipy/scipy](https://github.com/scipy/scipy)  
**License**: BSD-3-Clause License  
**Version**: 1.10.0+  
**URL**: https://scipy.org/  
**Usage**: Signal processing, filters, FFT

---

## 🌐 Web & Backend Libraries

### Flask
**Purpose**: Backend API framework  
**Source**: [pallets/flask](https://github.com/pallets/flask)  
**License**: BSD-3-Clause License  
**Version**: 3.0.0+  
**URL**: https://flask.palletsprojects.com/  
**Usage**: REST API endpoints, server infrastructure

### Flask-CORS
**Purpose**: Cross-origin resource sharing  
**Source**: [corydolphin/flask-cors](https://github.com/corydolphin/flask-cors)  
**License**: MIT License  
**Version**: 4.0.0+  
**URL**: https://flask-cors.readthedocs.io/  
**Usage**: Enable CORS for API access

---

## 📦 Utility Libraries

### numpy
**Purpose**: Numerical computing  
**Source**: [numpy/numpy](https://github.com/numpy/numpy)  
**License**: BSD-3-Clause License  
**Version**: 1.24.0+  
**URL**: https://numpy.org/  
**Usage**: Array operations, mathematical functions

### pydantic
**Purpose**: Data validation  
**Source**: [pydantic/pydantic](https://github.com/pydantic/pydantic)  
**License**: MIT License  
**Version**: 2.0.0+  
**URL**: https://docs.pydantic.dev/  
**Usage**: Request/response validation, data schemas

### pyyaml
**Purpose**: YAML parsing  
**Source**: [yaml/pyyaml](https://github.com/yaml/pyyaml)  
**License**: MIT License  
**Version**: 6.0+  
**URL**: https://pyyaml.org/  
**Usage**: Configuration file parsing

### python-dotenv
**Purpose**: Environment variable management  
**Source**: [theskumar/python-dotenv](https://github.com/theskumar/python-dotenv)  
**License**: BSD-3-Clause License  
**Version**: 1.0.0+  
**URL**: https://github.com/theskumar/python-dotenv  
**Usage**: Loading environment variables from .env files

### tqdm
**Purpose**: Progress bars  
**Source**: [tqdm/tqdm](https://github.com/tqdm/tqdm)  
**License**: MIT License / Mozilla Public License 2.0  
**Version**: 4.65.0+  
**URL**: https://tqdm.github.io/  
**Usage**: Training progress visualization

### gitpython
**Purpose**: Git repository interaction  
**Source**: [gitpython-developers/GitPython](https://github.com/gitpython-developers/GitPython)  
**License**: BSD-3-Clause License  
**Version**: 3.1.0+  
**URL**: https://gitpython.readthedocs.io/  
**Usage**: Version control operations

### safetensors
**Purpose**: Safe tensor serialization  
**Source**: [huggingface/safetensors](https://github.com/huggingface/safetensors)  
**License**: Apache 2.0  
**Version**: 0.3.0+  
**URL**: https://huggingface.co/docs/safetensors  
**Usage**: Secure model weight storage

---

## 🎓 Training & Monitoring

### tensorboard
**Purpose**: Training visualization  
**Source**: [tensorflow/tensorboard](https://github.com/tensorflow/tensorboard)  
**License**: Apache 2.0  
**Version**: 2.13.0+  
**URL**: https://www.tensorflow.org/tensorboard  
**Usage**: Loss curves, training metrics visualization

### wandb (Weights & Biases)
**Purpose**: Experiment tracking (optional)  
**Source**: [wandb/wandb](https://github.com/wandb/wandb)  
**License**: MIT License  
**Version**: 0.15.0+  
**URL**: https://wandb.ai/  
**Usage**: Advanced training monitoring and experiment tracking  
**Notes**: Optional dependency for enhanced tracking

### datasets (HuggingFace)
**Purpose**: Dataset management  
**Source**: [huggingface/datasets](https://github.com/huggingface/datasets)  
**License**: Apache 2.0  
**Version**: 2.14.0+  
**URL**: https://huggingface.co/docs/datasets  
**Usage**: Loading and processing training datasets

---

## 🗣️ NLP & Text Processing

### phonemizer
**Purpose**: Text to phoneme conversion  
**Source**: [bootphon/phonemizer](https://github.com/bootphon/phonemizer)  
**License**: GNU General Public License v3.0 (GPL-3.0)  
**Version**: 3.2.0+  
**URL**: https://github.com/bootphon/phonemizer  
**Usage**: Phonetic transcription for vocal synthesis

### sentencepiece
**Purpose**: Tokenization  
**Source**: [google/sentencepiece](https://github.com/google/sentencepiece)  
**License**: Apache 2.0  
**Version**: 0.1.99+  
**Author**: Google  
**URL**: https://github.com/google/sentencepiece  
**Usage**: Subword tokenization

### jieba
**Purpose**: Chinese text segmentation  
**Source**: [fxsjy/jieba](https://github.com/fxsjy/jieba)  
**License**: MIT License  
**Version**: 0.42.0+  
**URL**: https://github.com/fxsjy/jieba  
**Usage**: Chinese text processing for multilingual support

### pypinyin
**Purpose**: Chinese to Pinyin conversion  
**Source**: [mozillazg/python-pinyin](https://github.com/mozillazg/python-pinyin)  
**License**: MIT License  
**Version**: 0.50.0+  
**URL**: https://github.com/mozillazg/python-pinyin  
**Usage**: Chinese phonetic processing

### cn2an
**Purpose**: Chinese number conversion  
**Source**: [Ailln/cn2an](https://github.com/Ailln/cn2an)  
**License**: MIT License  
**Version**: 0.5.0+  
**URL**: https://github.com/Ailln/cn2an  
**Usage**: Chinese number to text conversion

### pykakasi
**Purpose**: Japanese text processing  
**Source**: [miurahr/pykakasi](https://github.com/miurahr/pykakasi)  
**License**: GNU General Public License v3.0 (GPL-3.0)  
**Version**: 2.3.0+  
**URL**: https://github.com/miurahr/pykakasi  
**Usage**: Japanese phonetic processing

### unidecode
**Purpose**: Unicode text normalization  
**Source**: [avian2/unidecode](https://github.com/avian2/unidecode)  
**License**: GNU General Public License v2.0 (GPL-2.0)  
**Version**: 1.3.0+  
**URL**: https://github.com/avian2/unidecode  
**Usage**: Text normalization across languages

---

## 📁 Datasets

### GTZAN
**Purpose**: Music genre classification dataset  
**Source**: [GTZAN Dataset](http://marsyas.info/downloads/datasets.html)  
**License**: Free for research use  
**Size**: 1,000 tracks (30 seconds each), 10 genres  
**Citation**:
```bibtex
@inproceedings{tzanetakis2002gtzan,
  title={Musical genre classification of audio signals},
  author={Tzanetakis, George and Cook, Perry},
  booktitle={IEEE Transactions on speech and audio processing},
  volume={10},
  number={5},
  pages={293--302},
  year={2002}
}
```
**Usage**: LoRA training for genre-specific models

### MusicCaps
**Purpose**: Music captioning dataset  
**Source**: [Google MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps)  
**License**: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)  
**Author**: Google Research  
**Size**: 5,521 music clips with text descriptions  
**Citation**:
```bibtex
@inproceedings{agostinelli2023musiclm,
  title={MusicLM: Generating Music From Text},
  author={Agostinelli, Andrea and Denk, Timo I. and Borsos, Zalán and Engel, Jesse and Verzetti, Mauro and Caillon, Antoine and Huang, Qingqing and Jansen, Aren and Roberts, Adam and Tagliasacchi, Marco and Sharifi, Matt and Zeghidour, Neil and Frank, Christian},
  booktitle={arXiv preprint arXiv:2301.11325},
  year={2023}
}
```
**Usage**: Training on music with natural language descriptions

### Free Music Archive (FMA)
**Purpose**: Large-scale music dataset  
**Source**: [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma)  
**License**: Creative Commons licenses (varies by track)  
**Authors**: Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson  
**Size**: 106,574 tracks from 16,341 artists  
**Citation**:
```bibtex
@inproceedings{fma_dataset,
  title={{FMA}: A Dataset for Music Analysis},
  author={Defferrard, Micha{\"e}l and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle={18th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2017}
}
```
**Usage**: Large-scale LoRA training

---

## 📜 License Summary

### Open Source Licenses Used

**MIT License** (Most permissive):
- DiffRhythm2
- Demucs
- Flask-CORS
- pydub
- pydantic
- pyyaml
- tqdm (dual license)
- wandb
- jieba
- pypinyin
- cn2an

**Apache 2.0** (Permissive with patent grant):
- MuQ-MuLan
- Transformers (HuggingFace)
- Diffusers (HuggingFace)
- PEFT (HuggingFace)
- Gradio
- HuggingFace Hub
- sentencepiece
- tensorboard
- datasets
- safetensors

**BSD-3-Clause** (Permissive):
- PyTorch
- Flask
- numpy
- python-dotenv
- gitpython
- scipy
- soundfile

**ISC License** (Very permissive):
- librosa
- resampy

**GPL-3.0** (Copyleft - requires open source distribution):
- Pedalboard (Spotify)
- phonemizer
- pykakasi

**GPL-2.0** (Copyleft):
- unidecode

**Creative Commons** (Content licenses):
- AudioSR: CC BY 4.0
- MusicCaps Dataset: CC BY-SA 4.0

### LEMM License

**LEMM** is licensed under the **MIT License**.

This means:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ No warranty provided
- ℹ️ License and copyright notice must be included

**Note on GPL Dependencies**:
- Pedalboard (GPL-3.0), phonemizer (GPL-3.0), pykakasi (GPL-3.0), and unidecode (GPL-2.0) are optional dependencies
- If you distribute LEMM with these components, GPL requirements apply
- For commercial closed-source distributions, consider removing GPL dependencies

---

## 🙏 Special Thanks

### Research Teams
- **ASLP Lab**: For DiffRhythm2 and advancing music generation
- **Meta AI**: For Demucs and open-source AI research
- **HuggingFace**: For transformers, diffusers, PEFT, and the Hub
- **Spotify**: For Pedalboard audio processing tools
- **Google**: For MusicCaps dataset and sentencepiece

### Open Source Communities
- PyTorch community
- Gradio team
- librosa developers
- All contributors to the libraries listed above

---

## 📞 Contact & Compliance

For licensing questions or compliance concerns:
- **GitHub**: [lemm-ai/LEMM-1.0.0-ALPHA](https://github.com/lemm-ai/LEMM-1.0.0-ALPHA)
- **Issues**: [Report licensing concerns](https://github.com/lemm-ai/LEMM-1.0.0-ALPHA/issues)

---

## 🔄 Updates

This attribution document is maintained to reflect the current state of LEMM dependencies. Last updated: **December 14, 2025**

To see current dependencies and versions:
```bash
pip list
```

For the complete dependency specification:
```bash
cat requirements.txt
```

---

**LEMM respects and acknowledges all open-source contributions.**

If you believe any attribution is missing or incorrect, please [open an issue](https://github.com/lemm-ai/LEMM-1.0.0-ALPHA/issues).
