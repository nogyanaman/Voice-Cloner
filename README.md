# 🎤 VoiceConv – Full Setup & Usage Guide

Convert a **source voice** (`B.wav`) into a **target voice** (`A.wav`) using a modern **RVC (Retrieval‑based Voice Conversion)** backend.

---

## ⚡ Features
- GPU‑accelerated voice conversion
- Simple CLI interface
- Supports **16 kHz / 32 kHz / 48 kHz** RVC models
- Docker + CUDA support
- Modular architecture
- Includes full pytest test suite

---

## 1️⃣ Preparing Target Voice (A.wav)
To clone a voice, you must train the RVC model using **clean, diverse, high‑quality audio samples** of the target speaker.

### Requirements
- Format: `.wav`
- Sample rate: `48 kHz`
- Minimum duration: **5–10 minutes** (recommended **10–20+ minutes**)
- Clean, noise‑free, no reverb, no background music
- Varied emotions, pitch, volume, and speaking styles

### FFmpeg Cleaning & Conversion Command
```bash
ffmpeg -i "audio-path-target" -af "highpass=f=80, lowpass=f=12000, afftdn=nf=-25, anlmdn=s=10:r=0.002, arnndn=m=cleanvoice, dynaudnorm" -ar 48000 -ac 1 -y "...\voice_conv\data\ref\A.wav"
```

### File Path
```
C:\Users\nogya\Documents\voice_conv\data\ref\A.wav
```

**Tip:** Record in a quiet room and include multiple speaking styles for the best cloning accuracy.

---

## 2️⃣ Training the Model
Place `A.wav` inside the `ref` directory and train your RVC model.

### Output Files
- `model.pth` – trained model parameters
- `model.index` (optional) – faster retrieval and inference

### File Path
```
C:\Users\nogya\Documents\voice_conv\models\vc\
    model.pth
    model.index
```

If you change the file names, update them in your configuration or CLI.

---

## 3️⃣ Preparing Source Audio (B.wav)
This is the voice you want to convert *into* your target voice.

### Requirements
- Format: `.wav`
- Sample rate: `48 kHz`
- Length: **10–60 seconds**
- Clean and expressive; no background noise

### FFmpeg Command
```bash
ffmpeg -i "audio-path-source" -af "highpass=f=80, lowpass=f=12000, afftdn=nf=-25, anlmdn=s=10:r=0.002, arnndn=m=cleanvoice, dynaudnorm" -ar 48000 -ac 1 -y "...\voice_conv\data\src\B.wav"
```

### File Path
```
C:\Users\nogya\Documents\voice_conv\data\src\B.wav
```

Short, expressive audio yields the best conversion quality.

---

## 4️⃣ Running Voice Conversion (Inference)
Once the model and audio files are ready, run the CLI.

### 48 kHz Model
```bash
python -m voice_conv.cli --ref data/ref/A.wav --src data/src/B.wav --out data/out/C.wav --vc-sr 48000
```

### 32 kHz Model
```bash
python -m voice_conv.cli --ref data/ref/A.wav --src data/src/B.wav --out data/out/C.wav --vc-sr 32000
```

### Workflow Summary
1. Load `model.pth` (+ optional `model.index`)
2. Load target reference (`A.wav`)
3. Load source audio (`B.wav`)
4. Convert source → target voice
5. Export to `C.wav`

Quality depends on model training, audio cleanliness, and pitch extraction.

---

## 5️⃣ Best Practices
- Use **10–20 minutes** of target voice for high‑quality voice cloning
- Avoid echo or background noise during recording
- Clean audio with spectral denoise tools
- Train and infer with **48 kHz** audio
- Provide multiple speaking styles in target voice
- Keep source audio **short (10–60 seconds)** and expressive

---

## 6️⃣ Installation (Local)
```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### One‑Command Setup (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/create_env.ps1
```

---

## 7️⃣ Docker Setup
```bash
cd docker
docker compose up --build
```

---

## 8️⃣ Testing & GPU Check
```bash
# Check GPU
python check_gpu.py

# Check model keys
python check_keys.pth

# Run tests
pytest
```

---

## ⚙️ System Info Example
- Python 3.10
- NVIDIA GeForce RTX 3050
- CUDA 12.8
- Windows 11


## 🧩 Developer Extras

- **Makefile**:
  - `make dev` – create venv + install deps
  - `make test` – run tests
  - `make run` – run sample conversion
- **Logging**:
  - Configure via `config.yaml` → `logging.level`
- **Config Loader**:
  - Edit `config.yaml` for defaults (`feature_sample_rate`, `vc_sample_rate`, `device`)
- **Tests**:
  - Run `pytest` for unit tests on I/O, preprocess, and config
