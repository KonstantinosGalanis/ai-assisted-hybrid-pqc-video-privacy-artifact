# ROI Video Encryption Artifact

This repository contains the research artifact for reproducing the experiments associated with the paper:

**AI-Assisted Region-of-Interest Video Protection with Hybrid Post-Quantum Key Establishment for Smart-City Surveillance: Design, Implementation, and Empirical Evaluation**

The artifact provides standalone implementations for:

1. Region-of-interest chaotic-map video encryption.
2. Region-of-interest ChaCha20-Poly1305 AEAD video encryption.
3. Full-frame encryption baseline.
4. Reproducibility test suites and experiment scripts.
5. Example input videos or dummy videos for verification.

---

## Authors

**Konstantinos Orestis Vasileios Galanis**  
**Achilleas Alexandros Vasileios Galanis**

*Both authors contributed equally to this work.*

Department of Computer and Information Science (IDA)  
Linköping University  
SE-581 83 Linköping, Sweden

Contact:

- konstantinegalanis@gmail.com
- achgalanis@gmail.com

---

## Repository Structure

```text
roi-video-crypto-artifact/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── requirements-lock.txt
├── environment.yml
├── .gitignore
├── .gitattributes
│
├── src/
│   ├── chaos.py
│   └── chacha.py
│
├── baselines/
│   └── full_frame_encrypt.py
│
├── configs/
│   └── custom_botsort_privacy_v2.yaml
│
├── data/
│   ├── README.md
│   ├── input/
│   │   ├── test1.mkv
│   │   ├── test2.mkv
│   │   └── ...
│   └── dummy/
│
├── docs/
│   ├── RUNNING.md
│   ├── EXPERIMENTS.md
│   ├── REPRODUCIBILITY.md
│   └── DATA.md
│
├── experiments/
│   └── tests/
│       ├── unified_framework/
│       ├── unified_framework2/
│       └── unified_framework3/
│
├── models/
│   └── README.md
│
└── runs/
    └── .gitkeep
```

---

## Main Files

| Path | Description |
|---|---|
| `src/chaos.py` | Standalone ROI chaotic-map encryption pipeline. |
| `src/chacha.py` | Standalone ROI ChaCha20-Poly1305 AEAD encryption pipeline. |
| `baselines/full_frame_encrypt.py` | Full-frame encryption baseline. |
| `configs/custom_botsort_privacy_v2.yaml` | Tracker configuration used by the ROI detection/tracking pipeline. |
| `experiments/tests/` | Evaluation and testing scripts. |
| `data/input/` | Input videos used for reproduction, if they can be redistributed. |
| `data/dummy/` | Optional dummy/synthetic videos for public reproducibility. |
| `runs/` | Output folder for generated artifacts. This folder is ignored by Git except for `.gitkeep`. |

---

## What the Pipelines Do

Both ROI pipelines follow the same high-level structure:

1. Decode the input video.
2. Detect sensitive regions of interest using YOLO segmentation and tracking.
3. Store ROI metadata in a sidecar file.
4. Store encrypted recoverable ROI/audio payloads in a payload file.
5. Write a protected public video.
6. Support authorized recovery using the same key, sidecar, payload, and manifest.

The difference is the cryptographic method used for ROI/audio protection.

### Chaos ROI Pipeline

`src/chaos.py` uses chaotic keystream generation with frame-type-aware map selection:

- I-frames: Chen map
- P-frames: Cubic map
- B-frames: Skew-Tent map
- Fallback: Cubic map

### ChaCha20-Poly1305 ROI Pipeline

`src/chacha.py` uses ChaCha20-Poly1305 AEAD for ROI and payload protection.

---

## Setup

### Option 1: Conda setup from minimal requirements

```powershell
conda create -n roi-video-crypto python=3.10 -y
conda activate roi-video-crypto
pip install -r requirements.txt
```

### Option 2: Recreate from exported environment

```powershell
conda env create -f environment.yml
conda activate roi-video-crypto
```

If `environment.yml` was exported from another machine, some packages may need adjustment depending on CUDA, PyTorch, and FFmpeg/PyAV availability.

---

## Required Model Files

The code expects a YOLO segmentation model, for example:

```text
yolo26x-seg.pt
```

You may place the model in the repository root or inside `models/`.

Example:

```powershell
$env:COCO_MODEL="models/yolo26x-seg.pt"
$env:DETECT_WIDTH="960"
$env:RETINA_MASKS="0"
```

TensorRT `.engine` files are not included because they are hardware-specific and should be generated locally.

See:

```text
models/README.md
```

---

## Input Videos

Place input videos in:

```text
data/input/
```

Example:

```text
data/input/test1.mkv
data/input/test2.mkv
data/input/test3.mkv
data/input/test4.mkv
data/input/test5.mkv
data/input/test6.mkv
```

If the original videos cannot be redistributed due to privacy or copyright restrictions, use dummy/synthetic videos in:

```text
data/dummy/
```

See:

```text
data/README.md
docs/DATA.md
```

---

## How to Run

For the complete command list, see:

```text
docs/RUNNING.md
```

Quick example for Chaos ROI encryption:

```powershell
python src/chaos.py --mode encrypt --in data/input/test6.mkv --out runs/chaos_cipher.mkv --key demo_key_change_me --roi_sidecar runs/rois.jsonl --payload runs/payload.jsonl --keystream_dump runs/keystream.bin --detect_width 960
```

Quick example for Chaos ROI decryption:

```powershell
python src/chaos.py --mode decrypt --in runs/chaos_cipher.mkv --out runs/chaos_decrypted.mkv --key demo_key_change_me --roi_sidecar runs/rois.jsonl --payload runs/payload.jsonl
```

Quick example for ChaCha20-Poly1305 ROI encryption:

```powershell
python src/chacha.py --mode encrypt --in data/input/test6.mkv --out runs/chacha_cipher.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha.jsonl --payload runs/payload_chacha.jsonl --keystream_dump runs/keystream_chacha.bin --detect_width 960
```

Quick example for ChaCha20-Poly1305 ROI decryption:

```powershell
python src/chacha.py --mode decrypt --in runs/chacha_cipher.mkv --out runs/chacha_decrypted.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha.jsonl --payload runs/payload_chacha.jsonl
```

Quick example for full-frame baseline:

```powershell
python baselines/full_frame_encrypt.py --mode encrypt --in data/input/test6.mkv --out runs/full_frame_cipher.mkv --key demo_key_change_me --keystream_dump runs/keystream_full_frame.bin
```

Quick example for full-frame decryption:

```powershell
python baselines/full_frame_encrypt.py --mode decrypt --in data/input/full_frame_cipher.mkv --out runs/full_frame_decrypted.mkv --key demo_key_change_me
```

---

## Experiment Documentation

| Document | Purpose |
|---|---|
| `docs/RUNNING.md` | Complete commands for all main codes. |
| `docs/EXPERIMENTS.md` | Explains the experiment branches, tests, and expected outputs. |
| `docs/REPRODUCIBILITY.md` | Step-by-step protocol for reviewers starting from a fresh clone. |
| `docs/DATA.md` | Dataset, privacy, dummy-video, and redistribution notes. |

---

## Important Runtime Files

### `rois.jsonl`

Stores ROI metadata such as frame index, bounding boxes, masks, class IDs, track IDs, policy tags, and mask hashes.

### `payload.jsonl`

Stores encrypted recoverable payload data such as original ROI pixels/audio segments, crypto metadata, and authentication information.

### `payload.jsonl.manifest.json`

Binds together the encrypted video, ROI sidecar, payload file, and session metadata.

### `keystream.bin`

Optional binary artifact used for ciphertext/randomness testing.

These generated files are ignored by Git by default.

---

## Reproducibility Notes

Generated files are ignored by Git by default:

```text
runs/
results/
cipher videos
decrypted videos
ROI sidecars
payload files
manifest files
keystream dumps
TensorRT engine files
Python cache files
```

This keeps the repository clean and prevents accidental upload of large or sensitive generated artifacts.

---

## Privacy and Dataset Notice

Videos may contain sensitive visual information such as people, vehicles, faces, license plates, or private locations. Only videos that are legally and ethically redistributable should be committed to this repository.

If original experimental videos cannot be shared, provide dummy or synthetic videos with the same approximate format, duration, frame size, and encoding properties.

---

## License

This artifact is released for academic review, verification, and reproducibility only.

See `LICENSE` for the exact terms.

---

## Citation

If you use this artifact, please cite the associated paper and this repository.

See `CITATION.cff`.
