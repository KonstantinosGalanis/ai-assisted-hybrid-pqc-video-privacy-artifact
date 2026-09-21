# Reproducibility Guide

This document describes a fresh-clone reproduction protocol for reviewers and readers.

---

## 1. Clone the Repository

```powershell
git clone https://github.com/YOUR_GITHUB_USERNAME_HERE/roi-video-crypto-artifact.git
cd roi-video-crypto-artifact
```

Replace `YOUR_GITHUB_USERNAME_HERE` with the actual GitHub username or organization.

---

## 2. Create Environment

Recommended:

```powershell
conda create -n roi-video-crypto python=3.10 -y
conda activate roi-video-crypto
pip install -r requirements.txt
```

Alternative:

```powershell
conda env create -f environment.yml
conda activate roi-video-crypto
```

If CUDA, PyTorch, TensorRT, or PyAV versions differ across machines, some package adjustments may be needed.

---

## 3. Place Model File

Place the YOLO segmentation model in either the repository root or `models/`.

Recommended:

```text
models/yolo26x-seg.pt
```

Set the model path:

```powershell
$env:COCO_MODEL="models/yolo26x-seg.pt"
```

Optional speed settings:

```powershell
$env:DETECT_WIDTH="960"
$env:RETINA_MASKS="0"
```

---

## 4. Place Input Videos

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

If real input videos cannot be shared, use dummy videos in:

```text
data/dummy/
```

---

## 5. Run Minimal Chaos Reproduction

```powershell
python src/chaos.py --mode encrypt --in data/input/test6.mkv --out runs/chaos_cipher.mkv --key demo_key_change_me --roi_sidecar runs/rois.jsonl --payload runs/payload.jsonl --keystream_dump runs/keystream.bin --detect_width 960
```

Then decrypt:

```powershell
python src/chaos.py --mode decrypt --in runs/chaos_cipher.mkv --out runs/chaos_decrypted.mkv --key demo_key_change_me --roi_sidecar runs/rois.jsonl --payload runs/payload.jsonl
```

Expected files:

```text
runs/chaos_cipher.mkv
runs/chaos_decrypted.mkv
runs/rois.jsonl
runs/payload.jsonl
runs/payload.jsonl.manifest.json
```

---

## 6. Run Minimal ChaCha20-Poly1305 Reproduction

```powershell
python src/chacha.py --mode encrypt --in data/input/test6.mkv --out runs/chacha_cipher.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha.jsonl --payload runs/payload_chacha.jsonl --keystream_dump runs/keystream_chacha.bin --detect_width 960
```

Then decrypt:

```powershell
python src/chacha.py --mode decrypt --in runs/chacha_cipher.mkv --out runs/chacha_decrypted.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha.jsonl --payload runs/payload_chacha.jsonl
```

Expected files:

```text
runs/chacha_cipher.mkv
runs/chacha_decrypted.mkv
runs/rois_chacha.jsonl
runs/payload_chacha.jsonl
runs/payload_chacha.jsonl.manifest.json
```

---

## 7. Run Full-Frame Baseline

```powershell
python baselines/full_frame_encrypt.py --mode encrypt --in data/input/test6.mkv --out runs/full_frame_cipher.mkv --key demo_key_change_me --keystream_dump runs/keystream_full_frame.bin
```

Then decrypt:

```powershell
python src/chacha.py --mode decrypt --in runs/chacha_cipher.mkv --out runs/chacha_decrypted.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha.jsonl --payload runs/payload_chacha.jsonl
```

If this fails because the script uses different arguments, run:

```powershell
python baselines/full_frame_encrypt.py --help
```

---

## 8. Run Test Framework

Use the final test framework folder:

```powershell
cd experiments/tests/unified_framework3
python run_all.py --help
```

Then run with paths appropriate to the script.

Example pattern:

```powershell
python run_all.py --input ..\..\..\data\input\test6.mkv --out_dir ..\..\..\runs\tests --key demo_key_change_me
```

Return to repository root:

```powershell
cd ..\..\..
```

---

## 9. Check Generated Outputs

After reproduction, check:

```powershell
dir runs
```

Expected generated file types include:

```text
*.mkv
*.jsonl
*.manifest.json
*.bin
```

These files are generated outputs and should not normally be committed.

---

## 10. Clean Generated Outputs

To reset generated outputs:

```powershell
Remove-Item runs\* -Recurse -Force
New-Item -ItemType File -Path runs\.gitkeep -Force
```

---

## 11. Common Problems

### Missing `cv2`

```powershell
pip install opencv-python
```

### YOLO model not found

```powershell
$env:COCO_MODEL="models/yolo26x-seg.pt"
```

### CUDA or PyTorch mismatch

Install the PyTorch version matching your CUDA setup.

### TensorRT engine not portable

Use the `.pt` model instead, or regenerate the `.engine` locally.
