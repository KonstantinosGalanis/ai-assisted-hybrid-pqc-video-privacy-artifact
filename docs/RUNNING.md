# Running the Artifact

This document explains how to run each main component of the ROI video encryption artifact.

Run all commands from the repository root unless a section says otherwise.

Expected structure:

```text
roi-video-crypto-artifact/
├── src/
│   ├── chaos.py
│   └── chacha.py
├── baselines/
│   └── full_frame_encrypt.py
├── configs/
│   └── custom_botsort_privacy_v2.yaml
├── data/
│   └── input/
│       └── test6.mkv
├── experiments/
│   └── tests/
└── runs/
```

---

## 1. Activate Environment

```powershell
conda activate roi-video-crypto
```

If you are using the original local environment:

```powershell
conda activate facecrypt
```

---

## 2. Optional Runtime Settings

These environment variables can be set before running the scripts.

```powershell
$env:COCO_MODEL="yolo26x-seg.pt"
$env:DETECT_WIDTH="960"
$env:RETINA_MASKS="0"
```

If the model is stored in the `models/` folder:

```powershell
$env:COCO_MODEL="models/yolo26x-seg.pt"
```

If using TensorRT:

```powershell
$env:COCO_MODEL="models/yolo26x-seg.engine"
```

TensorRT `.engine` files are hardware-specific and should be generated locally. Do not commit `.engine` files.

---

# Main Pipelines

---

## 3. Chaos ROI Encryption

```powershell
python src/chaos.py `
  --mode encrypt `
  --in data/input/test6.mkv `
  --out runs/chaos_cipher.mkv `
  --key demo_key_change_me `
  --roi_sidecar runs/rois.jsonl `
  --payload runs/payload.jsonl `
  --keystream_dump runs/keystream.bin `
  --detect_width 960
```

Output files:

```text
runs/chaos_cipher.mkv
runs/rois.jsonl
runs/payload.jsonl
runs/payload.jsonl.manifest.json
runs/keystream.bin
```

---

## 4. Chaos ROI Decryption

```powershell
python src/chaos.py `
  --mode decrypt `
  --in runs/chaos_cipher.mkv `
  --out runs/chaos_decrypted.mkv `
  --key demo_key_change_me `
  --roi_sidecar runs/rois.jsonl `
  --payload runs/payload.jsonl
```

Output file:

```text
runs/chaos_decrypted.mkv
```

---

## 5. Chaos ROI Encryption With Reused ROIs

Use this when `runs/rois.jsonl` already exists and you want to skip YOLO detection.

```powershell
python src/chaos.py `
  --mode encrypt `
  --in data/input/test6.mkv `
  --out runs/chaos_cipher_reuse.mkv `
  --key demo_key_change_me `
  --roi_sidecar runs/rois.jsonl `
  --payload runs/payload_reuse.jsonl `
  --keystream_dump runs/keystream_reuse.bin `
  --detect_width 960 `
  --reuse_rois
```

---

## 6. ChaCha20-Poly1305 ROI Encryption

```powershell
python src/chacha.py `
  --mode encrypt `
  --in data/input/test6.mkv `
  --out runs/chacha_cipher.mkv `
  --key demo_key_change_me `
  --roi_sidecar runs/rois_chacha.jsonl `
  --payload runs/payload_chacha.jsonl `
  --keystream_dump runs/keystream_chacha.bin `
  --detect_width 960
```

Output files:

```text
runs/chacha_cipher.mkv
runs/rois_chacha.jsonl
runs/payload_chacha.jsonl
runs/payload_chacha.jsonl.manifest.json
runs/keystream_chacha.bin
```

---

## 7. ChaCha20-Poly1305 ROI Decryption

```powershell
python src/chacha.py `
  --mode decrypt `
  --in runs/chacha_cipher.mkv `
  --out runs/chacha_decrypted.mkv `
  --key demo_key_change_me `
  --roi_sidecar runs/rois_chacha.jsonl `
  --payload runs/payload_chacha.jsonl
```

Output file:

```text
runs/chacha_decrypted.mkv
```

---

## 8. ChaCha20-Poly1305 ROI Encryption With Reused ROIs

Use this when `runs/rois_chacha.jsonl` already exists and you want to skip YOLO detection.

```powershell
python src/chacha.py `
  --mode encrypt `
  --in data/input/test6.mkv `
  --out runs/chacha_cipher_reuse.mkv `
  --key demo_key_change_me `
  --roi_sidecar runs/rois_chacha.jsonl `
  --payload runs/payload_chacha_reuse.jsonl `
  --keystream_dump runs/keystream_chacha_reuse.bin `
  --detect_width 960 `
  --reuse_rois
```

---

# Baseline

---

## 9. Full-Frame Encryption Baseline

```powershell
python baselines/full_frame_encrypt.py `
  --in data/input/test6.mkv `
  --out runs/full_frame_cipher.mkv `
  --key demo_key_change_me
```

If the baseline script uses a different CLI format, check:

```powershell
python baselines/full_frame_encrypt.py --help
```

---

# Running on Multiple Videos

---

## 10. Chaos ROI on Six Videos

```powershell
python src/chaos.py --mode encrypt --in data/input/test1.mkv --out runs/chaos_v1.mkv --key demo_key_change_me --roi_sidecar runs/rois_v1.jsonl --payload runs/payload_v1.jsonl --keystream_dump runs/keystream_v1.bin --detect_width 960
python src/chaos.py --mode encrypt --in data/input/test2.mkv --out runs/chaos_v2.mkv --key demo_key_change_me --roi_sidecar runs/rois_v2.jsonl --payload runs/payload_v2.jsonl --keystream_dump runs/keystream_v2.bin --detect_width 960
python src/chaos.py --mode encrypt --in data/input/test3.mkv --out runs/chaos_v3.mkv --key demo_key_change_me --roi_sidecar runs/rois_v3.jsonl --payload runs/payload_v3.jsonl --keystream_dump runs/keystream_v3.bin --detect_width 960
python src/chaos.py --mode encrypt --in data/input/test4.mkv --out runs/chaos_v4.mkv --key demo_key_change_me --roi_sidecar runs/rois_v4.jsonl --payload runs/payload_v4.jsonl --keystream_dump runs/keystream_v4.bin --detect_width 960
python src/chaos.py --mode encrypt --in data/input/test5.mkv --out runs/chaos_v5.mkv --key demo_key_change_me --roi_sidecar runs/rois_v5.jsonl --payload runs/payload_v5.jsonl --keystream_dump runs/keystream_v5.bin --detect_width 960
python src/chaos.py --mode encrypt --in data/input/test6.mkv --out runs/chaos_v6.mkv --key demo_key_change_me --roi_sidecar runs/rois_v6.jsonl --payload runs/payload_v6.jsonl --keystream_dump runs/keystream_v6.bin --detect_width 960
```

---

## 11. ChaCha20-Poly1305 ROI on Six Videos

```powershell
python src/chacha.py --mode encrypt --in data/input/test1.mkv --out runs/chacha_v1.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha_v1.jsonl --payload runs/payload_chacha_v1.jsonl --keystream_dump runs/keystream_chacha_v1.bin --detect_width 960
python src/chacha.py --mode encrypt --in data/input/test2.mkv --out runs/chacha_v2.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha_v2.jsonl --payload runs/payload_chacha_v2.jsonl --keystream_dump runs/keystream_chacha_v2.bin --detect_width 960
python src/chacha.py --mode encrypt --in data/input/test3.mkv --out runs/chacha_v3.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha_v3.jsonl --payload runs/payload_chacha_v3.jsonl --keystream_dump runs/keystream_chacha_v3.bin --detect_width 960
python src/chacha.py --mode encrypt --in data/input/test4.mkv --out runs/chacha_v4.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha_v4.jsonl --payload runs/payload_chacha_v4.jsonl --keystream_dump runs/keystream_chacha_v4.bin --detect_width 960
python src/chacha.py --mode encrypt --in data/input/test5.mkv --out runs/chacha_v5.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha_v5.jsonl --payload runs/payload_chacha_v5.jsonl --keystream_dump runs/keystream_chacha_v5.bin --detect_width 960
python src/chacha.py --mode encrypt --in data/input/test6.mkv --out runs/chacha_v6.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha_v6.jsonl --payload runs/payload_chacha_v6.jsonl --keystream_dump runs/keystream_chacha_v6.bin --detect_width 960
```

---

# Test Scripts

---

## 12. Run Experiment Tests

The repository may include one or more test folders:

```text
experiments/tests/unified_framework/
experiments/tests/unified_framework2/
experiments/tests/unified_framework3/
```

Run the final test framework first:

```powershell
cd experiments/tests/unified_framework3
python run_all.py
```

If arguments are required:

```powershell
python run_all.py --help
```

Example pattern:

```powershell
python run_all.py `
  --input ..\..\..\data\input\test6.mkv `
  --out_dir ..\..\..\runs\tests `
  --key demo_key_change_me
```

Return to repository root:

```powershell
cd ..\..\..
```

---

# NIST / Randomness Testing

---

## 13. Keystream and Ciphertext Randomness Input

Both ROI pipelines can produce keystream/ciphertext dump files:

```text
runs/keystream.bin
runs/keystream_chacha.bin
```

Example command pattern:

```powershell
python experiments/tests/unified_framework3/video_crypto_test_framework.py `
  --cipher runs/chaos_cipher.mkv `
  --out runs/nist_chaos `
  --nist `
  --nist_backend nistrng `
  --max_bits 2000000 `
  --max_frames 200
```

For ChaCha20-Poly1305:

```powershell
python experiments/tests/unified_framework3/video_crypto_test_framework.py `
  --cipher runs/chacha_cipher.mkv `
  --out runs/nist_chacha `
  --nist `
  --nist_backend nistrng `
  --max_bits 2000000 `
  --max_frames 200
```

Check exact script options with:

```powershell
python experiments/tests/unified_framework3/video_crypto_test_framework.py --help
```

---

# Expected Output Summary

After running the main commands, `runs/` may contain:

```text
chaos_cipher.mkv
chaos_decrypted.mkv
chacha_cipher.mkv
chacha_decrypted.mkv
full_frame_cipher.mkv

rois.jsonl
payload.jsonl
payload.jsonl.manifest.json

rois_chacha.jsonl
payload_chacha.jsonl
payload_chacha.jsonl.manifest.json

keystream.bin
keystream_chacha.bin
```

These files are generated artifacts and should not normally be committed to Git.

---

# Troubleshooting

## `ModuleNotFoundError: No module named 'cv2'`

Install OpenCV:

```powershell
pip install opencv-python
```

Or:

```powershell
conda install -c conda-forge opencv
```

## YOLO model not found

Set the model path:

```powershell
$env:COCO_MODEL="models/yolo26x-seg.pt"
```

Or place the model in the repository root.

## TensorRT engine not found

Use the `.pt` model instead:

```powershell
$env:COCO_MODEL="models/yolo26x-seg.pt"
```

TensorRT `.engine` files are local generated files and should not be committed.

## Generated files accidentally appear in Git

Check:

```powershell
git status
```

If generated files appear, remove them from staging:

```powershell
git restore --staged runs/
git restore --staged *.jsonl
git restore --staged *.bin
git restore --staged *.engine
```

---

# Recommended Minimal Reproduction

Run these four commands from the repository root:

```powershell
conda activate roi-video-crypto

python src/chaos.py --mode encrypt --in data/input/test6.mkv --out runs/chaos_cipher.mkv --key demo_key_change_me --roi_sidecar runs/rois.jsonl --payload runs/payload.jsonl --keystream_dump runs/keystream.bin --detect_width 960

python src/chaos.py --mode decrypt --in runs/chaos_cipher.mkv --out runs/chaos_decrypted.mkv --key demo_key_change_me --roi_sidecar runs/rois.jsonl --payload runs/payload.jsonl

python src/chacha.py --mode encrypt --in data/input/test6.mkv --out runs/chacha_cipher.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha.jsonl --payload runs/payload_chacha.jsonl --keystream_dump runs/keystream_chacha.bin --detect_width 960

python src/chacha.py --mode decrypt --in runs/chacha_cipher.mkv --out runs/chacha_decrypted.mkv --key demo_key_change_me --roi_sidecar runs/rois_chacha.jsonl --payload runs/payload_chacha.jsonl
```

If these commands finish successfully, the two main ROI encryption pipelines are working.
