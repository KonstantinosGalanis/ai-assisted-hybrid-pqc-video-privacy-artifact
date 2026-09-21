# Experiments

This document explains how the experiment code in this repository is organized and how it relates to the paper artifact.

---

## Experiment Branches

The artifact contains three main experimental branches.

### Branch A: Chaos ROI Encryption

Main script:

```text
src/chaos.py
```

Purpose:

- Detect sensitive regions of interest.
- Encrypt only ROI pixels/audio using the chaotic-map pipeline.
- Store recovery data in `payload.jsonl`.
- Store ROI geometry/masks in `rois.jsonl`.

Expected generated files:

```text
runs/chaos_cipher.mkv
runs/rois.jsonl
runs/payload.jsonl
runs/payload.jsonl.manifest.json
runs/keystream.bin
```

---

### Branch B: ChaCha20-Poly1305 ROI Encryption

Main script:

```text
src/chacha.py
```

Purpose:

- Detect sensitive regions of interest.
- Encrypt ROI/audio data using ChaCha20-Poly1305 AEAD.
- Store recovery data in `payload_chacha.jsonl`.
- Store ROI geometry/masks in `rois_chacha.jsonl`.

Expected generated files:

```text
runs/chacha_cipher.mkv
runs/rois_chacha.jsonl
runs/payload_chacha.jsonl
runs/payload_chacha.jsonl.manifest.json
runs/keystream_chacha.bin
```

---

### Baseline: Full-Frame Encryption

Main script:

```text
baselines/full_frame_encrypt.py
```

Purpose:

- Encrypt or protect the whole frame rather than only ROIs.
- Provide a comparison point for speed, privacy leakage, and output characteristics.

Expected generated files:

```text
runs/full_frame_cipher.mkv
```

---

## Test Suites

The repository may contain multiple test-suite folders:

```text
experiments/tests/unified_framework/
experiments/tests/unified_framework2/
experiments/tests/unified_framework3/
```

Use the most recent/final version for paper reproduction. In this artifact, `unified_framework3` is expected to be the final test-suite location unless the paper states otherwise.

Check each test suite with:

```powershell
python experiments/tests/unified_framework3/run_all.py --help
```

---

## Typical Experiment Flow

A normal experiment run is:

1. Place input videos in `data/input/`.
2. Run Chaos ROI encryption/decryption.
3. Run ChaCha20-Poly1305 ROI encryption/decryption.
4. Run the full-frame baseline.
5. Run test scripts from `experiments/tests/`.
6. Store all generated outputs in `runs/`.
7. Do not commit generated outputs unless intentionally publishing result samples.

---

## Suggested Input Naming

For paper figures/tables, use stable names:

```text
data/input/test1.mkv
data/input/test2.mkv
data/input/test3.mkv
data/input/test4.mkv
data/input/test5.mkv
data/input/test6.mkv
```

Or:

```text
data/input/v1.mkv
data/input/v2.mkv
data/input/v3.mkv
data/input/v4.mkv
data/input/v5.mkv
data/input/v6.mkv
```

Use the naming convention that matches the paper.

---

## Generated Files

Generated files should usually stay inside `runs/`.

Examples:

```text
runs/chaos_cipher.mkv
runs/chacha_cipher.mkv
runs/full_frame_cipher.mkv
runs/rois.jsonl
runs/payload.jsonl
runs/keystream.bin
```

These are ignored by Git by default.

---

## Notes About Full-Frame Baseline Tests

Some ROI-specific tests may not apply to the full-frame baseline, especially tests that require ROI masks, detector leakage on ROI-only protected frames, or payload/sidecar recovery behavior.

When reporting full-frame results, clearly mark ROI-only tests as not applicable.

---

## Re-running Experiments

For full command examples, see:

```text
docs/RUNNING.md
```

For a fresh-clone protocol, see:

```text
docs/REPRODUCIBILITY.md
```
