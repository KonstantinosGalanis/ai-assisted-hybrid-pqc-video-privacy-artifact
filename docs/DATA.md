# Data and Video Inputs

This document explains how videos should be handled in this artifact.

---

## Input Folder

Input videos should be placed in:

```text
data/input/
```

Suggested naming:

```text
data/input/test1.mkv
data/input/test2.mkv
data/input/test3.mkv
data/input/test4.mkv
data/input/test5.mkv
data/input/test6.mkv
```

or:

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

## Dummy / Synthetic Videos

If the original experimental videos contain people, faces, license plates, private locations, copyrighted material, or other sensitive content, do not upload them publicly.

Instead, place public dummy videos in:

```text
data/dummy/
```

Dummy videos should preserve the important technical properties needed for reproducibility:

```text
similar resolution
similar frame rate
similar duration
similar codec/container
similar object classes if possible
```

---

## Privacy Warning

ROI video encryption experiments may involve sensitive visual data.

Before committing any video to GitHub, verify that it is legal and ethical to redistribute it.

Do not upload videos that expose:

```text
faces without consent
license plates
private addresses
private surveillance scenes
copyrighted video material
sensitive locations
```

---

## Git LFS

Videos should be tracked using Git LFS.

Recommended commands:

```powershell
git lfs install
git lfs track "*.mkv"
git lfs track "*.mp4"
git add .gitattributes
```

---

## Generated Files

Generated outputs should go to:

```text
runs/
```

Do not commit generated files such as:

```text
cipher videos
decrypted videos
payload files
ROI sidecar files
manifest files
keystream dumps
```
