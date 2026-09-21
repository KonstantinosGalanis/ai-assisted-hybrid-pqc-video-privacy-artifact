# Models

This folder is reserved for YOLO segmentation models used by the ROI detection pipeline.

---

## Expected Model

The default model expected by the code is:

```text
yolo26x-seg.pt
```

Recommended location:

```text
models/yolo26x-seg.pt
```

Set the model path with:

```powershell
$env:COCO_MODEL="models/yolo26x-seg.pt"
```

If the model is placed in the repository root instead, use:

```powershell
$env:COCO_MODEL="yolo26x-seg.pt"
```

---

## Runtime Detection Settings

Recommended speed settings:

```powershell
$env:DETECT_WIDTH="960"
$env:RETINA_MASKS="0"
```

`DETECT_WIDTH=960` runs YOLO on resized frames for speed.

`RETINA_MASKS=0` may reduce mask precision but usually improves speed.

---

## TensorRT Engines

TensorRT `.engine` files are not included because they are hardware-specific.

Do not commit:

```text
*.engine
```

If TensorRT is needed, generate the engine locally on the target machine.

Example usage after local generation:

```powershell
$env:COCO_MODEL="models/yolo26x-seg.engine"
```

---

## Git LFS

Model weights can be large. If you commit `.pt` model files, track them with Git LFS:

```powershell
git lfs install
git lfs track "*.pt"
git add .gitattributes
```
