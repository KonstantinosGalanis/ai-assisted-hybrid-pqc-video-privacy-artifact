# Data Folder

This folder contains input videos and optional dummy videos for reproducing the artifact.

---

## Folder Layout

```text
data/
├── README.md
├── input/
│   ├── test1.mkv
│   ├── test2.mkv
│   ├── test3.mkv
│   ├── test4.mkv
│   ├── test5.mkv
│   └── test6.mkv
└── dummy/
```

---

## `data/input/`

Use this folder for the videos used in the paper experiments, if they can be redistributed.

Expected examples:

```text
data/input/test1.mkv
data/input/test2.mkv
data/input/test3.mkv
data/input/test4.mkv
data/input/test5.mkv
data/input/test6.mkv
```

---

## `data/dummy/`

Use this folder for dummy or synthetic videos when the original videos cannot be shared.

Dummy videos should preserve the approximate technical properties of the original videos:

```text
resolution
frame rate
duration
codec/container
object classes when possible
```

---

## Privacy Notice

Only commit videos that are legal and ethical to redistribute.

Do not commit videos containing private faces, license plates, private locations, copyrighted material, or surveillance footage without permission.

For more detail, see:

```text
docs/DATA.md
```
