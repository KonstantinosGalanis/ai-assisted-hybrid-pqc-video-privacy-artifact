# Full-frame testing suite

These files are adapted for `framework_full_encrypt.py`.

Main changes:

- Removed `--roi_sidecar`, `rois.jsonl`, `payload.jsonl`, and manifest dependencies.
- All metrics run on the whole frame because the encryption is full-frame AES-256-CTR.
- Key-dependent tests call:

```bash
python framework_full_encrypt.py --mode encrypt --in plain.mp4 --out encrypted.mp4 --key YOUR_KEY
python framework_full_encrypt.py --mode decrypt --in encrypted.mp4 --out decrypted.mp4 --key YOUR_KEY
```

Recommended basic run:

```bash
python run_all.py --dataset_dir /path/to/dataset --out results_full --framework_path framework_full_encrypt.py --master_key YOUR_KEY --generate_cipher_if_missing --generate_decrypted_if_missing
```

Expected dataset names:

- `plain.*`, `original.*`, or `source.*`
- `cipher.*`, `encrypted.*`, or `enc.*` unless you use `--generate_cipher_if_missing`
- optional `decrypted.*` or `recovered.*`
- optional `master_key.txt`
- optional `bitstream.bin` or `cipher_dump.bin` for NIST; otherwise decoded cipher frames are used

Even though some filenames still end in `_roi.py`, the code inside is full-frame and does not accept `--roi_sidecar`.
