# Unified Complete Encryption Evaluation Framework

Run the whole framework from **one dataset folder**.

## One command

```bash
python run_all.py /path/to/dataset_folder
```

That is the default mode now.

## What the dataset folder should contain

For a **full run** place these files in the folder (or subfolders):

- `plain.*` or `original.*` or `source.*`
- `cipher.*` or `encrypted.*` or `enc.*`
- `rois.jsonl` (or any `roi*.jsonl` sidecar)
- `bitstream.bin` or `keystream.bin`
- `master_key.txt` or `key.txt`

Optional files:

- `plain2.*`
- `cipher2.*`
- `plain_ref.*`
- `cipher_ref.*`

If some optional files are missing, `run_all.py` generates them automatically when enough inputs exist.

If `keystream.bin` is missing but `rois.jsonl` and `master_key.txt` exist, the runner derives the keystream automatically for NIST.

If `cipher.*` or `rois.jsonl` are missing but `plain.*` and `master_key.txt` exist, the runner tries to generate them using `framework_faster.py`.

## Output

By default results go to:

```text
<dataset_folder>/results_unified/
```

Main outputs:

- `master_report.json` — status of all suites
- `dataset_manifest.json` — resolved dataset assets and validation
- `01_*` to `11_*` subfolders — per-suite outputs
- `generated/` — helper assets created automatically

## Included suites

- `01_quality_metrics_roi.py`
- `02_differential_sensitivity_roi.py`
- `03_correlation_tests_roi.py`
- `04_hist_entropy_roi.py`
- `05_psd_estimation.py`
- `06_robustness_noise_occlusion_roi.py`
- `07_bandwidth_analysis.py`
- `08_perf_cycles_throughput.py`
- `09_missing_attack_eq_suite.py`
- `10_nist_runner.py`
- `11_param_sweep.py`

## Notes

- `run_all.py` is now **dataset-folder first**, not single-file first.
- It scans the dataset **recursively**, so subfolders are allowed.
- If a suite cannot run, the reason is recorded as `skipped` in `master_report.json`.
- `--master_key` still exists as an override, but the intended simple mode is to keep the key in `master_key.txt` and only pass the dataset folder.
