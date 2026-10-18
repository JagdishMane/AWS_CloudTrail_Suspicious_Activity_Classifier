# Data catalogue

All project data is stored in Amazon S3 (never in git). The bucket is the SageMaker default bucket of the
AWS account that runs the notebooks: `s3://sagemaker-<region>-<account-id>/`. Everything lives under one
prefix, `cloudtrail-mlops/`, so the whole system can be found and cleaned up in one place.

| S3 location (under `cloudtrail-mlops/`) | Written by | Content | Format |
|---|---|---|---|
| `raw/flaws_cloudtrail_logs.tar` | notebook 01 | Original public dataset, untouched (MD5 `4f0481c6be700ffc7610dbbf8cee5578`) | tar of 20 gzip-JSON files |
| `processed/events/part-000NN.parquet` | notebook 01 | 1,939,207 events flattened to 17 typed columns, one file per source file | Parquet (zstd) |
| `features/feature_mart.parquet` | notebook 02 | One row per actor × 5-minute window: 44 numeric features, 3 categorical features, policy audit columns, 3 weak targets | Parquet |
| `features/csv/feature_mart.csv` | notebook 02 | Same mart as CSV — input of the SageMaker Pipeline | CSV with header |
| `features/feature-store/` | Feature Store | Offline store of feature group `cloudtrail-actor-window-features` | Parquet (managed by SageMaker) |
| `model-input/train/train.csv`, `model-input/validation/validation.csv` | notebook 03 / pipeline | Encoded training data: label first, no header | CSV |
| `model-input/test/test_features.csv`, `test_labels.csv` | notebook 03 / pipeline | Encoded 2020 test features (no label) and labels | CSV |
| `model-input/encoder.json` | notebook 03 / pipeline | Category vocabulary for one-hot encoding at inference time | JSON |
| `model-input/baseline/train_with_header.csv` | notebook 03 | Model Monitor baseline dataset | CSV with header |
| `training/models/`, `training/tuning/` | SageMaker training / tuning jobs | `model.tar.gz` artefacts | tar.gz |
| `training/evaluation/evaluation.json` | notebook 03 | Thresholds, validation and test metrics, quality gate | JSON |
| `batch-transform/output/test_features.csv.out` | notebook 04 | Batch-inference risk scores for 2020 | CSV |
| `monitoring/datacapture/` | endpoint | Captured requests and responses | JSON Lines |
| `monitoring/baseline/`, `monitoring/reports/` | notebook 05 | Model Monitor statistics, constraints and violation reports | JSON |
| `pipeline/` | SageMaker Pipeline | Per-execution outputs of every pipeline step | mixed |
| `project_state.json` | all notebooks | Names of resources created by earlier notebooks (endpoint, model package, ...) | JSON |

## Source dataset
* **flAWS.cloud CloudTrail logs**, published by Summit Route:
  <https://summitroute.com/blog/2020/10/09/public_dataset_of_cloudtrail_logs_from_flaws_cloud/>
* 1,939,207 events (≈ 1.23 M unique after de-duplication on `eventID`), 2017-02-12 → 2020-10-07.
* Account IDs and IPs come from a public security-training (CTF) environment; there is no PHI or payment data.

## Feature mart columns
* **Keys:** `actor_id` (IAM ARN, principal, access key or source IP), `window_start` (UTC, 5-minute buckets).
* **Model inputs (38):** listed in `src/cloudtrail_mlops/config.py` → `NUMERIC_FEATURES` and `CATEGORICAL_FEATURES`.
* **Audit-only columns (never model inputs):** `*_count_current` policy counters, `label_signal_current`, `policy_signal_category_current`.
* **Targets:** `target_detection_current`, `target_forecast_30m`, `target_current_or_future_30m` (project target).
