"""Pipeline step 1 - ValidateAndPreprocess (SageMaker Processing job).

Reads the feature mart CSV from S3, runs the data-quality checks, then writes the
chronologically split, encoded train / validation / test files for XGBoost.
The step FAILS (non-zero exit) if any data-quality check fails, which stops the pipeline.
"""
import argparse
import json
import sys
from pathlib import Path

from cloudtrail_mlops import dataset
from cloudtrail_mlops.features import validate_feature_mart


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).glob("*.csv"))
    if not files:
        sys.exit(f"No feature mart CSV found in {args.input_dir}")
    mart = dataset.load_mart(files[0])
    print(f"Loaded feature mart {files[0].name}: {mart.shape}")

    checks = validate_feature_mart(mart)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_quality.json").write_text(json.dumps(checks, indent=2))
    print("Data-quality checks:", json.dumps(checks, indent=2))
    if not all(checks.values()):
        sys.exit("Data-quality checks failed - stopping the pipeline")

    result = dataset.prepare_model_inputs(mart, output_dir)
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
