"""Pipeline step 3 - EvaluateModel (SageMaker Processing job).

Scores the validation period (to set the alert thresholds) and the untouched test period,
then writes evaluation.json. The pipeline's QualityGate step reads
``metrics.test_roc_auc`` and ``metrics.test_ap_lift`` from that file.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from cloudtrail_mlops import evaluation, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/opt/ml/processing/model")
    parser.add_argument("--data-dir", default="/opt/ml/processing/data")
    parser.add_argument("--output-dir", default="/opt/ml/processing/evaluation")
    args = parser.parse_args()
    data = Path(args.data_dir)

    booster = model.load_booster(model.extract_model(Path(args.model_dir) / "model.tar.gz", Path("/tmp/model")))

    validation = pd.read_csv(data / "validation" / "validation.csv", header=None)
    val_meta = pd.read_csv(data / "metadata" / "validation_metadata.csv")
    test_x = pd.read_csv(data / "test" / "test_features.csv", header=None)
    test_y = pd.read_csv(data / "test" / "test_labels.csv")["label"].to_numpy()
    test_meta = pd.read_csv(data / "metadata" / "test_metadata.csv")

    report = evaluation.evaluation_report(
        val_y=validation[0].to_numpy(), val_scores=model.predict(booster, validation.iloc[:, 1:]),
        val_times=val_meta["window_start"],
        test_y=test_y, test_scores=model.predict(booster, test_x), test_times=test_meta["window_start"],
    )
    report["quality_gate"] = evaluation.quality_gate(
        {"roc_auc": report["metrics"]["test_roc_auc"], "ap_lift": report["metrics"]["test_ap_lift"]})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
