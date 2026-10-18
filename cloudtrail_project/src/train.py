import argparse
import json
import os
from pathlib import Path
import numpy as np
import xgboost as xgb
from common import TARGET, LABEL_VERSION, fit_encoder, transform, feature_names, load_rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--eta", type=float, default=.1)
    p.add_argument("--num-round", type=int, default=300)
    p.add_argument("--threshold", type=float, default=.50)
    a, _ = p.parse_known_args()
    if not 0 < a.threshold < 1: raise ValueError("Threshold must be between zero and one")
    train_rows = load_rows(Path(os.environ["SM_CHANNEL_TRAIN"]) / "events.jsonl")
    val_rows = load_rows(Path(os.environ["SM_CHANNEL_VALIDATION"]) / "events.jsonl")
    encoder = fit_encoder(train_rows)
    names = feature_names(encoder)
    y = np.array([r[TARGET] for r in train_rows])
    vy = np.array([r[TARGET] for r in val_rows])
    if len(set(y)) != 2 or len(set(vy)) != 2: raise ValueError("Both classes required")
    dtrain = xgb.DMatrix(transform(train_rows, encoder), label=y, feature_names=names)
    dval = xgb.DMatrix(transform(val_rows, encoder), label=vy, feature_names=names)
    # Unweighted logistic objective: avoid presenting class-weight-shifted scores as calibrated probabilities.
    params = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
              "max_depth": a.max_depth, "eta": a.eta, "subsample": .8, "colsample_bytree": .8,
              "min_child_weight": 5, "lambda": 1.0, "seed": 42, "nthread": max(1, os.cpu_count() or 1)}
    model = xgb.train(params, dtrain, num_boost_round=a.num_round, evals=[(dtrain, "train"), (dval, "validation")], early_stopping_rounds=30)
    # Save only the selected boosting rounds, so online predictions match evaluation.
    model = model[:model.best_iteration + 1]
    out = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "model.json"))
    (out / "encoder.json").write_text(json.dumps(encoder))
    (out / "metadata.json").write_text(json.dumps({"threshold": a.threshold, "target": TARGET,
        "label_version": LABEL_VERSION, "label_source": "weak_rules", "xgboost_version": xgb.__version__,
        "num_boosted_rounds": model.num_boosted_rounds(), "params": params, "train_rows": len(train_rows),
        "validation_rows": len(val_rows)}, indent=2))
    (out / "feature_importance.json").write_text(json.dumps(model.get_score(importance_type="gain"), indent=2))

if __name__ == "__main__": main()
