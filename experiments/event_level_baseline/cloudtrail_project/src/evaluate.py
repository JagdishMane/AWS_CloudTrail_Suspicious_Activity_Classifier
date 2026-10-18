import argparse
import json
import tarfile
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, brier_score_loss, log_loss, precision_recall_curve, roc_curve
from common import TARGET, load_rows, transform, feature_names

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/opt/ml/processing/model")
    p.add_argument("--data-dir", default="/opt/ml/processing")
    p.add_argument("--output", default="/opt/ml/processing/evaluation")
    a = p.parse_args()
    # Read only the exact files we need, never extract arbitrary archive paths.
    with tarfile.open(Path(a.model_dir) / "model.tar.gz", "r:gz") as t:
        files = {}
        for name in ("model.json", "encoder.json", "metadata.json", "feature_importance.json"):
            m = t.getmember(name)
            if not m.isfile(): raise ValueError("Invalid model artifact")
            files[name] = t.extractfile(m).read()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_bytes(files["model.json"])
    model = xgb.Booster()
    model.load_model(str(out / "model.json"))
    encoder = json.loads(files["encoder.json"])
    meta = json.loads(files["metadata.json"])
    report = {"label_source": "weak_rules", "threshold": meta["threshold"], "metadata": meta}
    for split in ("validation", "test"):
        rows = load_rows(Path(a.data_dir) / split / "events.jsonl")
        y = np.array([r[TARGET] for r in rows])
        if len(set(y)) != 2: raise ValueError("Both classes required for evaluation")
        scores = model.predict(xgb.DMatrix(transform(rows, encoder), feature_names=feature_names(encoder)))
        pred = (scores >= meta["threshold"]).astype(int)
        values = {"roc_auc": roc_auc_score(y, scores), "average_precision": average_precision_score(y, scores),
                  "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
                  "f1": f1_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred),
                  "brier": brier_score_loss(y, scores), "log_loss": log_loss(y, scores), "prevalence": y.mean()}
        report[split] = {k: {"value": float(v)} for k, v in values.items()}
        report[split]["confusion_matrix"] = confusion_matrix(y, pred, labels=[0,1]).tolist()
        report[split]["rows"] = len(rows)
        # A constant-normal reference is a metric baseline, not a second trained model.
        report[split]["always_normal_accuracy"] = float((y == 0).mean())
        precision, recall, _ = precision_recall_curve(y, scores)
        fpr, tpr, _ = roc_curve(y, scores)
        def thin(x): return np.linspace(0, len(x)-1, min(200, len(x)), dtype=int)
        pi, ri = thin(precision), thin(fpr)
        report[split]["curves"] = {"precision": precision[pi].tolist(), "recall": recall[pi].tolist(), "fpr": fpr[ri].tolist(), "tpr": tpr[ri].tolist()}
        report[split]["score_histogram"] = np.histogram(scores, bins=np.linspace(0,1,11))[0].tolist()
    report["feature_importance"] = json.loads(files["feature_importance.json"])
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / "model.json").unlink()
    print(json.dumps({s: {k:v for k,v in report[s].items() if k in ("precision","recall","f1","average_precision")} for s in ("validation", "test")}))

if __name__ == "__main__": main()
