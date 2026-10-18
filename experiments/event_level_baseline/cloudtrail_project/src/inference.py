import json
from pathlib import Path
import xgboost as xgb
from common import CATS, NUMS, transform, feature_names

def model_fn(model_dir):
    p = Path(model_dir)
    model = xgb.Booster()
    model.load_model(str(p / "model.json"))
    return model, json.loads((p / "encoder.json").read_text()), json.loads((p / "metadata.json").read_text())

def input_fn(body, content_type):
    if content_type.split(";")[0] != "application/json": raise ValueError("Use application/json")
    data = json.loads(body)
    rows = data.get("instances") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("Provide 1 to 100 normalized events in instances")
    required = set(CATS + NUMS + ["ip_hash"])
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row): raise ValueError("Feature schema mismatch")
        if set(row) != required: raise ValueError("Unexpected fields; send only the documented feature contract")
        if any(not isinstance(row[k], str) for k in CATS + ["ip_hash"]): raise ValueError("Categorical fields must be strings")
    return rows

def predict_fn(rows, bundle):
    model, encoder, meta = bundle
    scores = model.predict(xgb.DMatrix(transform(rows, encoder), feature_names=feature_names(encoder)))
    return {"predictions": [{"risk_score": float(s), "CloudTrail_Activity_Risk": int(s >= meta["threshold"]),
            "class_name": "suspicious" if s >= meta["threshold"] else "normal",
            "threshold": meta["threshold"], "label_source": meta["label_source"]} for s in scores]}

def output_fn(prediction, accept):
    if accept.split(";")[0] != "application/json": raise ValueError("Use application/json")
    return json.dumps(prediction), "application/json"
