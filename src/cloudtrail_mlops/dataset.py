"""Model dataset: chronological split, categorical encoding and XGBoost CSV export.

The SageMaker built-in XGBoost algorithm expects headerless CSV files with the label in the
first column and only numeric values. The three categorical features are therefore one-hot
encoded with a vocabulary learned on the training split only; the vocabulary is saved as
JSON so the exact same encoding can be applied at inference time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

OTHER = "__other__"
MISSING = "__missing__"


def leakage_violations(columns: list[str]) -> list[str]:
    """Return any proposed input column that would leak the label."""
    return [c for c in columns
            if c in config.POLICY_COLUMNS or c.startswith(config.FORBIDDEN_INPUT_PREFIXES)]


def load_mart(path_or_frame) -> pd.DataFrame:
    """Read the feature mart (Parquet or CSV) and normalise the timestamp column."""
    if isinstance(path_or_frame, pd.DataFrame):
        mart = path_or_frame.copy()
    elif str(path_or_frame).endswith(".csv"):
        mart = pd.read_csv(path_or_frame, low_memory=False)
    else:
        mart = pd.read_parquet(path_or_frame)
    mart["window_start"] = pd.to_datetime(mart["window_start"], utc=True)
    return mart.sort_values(["window_start", "actor_id"]).reset_index(drop=True)


def chronological_split(
    mart: pd.DataFrame,
    validation_start: str = config.VALIDATION_START,
    test_start: str = config.TEST_START,
    embargo_days: int = config.EMBARGO_DAYS,
    target: str = config.TARGET_COLUMN,
) -> dict[str, pd.DataFrame]:
    """Split by time: train < validation_start <= validation < test_start <= test.

    Rows in the ``embargo_days`` before each cut are dropped: their 30-minute look-ahead
    label could otherwise peek into the next split.
    """
    labelled = mart[mart[target].notna()].copy()
    labelled[target] = labelled[target].astype(int)
    val_cut, test_cut = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    embargo = pd.Timedelta(days=embargo_days)
    ts = labelled["window_start"]
    return {
        "train": labelled[ts < val_cut - embargo].reset_index(drop=True),
        "validation": labelled[(ts >= val_cut) & (ts < test_cut - embargo)].reset_index(drop=True),
        "test": labelled[ts >= test_cut].reset_index(drop=True),
    }


def split_summary(splits: dict[str, pd.DataFrame], target: str = config.TARGET_COLUMN) -> pd.DataFrame:
    """One row per split: size, period, positives and prevalence."""
    return pd.DataFrame([{
        "split": name, "rows": len(frame),
        "start": frame["window_start"].min(), "end": frame["window_start"].max(),
        "positives": int(frame[target].sum()), "prevalence": round(float(frame[target].mean()), 5),
        "actors": frame["actor_id"].nunique(),
    } for name, frame in splits.items()])


@dataclass
class FeatureEncoder:
    """Numeric pass-through + one-hot categories (rare / unseen values -> '__other__')."""

    numeric: list[str] = field(default_factory=lambda: list(config.NUMERIC_FEATURES))
    categorical: list[str] = field(default_factory=lambda: list(config.CATEGORICAL_FEATURES))
    min_frequency: int = config.MIN_CATEGORY_FREQUENCY
    vocab: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "FeatureEncoder":
        for column in self.categorical:
            counts = frame[column].fillna(MISSING).astype(str).value_counts()
            self.vocab[column] = sorted(counts[counts >= self.min_frequency].index.tolist())
        return self

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric)
        for column in self.categorical:
            names += [f"{column}={value}" for value in self.vocab[column] + [OTHER]]
        return names

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a purely numeric frame whose columns are ``feature_names``."""
        if not self.vocab:
            raise RuntimeError("FeatureEncoder must be fitted before transform")
        # Missing numeric values only occur in the rolling-history counters when an actor has
        # no earlier activity, so 0 is the semantically correct value (and keeps CSV payloads simple).
        out = frame.reindex(columns=self.numeric).apply(pd.to_numeric, errors="coerce").astype("float64").fillna(0.0)
        blocks = [out]
        for column in self.categorical:
            known = self.vocab[column]
            values = frame[column].fillna(MISSING).astype(str) if column in frame else pd.Series(MISSING, index=frame.index)
            values = values.where(values.isin(known), OTHER)
            onehot = pd.get_dummies(values).reindex(columns=known + [OTHER], fill_value=0).astype("float64")
            onehot.columns = [f"{column}={value}" for value in onehot.columns]
            blocks.append(onehot.set_index(out.index))
        return pd.concat(blocks, axis=1)[self.feature_names]

    def to_dict(self) -> dict:
        return {"numeric": self.numeric, "categorical": self.categorical,
                "min_frequency": self.min_frequency, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureEncoder":
        return cls(numeric=data["numeric"], categorical=data["categorical"],
                   min_frequency=data["min_frequency"], vocab=data["vocab"])

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "FeatureEncoder":
        return cls.from_dict(json.loads(Path(path).read_text()))


def to_xgboost_csv(features: pd.DataFrame, labels: np.ndarray | None, path: Path) -> None:
    """Write the built-in XGBoost format: no header, label first (omit label for inference)."""
    data = features.copy()
    if labels is not None:
        data.insert(0, "label", np.asarray(labels).astype(int))
    data.to_csv(path, header=False, index=False, float_format="%.6g")


def prepare_model_inputs(mart: pd.DataFrame, output_dir: Path) -> dict:
    """Split + encode the mart and write every file the training / inference steps need.

    Files written to ``output_dir``:
      train/train.csv, validation/validation.csv        label + features (training job channels)
      test/test_features.csv, test/test_labels.csv      features only + labels (batch transform / evaluation)
      encoder.json                                      vocabulary for inference-time encoding
      baseline/train_with_header.csv                    Model Monitor baseline (features only, header)
      metadata/{validation,test}_metadata.csv          actor / time keys (thresholds, reporting)
    """
    output_dir = Path(output_dir)
    for sub in ("train", "validation", "test", "baseline", "metadata"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    splits = chronological_split(mart)
    encoder = FeatureEncoder().fit(splits["train"])
    encoded = {name: encoder.transform(frame) for name, frame in splits.items()}
    labels = {name: frame[config.TARGET_COLUMN].to_numpy() for name, frame in splits.items()}

    to_xgboost_csv(encoded["train"], labels["train"], output_dir / "train" / "train.csv")
    to_xgboost_csv(encoded["validation"], labels["validation"], output_dir / "validation" / "validation.csv")
    to_xgboost_csv(encoded["test"], None, output_dir / "test" / "test_features.csv")
    pd.Series(labels["test"], name="label").to_csv(output_dir / "test" / "test_labels.csv", index=False)
    splits["test"][["actor_id", "window_start"]].to_csv(output_dir / "metadata" / "test_metadata.csv", index=False)
    splits["validation"][["actor_id", "window_start"]].to_csv(output_dir / "metadata" / "validation_metadata.csv", index=False)
    encoded["train"].to_csv(output_dir / "baseline" / "train_with_header.csv", index=False)
    encoder.save(output_dir / "encoder.json")

    summary = split_summary(splits)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    return {"splits": splits, "encoder": encoder, "encoded": encoded, "labels": labels, "summary": summary}
