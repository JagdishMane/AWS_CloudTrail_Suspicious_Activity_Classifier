"""Load a model trained by the SageMaker built-in XGBoost container and score feature rows."""
from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


def extract_model(model_tar_gz: Path, destination: Path) -> Path:
    """Unpack SageMaker's model.tar.gz and return the path of the 'xgboost-model' file."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(model_tar_gz, "r:gz") as tar:
        tar.extractall(destination)
    return destination / "xgboost-model"


def load_booster(model_file: Path):
    """Load the booster. Requires xgboost 1.7.x locally (same version as the container)."""
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(model_file))
    return booster


def predict(booster, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Probability of 'suspicious' for each row (columns must be in encoder.feature_names order)."""
    import xgboost as xgb

    values = features.to_numpy(dtype=float) if isinstance(features, pd.DataFrame) else np.asarray(features, dtype=float)
    # Training used early stopping: like the SageMaker serving container, only use the trees up to
    # the best validation iteration.
    best = booster.attr("best_iteration")
    iteration_range = (0, int(best) + 1) if best is not None else (0, 0)
    return booster.predict(xgb.DMatrix(values), iteration_range=iteration_range)
