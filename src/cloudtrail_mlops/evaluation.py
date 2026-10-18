"""Evaluation, alert thresholds, release gate and drift statistics.

The positive class is rare (~0.5 % of windows), so accuracy is meaningless: a model that
always says "normal" is 99.5 % accurate. We report ranking metrics (ROC-AUC, average
precision and its lift over the base rate) and recall at a fixed analyst alert budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from . import config


def days_covered(timestamps: pd.Series) -> float:
    """Length of the period in days (at least one)."""
    ts = pd.to_datetime(timestamps, utc=True)
    return max(1.0, (ts.max() - ts.min()).total_seconds() / 86_400)


def threshold_for_budget(scores: np.ndarray, timestamps: pd.Series, alerts_per_day: float) -> float:
    """Score cut-off that produces ``alerts_per_day`` alerts on average over the period."""
    n_alerts = int(round(alerts_per_day * days_covered(timestamps)))
    n_alerts = min(max(n_alerts, 1), len(scores))
    return float(np.sort(np.asarray(scores))[::-1][n_alerts - 1])


def alert_thresholds(scores: np.ndarray, timestamps: pd.Series) -> dict[str, float]:
    """MEDIUM / HIGH thresholds from analyst capacity on the validation period."""
    return {
        "medium": threshold_for_budget(scores, timestamps, config.MEDIUM_ALERTS_PER_DAY),
        "high": threshold_for_budget(scores, timestamps, config.HIGH_ALERTS_PER_DAY),
    }


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Ranking metrics + confusion-matrix metrics at ``threshold``."""
    y_true, scores = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    prevalence = float(y_true.mean())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y_true)) == 2
    ap = float(average_precision_score(y_true, scores)) if two_classes else float("nan")
    return {
        "rows": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y_true, scores)) if two_classes else float("nan"),
        "average_precision": ap,
        "ap_lift": ap / prevalence if prevalence > 0 else float("nan"),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
    }


def budget_curve(y_true: np.ndarray, scores: np.ndarray, timestamps: pd.Series,
                 budgets=(0.5, 1, 2, 5, 10, 20)) -> pd.DataFrame:
    """Precision / recall an analyst team would get at different daily alert budgets."""
    rows = []
    for budget in budgets:
        threshold = threshold_for_budget(scores, timestamps, budget)
        m = classification_metrics(y_true, scores, threshold)
        rows.append({"alerts_per_day": budget, "threshold": threshold, "alerts": m["true_positives"] + m["false_positives"],
                     "precision": m["precision"], "recall": m["recall"]})
    return pd.DataFrame(rows)


def evaluation_report(val_y, val_scores, val_times, test_y, test_scores, test_times) -> dict:
    """Thresholds from the validation period, then metrics on the untouched test period.

    The structure of this dictionary is what the pipeline's quality gate reads
    (``metrics.test_roc_auc`` / ``metrics.test_ap_lift``).
    """
    thresholds = alert_thresholds(val_scores, val_times)
    validation = classification_metrics(val_y, val_scores, thresholds["medium"])
    test = classification_metrics(test_y, test_scores, thresholds["medium"])
    test_high = classification_metrics(test_y, test_scores, thresholds["high"])
    return {
        "thresholds": thresholds,
        "validation": validation,
        "test_medium": test,
        "test_high": test_high,
        "test_alerts_per_day_medium": (test["true_positives"] + test["false_positives"]) / days_covered(test_times),
        "metrics": {"test_roc_auc": test["roc_auc"], "test_average_precision": test["average_precision"],
                    "test_ap_lift": test["ap_lift"], "test_recall_medium": test["recall"]},
        # format understood by the SageMaker Model Registry UI (shown under "Model quality")
        "binary_classification_metrics": {
            name: {"value": float(value), "standard_deviation": "NaN"}
            for name, value in (("auc", test["roc_auc"]), ("average_precision", test["average_precision"]),
                                ("precision", test["precision"]), ("recall", test["recall"]), ("f1", test["f1"]))
        },
    }


def quality_gate(test_metrics: dict, min_roc_auc: float = config.MIN_TEST_ROC_AUC,
                 min_ap_lift: float = config.MIN_TEST_AP_LIFT) -> dict:
    """Release gate: a model is only registered if it clearly beats random ranking."""
    checks = {
        f"roc_auc >= {min_roc_auc}": bool(test_metrics["roc_auc"] >= min_roc_auc),
        f"ap_lift >= {min_ap_lift}": bool(test_metrics["ap_lift"] >= min_ap_lift),
    }
    return {"passed": all(checks.values()), "checks": checks}


# ----------------------------------------------------------------------------------------
# Drift: Population Stability Index (PSI)
#   PSI < 0.10 stable | 0.10 - 0.25 moderate shift (warn) | > 0.25 major shift (alert)
# ----------------------------------------------------------------------------------------
PSI_WARN, PSI_ALERT = 0.10, 0.25
_EPS = 1e-4


def psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI of a numeric feature using baseline quantile bins."""
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    baseline, current = baseline[~np.isnan(baseline)], current[~np.isnan(current)]
    if len(baseline) == 0 or len(current) == 0:
        return 0.0
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1))[1:-1])
    if len(edges) == 0:  # (near-)constant feature: compare "equals baseline value" share
        b = np.array([1.0, _EPS])
        share_same = float(np.mean(current == baseline[0]))
        c = np.clip(np.array([share_same, 1 - share_same]), _EPS, None)
        return float(np.sum((c - b) * np.log(c / b)))
    b = np.bincount(np.digitize(baseline, edges), minlength=len(edges) + 1) / len(baseline)
    c = np.bincount(np.digitize(current, edges), minlength=len(edges) + 1) / len(current)
    b, c = np.clip(b, _EPS, None), np.clip(c, _EPS, None)
    return float(np.sum((c - b) * np.log(c / b)))


def drift_report(baseline: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """PSI per feature with a status column (OK / WARN / ALERT), worst first."""
    rows = []
    for column in columns:
        value = psi(baseline[column].to_numpy(), current[column].to_numpy())
        status = "ALERT" if value >= PSI_ALERT else "WARN" if value >= PSI_WARN else "OK"
        rows.append({"feature": column, "psi": round(value, 4), "status": status,
                     "baseline_mean": float(baseline[column].mean()), "current_mean": float(current[column].mean())})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
