"""Unit tests run by GitHub Actions on every push / pull request."""
import json

import numpy as np
import pandas as pd
import pytest

from cloudtrail_mlops import config, dataset, evaluation, ingest
from cloudtrail_mlops.features import validate_feature_mart


# ---------------------------------------------------------------- ingestion
def test_flatten_event_extracts_identity_and_booleans():
    event = {"eventID": "e1", "eventTime": "2019-01-01T00:00:00Z", "eventName": "StopLogging",
             "readOnly": "false", "sourceIPAddress": "1.2.3.4",
             "userIdentity": {"type": "Root", "arn": "arn:aws:iam::1:root",
                              "sessionContext": {"attributes": {"mfaAuthenticated": "true"}}}}
    row = ingest.flatten_event(event, "file.json.gz", 0)
    assert row["actor_id"] == "arn:aws:iam::1:root"
    assert row["read_only"] is False and row["mfa_authenticated"] is True
    assert row["identity_type"] == "Root"


def test_actor_falls_back_to_source_ip():
    assert ingest.actor_id_of({"sourceIPAddress": "9.9.9.9", "userIdentity": {}}) == "9.9.9.9"


def test_safe_extract_rejects_path_traversal(tmp_path):
    import io
    import tarfile
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("../evil.json.gz")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    with pytest.raises(ValueError):
        ingest.safe_extract(archive, tmp_path / "out")


# ---------------------------------------------------------------- feature mart
def test_feature_mart_passes_validation(feature_mart):
    checks = validate_feature_mart(feature_mart)
    assert all(checks.values()), checks


def test_deduplication_removed_repeated_events(feature_mart):
    audit = feature_mart.attrs["audit"].iloc[0]
    assert audit["unique_events"] < audit["raw_events"]


def test_history_features_only_use_the_past(feature_mart):
    # The first window of every actor has no history, so its prior counters must be empty / zero.
    first = feature_mart.sort_values("window_start").groupby("actor_id").head(1)
    assert first["prior_writes_24h"].fillna(0).eq(0).all()
    assert first["first_actor_event_count"].ge(1).all()


def test_target_contains_detection_label(feature_mart):
    labelled = feature_mart[feature_mart[config.TARGET_COLUMN].notna()]
    assert (labelled[config.TARGET_COLUMN] >= labelled["target_detection_current"]).all()


# ---------------------------------------------------------------- dataset
def test_feature_contract_has_no_leakage():
    assert dataset.leakage_violations(config.FEATURE_COLUMNS) == []
    assert dataset.leakage_violations(["target_x", "audit_tamper_count_current"]) != []
    assert not set(config.DRIFT_PRONE_FEATURES) & set(config.FEATURE_COLUMNS)


def test_split_is_chronological_with_embargo(feature_mart):
    splits = dataset.chronological_split(feature_mart)
    assert splits["train"].window_start.max() < pd.Timestamp(config.VALIDATION_START)
    assert splits["validation"].window_start.max() < pd.Timestamp(config.TEST_START)
    assert splits["test"].window_start.min() >= pd.Timestamp(config.TEST_START)


def test_encoder_handles_unseen_categories_and_roundtrips(feature_mart):
    enc = dataset.FeatureEncoder(min_frequency=1).fit(feature_mart)
    sample = feature_mart.head(3).copy()
    sample.loc[sample.index[0], "last_service"] = "never-seen.amazonaws.com"
    encoded = enc.transform(sample)
    assert list(encoded.columns) == enc.feature_names
    assert encoded.iloc[0]["last_service=__other__"] == 1.0
    assert not encoded.isna().any().any()
    again = dataset.FeatureEncoder.from_dict(json.loads(json.dumps(enc.to_dict())))
    pd.testing.assert_frame_equal(again.transform(sample), encoded)


def test_prepare_model_inputs_writes_xgboost_format(feature_mart, tmp_path):
    result = dataset.prepare_model_inputs(feature_mart, tmp_path)
    train = pd.read_csv(tmp_path / "train" / "train.csv", header=None)
    assert set(train[0].unique()) <= {0, 1}                       # label is the first column
    assert train.shape[1] == len(result["encoder"].feature_names) + 1
    test = pd.read_csv(tmp_path / "test" / "test_features.csv", header=None)
    assert test.shape[1] == len(result["encoder"].feature_names)  # no label at inference


# ---------------------------------------------------------------- evaluation
def test_threshold_for_budget_produces_requested_alert_volume():
    ts = pd.Series(pd.date_range("2020-01-01", periods=1000, freq="h", tz="UTC"))
    scores = np.linspace(0, 1, 1000)
    threshold = evaluation.threshold_for_budget(scores, ts, alerts_per_day=2)
    expected = round(2 * evaluation.days_covered(ts))
    assert (scores >= threshold).sum() == expected


def test_quality_gate_pass_and_fail():
    assert evaluation.quality_gate({"roc_auc": 0.95, "ap_lift": 20})["passed"]
    assert not evaluation.quality_gate({"roc_auc": 0.60, "ap_lift": 20})["passed"]


def test_psi_detects_shift_and_ignores_identical_data():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    assert evaluation.psi(base, rng.normal(0, 1, 5000)) < evaluation.PSI_WARN
    assert evaluation.psi(base, rng.normal(2, 1, 5000)) > evaluation.PSI_ALERT


def test_evaluation_report_exposes_keys_used_by_the_pipeline_gate():
    rng = np.random.default_rng(0)
    times = pd.Series(pd.date_range("2019-01-01", periods=2000, freq="h", tz="UTC"))
    y = (rng.random(2000) < 0.05).astype(int)
    scores = np.clip(y * 0.6 + rng.random(2000) * 0.5, 0, 1)
    report = evaluation.evaluation_report(y, scores, times, y, scores, times)
    assert {"test_roc_auc", "test_ap_lift"} <= set(report["metrics"])
    assert report["thresholds"]["high"] >= report["thresholds"]["medium"]
    json.dumps(report)  # must be serialisable for evaluation.json
