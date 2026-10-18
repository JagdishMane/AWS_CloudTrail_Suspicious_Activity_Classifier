"""Project-wide constants: data locations, split dates, feature contract and model settings.

Every notebook and pipeline step imports these values instead of hard-coding them.
"""

# ---------------------------------------------------------------------------------------
# Data source and S3 layout
# ---------------------------------------------------------------------------------------
DATA_URL = "https://summitroute.com/downloads/flaws_cloudtrail_logs.tar"
DATA_MD5 = "4f0481c6be700ffc7610dbbf8cee5578"
EXPECTED_RAW_EVENTS = 1_939_207

S3_PREFIX = "cloudtrail-mlops"  # all project objects live under s3://<bucket>/cloudtrail-mlops/
S3_LAYOUT = {
    "raw": "raw",                         # original tar archive, untouched
    "events": "processed/events",         # normalised events, one Parquet file per source file
    "features": "features",               # actor x 5-minute feature mart (Parquet + CSV)
    "model_input": "model-input",         # encoded train / validation / test CSVs for XGBoost
    "training": "training",               # SageMaker training + tuning job outputs
    "batch": "batch-transform",           # batch inference input / output
    "datacapture": "monitoring/datacapture",
    "monitoring": "monitoring",
    "pipeline": "pipeline",
}

# ---------------------------------------------------------------------------------------
# Unit of prediction and target
# ---------------------------------------------------------------------------------------
WINDOW_MINUTES = 5          # one row = one actor's activity in one 5-minute window
TARGET_HORIZON_MINUTES = 30  # label looks at the current window and the next 30 minutes
TARGET_COLUMN = "target_current_or_future_30m"
KEY_COLUMNS = ["actor_id", "window_start"]

# ---------------------------------------------------------------------------------------
# Chronological split (never random: CloudTrail is a time series)
#   train      : 2017-02-12 .. 2018-12-31
#   validation : 2019 (early stopping, tuning, alert thresholds)
#   test       : 2020 (touched once, for the final report and the quality gate)
# A 1-day embargo before each cut stops the 30-minute look-ahead label leaking across splits.
# ---------------------------------------------------------------------------------------
VALIDATION_START = "2019-01-01T00:00:00Z"
TEST_START = "2020-01-01T00:00:00Z"
EMBARGO_DAYS = 1

# ---------------------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------------------
NUMERIC_FEATURES = [
    # activity inside the window
    "distinct_services", "distinct_apis", "distinct_regions", "distinct_source_ips",
    "distinct_ua_families", "error_count", "auth_error_count", "write_count",
    "read_only_known_count", "read_only_coverage_current", "management_count",
    "readonly_count", "destructive_count", "authentication_count", "off_hours_count",
    "weekend_count",
    # novelty of the behaviour compared with the actor's own past
    "first_actor_event_count", "new_actor_api_count", "new_actor_ip_count",
    "new_actor_ua_count", "globally_rare_api_count",
    # strictly-prior rolling history of the actor
    "prior_errors_24h", "prior_auth_errors_24h", "prior_writes_24h", "prior_new_apis_7d",
    "prior_ip_mass_24h", "prior_active_windows_24h",
    # rates and calendar encodings
    "error_rate_current", "write_rate_current", "new_api_rate_current", "new_ip_rate_current",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
CATEGORICAL_FEATURES = ["last_service", "last_region", "last_ua_family"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that DEFINE the weak label. They are kept in the feature mart for auditing but
# must never be used as model inputs (that would be target leakage).
POLICY_COLUMNS = [
    "root_event_count_current", "root_no_mfa_count_current", "audit_tamper_count_current",
    "public_exposure_count_current", "credential_privilege_count_current",
    "console_no_mfa_count_current", "label_signal_current", "policy_signal_category_current",
]
# Features that grow with the cumulative size of the log rather than with behaviour.
# They drift by construction, so they are excluded from the model (see notebook 03).
DRIFT_PRONE_FEATURES = [
    "event_count", "mean_log_actor_prior", "mean_log_actor_api_prior", "prior_events_1h",
    "prior_events_24h", "prior_events_7d", "prior_mean_events_per_window_7d",
    "prior_active_windows_7d", "event_burst_vs_7d",
]
FORBIDDEN_INPUT_PREFIXES = ("target_", "label_")

# Categories seen fewer times than this in training are folded into "__other__".
MIN_CATEGORY_FREQUENCY = 20

# ---------------------------------------------------------------------------------------
# Model and operating point
# ---------------------------------------------------------------------------------------
XGBOOST_VERSION = "1.7-1"  # SageMaker built-in XGBoost container version
DEFAULT_HYPERPARAMETERS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "num_round": 400,
    "early_stopping_rounds": 40,
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.6,
    "min_child_weight": 4,
    "gamma": 0.1,
    "lambda": 2.0,
    "scale_pos_weight": 5,
}

# Alert thresholds are set from analyst capacity, not from a fixed 0.5 cut-off:
#   MEDIUM = top 5 alerts/day on the validation period, HIGH = top 1 alert/day.
MEDIUM_ALERTS_PER_DAY = 5.0
HIGH_ALERTS_PER_DAY = 1.0

# Release gate used by the CI/CD pipeline before a model may be registered.
MIN_TEST_ROC_AUC = 0.80
MIN_TEST_AP_LIFT = 5.0
