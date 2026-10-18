# Experiment: event-level weak-label baseline (Jagdish Mane)

`CloudTrail_SageMaker_MLOps.ipynb` is an earlier end-to-end workflow that scores **individual CloudTrail events**
(10 % sample of the archive) with an XGBoost model trained in the notebook kernel, registers it in the
SageMaker Model Registry, deploys a real-time endpoint and creates CloudWatch alarms. Screenshots of that run
are in `resources/`.

## Why the final system works on actor × 5-minute windows instead

The event-level weak label is a **deterministic function of the model's own inputs**: an event is labelled
suspicious if `identity_type == "Root"`, if `event_name` is in a list of sensitive APIs, or if the call failed
authorisation — and `identity_type`, `event_name`, `event_source`, `mfa` and the error flag are all model features.
The model can therefore reproduce the rule exactly, which is why it reaches validation ROC-AUC 1.000 and test F1 0.995
on a 28–31 % positive rate. These scores measure agreement with the rule, not the ability to find suspicious
*behaviour*, and the model still missed a `CreateAccessKey` scenario in the endpoint test.

The final system (`notebooks/01`–`06`) addresses this by:
* aggregating events into **actor × 5-minute windows** with strictly-prior history features,
* **excluding every column that defines the label** from the model inputs (checked by unit tests),
* predicting suspicious activity **in the window or in the next 30 minutes** (early warning),
* evaluating on an untouched future year with ranking metrics and analyst alert budgets.

This experiment is kept as the documented baseline that motivated those design decisions.
