import json
CATS = ["event_source", "event_name", "region", "identity_type", "mfa", "error_code", "ip_kind", "agent_family"]
NUMS = ["hour_sin", "hour_cos", "weekday", "is_error", "is_auth_failure", "read_only"]
def preprocess_handler(inference_record):
    payload = json.loads(inference_record.endpoint_input.data)
    rows = payload["instances"]
    # This schedule expects one event per invocation. Batch clients use a separate monitor.
    if len(rows) != 1: raise ValueError("Monitoring requires one event per invocation")
    return {k: rows[0][k] for k in CATS + NUMS}
