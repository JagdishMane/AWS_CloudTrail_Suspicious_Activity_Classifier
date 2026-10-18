import hashlib
import ipaddress
import json
import math
from collections import Counter
from datetime import datetime, timezone

TARGET = "CloudTrail_Activity_Risk"
LABEL_VERSION = "cloudtrail-weak-label-v1"
CATS = ["event_source", "event_name", "region", "identity_type", "mfa", "error_code", "ip_kind", "agent_family"]
NUMS = ["hour_sin", "hour_cos", "weekday", "is_error", "is_auth_failure", "read_only"]
TAMPER = {"StopLogging", "DeleteTrail", "UpdateTrail", "PutEventSelectors", "DeleteEventDataStore", "StopEventDataStoreIngestion"}
IAM = {"CreateAccessKey", "UpdateAccessKey", "CreateLoginProfile", "UpdateLoginProfile", "AttachUserPolicy", "AttachRolePolicy", "AttachGroupPolicy", "PutUserPolicy", "PutRolePolicy", "PutGroupPolicy", "CreatePolicyVersion", "SetDefaultPolicyVersion", "UpdateAssumeRolePolicy", "AddUserToGroup", "DeactivateMFADevice", "DeleteVirtualMFADevice"}
S3 = {"PutBucketPolicy", "DeleteBucketPolicy", "PutBucketAcl", "PutBucketPublicAccessBlock", "DeleteBucketPublicAccessBlock"}
KMS = {"DisableKey", "ScheduleKeyDeletion", "PutKeyPolicy", "CreateGrant"}
CONTEXTUAL = {"ListBuckets", "ListObjects", "ListObjectsV2", "AssumeRole", "GetSessionToken", "GetFederationToken", "RunInstances", "AuthorizeSecurityGroupIngress", "CreateSnapshot", "ModifySnapshotAttribute"}

def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()

def normalize(event):
    """Only request-time fields. Missing MFA remains unknown, never false."""
    if not isinstance(event, dict):
        raise ValueError("Event must be an object")
    for key in ("eventTime", "eventName", "eventSource"):
        if not isinstance(event.get(key), str) or not event[key]:
            raise ValueError("Required CloudTrail field missing: " + key)
    dt = datetime.fromisoformat(event["eventTime"].replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("eventTime must include a timezone")
    dt = dt.astimezone(timezone.utc)
    identity = event.get("userIdentity") or {}
    if not isinstance(identity, dict):
        raise ValueError("userIdentity must be an object")
    session = identity.get("sessionContext") or {}
    attrs = session.get("attributes") or {}
    mfa = str(attrs.get("mfaAuthenticated", "unknown")).lower()
    if mfa not in ("true", "false"):
        mfa = "unknown"
    source_ip = str(event.get("sourceIPAddress") or "")
    try:
        ip = ipaddress.ip_address(source_ip)
        ip_kind = "public" if ip.is_global else "nonpublic"
    except ValueError:
        ip_kind = "aws_service" if source_ip.endswith("amazonaws.com") else "unknown"
    error = str(event.get("errorCode") or "none")
    auth = any(x in error.lower() for x in ("accessdenied", "unauthorized", "authfailure", "invalidclienttoken", "unrecognizedclient"))
    agent = str(event.get("userAgent") or "").lower()
    family = next((label for needle, label in [("aws-cli", "cli"), ("boto", "python_sdk"), ("console", "console"), ("mozilla", "browser"), ("aws-sdk", "sdk"), ("amazonaws.com", "aws_service")] if needle in agent), "other")
    angle = 2 * math.pi * dt.hour / 24
    return {
        "event_time": dt.isoformat(), "event_source": event["eventSource"],
        "event_name": event["eventName"], "region": str(event.get("awsRegion") or "unknown"),
        "identity_type": str(identity.get("type") or "unknown"), "mfa": mfa,
        "error_code": error, "ip_kind": ip_kind, "agent_family": family,
        "hour_sin": math.sin(angle), "hour_cos": math.cos(angle), "weekday": dt.weekday(),
        "is_error": int(error != "none"), "is_auth_failure": int(auth),
        "read_only": {"true": 1, "false": 0}.get(str(event.get("readOnly", "unknown")).lower(), -1),
        "ip_hash": digest(source_ip) if source_ip else "missing",
        "principal_hash": digest(identity.get("arn") or identity.get("principalId") or "missing"),
    }

def weak_label(row):
    """Risk policy, not evidence of a confirmed attack. Reasons never enter X."""
    name, service = row["event_name"], row["event_source"]
    reasons = []
    if row["identity_type"] == "Root": reasons.append("root_activity")
    if service == "cloudtrail.amazonaws.com" and name in TAMPER: reasons.append("trail_change")
    if service == "iam.amazonaws.com" and name in IAM: reasons.append("iam_sensitive_change")
    if service == "s3.amazonaws.com" and name in S3: reasons.append("s3_access_change")
    if service == "kms.amazonaws.com" and name in KMS: reasons.append("kms_sensitive_change")
    if name in CONTEXTUAL and (row["mfa"] == "false" or row["is_auth_failure"]):
        reasons.append("sensitive_api_with_context")
    if row["is_auth_failure"]: reasons.append("authorization_failure")
    return int(bool(reasons)), reasons

def fit_encoder(rows):
    # Vocabulary and frequency references are fit on training rows only.
    vocab = {c: [v for v, _ in Counter(str(r[c]) for r in rows).most_common(64)] for c in CATS}
    event_counts = dict(Counter(r["event_source"] + "/" + r["event_name"] for r in rows))
    ips = sorted({r["ip_hash"] for r in rows if r["ip_hash"] != "missing"})
    return {"vocab": vocab, "event_counts": event_counts, "known_ips": ips,
            "n_train": len(rows), "label_version": LABEL_VERSION, "schema_version": 1}

def feature_names(encoder):
    # Fixed safe numeric names avoid forbidden characters in XGBoost feature names.
    names = NUMS + ["event_frequency", "unseen_ip"]
    for col in CATS:
        names += [col + "__" + str(i) for i in range(len(encoder["vocab"][col]) + 1)]
    return names

def transform(rows, encoder):
    import numpy as np
    out = np.zeros((len(rows), len(feature_names(encoder))), dtype=np.float32)
    maps = {c: {v: i for i, v in enumerate(encoder["vocab"][c])} for c in CATS}
    known_ips = set(encoder["known_ips"])
    for i, r in enumerate(rows):
        for j, col in enumerate(NUMS): out[i, j] = float(r[col])
        out[i, len(NUMS)] = encoder["event_counts"].get(r["event_source"] + "/" + r["event_name"], 0) / encoder["n_train"]
        out[i, len(NUMS) + 1] = int(r["ip_hash"] != "missing" and r["ip_hash"] not in known_ips)
        offset = len(NUMS) + 2
        for col in CATS:
            out[i, offset + maps[col].get(str(r[col]), len(maps[col]))] = 1
            offset += len(maps[col]) + 1
    if not np.isfinite(out).all(): raise ValueError("Nonfinite features")
    return out

def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
