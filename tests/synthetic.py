"""Small synthetic CloudTrail generator so the unit tests run without AWS or the real dataset."""
from __future__ import annotations

import gzip
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROUTINE = [("ec2.amazonaws.com", "DescribeInstances"), ("s3.amazonaws.com", "ListBuckets"),
           ("sts.amazonaws.com", "AssumeRole"), ("s3.amazonaws.com", "GetBucketAcl"),
           ("ec2.amazonaws.com", "RunInstances"), ("lambda.amazonaws.com", "ListFunctions")]
RISKY = [("cloudtrail.amazonaws.com", "StopLogging"), ("iam.amazonaws.com", "CreateAccessKey"),
         ("s3.amazonaws.com", "PutBucketPolicy"), ("signin.amazonaws.com", "ConsoleLogin")]
AGENTS = ["aws-cli/1.16.0 Python/3.6", "Boto3/1.9.86 Python/3.7.3", "signin.amazonaws.com", "[S3Console/0.4]"]


def make_events(n_events: int = 6000, seed: int = 0,
                start: str = "2017-02-12", end: str = "2020-10-07") -> list[dict]:
    """Random CloudTrail-like events; ~2 % are risky and arrive in short bursts with errors."""
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    times = pd.to_datetime(np.sort(rng.uniform(t0.value, t1.value, n_events)).astype("int64"), utc=True)
    actors = [f"arn:aws:iam::123456789012:user/user{i}" for i in range(12)] + ["arn:aws:iam::123456789012:root"]
    events = []
    for i, ts in enumerate(times):
        risky = rng.random() < 0.02
        source, name = RISKY[rng.integers(len(RISKY))] if risky else ROUTINE[rng.integers(len(ROUTINE))]
        actor = actors[rng.integers(len(actors))]
        burst = 1 + int(risky) * int(rng.integers(1, 5))  # risky actors do several calls in a burst
        for j in range(burst):
            event = {
                "eventVersion": "1.05",
                "eventID": str(uuid.UUID(int=int(rng.integers(0, 2**63)) * 2**64 + i * 10 + j)),
                "eventTime": (ts + pd.Timedelta(seconds=30 * j)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "eventSource": source, "eventName": name if j == 0 else ROUTINE[0][1],
                "eventType": "AwsApiCall", "awsRegion": rng.choice(["us-east-1", "us-west-2", "eu-west-1"]),
                "sourceIPAddress": f"10.0.{rng.integers(0, 4)}.{rng.integers(1, 50)}",
                "userAgent": AGENTS[rng.integers(len(AGENTS))],
                "readOnly": "false" if risky else rng.choice(["true", "false"]),
                "managementEvent": True,
                "userIdentity": {"type": "Root" if actor.endswith("root") else "IAMUser", "arn": actor,
                                 "sessionContext": {"attributes": {"mfaAuthenticated": "false"}}},
            }
            if risky and rng.random() < 0.5:
                event["errorCode"] = "AccessDenied"
            events.append(event)
    # duplicate a few events, as the public archive does
    events += events[:50]
    return events


def write_cloudtrail_files(directory: Path, n_files: int = 3, **kwargs) -> list[Path]:
    """Write synthetic events as CloudTrail-style .json.gz files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    events = make_events(**kwargs)
    paths = []
    for k, chunk in enumerate(np.array_split(np.arange(len(events)), n_files)):
        path = directory / f"flaws_cloudtrail{k:02d}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"Records": [events[i] for i in chunk]}, handle)
        paths.append(path)
    return paths
