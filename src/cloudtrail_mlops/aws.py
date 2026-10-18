"""Small helpers shared by the notebooks: SageMaker session, S3 paths and project state.

Resource names created by one notebook (training job, endpoint, ...) are stored in a small
JSON "state" object in S3 so the next notebook - or a teammate in another Studio space of
the same AWS account - can pick them up.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import config

STATE_KEY = f"{config.S3_PREFIX}/project_state.json"


@dataclass
class AwsContext:
    session: object          # sagemaker.Session
    region: str
    bucket: str
    role: str

    def s3_uri(self, area: str, *parts: str) -> str:
        """s3://<bucket>/cloudtrail-mlops/<area folder>/<parts...>"""
        key = "/".join([config.S3_PREFIX, config.S3_LAYOUT.get(area, area), *parts])
        return f"s3://{self.bucket}/{key}"

    def prefix(self, area: str) -> str:
        return f"{config.S3_PREFIX}/{config.S3_LAYOUT.get(area, area)}"


def get_context() -> AwsContext:
    """Create the SageMaker session. In Studio the execution role is the Learner Lab 'LabRole'."""
    import sagemaker

    session = sagemaker.Session()
    try:
        role = sagemaker.get_execution_role()
    except ValueError:  # running outside SageMaker: fall back to the Learner Lab role name
        import boto3
        account = boto3.client("sts").get_caller_identity()["Account"]
        role = f"arn:aws:iam::{account}:role/LabRole"
    return AwsContext(session=session, region=session.boto_region_name,
                      bucket=session.default_bucket(), role=role)


def load_state(ctx: AwsContext) -> dict:
    import boto3
    from botocore.exceptions import ClientError

    try:
        body = boto3.client("s3").get_object(Bucket=ctx.bucket, Key=STATE_KEY)["Body"].read()
        return json.loads(body)
    except ClientError:  # no state saved yet
        return {}


def save_state(ctx: AwsContext, **values) -> dict:
    """Merge ``values`` into the project state stored in S3 and return the new state."""
    import boto3

    state = load_state(ctx)
    state.update(values)
    boto3.client("s3").put_object(Bucket=ctx.bucket, Key=STATE_KEY,
                                  Body=json.dumps(state, indent=2, default=str).encode())
    return state
