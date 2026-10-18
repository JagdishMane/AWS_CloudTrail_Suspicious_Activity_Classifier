"""Raw CloudTrail ingestion: download, verify, extract, and flatten JSON events to Parquet.

CloudTrail files are gzip-compressed JSON documents of the form {"Records": [event, ...]}.
Each event is a nested dictionary; we flatten the fields the feature pipeline needs into
one typed row per event and write one Parquet file per source file.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

BOOLEAN_COLUMNS = ("management_event", "read_only", "mfa_authenticated")


def download_file(url: str, destination: Path, chunk_size: int = 2**20) -> Path:
    """Stream ``url`` to ``destination`` (skipped if the file already exists)."""
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size):
                handle.write(chunk)
    partial.replace(destination)
    return destination


def md5_of(path: Path, block_size: int = 2**20) -> str:
    """MD5 checksum, used to confirm the archive matches the published dataset."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> list[Path]:
    """Extract a tar archive, refusing absolute paths, '..' and links (tar-slip protection)."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe path in archive: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Link in archive: {member.name}")
        tar.extractall(destination)
    return sorted(destination.rglob("*.json.gz"))


def _get(event: dict, dotted_path: str) -> Any:
    """Safe nested lookup: _get(e, 'userIdentity.arn') -> value or None."""
    current: Any = event
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _to_bool(value: Any) -> bool | None:
    """CloudTrail stores some booleans as the strings 'true' / 'false'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def actor_id_of(event: dict) -> str:
    """Who performed the call: IAM ARN, else principal id, else access key, else source IP."""
    for path in ("userIdentity.arn", "userIdentity.principalId", "userIdentity.accessKeyId"):
        value = _get(event, path)
        if value:
            return str(value)
    return str(event.get("sourceIPAddress") or "UNKNOWN")


def flatten_event(event: dict, source_file: str, record_index: int) -> dict:
    """Flatten one CloudTrail event into the columns used downstream."""
    return {
        "source_file": source_file,
        "source_record_index": record_index,
        "event_id": event.get("eventID"),
        "event_time": event.get("eventTime"),
        "event_source": event.get("eventSource"),
        "event_name": event.get("eventName"),
        "event_type": event.get("eventType"),
        "aws_region": event.get("awsRegion"),
        "source_ip": event.get("sourceIPAddress"),
        "user_agent": event.get("userAgent"),
        "error_code": event.get("errorCode"),
        "error_message": event.get("errorMessage"),
        "management_event": _to_bool(event.get("managementEvent")),
        "read_only": _to_bool(event.get("readOnly")),
        "identity_type": _get(event, "userIdentity.type"),
        "mfa_authenticated": _to_bool(_get(event, "userIdentity.sessionContext.attributes.mfaAuthenticated")),
        "actor_id": actor_id_of(event),
    }


def events_to_frame(records: Iterable[dict], source_file: str) -> pd.DataFrame:
    """Flatten a list of raw events into a typed DataFrame."""
    frame = pd.DataFrame([flatten_event(e, source_file, i) for i, e in enumerate(records)])
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    frame["source_record_index"] = frame["source_record_index"].astype("int64")
    for column in frame.columns:
        if column in BOOLEAN_COLUMNS:
            frame[column] = frame[column].astype("boolean")
        elif column not in ("event_time", "source_record_index"):
            frame[column] = frame[column].astype("string")
    return frame


def read_cloudtrail_file(path: Path) -> list[dict]:
    """Read one .json.gz (or .json) CloudTrail file and return its Records list."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle).get("Records", [])


def convert_to_parquet(json_files: Iterable[Path], output_dir: Path) -> pd.DataFrame:
    """Convert each CloudTrail file to one Parquet file; returns a per-file row count summary.

    Processing one file at a time keeps memory low enough for a small Studio instance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for part, path in enumerate(sorted(json_files)):
        frame = events_to_frame(read_cloudtrail_file(path), Path(path).name)
        frame.to_parquet(output_dir / f"part-{part:05d}.parquet", index=False, compression="zstd")
        summary.append({"source_file": Path(path).name, "events": len(frame),
                        "first_event": frame["event_time"].min(), "last_event": frame["event_time"].max()})
    return pd.DataFrame(summary)
