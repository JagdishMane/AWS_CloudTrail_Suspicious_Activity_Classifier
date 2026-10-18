import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import tarfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
import ijson
from common import CATS, NUMS, TARGET, LABEL_VERSION, normalize, weak_label, digest

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/opt/ml/processing/input")
    p.add_argument("--output", default="/opt/ml/processing/output")
    p.add_argument("--sample-fraction", type=float, default=1.0)
    p.add_argument("--gap-hours", type=float, default=1.0)
    a = p.parse_args()
    if not 0 < a.sample_fraction <= 1: raise ValueError("sample-fraction must be in (0,1]")
    if a.gap_hours < 0: raise ValueError("gap-hours must be nonnegative")
    archives = list(Path(a.input).glob("*.tar"))
    if len(archives) != 1: raise ValueError("Input must contain exactly one .tar archive")
    out = Path(a.output)
    for name in ("train", "validation", "test", "reports"):
        (out / name).mkdir(parents=True, exist_ok=True)
    db_path = out.parent / "events.sqlite"
    if db_path.exists(): db_path.unlink()  # This job's disposable staging database only.
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE events (id TEXT PRIMARY KEY, ts TEXT, principal TEXT, label INTEGER, body TEXT)")
    counts, fields, names, reasons, missing_fields = Counter(), Counter(), Counter(), Counter(), Counter()
    tracked_fields = ("eventTime", "eventName", "eventSource", "awsRegion", "sourceIPAddress",
                      "userAgent", "errorCode", "readOnly", "userIdentity")
    first, last = None, None
    h = hashlib.sha256()
    with archives[0].open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    with tarfile.open(archives[0], "r:*") as archive:
        for member in archive:
            path = PurePosixPath(member.name.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or ":" in member.name or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
            if member.isdir(): continue
            if not member.isfile() or not member.name.endswith(".json.gz"):
                raise ValueError("Expected only gzip JSON files")
            counts["gzip_members"] += 1
            # No extraction to disk: incrementally parse Records.item from gzip.
            with archive.extractfile(member) as raw, gzip.GzipFile(fileobj=raw) as stream:
                member_events = 0
                for event in ijson.items(stream, "Records.item", use_float=True):
                    member_events += 1
                    counts["raw_events"] += 1
                    if isinstance(event, dict):
                        fields.update(event.keys())
                        for field in tracked_fields:
                            if field not in event or event[field] is None or event[field] == "":
                                missing_fields[field] += 1
                        identity = event.get("userIdentity")
                        if not isinstance(identity, dict) or not identity.get("type"):
                            missing_fields["userIdentity.type"] += 1
                        attrs = identity.get("sessionContext", {}).get("attributes", {}) if isinstance(identity, dict) else {}
                        if (not isinstance(attrs, dict) or "mfaAuthenticated" not in attrs
                                or attrs.get("mfaAuthenticated") is None
                                or str(attrs.get("mfaAuthenticated")).strip() == ""):
                            missing_fields["userIdentity.sessionContext.attributes.mfaAuthenticated"] += 1
                    try: row = normalize(event)
                    except (ValueError, TypeError, AttributeError):
                        counts["invalid_events"] += 1
                        continue
                    first = min(first, row["event_time"]) if first else row["event_time"]
                    last = max(last, row["event_time"]) if last else row["event_time"]
                    key = str(event.get("eventID") or digest(json.dumps(event, sort_keys=True)))
                    if int(digest(key)[:16], 16) / 2**64 >= a.sample_fraction:
                        counts["sampled_out"] += 1
                        continue
                    y, why = weak_label(row)
                    row[TARGET] = y
                    row["label_reasons"] = why
                    row["event_key"] = digest(key)
                    cur = db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?)",
                                     (key, row["event_time"], row["principal_hash"], y, json.dumps(row)))
                    if cur.rowcount:
                        counts["unique_selected"] += 1
                        names[row["event_name"]] += 1
                        reasons.update(why)
                    else: counts["duplicates_selected"] += 1
                    if counts["raw_events"] % 10000 == 0: db.commit()
                if member_events == 0: raise ValueError("Empty or unsupported JSON member; expected Records array")
    db.commit()
    db.execute("CREATE INDEX event_time_idx ON events(ts)")
    n = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if n < 100: raise ValueError("Too few usable events")
    def cutoff(fraction):
        return db.execute("SELECT ts FROM events ORDER BY ts,id LIMIT 1 OFFSET ?", (int(n*fraction),)).fetchone()[0]
    t1, t2 = cutoff(.70), cutoff(.85)
    if t1 >= t2: raise ValueError("Insufficient distinct timestamps for temporal splitting")
    gap = timedelta(hours=a.gap_hours)
    train_end = (datetime.fromisoformat(t1) - gap).isoformat()
    val_end = (datetime.fromisoformat(t2) - gap).isoformat()
    splits = {"train": ("ts < ?", (train_end,)),
              "validation": ("ts >= ? AND ts < ?", (t1, val_end)),
              "test": ("ts >= ?", (t2,))}
    summaries, principals = {}, {}
    for name, (where, params) in splits.items():
        summary = Counter()
        principal_set = set()
        times = []
        with (out / name / "events.jsonl").open("w") as f:
            for ts, principal, y, body in db.execute("SELECT ts,principal,label,body FROM events WHERE " + where + " ORDER BY ts,id", params):
                f.write(body + "\n")
                summary[str(y)] += 1
                principal_set.add(principal)
                if not times: times.append(ts)
                end = ts
        if min(summary.get("0", 0), summary.get("1", 0)) < 10:
            raise ValueError("Each temporal split needs at least 10 examples of each class; inspect data, do not silently randomize")
        summaries[name] = {"rows": sum(summary.values()), "classes": dict(summary), "start": times[0], "end": end}
        principals[name] = principal_set
    # Train-only monitor reference: deterministic first 10,000 chronologically ordered rows.
    with (out / "train" / "events.jsonl").open() as f, (out / "reports" / "monitor_baseline.csv").open("w", newline="") as g:
        writer = csv.DictWriter(g, fieldnames=CATS + NUMS)
        writer.writeheader()
        for i, line in enumerate(f):
            if i >= 10000: break
            r = json.loads(line)
            writer.writerow({k: r[k] for k in CATS + NUMS})
    profile = {"archive_sha256": h.hexdigest(), "archive_bytes": archives[0].stat().st_size,
               "counts": dict(counts), "observed_fields": dict(fields),
               "missing_field_counts": {k: {"count": v, "fraction": v / max(counts["raw_events"], 1)} for k, v in sorted(missing_fields.items())},
               "time_min": first, "time_max": last,
               "top_events_selected": names.most_common(20), "label_reason_counts": dict(reasons),
               "label_version": LABEL_VERSION, "sample_fraction": a.sample_fraction, "gap_hours": a.gap_hours,
               "splits": summaries, "embargoed_rows": n - sum(s["rows"] for s in summaries.values()),
               "validation_principals_also_in_train": len(principals["validation"] & principals["train"]),
               "test_principals_also_in_train": len(principals["test"] & principals["train"])}
    (out / "reports" / "data_profile.json").write_text(json.dumps(profile, indent=2))
    print(json.dumps({"counts": dict(counts), "splits": summaries}))
    db.close()

if __name__ == "__main__": main()
