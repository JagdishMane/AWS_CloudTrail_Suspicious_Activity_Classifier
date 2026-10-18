"""Feature engineering: normalised events (Parquet) -> actor x 5-minute feature mart.

Implemented in DuckDB SQL because the windowed history features over ~1.2 M events are far
faster (and use less memory) in a columnar SQL engine than in pandas.

Pipeline (each step is one SQL statement below):
  1. events_dedup     drop duplicate eventIDs (the public archive repeats ~37 % of events)
  2. event_semantic   derive action / error / user-agent families and the weak-label policy flags
  3. event_causal     per-actor novelty counters that only look at the PAST (no leakage)
  4. actor_windows    aggregate events to one row per actor per 5-minute window
  5. feature_mart     strictly-prior rolling history (1 h / 24 h / 7 d) + the weak targets
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

# ----------------------------------------------------------------------------------------
# 1. De-duplicate on eventID (fall back to file + position when eventID is missing)
# ----------------------------------------------------------------------------------------
SQL_DEDUP = """
CREATE OR REPLACE TABLE events_dedup AS
SELECT * EXCLUDE (dedup_key, rn)
FROM (
    SELECT *,
           COALESCE(NULLIF(TRIM(event_id), ''), source_file || '|' || CAST(source_record_index AS VARCHAR)) AS dedup_key,
           ROW_NUMBER() OVER (
               PARTITION BY COALESCE(NULLIF(TRIM(event_id), ''), source_file || '|' || CAST(source_record_index AS VARCHAR))
               ORDER BY event_time, source_file, source_record_index) AS rn
    FROM events
    WHERE event_time IS NOT NULL
)
WHERE rn = 1
"""

# ----------------------------------------------------------------------------------------
# 2. Event semantics + weak-label policy flags
#    The four policy rules below DEFINE "suspicious". They are security-team rules of thumb,
#    not analyst-confirmed incidents, which is why we call the labels "weak labels".
# ----------------------------------------------------------------------------------------
SQL_SEMANTIC = """
CREATE OR REPLACE TABLE event_semantic AS
WITH base AS (
    SELECT *,
        COALESCE(NULLIF(TRIM(actor_id), ''), '<unknown_actor>') AS actor_key,
        CASE
            WHEN LOWER(COALESCE(user_agent, '')) LIKE '%console%'        THEN 'console'
            WHEN LOWER(COALESCE(user_agent, '')) LIKE '%aws-cli%'        THEN 'cli'
            WHEN LOWER(COALESCE(user_agent, '')) LIKE '%boto%'           THEN 'python_sdk'
            WHEN LOWER(COALESCE(user_agent, '')) LIKE '%terraform%'      THEN 'terraform'
            WHEN LOWER(COALESCE(user_agent, '')) LIKE '%cloudformation%' THEN 'cloudformation'
            WHEN LOWER(COALESCE(user_agent, '')) LIKE '%java%'           THEN 'java_sdk'
            WHEN NULLIF(TRIM(COALESCE(user_agent, '')), '') IS NULL       THEN 'missing'
            ELSE 'other'
        END AS ua_family,
        CASE
            WHEN regexp_matches(LOWER(COALESCE(event_name, '')), '^(get|list|describe|lookup|head|download|select)') THEN 'read'
            WHEN regexp_matches(LOWER(COALESCE(event_name, '')), '^(create|put|update|modify|attach|associate|enable|start|run|invoke|upload)') THEN 'change'
            WHEN regexp_matches(LOWER(COALESCE(event_name, '')), '^(delete|remove|detach|disable|stop|terminate|revoke)') THEN 'destructive'
            WHEN LOWER(COALESCE(event_name, '')) LIKE '%login%'
              OR LOWER(COALESCE(event_name, '')) LIKE '%authenticate%' THEN 'authentication'
            ELSE 'other'
        END AS action_family,
        CASE
            WHEN NULLIF(TRIM(COALESCE(error_code, '')), '') IS NULL THEN 'none'
            WHEN LOWER(error_code) LIKE '%accessdenied%' OR LOWER(error_code) LIKE '%unauthorized%' THEN 'authorization'
            WHEN LOWER(error_code) LIKE '%throttl%' THEN 'throttling'
            WHEN LOWER(error_code) LIKE '%notfound%' OR LOWER(error_code) LIKE '%nosuch%' THEN 'not_found'
            ELSE 'other'
        END AS error_family
    FROM events_dedup
), flags AS (
    SELECT *,
        CASE WHEN read_only IS NOT NULL THEN CAST(read_only = FALSE AS INTEGER)
             WHEN action_family IN ('change', 'destructive', 'authentication') THEN 1 ELSE 0 END AS is_write_event,
        CAST(read_only IS NOT NULL AS INTEGER) AS read_only_known,
        CAST(COALESCE(identity_type, '') = 'Root' AS INTEGER) AS is_root,
        CAST(COALESCE(identity_type, '') = 'Root' AND NOT COALESCE(mfa_authenticated, FALSE) AS INTEGER) AS is_root_no_mfa,
        CAST(error_family = 'authorization' AS INTEGER) AS is_auth_error,
        -- policy rule 1: someone tampers with the audit trail
        CAST(event_name IN ('StopLogging', 'DeleteTrail', 'UpdateTrail', 'PutEventSelectors', 'DeleteEventDataStore') AS INTEGER) AS flag_audit_tamper,
        -- policy rule 2: data is exposed publicly
        CAST(event_name IN ('PutBucketPolicy', 'PutBucketAcl', 'DeletePublicAccessBlock', 'PutBucketPublicAccessBlock',
                            'PutAccountPublicAccessBlock') AS INTEGER) AS flag_public_exposure,
        -- policy rule 3: new credentials or privilege escalation
        CAST(event_name IN ('CreateAccessKey', 'AttachUserPolicy', 'AttachRolePolicy', 'PutUserPolicy', 'PutRolePolicy',
                            'CreatePolicyVersion', 'SetDefaultPolicyVersion') AS INTEGER) AS flag_credential_or_privilege,
        -- policy rule 4: console login without MFA
        CAST(event_name = 'ConsoleLogin' AND NOT COALESCE(mfa_authenticated, FALSE) AS INTEGER) AS flag_console_no_mfa
    FROM base
)
SELECT *,
    CAST(flag_audit_tamper + flag_public_exposure + flag_credential_or_privilege + flag_console_no_mfa > 0 AS INTEGER) AS label_signal_current,
    CASE WHEN flag_audit_tamper = 1 THEN 'audit_tamper'
         WHEN flag_public_exposure = 1 THEN 'public_exposure'
         WHEN flag_credential_or_privilege = 1 THEN 'credential_or_privilege'
         WHEN flag_console_no_mfa = 1 THEN 'console_no_mfa'
         ELSE 'none' END AS policy_signal_category
FROM flags
"""

# ----------------------------------------------------------------------------------------
# 3. Causal novelty counters: "how many times has this actor done X BEFORE this event?"
#    ROWS ... 1 PRECEDING guarantees the current and future events are never counted.
# ----------------------------------------------------------------------------------------
SQL_CAUSAL = """
CREATE OR REPLACE TABLE event_causal AS
SELECT *,
    COUNT(*) OVER (PARTITION BY actor_key ORDER BY event_time, source_file, source_record_index
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS actor_prior_events,
    COUNT(*) OVER (PARTITION BY actor_key, event_source, event_name ORDER BY event_time, source_file, source_record_index
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS actor_api_prior_events,
    COUNT(*) OVER (PARTITION BY event_source, event_name ORDER BY event_time, source_file, source_record_index
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS global_api_prior_events,
    COUNT(*) OVER (PARTITION BY actor_key, COALESCE(source_ip, '<missing>') ORDER BY event_time, source_file, source_record_index
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS actor_ip_prior_events,
    COUNT(*) OVER (PARTITION BY actor_key, ua_family ORDER BY event_time, source_file, source_record_index
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS actor_ua_prior_events
FROM event_semantic
"""

# ----------------------------------------------------------------------------------------
# 4. Aggregate to one row per actor per window
# ----------------------------------------------------------------------------------------
SQL_WINDOWS = """
CREATE OR REPLACE TABLE actor_windows AS
SELECT
    actor_key AS actor_id,
    time_bucket(INTERVAL '{window} minutes', event_time) AS window_start,
    COUNT(*)                          AS event_count,
    COUNT(DISTINCT event_source)      AS distinct_services,
    COUNT(DISTINCT event_name)        AS distinct_apis,
    COUNT(DISTINCT aws_region)        AS distinct_regions,
    COUNT(DISTINCT source_ip)         AS distinct_source_ips,
    COUNT(DISTINCT ua_family)         AS distinct_ua_families,
    SUM(CAST(error_family <> 'none' AS INTEGER))                 AS error_count,
    SUM(is_auth_error)                                           AS auth_error_count,
    SUM(is_write_event)                                          AS write_count,
    SUM(read_only_known)                                         AS read_only_known_count,
    SUM(CAST(COALESCE(management_event, FALSE) AS INTEGER))      AS management_count,
    SUM(CAST(COALESCE(read_only, FALSE) AS INTEGER))             AS readonly_count,
    SUM(CAST(action_family = 'destructive' AS INTEGER))          AS destructive_count,
    SUM(CAST(action_family = 'authentication' AS INTEGER))       AS authentication_count,
    SUM(CAST(EXTRACT(hour FROM event_time) NOT BETWEEN 7 AND 19 AS INTEGER)) AS off_hours_count,
    SUM(CAST(EXTRACT(isodow FROM event_time) IN (6, 7) AS INTEGER))          AS weekend_count,
    SUM(CAST(actor_prior_events = 0 AS INTEGER))                 AS first_actor_event_count,
    SUM(CAST(actor_api_prior_events = 0 AS INTEGER))             AS new_actor_api_count,
    SUM(CAST(actor_ip_prior_events = 0 AS INTEGER))              AS new_actor_ip_count,
    SUM(CAST(actor_ua_prior_events = 0 AS INTEGER))              AS new_actor_ua_count,
    SUM(CAST(global_api_prior_events < 20 AS INTEGER))           AS globally_rare_api_count,
    AVG(LN(1 + actor_prior_events))                              AS mean_log_actor_prior,
    AVG(LN(1 + actor_api_prior_events))                          AS mean_log_actor_api_prior,
    arg_max(event_source, event_time)                            AS last_service,
    arg_max(aws_region, event_time)                              AS last_region,
    arg_max(ua_family, event_time)                               AS last_ua_family,
    -- policy columns: kept for auditing, never used as model inputs
    SUM(is_root)                        AS root_event_count_current,
    SUM(is_root_no_mfa)                 AS root_no_mfa_count_current,
    SUM(flag_audit_tamper)              AS audit_tamper_count_current,
    SUM(flag_public_exposure)           AS public_exposure_count_current,
    SUM(flag_credential_or_privilege)   AS credential_privilege_count_current,
    SUM(flag_console_no_mfa)            AS console_no_mfa_count_current,
    MAX(label_signal_current)           AS label_signal_current,
    arg_max(policy_signal_category, event_time) FILTER (WHERE label_signal_current = 1) AS policy_signal_category_current
FROM event_causal
GROUP BY actor_key, time_bucket(INTERVAL '{window} minutes', event_time)
"""

# ----------------------------------------------------------------------------------------
# 5. Rolling history (strictly before the window) + targets
#    target_detection_current      : this window contains a policy-flagged event
#    target_forecast_30m           : a policy-flagged event follows within 30 minutes
#    target_current_or_future_30m  : either of the above  <- the project target
#    The final 30 minutes of the log cannot be labelled (no future to look at) -> NULL.
# ----------------------------------------------------------------------------------------
SQL_MART = """
CREATE OR REPLACE TABLE feature_mart AS
WITH history AS (
    SELECT *,
        SUM(event_count)          OVER w1h  AS prior_events_1h,
        SUM(event_count)          OVER w24h AS prior_events_24h,
        SUM(event_count)          OVER w7d  AS prior_events_7d,
        SUM(error_count)          OVER w24h AS prior_errors_24h,
        SUM(auth_error_count)     OVER w24h AS prior_auth_errors_24h,
        SUM(write_count)          OVER w24h AS prior_writes_24h,
        SUM(new_actor_api_count)  OVER w7d  AS prior_new_apis_7d,
        SUM(distinct_source_ips)  OVER w24h AS prior_ip_mass_24h,
        AVG(event_count)          OVER w7d  AS prior_mean_events_per_window_7d,
        COUNT(*)                  OVER w24h AS prior_active_windows_24h,
        COUNT(*)                  OVER w7d  AS prior_active_windows_7d,
        MAX(label_signal_current) OVER (PARTITION BY actor_id ORDER BY window_start
            RANGE BETWEEN INTERVAL '1 microsecond' FOLLOWING AND INTERVAL '{horizon} minutes' FOLLOWING) AS future_signal,
        MAX(window_start) OVER () AS last_window_in_log
    FROM actor_windows
    WINDOW
        w1h  AS (PARTITION BY actor_id ORDER BY window_start RANGE BETWEEN INTERVAL '1 hour'   PRECEDING AND INTERVAL '1 microsecond' PRECEDING),
        w24h AS (PARTITION BY actor_id ORDER BY window_start RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING),
        w7d  AS (PARTITION BY actor_id ORDER BY window_start RANGE BETWEEN INTERVAL '7 days'   PRECEDING AND INTERVAL '1 microsecond' PRECEDING)
)
SELECT * EXCLUDE (future_signal, last_window_in_log),
    label_signal_current AS target_detection_current,
    CASE WHEN window_start > last_window_in_log - INTERVAL '{horizon} minutes' THEN NULL
         ELSE COALESCE(future_signal, 0) END AS target_forecast_30m,
    CASE WHEN window_start > last_window_in_log - INTERVAL '{horizon} minutes' THEN NULL
         ELSE GREATEST(label_signal_current, COALESCE(future_signal, 0)) END AS target_current_or_future_30m,
    COALESCE(event_count / NULLIF(prior_mean_events_per_window_7d, 0), 0.0)    AS event_burst_vs_7d,
    COALESCE(error_count::DOUBLE / NULLIF(event_count, 0), 0.0)                AS error_rate_current,
    COALESCE(write_count::DOUBLE / NULLIF(event_count, 0), 0.0)                AS write_rate_current,
    COALESCE(read_only_known_count::DOUBLE / NULLIF(event_count, 0), 0.0)      AS read_only_coverage_current,
    COALESCE(new_actor_api_count::DOUBLE / NULLIF(event_count, 0), 0.0)        AS new_api_rate_current,
    COALESCE(new_actor_ip_count::DOUBLE / NULLIF(event_count, 0), 0.0)         AS new_ip_rate_current,
    SIN(2 * PI() * EXTRACT(hour FROM window_start) / 24.0)  AS hour_sin,
    COS(2 * PI() * EXTRACT(hour FROM window_start) / 24.0)  AS hour_cos,
    SIN(2 * PI() * EXTRACT(isodow FROM window_start) / 7.0) AS dow_sin,
    COS(2 * PI() * EXTRACT(isodow FROM window_start) / 7.0) AS dow_cos
FROM history
ORDER BY window_start, actor_id
"""


def connect(temp_dir: Path | None = None, memory_fraction: float = 0.6):
    """In-memory DuckDB pinned to UTC, using at most ``memory_fraction`` of the machine's RAM.

    Large window functions spill to ``temp_dir`` instead of crashing a small Studio instance.
    """
    import os

    import duckdb  # imported lazily: the pipeline containers only need validate_feature_mart

    total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    con = duckdb.connect(database=":memory:")
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"SET memory_limit = '{max(1.0, total_ram_gb * memory_fraction):.1f}GB'")
    con.execute("SET preserve_insertion_order = false")
    if temp_dir is not None:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{Path(temp_dir).as_posix()}'")
    return con


def build_feature_mart(
    con,
    events_glob: str,
    window_minutes: int = config.WINDOW_MINUTES,
    horizon_minutes: int = config.TARGET_HORIZON_MINUTES,
) -> pd.DataFrame:
    """Run the five SQL steps over the event Parquet files and return an audit summary."""
    con.execute(f"CREATE OR REPLACE VIEW events AS SELECT * FROM read_parquet('{events_glob}', union_by_name = true)")
    con.execute(SQL_DEDUP)
    con.execute(SQL_SEMANTIC)
    con.execute(SQL_CAUSAL)
    con.execute(SQL_WINDOWS.format(window=int(window_minutes)))
    con.execute(SQL_MART.format(horizon=int(horizon_minutes)))
    return con.execute("""
        SELECT (SELECT COUNT(*) FROM events)                    AS raw_events,
               (SELECT COUNT(*) FROM events_dedup)              AS unique_events,
               (SELECT SUM(label_signal_current) FROM event_semantic) AS policy_flagged_events,
               COUNT(*)                                         AS actor_windows,
               COUNT(DISTINCT actor_id)                         AS actors,
               MIN(window_start)                                AS first_window,
               MAX(window_start)                                AS last_window,
               SUM(target_current_or_future_30m)                AS positive_windows,
               COUNT(*) - COUNT(target_current_or_future_30m)   AS unlabelled_tail_windows
        FROM feature_mart
    """).fetchdf()


def validate_feature_mart(mart: pd.DataFrame) -> dict[str, bool]:
    """Data-quality checks run before training (also used by the CI/CD pipeline)."""
    labelled = mart[mart[config.TARGET_COLUMN].notna()]
    checks = {
        "all_feature_columns_present": set(config.FEATURE_COLUMNS) <= set(mart.columns),
        "target_column_present": config.TARGET_COLUMN in mart.columns,
        "unique_actor_window_keys": not mart.duplicated(config.KEY_COLUMNS).any(),
        "target_is_binary": set(labelled[config.TARGET_COLUMN].unique()) <= {0, 1},
        "both_classes_present": labelled[config.TARGET_COLUMN].nunique() == 2,
        "no_policy_columns_in_features": not set(config.POLICY_COLUMNS) & set(config.FEATURE_COLUMNS),
        "no_infinite_numeric_values": not bool(
            mart[config.NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").isin([float("inf"), float("-inf")]).any().any()
        ),
    }
    return checks


def export_feature_mart(con, parquet_path: Path, csv_path: Path | None = None) -> None:
    """Write the mart to Parquet (analytics) and optionally CSV (pipeline input)."""
    con.execute(f"COPY feature_mart TO '{Path(parquet_path).as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    if csv_path is not None:
        con.execute(f"COPY feature_mart TO '{Path(csv_path).as_posix()}' (FORMAT CSV, HEADER)")
