#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loganalyzer.py — анализатор лог-файлов (один файл)

Поддерживает:
- Вход: --file PATH или stdin
- Форматы: classic, bracketed, regex (--pattern с именованными группами ts/level/logger/msg)
- Фильтры: --level, --since/--until, --match/--ignore, --logger/--logger-regex
- Метрики: summary, levels, top_messages, top_loggers, time_histogram (+ optional top_codes)
- Вывод: консоль (по умолчанию), --json PATH, --csv-dir DIR
- Непарсенные строки: счётчик + --save-unparsed PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import Counter
from typing import Iterable, Iterator, Optional, TextIO, Dict, Any, Tuple


# ----------------------------
# Models
# ----------------------------

@dataclass(frozen=True)
class LogRecord:
    timestamp: datetime
    level: str
    logger: str
    message: str
    raw: str


# ----------------------------
# Parsing helpers
# ----------------------------

LEVEL_ALIASES = {
    "WARN": "WARNING",
}

DEFAULT_LEVELS = {
    "TRACE", "DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"
}

# classic example:
# 2025-12-14 10:15:02,123 INFO auth: User login success id=42 ip=1.2.3.4
CLASSIC_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+"
    r"(?P<level>[A-Za-z]+)\s+"
    r"(?P<logger>[^:]+):\s+"
    r"(?P<msg>.*)$"
)

# bracketed example:
# [2025-12-14T10:15:02.123+00:00] [INFO] [auth] User login success id=42
BRACKETED_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+\[(?P<level>[^\]]+)\]\s+\[(?P<logger>[^\]]+)\]\s+(?P<msg>.*)$"
)


def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse supported timestamps. Returns timezone-aware datetime when possible."""
    ts = ts.strip()
    # classic: YYYY-MM-DD HH:MM:SS,mmm
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
        # no timezone in classic -> treat as naive local; keep naive.
        return dt
    except ValueError:
        pass

    # ISO 8601 (e.g. 2025-12-14T10:15:02.123+00:00 / 2025-12-14T10:15:02+00:00)
    # datetime.fromisoformat supports offsets.
    try:
        dt = datetime.fromisoformat(ts)
        return dt
    except ValueError:
        return None


def normalize_level(level: str) -> str:
    level = (level or "").strip().upper()
    level = LEVEL_ALIASES.get(level, level)
    return level


def build_parser(fmt: str, pattern: Optional[str] = None) -> re.Pattern:
    fmt = fmt.lower().strip()
    if fmt == "classic":
        return CLASSIC_RE
    if fmt == "bracketed":
        return BRACKETED_RE
    if fmt == "regex":
        if not pattern:
            raise ValueError("Format 'regex' requires --pattern with named groups: ts, level, logger, msg")
        try:
            return re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid --pattern regex: {e}")
    raise ValueError(f"Unknown format: {fmt}")


def parse_line(line: str, parser: re.Pattern) -> Optional[Tuple[LogRecord, Dict[str, str]]]:
    """
    Parse a single line into LogRecord.
    Returns (record, extracted_kv) or None if unparsed.
    extracted_kv: optional extracted fields (e.g., 'code') from message.
    """
    raw = line.rstrip("\n")
    m = parser.match(raw)
    if not m:
        return None

    gd = m.groupdict()
    ts_raw = gd.get("ts", "") or ""
    level_raw = gd.get("level", "") or ""
    logger_raw = gd.get("logger", "") or "unknown"
    msg_raw = gd.get("msg", "") or ""

    ts = parse_timestamp(ts_raw)
    if ts is None:
        return None

    level = normalize_level(level_raw)
    logger = logger_raw.strip() or "unknown"
    message = msg_raw.strip()

    # Extract common fields from message (optional metrics)
    extracted = {}
    # code=XYZ
    cm = re.search(r"\bcode=(?P<code>[A-Za-z0-9_\-\.]+)\b", message)
    if cm:
        extracted["code"] = cm.group("code")

    return LogRecord(timestamp=ts, level=level, logger=logger, message=message, raw=raw), extracted


# ----------------------------
# Filters
# ----------------------------

def parse_user_datetime(s: str) -> datetime:
    """
    Parse --since/--until.
    Accepts:
    - YYYY-MM-DD
    - YYYY-MM-DD HH:MM:SS
    - ISO8601 (fromisoformat)
    """
    s = s.strip()
    # date only
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return datetime.strptime(s, "%Y-%m-%d")
    # datetime
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", s):
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    # ISO 8601
    try:
        return datetime.fromisoformat(s)
    except ValueError as e:
        raise ValueError(f"Invalid datetime: {s}") from e


@dataclass(frozen=True)
class Filters:
    levels: Optional[set[str]]
    since: Optional[datetime]
    until: Optional[datetime]
    match: Optional[re.Pattern]
    ignore: Optional[re.Pattern]
    logger: Optional[str]
    logger_regex: Optional[re.Pattern]


def record_passes_filters(rec: LogRecord, f: Filters) -> bool:
    if f.levels is not None and rec.level not in f.levels:
        return False

    if f.logger is not None and rec.logger != f.logger:
        return False

    if f.logger_regex is not None and not f.logger_regex.search(rec.logger):
        return False

    # since/until compare: allow comparing aware to naive by converting aware->naive in UTC where possible.
    ts = rec.timestamp
    if f.since is not None:
        if not datetime_leq(f.since, ts):
            # since > ts  => fail
            return False

    if f.until is not None:
        if not datetime_lt(ts, f.until):
            # ts >= until => fail (until is exclusive)
            return False

    if f.match is not None and not f.match.search(rec.message):
        return False

    if f.ignore is not None and f.ignore.search(rec.message):
        return False

    return True


def _to_comparable(dt: datetime) -> datetime:
    """
    Convert datetime to comparable form:
    - If aware -> convert to UTC and drop tzinfo (naive UTC)
    - If naive -> keep as-is
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def datetime_leq(a: datetime, b: datetime) -> bool:
    return _to_comparable(a) <= _to_comparable(b)


def datetime_lt(a: datetime, b: datetime) -> bool:
    return _to_comparable(a) < _to_comparable(b)


# ----------------------------
# Message normalization (optional)
# ----------------------------

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
NUM_RE = re.compile(r"\b\d+\b")
# very rough *nix and windows paths; optional / best-effort
PATH_RE = re.compile(
    r"(?:(?:[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]+)|(?:/(?:[^/\s]+/)*[^/\s]+))"
)


def normalize_message(msg: str) -> str:
    s = msg
    s = UUID_RE.sub("<uuid>", s)
    s = IP_RE.sub("<ip>", s)
    s = PATH_RE.sub("<path>", s)
    s = NUM_RE.sub("<num>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------
# Metrics computation (streaming)
# ----------------------------

@dataclass
class Aggregates:
    total_lines: int = 0
    parsed_lines: int = 0
    unparsed_lines: int = 0

    min_ts: Optional[datetime] = None
    max_ts: Optional[datetime] = None

    level_counts: Counter = None  # type: ignore
    logger_counts: Counter = None  # type: ignore
    message_counts: Counter = None  # type: ignore
    time_hist: Counter = None  # type: ignore
    code_counts: Counter = None  # type: ignore

    def __post_init__(self) -> None:
        self.level_counts = Counter()
        self.logger_counts = Counter()
        self.message_counts = Counter()
        self.time_hist = Counter()
        self.code_counts = Counter()


def bucket_time(ts: datetime, bucket: str) -> str:
    # Use comparable naive UTC for aware dt (stable)
    dt = _to_comparable(ts)
    if bucket == "hour":
        dtb = dt.replace(minute=0, second=0, microsecond=0)
        return dtb.strftime("%Y-%m-%d %H:00")
    if bucket == "day":
        dtb = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return dtb.strftime("%Y-%m-%d")
    raise ValueError("time bucket must be 'hour' or 'day'")


def compute_metrics(
    stream: TextIO,
    parser: re.Pattern,
    filters: Filters,
    top_n: int,
    time_bucket: str,
    do_normalize: bool,
    max_lines: Optional[int],
    save_unparsed_path: Optional[str],
    encoding: str,
) -> Dict[str, Any]:
    ag = Aggregates()

    unparsed_fp: Optional[TextIO] = None
    if save_unparsed_path:
        unparsed_fp = open(save_unparsed_path, "w", encoding=encoding, errors="replace")

    try:
        for i, line in enumerate(stream):
            ag.total_lines += 1
            if max_lines is not None and ag.total_lines > max_lines:
                break

            parsed = parse_line(line, parser)
            if parsed is None:
                ag.unparsed_lines += 1
                if unparsed_fp:
                    unparsed_fp.write(line)
                continue

            rec, extracted = parsed
            ag.parsed_lines += 1

            if not record_passes_filters(rec, filters):
                continue

            # range summary
            if ag.min_ts is None or datetime_lt(rec.timestamp, ag.min_ts):
                ag.min_ts = rec.timestamp
            if ag.max_ts is None or datetime_lt(ag.max_ts, rec.timestamp):
                ag.max_ts = rec.timestamp

            ag.level_counts[rec.level] += 1
            ag.logger_counts[rec.logger] += 1

            msg_key = normalize_message(rec.message) if do_normalize else rec.message
            ag.message_counts[msg_key] += 1

            b = bucket_time(rec.timestamp, time_bucket)
            ag.time_hist[b] += 1

            code = extracted.get("code")
            if code:
                ag.code_counts[code] += 1

    finally:
        if unparsed_fp:
            unparsed_fp.close()

    # Prepare output
    filtered_total = sum(ag.level_counts.values())  # only records passing filters
    summary = {
        "total_lines": ag.total_lines,
        "parsed_lines": ag.parsed_lines,
        "unparsed_lines": ag.unparsed_lines,
        "filtered_records": filtered_total,
        "log_period": {
            "start": dt_to_str(ag.min_ts),
            "end": dt_to_str(ag.max_ts),
        },
    }

    levels_list = []
    for lvl, cnt in ag.level_counts.most_common():
        pct = (cnt / filtered_total * 100.0) if filtered_total else 0.0
        levels_list.append({"level": lvl, "count": cnt, "pct": round(pct, 2)})

    top_messages = [{"message": m, "count": c} for m, c in ag.message_counts.most_common(top_n)]
    top_loggers = [{"logger": l, "count": c} for l, c in ag.logger_counts.most_common(top_n)]

    # histogram sorted by bucket key (string is sortable for these formats)
    time_histogram = [{"bucket": k, "count": ag.time_hist[k]} for k in sorted(ag.time_hist.keys())]

    metrics: Dict[str, Any] = {
        "summary": summary,
        "levels": levels_list,
        "top_messages": top_messages,
        "top_loggers": top_loggers,
        "time_histogram": time_histogram,
    }

    if ag.code_counts:
        metrics["top_codes"] = [{"code": c, "count": n} for c, n in ag.code_counts.most_common(top_n)]

    return metrics


def dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    # Preserve timezone if present
    if dt.tzinfo is not None:
        return dt.isoformat()
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ----------------------------
# Output rendering
# ----------------------------

def render_console_report(metrics: Dict[str, Any], width: int = 120) -> str:
    s = []
    summ = metrics["summary"]
    s.append("loganalyzer.py report")
    s.append("=" * 60)
    s.append(f"Total lines:        {summ['total_lines']}")
    s.append(f"Parsed lines:       {summ['parsed_lines']}")
    s.append(f"Unparsed lines:     {summ['unparsed_lines']}")
    s.append(f"Filtered records:   {summ['filtered_records']}")
    s.append(f"Log period start:   {summ['log_period']['start']}")
    s.append(f"Log period end:     {summ['log_period']['end']}")
    s.append("")

    s.append("Levels")
    s.append("-" * 60)
    s.extend(render_table(
        headers=["LEVEL", "COUNT", "PCT"],
        rows=[(x["level"], str(x["count"]), f"{x['pct']:.2f}%") for x in metrics.get("levels", [])],
        col_widths=[12, 10, 10],
    ))
    s.append("")

    s.append("Top messages")
    s.append("-" * 60)
    s.extend(render_table(
        headers=["COUNT", "MESSAGE"],
        rows=[(str(x["count"]), truncate(x["message"], width - 16)) for x in metrics.get("top_messages", [])],
        col_widths=[10, width - 14],
    ))
    s.append("")

    s.append("Top loggers")
    s.append("-" * 60)
    s.extend(render_table(
        headers=["COUNT", "LOGGER"],
        rows=[(str(x["count"]), x["logger"]) for x in metrics.get("top_loggers", [])],
        col_widths=[10, 40],
    ))
    s.append("")

    if "top_codes" in metrics:
        s.append("Top codes (from message code=...)")
        s.append("-" * 60)
        s.extend(render_table(
            headers=["COUNT", "CODE"],
            rows=[(str(x["count"]), x["code"]) for x in metrics.get("top_codes", [])],
            col_widths=[10, 30],
        ))
        s.append("")

    s.append("Time histogram")
    s.append("-" * 60)
    s.extend(render_table(
        headers=["BUCKET", "COUNT"],
        rows=[(x["bucket"], str(x["count"])) for x in metrics.get("time_histogram", [])],
        col_widths=[20, 10],
    ))

    return "\n".join(s) + "\n"


def render_table(headers: list[str], rows: list[Tuple[str, ...]], col_widths: list[int]) -> list[str]:
    def fmt_row(cols: Tuple[str, ...]) -> str:
        parts = []
        for i, c in enumerate(cols):
            w = col_widths[i] if i < len(col_widths) else 20
            parts.append(c[:w].ljust(w))
        return "  ".join(parts).rstrip()

    out = [fmt_row(tuple(headers))]
    out.append(fmt_row(tuple("-" * min(len(h), (col_widths[i] if i < len(col_widths) else 20)) for i, h in enumerate(headers))))
    for r in rows:
        out.append(fmt_row(r))
    return out


def truncate(s: str, max_len: int) -> str:
    if max_len <= 3:
        return s[:max_len]
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def write_json(metrics: Dict[str, Any], path: str, encoding: str) -> None:
    with open(path, "w", encoding=encoding) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def write_csv_dir(metrics: Dict[str, Any], out_dir: str, encoding: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def write_rows(filename: str, headers: list[str], rows: list[dict]) -> None:
        p = os.path.join(out_dir, filename)
        with open(p, "w", newline="", encoding=encoding) as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    if "levels" in metrics:
        write_rows("levels.csv", ["level", "count", "pct"], metrics["levels"])
    if "top_messages" in metrics:
        write_rows("top_messages.csv", ["message", "count"], metrics["top_messages"])
    if "top_loggers" in metrics:
        write_rows("top_loggers.csv", ["logger", "count"], metrics["top_loggers"])
    if "time_histogram" in metrics:
        write_rows("time_histogram.csv", ["bucket", "count"], metrics["time_histogram"])
    if "top_codes" in metrics:
        write_rows("top_codes.csv", ["code", "count"], metrics["top_codes"])


# ----------------------------
# CLI
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loganalyzer.py",
        description="Analyze log files and produce summary metrics.",
    )
    p.add_argument("--file", type=str, default=None, help="Path to log file. If omitted, reads from stdin.")
    p.add_argument("--format", type=str, default="classic", choices=["classic", "bracketed", "regex"],
                   help="Log line format.")
    p.add_argument("--pattern", type=str, default=None,
                   help="Custom regex pattern for --format regex. Must define named groups: ts, level, logger, msg.")
    p.add_argument("--encoding", type=str, default="utf-8", help="Input/output encoding. Default: utf-8")

    p.add_argument("--top", type=int, default=10, help="Top N entries for messages/loggers/codes.")
    p.add_argument("--time-bucket", type=str, default="hour", choices=["hour", "day"],
                   help="Histogram bucketing by hour or day.")

    p.add_argument("--level", type=str, default=None,
                   help="Comma-separated levels to include, e.g. ERROR,WARNING")
    p.add_argument("--since", type=str, default=None, help="Include records with timestamp >= since.")
    p.add_argument("--until", type=str, default=None, help="Include records with timestamp < until (exclusive).")

    p.add_argument("--match", type=str, default=None, help="Regex filter: include only if message matches.")
    p.add_argument("--ignore", type=str, default=None, help="Regex filter: exclude if message matches.")
    p.add_argument("--logger", type=str, default=None, help="Filter by exact logger name.")
    p.add_argument("--logger-regex", type=str, default=None, help="Filter by regex on logger name.")

    p.add_argument("--normalize", action="store_true", help="Normalize message for grouping (numbers, uuid, ip, paths).")

    p.add_argument("--json", type=str, default=None, help="Write full metrics JSON to path.")
    p.add_argument("--csv-dir", type=str, default=None, help="Write metrics tables as CSV files into directory.")
    p.add_argument("--save-unparsed", type=str, default=None, help="Save unparsed lines into file.")
    p.add_argument("--max-lines", type=int, default=None, help="Stop after reading N lines (debug/testing).")
    p.add_argument("--quiet", action="store_true", help="Do not print console report.")
    return p


def parse_filters(args: argparse.Namespace) -> Filters:
    levels = None
    if args.level:
        parts = [normalize_level(x) for x in args.level.split(",") if x.strip()]
        levels = set(parts)

    since = parse_user_datetime(args.since) if args.since else None
    until = parse_user_datetime(args.until) if args.until else None

    match = re.compile(args.match) if args.match else None
    ignore = re.compile(args.ignore) if args.ignore else None

    logger_regex = re.compile(args.logger_regex) if args.logger_regex else None

    return Filters(
        levels=levels,
        since=since,
        until=until,
        match=match,
        ignore=ignore,
        logger=args.logger,
        logger_regex=logger_regex,
    )


def open_input_stream(args: argparse.Namespace) -> TextIO:
    if args.file:
        return open(args.file, "r", encoding=args.encoding, errors="replace")
    # stdin text stream; ensure it is text
    return sys.stdin


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    try:
        parser = build_parser(args.format, args.pattern)
        filters = parse_filters(args)

        # Validate bucket
        if args.time_bucket not in ("hour", "day"):
            raise ValueError("--time-bucket must be hour|day")

        # Validate top
        if args.top <= 0:
            raise ValueError("--top must be > 0")

        # until should not be earlier than since (best-effort validation)
        if filters.since is not None and filters.until is not None:
            if not datetime_lt(filters.since, filters.until) and _to_comparable(filters.since) != _to_comparable(filters.until):
                # if since > until
                if _to_comparable(filters.since) > _to_comparable(filters.until):
                    raise ValueError("--since must be <= --until")

        with open_input_stream(args) as stream:
            metrics = compute_metrics(
                stream=stream,
                parser=parser,
                filters=filters,
                top_n=args.top,
                time_bucket=args.time_bucket,
                do_normalize=args.normalize,
                max_lines=args.max_lines,
                save_unparsed_path=args.save_unparsed,
                encoding=args.encoding,
            )

        if args.json:
            write_json(metrics, args.json, args.encoding)

        if args.csv_dir:
            write_csv_dir(metrics, args.csv_dir, args.encoding)

        if not args.quiet:
            sys.stdout.write(render_console_report(metrics))

        # Optional: return 1 if too many unparsed lines
        # (kept simple; always 0 on success)
        return 0

    except FileNotFoundError as e:
        sys.stderr.write(f"ERROR: file not found: {e}\n")
        return 2
    except PermissionError as e:
        sys.stderr.write(f"ERROR: permission denied: {e}\n")
        return 2
    except ValueError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("ERROR: interrupted\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())