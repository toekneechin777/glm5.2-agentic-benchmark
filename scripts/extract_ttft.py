#!/usr/bin/env python3
"""Extract time-to-first-token (TTFT) and generation latency metrics from a serving engine.

Two backends are supported:

1. Metrics endpoint (vLLM / SGLang): both expose Prometheus metrics on
   `/metrics` (vLLM) and `/metrics` (SGLang) by default. Hit it after every
   agent turn to capture per-request TTFT / token rates.

2. vLLM stdout log parsing: if you only have the engine's stdout, this script
   can grep for the request summary lines vLLM prints, e.g.:

       prompt_tokens=1234, generated_tokens=567, e2e_latency=4.32s,
       ttft=0.42s, itl=0.007s

Usage:
    # Pull from a running /metrics endpoint and append to CSV
    python extract_ttft.py --metrics-url http://localhost:8000/metrics \
        --output ttft_log.csv --turn 1

    # Parse a vLLM stdout log file
    python extract_ttft.py --log-file /workspace/vllm.log \
        --output ttft_log.csv --turn 2

Run it after each OpenHands agent turn. The `--turn` integer is just a label
column so you can correlate latency growth as the agent context grows.
"""

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso():
    """UTC timestamp in ISO-8601 with a trailing Z (e.g. 2026-07-11T03:20:07.035Z)."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

FIELDS = [
    "timestamp",
    "iso_time",
    "turn",
    "backend",
    "ttft_s",
    "e2e_latency_s",
    "itl_s",
    "prompt_tokens",
    "generated_tokens",
    "tokens_per_second",
    "request_id",
    "raw",
]


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# --- vLLM /metrics endpoint -------------------------------------------------

# vLLM exposes e.g. vllm:time_to_first_token_seconds (a Prometheus summary).
# SGLang exposes e.g. sglang:ttft_seconds or similar; we match broadly.
METRIC_PATTERNS = [
    # name -> attribute key in the CSV row
    re.compile(r'time_to_first_token_seconds.*?quantile="0\.5".*?([\d.eE+-]+)'),
    re.compile(r"ttft_seconds.*?([\d.eE+-]+)"),
    re.compile(r"vllm:time_to_first_token_seconds.*?([\d.eE+-]+)"),
]


def fetch_metrics(url, timeout=10):
    if requests is None:
        raise RuntimeError("requests not installed; pip install requests")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_metrics(prometheus_text):
    """Extract TTFT (and any latency) values from a Prometheus metrics dump.

    Returns a list of dicts; each dict maps FIELDS keys to values.
    """
    rows = []
    ttfts = []
    e2es = []
    itls = []
    for line in prometheus_text.splitlines():
        if line.startswith("#"):
            continue
        low = line.lower()
        if "time_to_first_token" in low or "ttft" in low:
            m = re.search(r"\}\s*([\d.eE+-]+)\s*$", line) or re.search(
                r"\s([\d.eE+-]+)\s*$", line
            )
            if m:
                ttfts.append(_to_float(m.group(1)))
        if "e2e" in low or "end_to_end" in low or "request_latency" in low:
            m = re.search(r"([\d.eE+-]+)\s*$", line)
            if m:
                e2es.append(_to_float(m.group(1)))
        if "inter_token" in low or "itl" in low or "time_per_output_token" in low:
            m = re.search(r"([\d.eE+-]+)\s*$", line)
            if m:
                itls.append(_to_float(m.group(1)))

    # We can't easily attribute per-request from the metrics dump; record the
    # latest observed quantile/value per field.
    row = {
        "ttft_s": ttfts[-1] if ttfts else None,
        "e2e_latency_s": e2es[-1] if e2es else None,
        "itl_s": itls[-1] if itls else None,
    }
    if any(row.values()):
        rows.append(row)
    return rows


# --- vLLM stdout log parsing ------------------------------------------------

# Matches lines like:
#   prompt_tokens=1234, generated_tokens=567, e2e_latency=4.32s, ttft=0.42s, itl=0.007s
LOG_KV_RE = re.compile(r"(\w+)\s*=\s*([\d.eE+-]+)")


def parse_vllm_log(text):
    """Parse vLLM stdout summary lines into per-request latency rows."""
    rows = []
    for line in text.splitlines():
        if "ttft" not in line.lower() and "time_to_first_token" not in line.lower():
            continue
        kvs = dict(LOG_KV_RE.findall(line))
        if not kvs:
            continue
        row = {}
        for src, dst, cast in [
            ("ttft", "ttft_s", _to_float),
            ("time_to_first_token", "ttft_s", _to_float),
            ("e2e_latency", "e2e_latency_s", _to_float),
            ("end_to_end_latency", "e2e_latency_s", _to_float),
            ("itl", "itl_s", _to_float),
            ("inter_token_latency", "itl_s", _to_float),
            ("prompt_tokens", "prompt_tokens", _to_int),
            ("generated_tokens", "generated_tokens", _to_int),
            ("output_tokens", "generated_tokens", _to_int),
            ("request_id", "request_id", str),
        ]:
            if src in kvs:
                val = cast(kvs[src])
                if val is not None:
                    row[dst] = val
        if "generated_tokens" in row and "e2e_latency_s" in row and row["e2e_latency_s"]:
            row["tokens_per_second"] = round(
                row["generated_tokens"] / row["e2e_latency_s"], 4
            )
        row["raw"] = line.strip()[:200]
        rows.append(row)
    return rows


# --- CSV writing ------------------------------------------------------------


def ensure_header(csv_path):
    if not Path(csv_path).exists() or Path(csv_path).stat().st_size == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def write_rows(csv_path, turn, backend, rows):
    if not rows:
        print("[ttft] no latency metrics found", file=sys.stderr)
        return 0
    now_mono = _utc_now_iso()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        written = 0
        for row in rows:
            out = {k: row.get(k) for k in FIELDS}
            out.update(
                {
                    "iso_time": now_mono,
                    "turn": turn,
                    "backend": backend,
                    "ttft_s": row.get("ttft_s"),
                    "e2e_latency_s": row.get("e2e_latency_s"),
                    "itl_s": row.get("itl_s"),
                    "prompt_tokens": row.get("prompt_tokens"),
                    "generated_tokens": row.get("generated_tokens"),
                    "tokens_per_second": row.get("tokens_per_second"),
                    "request_id": row.get("request_id"),
                    "raw": row.get("raw", ""),
                }
            )
            writer.writerow(out)
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-url",
        help="Prometheus /metrics endpoint (vLLM or SGLang), e.g. http://localhost:8000/metrics",
    )
    parser.add_argument(
        "--log-file",
        help="vLLM stdout log file to parse for per-request latency lines",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="ttft_log.csv",
        help="CSV to append rows to (default: ttft_log.csv)",
    )
    parser.add_argument(
        "--turn",
        type=int,
        default=0,
        help="Agent turn number to label this sample with",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout for /metrics fetch",
    )
    args = parser.parse_args()

    if not args.metrics_url and not args.log_file:
        parser.error("provide either --metrics-url or --log-file")

    ensure_header(args.output)

    if args.metrics_url:
        backend = "metrics"
        text = fetch_metrics(args.metrics_url, timeout=args.timeout)
        rows = parse_metrics(text)
    else:
        backend = "vllm-log"
        text = Path(args.log_file).read_text(errors="ignore")
        rows = parse_vllm_log(text)

    n = write_rows(args.output, args.turn, backend, rows)
    print(f"[ttft] wrote {n} row(s) to {args.output} (turn={args.turn})")


if __name__ == "__main__":
    main()
