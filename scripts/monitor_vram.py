#!/usr/bin/env python3
"""Poll GPU VRAM usage on a RunPod instance while an OpenHands agent is working.

Logs peak (and current) VRAM usage every `interval` seconds to a CSV file.

Usage:
    python monitor_vram.py --output vram_log.csv
    python monitor_vram.py --interval 5 --output /workspace/vram_log.csv
    python monitor_vram.py --use-nvidia-smi  # fall back to nvidia-smi if no pynvml

Run this in the background while your agent task is executing, e.g.:
    nohup python scripts/monitor_vram.py --output vram_log.csv &
"""

import argparse
import csv
import subprocess
import sys
import time
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
    import pynvml
except ImportError:  # pragma: no cover - handled by --use-nvidia-smi fallback
    pynvml = None

FIELDS = [
    "timestamp",
    "iso_time",
    "gpu_index",
    "gpu_name",
    "vram_used_mb",
    "vram_total_mb",
    "vram_free_mb",
    "vram_util_pct",
    "peak_vram_used_mb",
    "temperature_c",
    "power_draw_w",
]


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def query_nvidia_smi():
    """Query all GPUs via a single nvidia-smi call and return per-GPU stats."""
    query = ",".join(
        [
            "index",
            "name",
            "memory.used",
            "memory.total",
            "memory.free",
            "utilization.gpu",
            "temperature.gpu",
            "power.draw",
        ]
    )
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    results = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        idx, name, used, total, free, util, temp, power = parts[:8]
        results.append(
            {
                "gpu_index": _to_int(idx),
                "gpu_name": name,
                "vram_used_mb": _to_int(used),
                "vram_total_mb": _to_int(total),
                "vram_free_mb": _to_int(free),
                "vram_util_pct": _to_int(util),
                "temperature_c": _to_int(temp),
                "power_draw_w": _to_float(power),
            }
        )
    return results


def query_pynvml():
    """Query all GPUs via pynvml."""
    if pynvml is None:
        raise RuntimeError("pynvml is not installed; pip install nvidia-ml-py3")
    pynvml.nvmlInit()
    results = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            temp = None
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError:
            power = None
        try:
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "ignore")
        except pynvml.NVMLError:
            name = "unknown"
        results.append(
            {
                "gpu_index": i,
                "gpu_name": name,
                "vram_used_mb": int(mem.used / (1024 * 1024)),
                "vram_total_mb": int(mem.total / (1024 * 1024)),
                "vram_free_mb": int(mem.free / (1024 * 1024)),
                "vram_util_pct": int(util.gpu),
                "temperature_c": temp,
                "power_draw_w": power,
            }
        )
    pynvml.nvmlShutdown()
    return results


def write_header(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()


def append_rows(csv_path, sample_time, gpu_stats, peaks):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        for stat in gpu_stats:
            idx = stat["gpu_index"]
            used = stat["vram_used_mb"] or 0
            peaks[idx] = max(peaks.get(idx, 0), used)
            writer.writerow(
                {
                    "timestamp": f"{sample_time:.3f}",
                    "iso_time": _utc_now_iso(),
                    "gpu_index": idx,
                    "gpu_name": stat["gpu_name"],
                    "vram_used_mb": used,
                    "vram_total_mb": stat["vram_total_mb"],
                    "vram_free_mb": stat["vram_free_mb"],
                    "vram_util_pct": stat["vram_util_pct"],
                    "peak_vram_used_mb": peaks[idx],
                    "temperature_c": stat["temperature_c"],
                    "power_draw_w": stat["power_draw_w"],
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        default="vram_log.csv",
        help="Path to output CSV file (default: vram_log.csv)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=5.0,
        help="Seconds between samples (default: 5)",
    )
    parser.add_argument(
        "--use-nvidia-smi",
        action="store_true",
        help="Use nvidia-smi subprocess calls instead of pynvml",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Stop after N seconds (default: run forever / until Ctrl-C)",
    )
    args = parser.parse_args()

    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header(csv_path)

    query = query_nvidia_smi if args.use_nvidia_smi else query_pynvml
    backend = "nvidia-smi" if args.use_nvidia_smi else "pynvml"
    print(
        f"[vram] logging every {args.interval}s to {csv_path} (backend={backend})"
    )

    peaks = {}
    start = time.monotonic()
    sample_no = 0
    try:
        while True:
            try:
                stats = query()
            except Exception as e:
                print(f"[vram] query error: {e}", file=sys.stderr)
                stats = []
            append_rows(csv_path, time.time() - start, stats, peaks)
            sample_no += 1
            if stats:
                peak_overall = max(peaks.values()) if peaks else 0
                print(
                    f"[vram] sample={sample_no} "
                    f"peak={peak_overall}MB gpus={len(stats)}",
                    flush=True,
                )
            if args.duration is not None and (time.monotonic() - start) >= args.duration:
                print(f"[vram] reached duration limit ({args.duration}s), stopping")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[vram] stopped by user")


if __name__ == "__main__":
    main()
