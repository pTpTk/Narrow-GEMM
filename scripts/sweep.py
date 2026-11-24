#!/usr/bin/env python3
"""Run every case, then print one performance table. Progress goes to stderr."""
import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys


STAGE_NAMES = {
    0: "cuBLAS", 1: "Scalar", 2: "Warp MMA", 3: "Async MMA",
    4: "Split-K MMA", 5: "TMA/WGMMA", 6: "Warp specialized", 7: "Persistent",
    8: "TMA cooperative", 9: "Fused 3D/vector", 10: "Outstanding MMA",
    11: "Async FP32 TMA", 12: "Stream-K fixup", 13: "A multicast",
    14: "Ping-pong/reg",
}


def configurations(profile, include_scalar):
    # (kernel, BN, split, ring slots, K boxes per group, persistent CTA cap)
    configs = [(0, 64, 1, 2, 2, 0)]
    if include_scalar:
        configs.append((1, 64, 1, 2, 2, 0))
    if profile == "advanced":
        configs += [(k, 64, s, 2, 2, 0) for k in (5, 7, 8) for s in (1, 2, 4)]
        configs += [(k, bn, s, 2, 2, 0) for bn in (32, 64, 128)
                    for k in range(9, 14) for s in (1, 2, 4)]
        # Two independent input rings need a smaller slab to fit larger BN.
        configs += [(14, bn, s, 2, 1, 0) for bn in (16, 32, 64, 128, 256) for s in (1, 2, 4)]
        configs += [(k, bn, s, 2, 2, 0) for bn in (16, 256)
                    for k in (9, 10, 11, 12, 13) for s in (1, 4)]
        # DeepGEMM-like six merged slots, intermediate depths, and K64/deep
        # alternatives. Runtime resource checks, rather than guesses, decide
        # whether each is legal on the visible MIG device.
        configs += [(k, 64, s, slots, 2, 0) for k in (10, 11)
                    for s in (1, 2) for slots in (3, 4, 5, 6)]
        configs += [(k, 64, 2, 8, 1, 0) for k in (10, 11)]
        configs += [(12, 64, 4, 2, 2, cap) for cap in (1, 2)]
    elif profile == "full":
        for bn in (32, 64, 128):
            configs += [(k, bn, 1, 2, 2, 0) for k in (2, 3)]
            configs += [(4, bn, s, 2, 2, 0) for s in (2, 4, 8, 16)]
            configs += [(k, bn, s, 2, 2, 0) for k in (5, 6, 7, 8) for s in (1, 2, 4, 8, 16)]
    else:
        # Offline hypotheses, not measured winners. Emphasize low split counts:
        # a MIG device can need less K parallelism than the full-chip example.
        for bn in (64, 128):
            configs += [(k, bn, s, 2, 2, 0) for k in (5, 6, 7, 8) for s in (1, 2, 4)]
            configs += [(k, bn, s, slots, 1, 0)
                        for k in (6, 8) for s in (1, 2) for slots in (2, 4)]
            configs.append((7, bn, 2, 2, 2, 1))  # Isolate the old one-CTA cap.
            configs += [(4, bn, s, 2, 2, 0) for s in (2, 4)]
            configs += [(k, bn, 1, 2, 2, 0) for k in (2, 3)]
        configs += [(5, 32, 1, 2, 2, 0), (8, 32, 1, 2, 2, 0), (8, 32, 2, 2, 2, 0)]
    if profile in ("full", "h200-mig"):
        configs += [(k, 64, s, 2, 2, 0) for k in range(9, 14) for s in (1, 2, 4)]
        configs += [(14, 64, s, 2, 1, 0) for s in (1, 2, 4)]
    return list(dict.fromkeys(configs))


def write_csv(records, fields, stream):
    if fields:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def print_table(measurements, timing):
    if not measurements:
        return
    headers = ("Kernel", "BN", "Split", "K tile", "Stages", "CTA/SM", "Median us", "Min us", "TFLOP/s",
               "cuBLAS us", "Speedup", "Plan", "Status")
    rows = []
    for row, baseline in sorted(measurements, key=lambda item: tuple(
            int(item[0][key]) for key in ("kernel", "bn", "split", "tile_k", "stages", "workers_per_sm"))):
        kernel = int(row["kernel"])
        workers = row["workers_per_sm"] if int(row["workers_per_sm"]) else "auto"
        skipped = row.get("status", "ok") == "skipped"
        rows.append((f"{kernel} {STAGE_NAMES[kernel]}",
                     row["bn"] if int(row["bn"]) else "-", row["split"],
                     row["tile_k"], row["stages"], ("1" if kernel == 14 else workers) if kernel == 7 or kernel >= 9 else "-",
                     "SKIP" if skipped else f"{float(row['median_us']):.3f}", "-" if skipped else f"{float(row['min_us']):.3f}",
                     "-" if skipped else f"{float(row['tflops']):.2f}", f"{baseline:.3f}",
                     ("-" if skipped else f"{float(row['speedup_vs_cublas']):.3f}x"),
                     row.get("decomposition", "uniform"), row.get("status", "ok")))
    minimum_widths = (20, 4, 5, 6, 6, 6, 12, 12, 12, 12, 10, 8, 7)
    widths = [max(minimum_widths[i], *(len(row[i]) for row in [headers, *rows]))
              for i in range(len(headers))]

    def line(row):
        return "  ".join(value.ljust(width) if i == 0 else value.rjust(width)
                         for i, (value, width) in enumerate(zip(row, widths)))

    print(f"Performance summary ({timing}; M=64, N=4096, K=8192)")
    print(line(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(line(row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[2] / "build/narrow_gemm")
    parser.add_argument("--include-scalar", action="store_true")
    parser.add_argument("--profile", choices=("full", "h200-mig", "advanced"), default="full",
                        help="full checkpoint sweep or focused H200 MIG candidates")
    parser.add_argument("--kernels", help="optional comma-separated IDs, e.g. 9,10,11,12,13,14")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timing", choices=("graph", "events"), default="graph")
    parser.add_argument("--format", choices=("table", "csv"), default="table",
                        help="stdout format, emitted once after all cases (default: table)")
    parser.add_argument("--csv", type=Path,
                        help="also save full-precision CSV, including each local cuBLAS baseline")
    args = parser.parse_args()
    if args.iterations < 1 or args.samples < 1:
        parser.error("iterations and samples must be positive")
    selected = set(range(15))
    if args.kernels:
        try:
            selected = {int(value) for value in args.kernels.split(",")}
        except ValueError:
            parser.error("kernels must be comma-separated IDs from 0 to 14")
        if not selected or not selected <= set(range(15)):
            parser.error("kernels must be comma-separated IDs from 0 to 14")
    configs = configurations(args.profile, args.include_scalar or (args.kernels and 1 in selected))
    configs = [case for case in configs if case[0] == 0 or case[0] in selected]
    records, measurements, fields = [], [], None
    failed = False
    for index, (kernel, bn, split, stages, batch_k, workers) in enumerate(configs, 1):
        cmd = [str(args.binary.resolve()), "--kernel", str(kernel), "--bn", str(bn),
               "--split", str(split), "--iterations", str(args.iterations),
               "--samples", str(args.samples), "--timing", args.timing, "--format", "csv",
               "--stages", str(stages), "--batch-k", str(batch_k), "--workers-per-sm", str(workers)]
        print(f"[{index}/{len(configs)}] kernel={kernel} BN={bn} split={split} stages={stages} batch_k={batch_k} workers_per_sm={workers}",
              file=sys.stderr, flush=True)
        try:
            result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE)
            if result.returncode not in (0, 77):
                raise subprocess.CalledProcessError(result.returncode, cmd)
            reader = csv.DictReader(result.stdout.splitlines())
            if not reader.fieldnames:
                raise ValueError("benchmark returned no CSV header")
            if fields is not None and fields != reader.fieldnames:
                raise ValueError("benchmark CSV columns changed between cases")
            rows = list(reader)
            baseline = next((row for row in rows if row["kernel"] == "0"), None)
            measured = next((row for row in rows if row["kernel"] == str(kernel)), None)
            if baseline is None or measured is None:
                raise ValueError("benchmark CSV is missing the baseline or requested case")
            if measured.get("status", "ok") not in ("ok", "skipped"):
                raise ValueError("unknown result status")
            if result.returncode == 77 and measured.get("status") != "skipped":
                raise ValueError("exit 77 without an explicit skipped row")
            baseline_us = float(baseline["median_us"])
            if not math.isfinite(baseline_us) or baseline_us <= 0:
                raise ValueError("invalid cuBLAS timing")
            for column in ("bn", "split", "tile_k", "stages", "workers_per_sm"):
                int(measured[column])
            for column in ("median_us", "min_us", "tflops", "speedup_vs_cublas"):
                value = float(measured[column])
                if measured.get("status", "ok") == "ok" and (not math.isfinite(value) or value <= 0):
                    raise ValueError("invalid successful timing/throughput")
            fields = reader.fieldnames
            records.extend(rows)
            measurements.append((measured, baseline_us))
        except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as error:
            print(f"Sweep stopped: {error}. Summary contains completed cases only.",
                  file=sys.stderr, flush=True)
            failed = True
            break

    # No child process remains when the summary is emitted, so progress and
    # correctness diagnostics cannot split the table, even with stderr merged.
    if args.csv:
        try:
            with args.csv.open("w", newline="") as stream:
                write_csv(records, fields, stream)
        except OSError as error:
            print(f"Cannot write CSV: {error}", file=sys.stderr)
            failed = True
    if args.format == "csv":
        write_csv(records, fields, sys.stdout)
    else:
        print_table(measurements, args.timing)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
