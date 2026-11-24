#!/usr/bin/env python3
"""Run every case, then print one performance table. Progress goes to stderr."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys


STAGE_NAMES = {
    0: "cuBLAS", 1: "Scalar", 2: "Warp MMA", 3: "Async MMA",
    4: "Split-K MMA", 5: "TMA/WGMMA", 6: "Warp specialized", 7: "Persistent",
    8: "TMA cooperative",
}


def configurations(profile, include_scalar):
    # (kernel, BN, split, ring slots, K boxes per group, persistent CTA cap)
    configs = [(0, 64, 1, 2, 2, 0)]
    if include_scalar:
        configs.append((1, 64, 1, 2, 2, 0))
    if profile == "full":
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
    return configs


def write_csv(records, fields, stream):
    if fields:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def print_table(measurements, timing):
    if not measurements:
        return
    headers = ("Kernel", "BN", "Split", "K tile", "Stages", "CTA/SM", "Median us", "Min us", "TFLOP/s",
               "cuBLAS us", "Speedup")
    rows = []
    for row, baseline in sorted(measurements, key=lambda item: tuple(
            int(item[0][key]) for key in ("kernel", "bn", "split", "tile_k", "stages", "workers_per_sm"))):
        kernel = int(row["kernel"])
        workers = row["workers_per_sm"] if int(row["workers_per_sm"]) else "auto"
        rows.append((f"{kernel} {STAGE_NAMES[kernel]}",
                     row["bn"] if int(row["bn"]) else "-", row["split"],
                     row["tile_k"], row["stages"], workers if kernel == 7 else "-",
                     f"{float(row['median_us']):.3f}", f"{float(row['min_us']):.3f}",
                     f"{float(row['tflops']):.2f}", f"{baseline:.3f}",
                     f"{float(row['speedup_vs_cublas']):.3f}x"))
    minimum_widths = (20, 4, 5, 6, 6, 6, 12, 12, 12, 12, 10)
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
    parser.add_argument("--profile", choices=("full", "h200-mig"), default="full",
                        help="full checkpoint sweep or focused H200 MIG candidates")
    parser.add_argument("--kernels", help="optional comma-separated IDs, e.g. 5,6,7,8")
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
    selected = set(range(9))
    if args.kernels:
        try:
            selected = {int(value) for value in args.kernels.split(",")}
        except ValueError:
            parser.error("kernels must be comma-separated IDs from 0 to 8")
        if not selected or not selected <= set(range(9)):
            parser.error("kernels must be comma-separated IDs from 0 to 8")
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
            result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
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
            baseline_us = float(baseline["median_us"])
            for column in ("bn", "split", "tile_k", "stages", "workers_per_sm"):
                int(measured[column])
            for column in ("median_us", "min_us", "tflops", "speedup_vs_cublas"):
                float(measured[column])
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
