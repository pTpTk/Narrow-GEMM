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
}


def write_csv(records, fields, stream):
    if fields:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def print_table(measurements, timing):
    if not measurements:
        return
    headers = ("Kernel", "BN", "Split", "Median us", "Min us", "TFLOP/s",
               "cuBLAS us", "Speedup")
    rows = []
    for row, baseline in sorted(measurements, key=lambda item: (
            int(item[0]["kernel"]), int(item[0]["bn"]), int(item[0]["split"]))):
        kernel = int(row["kernel"])
        rows.append((f"{kernel} {STAGE_NAMES[kernel]}",
                     row["bn"] if int(row["bn"]) else "-", row["split"],
                     f"{float(row['median_us']):.3f}", f"{float(row['min_us']):.3f}",
                     f"{float(row['tflops']):.2f}", f"{baseline:.3f}",
                     f"{float(row['speedup_vs_cublas']):.3f}x"))
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]

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
    configs = [(0, 64, 1)]
    if args.include_scalar:
        configs.append((1, 64, 1))
    for bn in (32, 64, 128):
        configs += [(2, bn, 1), (3, bn, 1)]
        configs += [(4, bn, s) for s in (2, 4, 8, 16)]
        for stage in (5, 6, 7):
            configs += [(stage, bn, s) for s in (1, 2, 4, 8, 16)]
    records, measurements, fields = [], [], None
    failed = False
    for index, (kernel, bn, split) in enumerate(configs, 1):
        cmd = [str(args.binary.resolve()), "--kernel", str(kernel), "--bn", str(bn),
               "--split", str(split), "--iterations", str(args.iterations),
               "--samples", str(args.samples), "--timing", args.timing]
        print(f"[{index}/{len(configs)}] kernel={kernel} BN={bn} split={split}",
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
            for column in ("bn", "split"):
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
