#!/usr/bin/env python3
"""Run the staged experiment. Redirect stdout to ../results.csv, not the repo."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[2] / "build/narrow_gemm")
    parser.add_argument("--include-scalar", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timing", choices=("graph", "events"), default="graph")
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
    writer = None
    for kernel, bn, split in configs:
        cmd = [str(args.binary.resolve()), "--kernel", str(kernel), "--bn", str(bn),
               "--split", str(split), "--iterations", str(args.iterations),
               "--samples", str(args.samples), "--timing", args.timing]
        print("+ " + " ".join(cmd), file=sys.stderr)
        result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
        reader = csv.DictReader(result.stdout.splitlines())
        if writer is None:
            if not reader.fieldnames:
                raise RuntimeError("benchmark returned no CSV header")
            writer = csv.DictWriter(sys.stdout, fieldnames=reader.fieldnames)
            writer.writeheader()
        writer.writerows(reader)  # Keep each local cuBLAS baseline for auditability.
        sys.stdout.flush()


if __name__ == "__main__":
    main()
