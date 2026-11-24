#!/usr/bin/env python3
"""Arithmetic and work-grid model; none of these numbers are measurements."""
import argparse
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sms", type=int, default=132)
    parser.add_argument("--hbm-tb-s", type=float, default=3.35)
    args = parser.parse_args()
    if args.sms <= 0 or args.hbm_tb_s <= 0:
        parser.error("SM count and bandwidth must be positive")
    m, n, k = 64, 4096, 8192
    flops = 2 * m * n * k
    a, b, c = 2 * m * k, 2 * n * k, 4 * m * n
    print("Analytical estimates only; one CTA per SM is assumed for waves.")
    print(f"FLOPs={flops:,}; A={a:,} B={b:,} C={c:,} bytes")
    print(f"Compulsory intensity={flops / (a+b+c):.3f} FLOP/byte")
    print(f"Compulsory HBM floor={(a+b+c)/(args.hbm_tb_s*1e6):.3f} us")
    print("BN split CTAs waves slot_util A_CTA_MiB extra_partial_MiB K_per_CTA")
    for bn in (32, 64, 128, 256):
        for split in (1, 2, 4, 8, 16):
            ctas = (n // bn) * split
            waves = math.ceil(ctas / args.sms)
            # Counts requests at the CTA/global interface, NOT HBM traffic.
            a_requested = a * (n // bn)
            # S partial writes + S partial reads; final C store already counted.
            extra = 2 * split * c if split > 1 else 0
            print(f"{bn:3d} {split:5d} {ctas:4d} {waves:5d} "
                  f"{ctas/(waves*args.sms):9.3f} {a_requested/2**20:9.1f} "
                  f"{extra/2**20:17.1f} {k//split:9d}")


if __name__ == "__main__":
    main()
