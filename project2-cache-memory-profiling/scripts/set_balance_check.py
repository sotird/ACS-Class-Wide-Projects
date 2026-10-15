#!/usr/bin/env python3
"""Offline check of the cache-set balance of the exp7 'spread' layouts on 2 MiB pages (report, pitfall 8).

Inside a 2 MiB page physical and virtual offsets coincide, so the address bits that select the L2 set (bits 15:6 for a 1 MiB,
16-way, 64 B-line cache = 1024 sets) are fixed by the layout. This counts how many of the L lines land in each set for the
address-order layout (page slot = logical page) and for randomly permuted page slots (what exp7 uses), and how many lines
exceed the 16 ways. It is a simple-index MODEL: the real index function may be hashed, so the count motivates the control
(permuted slots) but does not predict individual measurements. Writes data/processed/set_balance_check.txt.
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "processed", "set_balance_check.txt")


def spread(L, P, permute, seed=31337):
    """Same construction as layout_spread() in src/chase.hpp (byte offsets of the L lines)."""
    j = np.arange(L)
    page, k = j % P, j // P
    base = page * 64 // P if P < 64 else page % 64
    line = (base + k) % 64
    slot = np.random.default_rng(seed).permutation(P)[page] if permute else page
    return slot * 4096 + line * 64


lines = []
for L, Ps in ((8192, (2048, 3072, 4096, 6144, 8192)), (65536, (16384, 32768, 65536))):
    for P in Ps:
        for perm in (False, True):
            sets = (spread(L, P, perm) >> 6) & 1023
            c = np.bincount(sets, minlength=1024)
            lines.append(f"L={L:6d} lines  P={P:6d} slots  permuted={str(perm):5}  L2 sets used {int((c > 0).sum()):4d}/1024  "
                         f"max lines in a set {int(c.max()):5d}  lines beyond 16 ways {int((c - 16).clip(0).sum()):6d}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
