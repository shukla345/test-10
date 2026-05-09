#!/usr/bin/env python3
"""Generate 10,000 Dogecoin (DOGE) P2PKH addresses → doge_addresses.csv"""
import csv, secrets
from crypto_core import _ec_keys, hash160, b58check, _N

OUTPUT = "doge_addresses.csv"
ROWS   = 10_000

def gen(n: int) -> str:
    comp, _ = _ec_keys(n)
    return b58check(0x1e, hash160(comp))   # P2PKH  D…

with open(OUTPUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["private_key", "address"])
    for _ in range(ROWS):
        n = secrets.randbelow(_N - 1) + 1
        w.writerow([f"{n:064x}", gen(n)])

print(f"Done → {OUTPUT}")
