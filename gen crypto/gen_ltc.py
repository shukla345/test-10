#!/usr/bin/env python3
"""Generate 10,000 Litecoin (LTC) P2WPKH addresses → ltc_addresses.csv"""
import csv, secrets
from crypto_core import _ec_keys, hash160, bech32_segwit, _N

OUTPUT = "ltc_addresses.csv"
ROWS   = 10_000

def gen(n: int) -> str:
    comp, _ = _ec_keys(n)
    return bech32_segwit("ltc", 0, hash160(comp))   # Native SegWit ltc1q…

with open(OUTPUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["private_key", "address"])
    for _ in range(ROWS):
        n = secrets.randbelow(_N - 1) + 1
        w.writerow([f"{n:064x}", gen(n)])

print(f"Done → {OUTPUT}")
