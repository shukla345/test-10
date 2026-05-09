#!/usr/bin/env python3
"""Generate 10,000 Tron (TRX) addresses → trx_addresses.csv"""
import csv, secrets
from crypto_core import _ec_keys, keccak256, _b58encode, sha256d, _N

OUTPUT = "trx_addresses.csv"
ROWS   = 10_000

def gen(n: int) -> str:
    _, raw64 = _ec_keys(n)
    raw = b'\x41' + keccak256(raw64)[-20:]
    return _b58encode(raw + sha256d(raw)[:4])   # T…

with open(OUTPUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["private_key", "address"])
    for _ in range(ROWS):
        n = secrets.randbelow(_N - 1) + 1
        w.writerow([f"{n:064x}", gen(n)])

print(f"Done → {OUTPUT}")
