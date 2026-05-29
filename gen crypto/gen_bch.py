#!/usr/bin/env python3
"""Generate 10,000 Bitcoin Cash (BCH) CashAddr addresses → bch_addresses.csv"""
import csv, secrets
from crypto_core import _ec_keys, hash160, cashaddr, _N

OUTPUT = "bch_addresses.csv"
ROWS   = 10_000

def gen(n: int) -> str:
    comp, _ = _ec_keys(n)
    return cashaddr("bitcoincash", hash160(comp), is_p2sh=False)   # bitcoincash:q…

with open(OUTPUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["private_key", "address"])
    for _ in range(ROWS):
        n = secrets.randbelow(_N - 1) + 1
        w.writerow([f"{n:064x}", gen(n)])

print(f"Done → {OUTPUT}")
