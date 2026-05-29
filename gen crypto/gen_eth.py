#!/usr/bin/env python3
"""Generate 10,000 Ethereum (ETH) addresses → eth_addresses.csv"""
import csv, secrets
from crypto_core import _ec_keys, keccak256, eip55, _N

OUTPUT = "eth_addresses.csv"
ROWS   = 10_000

def gen(n: int) -> str:
    _, raw64 = _ec_keys(n)
    return eip55(keccak256(raw64)[-20:].hex())   # EIP-55 checksum 0x…

with open(OUTPUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["private_key", "address"])
    for _ in range(ROWS):
        n = secrets.randbelow(_N - 1) + 1
        w.writerow([f"{n:064x}", gen(n)])

print(f"Done → {OUTPUT}")
