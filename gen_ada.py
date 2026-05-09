#!/usr/bin/env python3
"""Generate 10,000 Cardano (ADA) Shelley enterprise addresses → ada_addresses.csv"""
import csv, secrets
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from crypto_core import blake2b_224, bech32_raw

OUTPUT = "ada_addresses.csv"
ROWS   = 10_000
_MAX   = (1 << 256) - 1

def gen(n: int) -> str:
    seed     = n.to_bytes(32, 'big')
    sk       = Ed25519PrivateKey.from_private_bytes(seed)
    pub      = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    pay_cred = blake2b_224(pub)
    return bech32_raw("addr", bytes([0x61]) + pay_cred)   # Enterprise addr1v…

with open(OUTPUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["private_key", "address"])
    for _ in range(ROWS):
        n = secrets.randbelow(_MAX - 1) + 1
        w.writerow([f"{n:064x}", gen(n)])

print(f"Done → {OUTPUT}")
