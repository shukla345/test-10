#!/usr/bin/env python3
"""Generate all per-chain CSV files in one run."""

import csv
import secrets

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from crypto_core import (
    _N,
    _b58encode,
    _ec_keys,
    bech32_raw,
    bech32_segwit,
    blake2b_224,
    cashaddr,
    eip55,
    hash160,
    keccak256,
    sha256d,
    xrp_b58check,
)

ROWS = 10_000
ED25519_MAX = 1 << 256


def gen_btc(n: int) -> str:
    comp, _ = _ec_keys(n)
    return bech32_segwit("bc", 0, hash160(comp))


def gen_eth(n: int) -> str:
    _, raw64 = _ec_keys(n)
    return eip55(keccak256(raw64)[-20:].hex())


def gen_xrp(n: int) -> str:
    comp, _ = _ec_keys(n)
    return xrp_b58check(b"\x00", hash160(comp))


def gen_bnb(n: int) -> str:
    _, raw64 = _ec_keys(n)
    return eip55(keccak256(raw64)[-20:].hex())


def gen_sol(n: int) -> str:
    seed = n.to_bytes(32, "big")
    sk = Ed25519PrivateKey.from_private_bytes(seed)
    pub = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return _b58encode(pub)


def gen_doge(n: int) -> str:
    comp, _ = _ec_keys(n)
    return crypto_core_b58check(0x1E, hash160(comp))


def gen_ada(n: int) -> str:
    seed = n.to_bytes(32, "big")
    sk = Ed25519PrivateKey.from_private_bytes(seed)
    pub = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    pay_cred = blake2b_224(pub)
    return bech32_raw("addr", bytes([0x61]) + pay_cred)


def gen_trx(n: int) -> str:
    _, raw64 = _ec_keys(n)
    raw = b"\x41" + keccak256(raw64)[-20:]
    return _b58encode(raw + sha256d(raw)[:4])


def gen_avax(n: int) -> str:
    _, raw64 = _ec_keys(n)
    return eip55(keccak256(raw64)[-20:].hex())


def gen_ltc(n: int) -> str:
    comp, _ = _ec_keys(n)
    return bech32_segwit("ltc", 0, hash160(comp))


def gen_bch(n: int) -> str:
    comp, _ = _ec_keys(n)
    return cashaddr("bitcoincash", hash160(comp), is_p2sh=False)


def crypto_core_b58check(version_byte: int, payload: bytes) -> str:
    raw = bytes([version_byte]) + payload
    return _b58encode(raw + sha256d(raw)[:4])


JOBS = [
    ("btc_addresses.csv", gen_btc, _N),
    ("eth_addresses.csv", gen_eth, _N),
    ("xrp_addresses.csv", gen_xrp, _N),
    ("bnb_addresses.csv", gen_bnb, _N),
    ("sol_addresses.csv", gen_sol, ED25519_MAX),
    ("doge_addresses.csv", gen_doge, _N),
    ("ada_addresses.csv", gen_ada, ED25519_MAX),
    ("trx_addresses.csv", gen_trx, _N),
    ("avax_addresses.csv", gen_avax, _N),
    ("ltc_addresses.csv", gen_ltc, _N),
    ("bch_addresses.csv", gen_bch, _N),
]


def write_csv(filename: str, gen_fn, upper_bound: int) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        #writer.writerow(["private_key", "address"])
        for _ in range(ROWS):
            n = secrets.randbelow(upper_bound - 1) + 1
            writer.writerow([f"{n:064x}", gen_fn(n)])


def main() -> None:
    for filename, gen_fn, upper_bound in JOBS:
        write_csv(filename, gen_fn, upper_bound)
        print(f"Done -> {filename}")


if __name__ == "__main__":
    main()
