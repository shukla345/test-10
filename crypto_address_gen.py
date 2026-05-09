#!/usr/bin/env python3
"""
Multi-chain crypto address generator.

Input is treated directly as either:
- a secp256k1 private scalar for BTC/ETH/XRP/BNB/DOGE/TRX/AVAX/LTC/BCH, or
- a 32-byte Ed25519 seed for SOL/ADA.

Only classic XRP addresses are emitted here. X-addresses require the XRPL
codec format and should be derived with a verified XRPL library.
"""

import hashlib
import struct
import sys

try:
    from cryptography.hazmat.primitives.asymmetric.ec import derive_private_key, SECP256K1
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
except ImportError:
    print("[!] Missing dependency: pip install cryptography")
    sys.exit(1)

_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
_ROTC = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
_PIL = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]


def _keccak_f(state):
    for rc in _RC:
        c = [state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20] for x in range(5)]
        d = [c[(x - 1) % 5] ^ ((c[(x + 1) % 5] << 1 | c[(x + 1) % 5] >> 63) & 0xFFFFFFFFFFFFFFFF) for x in range(5)]
        state = [state[x] ^ d[x % 5] for x in range(25)]

        b = [0] * 25
        b[0] = state[0]
        last = state[1]
        for i, r in zip(_PIL, _ROTC):
            b[i] = ((last << r | last >> (64 - r)) & 0xFFFFFFFFFFFFFFFF)
            last = state[i]

        state = [b[x] ^ (~b[(x + 1) % 5 + (x // 5) * 5] & b[(x + 2) % 5 + (x // 5) * 5]) for x in range(25)]
        state[0] ^= rc
    return state


def keccak256(data: bytes) -> bytes:
    rate = 136
    msg = bytearray(data)
    msg.append(0x01)
    while len(msg) % rate != 0:
        msg.append(0x00)
    msg[-1] |= 0x80

    state = [0] * 25
    for i in range(0, len(msg), rate):
        block = msg[i : i + rate]
        for j in range(rate // 8):
            state[j] ^= struct.unpack_from("<Q", block, j * 8)[0]
        state = _keccak_f(state)

    out = b""
    for word in state[:4]:
        out += struct.pack("<Q", word)
    return out[:32]


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256d(data: bytes) -> bytes:
    return sha256(sha256(data))


def ripemd160(data: bytes) -> bytes:
    return hashlib.new("ripemd160", data).digest()


def hash160(data: bytes) -> bytes:
    return ripemd160(sha256(data))


def blake2b_224(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=28).digest()


_B58_ALPHA = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_XRP_ALPHA = b"rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz"


def _b58encode(data: bytes, alphabet: bytes = _B58_ALPHA) -> str:
    n = int.from_bytes(data, "big")
    out = []
    while n:
        n, r = divmod(n, 58)
        out.append(alphabet[r])
    leading_zeros = len(data) - len(data.lstrip(b"\x00"))
    out.extend([alphabet[0]] * leading_zeros)
    return bytes(reversed(out)).decode("ascii")


def b58check(version_byte: int, payload: bytes) -> str:
    raw = bytes([version_byte]) + payload
    return _b58encode(raw + sha256d(raw)[:4])


def xrp_b58check(version_bytes: bytes, payload: bytes) -> str:
    raw = version_bytes + payload
    return _b58encode(raw + sha256d(raw)[:4], _XRP_ALPHA)


_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_CASH_CHARSET = _BECH32_CHARSET


def _bech32_polymod(values):
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ v
        for i in range(5):
            if (top >> i) & 1:
                chk ^= gen[i]
    return chk


def _hrp_expand(hrp: str):
    return [ord(ch) >> 5 for ch in hrp] + [0] + [ord(ch) & 31 for ch in hrp]


def _to_5bit(data: bytes) -> list:
    acc = 0
    bits = 0
    out = []
    for byte in data:
        acc = (acc << 8) | byte
        bits += 8
        while bits >= 5:
            bits -= 5
            out.append((acc >> bits) & 31)
    if bits:
        out.append((acc << (5 - bits)) & 31)
    return out


def bech32_segwit(hrp: str, witver: int, witprog: bytes, bech32m: bool = False) -> str:
    data = [witver] + _to_5bit(witprog)
    const = 0x2BC830A3 if bech32m else 1
    chk = _bech32_polymod(_hrp_expand(hrp) + data + [0] * 6) ^ const
    checksum = [(chk >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(_BECH32_CHARSET[d] for d in data + checksum)


def bech32_raw(hrp: str, payload: bytes) -> str:
    data = _to_5bit(payload)
    chk = _bech32_polymod(_hrp_expand(hrp) + data + [0] * 6) ^ 1
    checksum = [(chk >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(_BECH32_CHARSET[d] for d in data + checksum)


def _cash_polymod(values):
    c = 1
    for d in values:
        c0 = c >> 35
        c = ((c & 0x07FFFFFFFF) << 5) ^ d
        for bit, poly in [
            (0x01, 0x98F2BC8E61),
            (0x02, 0x79B76D99E2),
            (0x04, 0xF33E5FB3C4),
            (0x08, 0xAE2EABE2A8),
            (0x10, 0x1E4F43E470),
        ]:
            if c0 & bit:
                c ^= poly
    return c ^ 1


def cashaddr(prefix: str, payload_hash: bytes, is_p2sh: bool = False) -> str:
    version_byte = 0x08 if is_p2sh else 0x00
    data = _to_5bit(bytes([version_byte]) + payload_hash)
    tmpl = [ord(ch) & 0x1F for ch in prefix] + [0] + data + [0] * 8
    checksum = _cash_polymod(tmpl)
    cs = [(checksum >> 5 * (7 - i)) & 31 for i in range(8)]
    return prefix + ":" + "".join(_CASH_CHARSET[d] for d in data + cs)


_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_ED25519_MAX = 1 << 256


def _is_valid_secp256k1_scalar(n: int) -> bool:
    return 0 < n < _N


def _ec_keys(n: int):
    sk = derive_private_key(n, SECP256K1())
    vk = sk.public_key()
    compressed = vk.public_bytes(Encoding.X962, PublicFormat.CompressedPoint)
    uncompressed = vk.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    return compressed, uncompressed[1:]


def _tagged_hash(tag: str, data: bytes) -> bytes:
    h = sha256(tag.encode())
    return sha256(h + h + data)


def taproot_output_key(privkey_int: int) -> bytes:
    comp, _ = _ec_keys(privkey_int)
    if comp[0] == 0x03:
        privkey_int = _N - privkey_int
        comp, _ = _ec_keys(privkey_int)
    x_only = comp[1:]
    tweak = _tagged_hash("TapTweak", x_only)
    tweaked_int = (privkey_int + int.from_bytes(tweak, "big")) % _N
    comp_tweaked, _ = _ec_keys(tweaked_int)
    return comp_tweaked[1:]


def eip55(addr_hex: str) -> str:
    digest = keccak256(addr_hex.encode()).hex()
    return "0x" + "".join(c.upper() if digest[i] >= "8" else c for i, c in enumerate(addr_hex))


def gen_bitcoin(n: int) -> list:
    comp, _ = _ec_keys(n)
    h160 = hash160(comp)
    redeem = b"\x00\x14" + h160
    ws = b"\x51" + bytes([len(comp)]) + comp + b"\x51\xae"
    return [
        ("P2PKH", b58check(0x00, h160)),
        ("P2SH", b58check(0x05, hash160(redeem))),
        ("P2WPKH", bech32_segwit("bc", 0, h160)),
        ("P2WSH", bech32_segwit("bc", 0, sha256(ws))),
        ("P2TR", bech32_segwit("bc", 1, taproot_output_key(n), bech32m=True)),
    ]


def gen_ethereum(n: int) -> list:
    _, raw64 = _ec_keys(n)
    addr = keccak256(raw64)[-20:].hex()
    return [
        ("EOA checksum", eip55(addr)),
        ("EOA lowercase", "0x" + addr),
        ("Contract nonce=0", "0x" + keccak256(b"\xd6\x94" + bytes.fromhex(addr) + b"\x80")[-20:].hex()),
    ]


def gen_xrp(n: int) -> list:
    comp, _ = _ec_keys(n)
    return [("Classic Address", xrp_b58check(b"\x00", hash160(comp)))]


def gen_bnb(n: int) -> list:
    comp, raw64 = _ec_keys(n)
    return [
        ("BEP-2", bech32_raw("bnb", hash160(comp))),
        ("BEP-20 / BSC", eip55(keccak256(raw64)[-20:].hex())),
    ]


def gen_solana(n: int) -> list:
    seed = n.to_bytes(32, "big")
    sk = Ed25519PrivateKey.from_private_bytes(seed)
    pub = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return [("Account Address", _b58encode(pub))]


def gen_dogecoin(n: int) -> list:
    comp, _ = _ec_keys(n)
    h160 = hash160(comp)
    return [
        ("P2PKH", b58check(0x1E, h160)),
        ("P2SH", b58check(0x16, hash160(b"\x00\x14" + h160))),
    ]


def gen_cardano(n: int) -> list:
    seed = n.to_bytes(32, "big")
    sk = Ed25519PrivateKey.from_private_bytes(seed)
    pub = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    pay_cred = blake2b_224(pub)
    return [
        ("Shelley Enterprise", bech32_raw("addr", bytes([0x61]) + pay_cred)),
        ("Shelley Base", bech32_raw("addr", bytes([0x01]) + pay_cred + pay_cred)),
        ("Stake / Reward", bech32_raw("stake", bytes([0xE1]) + pay_cred)),
    ]


def gen_tron(n: int) -> list:
    _, raw64 = _ec_keys(n)
    raw = b"\x41" + keccak256(raw64)[-20:]
    return [("TRX Address", _b58encode(raw + sha256d(raw)[:4]))]


def gen_avalanche(n: int) -> list:
    comp, raw64 = _ec_keys(n)
    h160 = hash160(comp)
    c_eth = keccak256(raw64)[-20:].hex()
    return [
        ("C-Chain", eip55(c_eth)),
        ("X-Chain", "X-" + bech32_raw("avax", h160)),
        ("P-Chain", "P-" + bech32_raw("avax", h160)),
    ]


def gen_litecoin(n: int) -> list:
    comp, _ = _ec_keys(n)
    h160 = hash160(comp)
    return [
        ("P2PKH", b58check(0x30, h160)),
        ("P2SH", b58check(0x32, hash160(b"\x00\x14" + h160))),
        ("P2WPKH", bech32_segwit("ltc", 0, h160)),
        ("P2TR", bech32_segwit("ltc", 1, taproot_output_key(n), bech32m=True)),
    ]


def gen_bch(n: int) -> list:
    comp, _ = _ec_keys(n)
    h160 = hash160(comp)
    redeem_h = hash160(b"\x00\x14" + h160)
    return [
        ("Legacy P2PKH", b58check(0x00, h160)),
        ("Legacy P2SH", b58check(0x05, redeem_h)),
        ("CashAddr P2PKH", cashaddr("bitcoincash", h160, is_p2sh=False)),
        ("CashAddr P2SH", cashaddr("bitcoincash", redeem_h, is_p2sh=True)),
    ]


CHAINS = [
    ("BTC", gen_bitcoin, "https://blockstream.info/address/{}", "secp256k1"),
    ("ETH", gen_ethereum, "https://etherscan.io/address/{}", "secp256k1"),
    ("XRP", gen_xrp, "https://xrpscan.com/account/{}", "secp256k1"),
    ("BNB", gen_bnb, "https://bscscan.com/address/{}", "secp256k1"),
    ("SOL", gen_solana, "https://solscan.io/account/{}", "ed25519"),
    ("DOGE", gen_dogecoin, "https://dogechain.info/address/{}", "secp256k1"),
    ("ADA", gen_cardano, "https://cardanoscan.io/address/{}", "ed25519"),
    ("TRX", gen_tron, "https://tronscan.org/#/address/{}", "secp256k1"),
    ("AVAX", gen_avalanche, "https://snowtrace.io/address/{}", "secp256k1"),
    ("LTC", gen_litecoin, "https://blockchair.com/litecoin/address/{}", "secp256k1"),
    ("BCH", gen_bch, "https://blockchair.com/bitcoin-cash/address/{}", "secp256k1"),
]


_W = 78


def _box_top():
    print("+" + "-" * _W + "+")


def _box_bot():
    print("+" + "-" * _W + "+")


def _box_mid():
    print("+" + "-" * _W + "+")


def _box_line(text):
    print("| " + str(text).ljust(_W - 1) + "|")


def _box_blank():
    _box_line("")


def print_results(n: int):
    _box_top()
    _box_line(f"  Private Key (integer)  : {n}")
    _box_line(f"  Private Key (hex)      : {n:064x}")
    if _is_valid_secp256k1_scalar(n):
        comp, _ = _ec_keys(n)
        _box_line(f"  Public Key (compressed): {comp.hex()}")
    else:
        _box_line("  Public Key (compressed): N/A for secp256k1-derived chains")
    _box_bot()
    print()

    for chain_name, gen_fn, explorer_tmpl, key_family in CHAINS:
        _box_top()
        _box_line(f"  {chain_name}")
        _box_mid()
        try:
            if key_family == "secp256k1" and not _is_valid_secp256k1_scalar(n):
                raise ValueError("requires a secp256k1 scalar smaller than the curve order")
            for label, addr in gen_fn(n):
                _box_line(f"    {label}")
                _box_line(f"      Address : {addr}")
                url_arg = addr.split(":")[-1] if ":" in addr and not addr.startswith("0x") else addr
                _box_line(f"      Explorer: {explorer_tmpl.format(url_arg)}")
                _box_blank()
        except Exception as exc:
            _box_line(f"    ERROR: {exc}")
            _box_blank()
        _box_bot()
        print()


def main():
    print()
    print("=" * (_W + 2))
    print("  Multi-Chain Crypto Address Generator")
    print("  Uses integer directly as secp256k1 scalar / Ed25519 seed")
    print("=" * (_W + 2))
    print()

    while True:
        raw = input("  Enter a positive integer (or 'q' to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            print("  Bye!\n")
            break
        try:
            n = int(raw)
            if n <= 0:
                print("  [!] Must be a positive integer.\n")
                continue
            if n >= _ED25519_MAX:
                print(f"  [!] Too large - must fit in 32 bytes:\n      {_ED25519_MAX - 1}\n")
                continue
        except ValueError:
            print("  [!] Not a valid integer.\n")
            continue

        if not _is_valid_secp256k1_scalar(n):
            print("  [!] This value is valid as a 32-byte Ed25519 seed.")
            print("      secp256k1 chains will be skipped because they require n < secp256k1 order.\n")

        print()
        print_results(n)


if __name__ == "__main__":
    main()
