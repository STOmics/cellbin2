"""Pure-stdlib encryption/decryption for the chip mask JSON.

Replaces the old Cython-compiled ``stereo_chip_name_c`` extension (which was
pinned to the cp38 ABI) with a portable scheme that runs on any Python >= 3.8
with no compiled extensions and no third-party crypto dependency.

File format ``CBMX1`` (all bytes, little-endian)::

    offset  size  field
    0       4     magic            b"CBMX"
    4       1     version          0x01
    5       16    salt             random per encryption
    21      4     iterations       PBKDF2 round count (uint32 LE)
    25      4     payload_len      length of the ciphertext (uint32 LE)
    29      N     ciphertext       XOR(zlib.compress(plaintext), keystream)
    29+N    32    hmac tag         HMAC-SHA256(master, header + ciphertext)

Keystream = SHAKE256(master_key + salt).digest(payload_len), where
master_key = PBKDF2-HMAC-SHA256(password, salt, iterations, 32).

The password is obfuscation-grade (it ships with the code, same as the old
compiled key). PBKDF2 + the HMAC tag still give real value: offline brute-force
of a leaked ``.enc`` is slowed, and any wrong key / corruption / tampering is
detected loudly on decrypt instead of silently producing garbage.
"""

import argparse
import hashlib
import hmac
import os
import struct
import sys
import zlib

_MAGIC = b"CBMX"
_VERSION = 1
_SALT_LEN = 16
_ITERATIONS = 200_000  # PBKDF2 rounds; only derives 32 bytes, so ~tens of ms
_KEY_LEN = 32  # SHA-256 output
_TAG_LEN = 32  # HMAC-SHA256 output
_HEADER_LEN = 4 + 1 + _SALT_LEN + 4 + 4  # magic + version + salt + iter + len

# Obfuscation-grade. To rotate: change this constant and re-encrypt the
# plaintext with ``python -m cellbin2.utils.chip_mask_crypto enc ...``.
_DEFAULT_PASSWORD = b"stOmIcs.cellbin2.chipMask/v1"


def _resolve_password(password):
    return _DEFAULT_PASSWORD if password is None else bytes(password)


def _keystream(master_key, salt, length):
    return hashlib.shake_256(master_key + salt).digest(length)


def _xor(data, ks):
    return bytes(b ^ k for b, k in zip(data, ks))


def encrypt_mask(plaintext: bytes, password: "bytes | None" = None) -> bytes:
    """Encrypt ``plaintext`` (raw bytes, e.g. UTF-8 JSON) into the CBMX1 blob."""
    if not isinstance(plaintext, (bytes, bytearray)):
        raise TypeError(f"plaintext must be bytes, got {type(plaintext).__name__}")
    pw = _resolve_password(password)

    payload = zlib.compress(bytes(plaintext), 9)  # smaller + mild obfuscation
    salt = os.urandom(_SALT_LEN)
    master_key = hashlib.pbkdf2_hmac("sha256", pw, salt, _ITERATIONS, _KEY_LEN)
    ks = _keystream(master_key, salt, len(payload))
    ct = _xor(payload, ks)

    header = _MAGIC + bytes([_VERSION]) + salt + struct.pack("<II", _ITERATIONS, len(ct))
    tag = hmac.new(master_key, header + ct, hashlib.sha256).digest()
    return header + ct + tag


def decrypt_mask(data: bytes, password: "bytes | None" = None) -> str:
    """Decrypt a CBMX1 blob produced by :func:`encrypt_mask`.

    Returns the original plaintext decoded as UTF-8 text (JSON string), matching
    the signature of the old compiled ``decrypt_mask``. Raises ``ValueError`` on
    a truncated blob, an unsupported format, a wrong password, or any tampering.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"data must be bytes, got {type(data).__name__}")
    data = bytes(data)
    if len(data) < _HEADER_LEN + _TAG_LEN:
        raise ValueError("truncated chip-mask blob (too short for header+tag)")

    magic = data[:4]
    if magic != _MAGIC:
        raise ValueError(f"not a CBMX chip-mask blob (bad magic {magic!r})")
    version = data[4]
    if version != _VERSION:
        raise ValueError(f"unsupported CBMX version {version}")
    salt = data[5:5 + _SALT_LEN]
    iterations, payload_len = struct.unpack("<II", data[21:29])
    ct = data[29:29 + payload_len]
    tag = data[29 + payload_len:29 + payload_len + _TAG_LEN]
    if len(ct) != payload_len or len(tag) != _TAG_LEN:
        raise ValueError("truncated chip-mask ciphertext/tag")

    pw = _resolve_password(password)
    master_key = hashlib.pbkdf2_hmac("sha256", pw, salt, iterations, _KEY_LEN)
    expected_tag = hmac.new(master_key, data[:29] + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError(
            "chip-mask authentication failed (wrong password or tampered data)"
        )

    ks = _keystream(master_key, salt, payload_len)
    payload = zlib.decompress(_xor(ct, ks))
    return payload.decode("utf-8")


def _main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m cellbin2.utils.chip_mask_crypto",
        description="Encrypt/decrypt chip_mask JSON (CBMX1 format).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enc", help="encrypt <in.json> -> <out.enc>")
    e.add_argument("infile")
    e.add_argument("outfile")
    d = sub.add_parser("dec", help="decrypt <in.enc> -> <out.json>")
    d.add_argument("infile")
    d.add_argument("outfile")
    args = parser.parse_args(argv)

    if args.cmd == "enc":
        with open(args.infile, "rb") as f:
            blob = encrypt_mask(f.read())
        with open(args.outfile, "wb") as f:
            f.write(blob)
        print(f"encrypted {args.infile} -> {args.outfile} ({len(blob)} bytes)")
    else:  # dec
        with open(args.infile, "rb") as f:
            text = decrypt_mask(f.read())
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"decrypted {args.infile} -> {args.outfile} ({len(text)} chars)")


if __name__ == "__main__":
    _main()
