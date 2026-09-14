#!/usr/bin/env python3
"""Internal Ed25519 key and offline license issuer.

This tool is intentionally not installed by the runtime package. Keep private
keys and issuance ledgers outside the source tree.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


PRODUCT = "deepstream-app-custom"
FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
KEY_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _read_json(path: Path, maximum: int = 16 * 1024) -> Dict[str, Any]:
    data = path.read_bytes()
    if not data or len(data) > maximum:
        raise ValueError(f"{path} is empty or exceeds {maximum} bytes")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("private key is not Ed25519")
    return key


def _load_public_key(path: Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("public key is not Ed25519")
    return key


def _validate_request(request: Dict[str, Any]) -> None:
    if request.get("schema") != 1:
        raise ValueError("unsupported request schema")
    if request.get("product") != PRODUCT:
        raise ValueError("request is for a different product")
    if not isinstance(request.get("request_id"), str) or not REQUEST_ID_RE.fullmatch(
        request["request_id"]
    ):
        raise ValueError("invalid request_id")
    fingerprint = request.get("device_fingerprint")
    if not isinstance(fingerprint, str) or not FINGERPRINT_RE.fullmatch(fingerprint):
        raise ValueError("invalid device fingerprint")


def command_keygen(args: argparse.Namespace) -> None:
    if not KEY_ID_RE.fullmatch(args.key_id):
        raise ValueError("key_id must contain 1-64 safe ASCII characters")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _atomic_write(args.private_key, private_pem, 0o600)
    _atomic_write(args.public_key, public_pem, 0o644)
    raw_public = public_key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    print(f"private key: {args.private_key}")
    print(f"public key:  {args.public_key}")
    print("Configure the production build with:")
    print(f"  -DLICENSE_ED25519_PUBLIC_KEY_HEX={raw_public}")
    print(f"  -DLICENSE_KEY_ID={args.key_id}")


def command_issue(args: argparse.Namespace) -> None:
    if not KEY_ID_RE.fullmatch(args.key_id):
        raise ValueError("key_id must contain 1-64 safe ASCII characters")
    customer = args.customer.strip()
    if not customer or len(customer.encode("utf-8")) >= 256:
        raise ValueError("customer must contain 1-255 UTF-8 bytes")
    request = _read_json(args.request)
    _validate_request(request)
    private_key = _load_private_key(args.private_key)
    issued_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    issued_at = issued_at.replace("+00:00", "Z")
    license_id = str(uuid4())
    payload: Dict[str, Any] = {
        "customer": customer,
        "device_fingerprint": request["device_fingerprint"],
        "issued_at": issued_at,
        "license_id": license_id,
        "perpetual": True,
        "product": PRODUCT,
        "request_id": request["request_id"],
        "schema": 1,
    }
    payload_bytes = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    signature = private_key.sign(payload_bytes)
    envelope = {
        "key_id": args.key_id,
        "payload": base64.b64encode(payload_bytes).decode("ascii"),
        "schema": 1,
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    license_bytes = (
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    _atomic_write(args.output, license_bytes, 0o644)

    ledger = args.ledger or args.private_key.with_suffix(
        args.private_key.suffix + ".issuance.jsonl"
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "customer": customer,
        "device_fingerprint": request["device_fingerprint"],
        "issued_at": issued_at,
        "license_file": str(args.output),
        "license_id": license_id,
        "request_id": request["request_id"],
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(ledger, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "ab") as stream:
            fd = -1
            stream.write(
                (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    print(f"issued license {license_id} for {request['device_fingerprint']}")
    print(f"license: {args.output}")
    print(f"ledger:  {ledger}")


def _decode_and_verify_license(
    license_path: Path, public_key: Ed25519PublicKey
) -> Dict[str, Any]:
    envelope = _read_json(license_path)
    if envelope.get("schema") != 1:
        raise ValueError("unsupported license envelope schema")
    try:
        payload = base64.b64decode(envelope["payload"], validate=True)
        signature = base64.b64decode(envelope["signature"], validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid license encoding") from error
    try:
        public_key.verify(signature, payload)
    except InvalidSignature as error:
        raise ValueError("license signature verification failed") from error
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("signed payload must be a JSON object")
    return decoded


def command_inspect(args: argparse.Namespace) -> None:
    payload = _decode_and_verify_license(
        args.license, _load_public_key(args.public_key)
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    keygen = subparsers.add_parser("keygen", help="generate an Ed25519 key pair")
    keygen.add_argument("--private-key", required=True, type=Path)
    keygen.add_argument("--public-key", required=True, type=Path)
    keygen.add_argument("--key-id", default="prod-2026-01")
    keygen.set_defaults(handler=command_keygen)

    issue = subparsers.add_parser("issue", help="issue a perpetual device license")
    issue.add_argument("--request", required=True, type=Path)
    issue.add_argument("--private-key", required=True, type=Path)
    issue.add_argument("--customer", required=True)
    issue.add_argument("--output", required=True, type=Path)
    issue.add_argument("--ledger", type=Path)
    issue.add_argument("--key-id", default="prod-2026-01")
    issue.set_defaults(handler=command_issue)

    inspect = subparsers.add_parser("inspect", help="verify and print a license")
    inspect.add_argument("--license", required=True, type=Path)
    inspect.add_argument("--public-key", required=True, type=Path)
    inspect.set_defaults(handler=command_inspect)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
