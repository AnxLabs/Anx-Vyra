"""Compressed, deterministic embedding of notebook assets, and the Kaggle 1 MB notebook-source size gate.

Kaggle's SaveKernel refuses a kernel source of 1 MB or more and the CLI reports only "400 Bad Request" (observed on the
rehearsal2 push of notebook c3ab200b, 1,708,020 bytes). This module stores the same asset texts and the same rehearsal
bytes more compactly; it changes how they are stored, never what they are:

* assets: json.dumps(sort_keys, compact separators, ASCII) -> zlib level 9 -> base64, one line ``assets_z=...``.
  The notebook checks the blob's SHA-256 (BOOTSTRAP_PACK_CORRUPT), decompresses it with the standard library, and then the
  unchanged per-asset SHA-256 loop (BOOTSTRAP_CORRUPT name) runs on the decompressed texts.
* rehearsal artifact: zlib level 9 -> base64; after decompression the pinned SHA-256 and byte count are checked as before.
* size gate: a built notebook must be strictly smaller than KAGGLE_SOURCE_LIMIT bytes (NOTEBOOK_TOO_LARGE); the builder
  refuses to write one that is not, and the CPU preflight re-checks the file on disk.

Usage: python notebook_pack.py check NOTEBOOK.ipynb   (exit 0 and NOTEBOOK_SIZE_OK, or exit 1 and NOTEBOOK_TOO_LARGE)
"""
import base64
import hashlib
import json
import sys
import zlib
from pathlib import Path

KAGGLE_SOURCE_LIMIT = 1048576
ZLIB_LEVEL = 9
METHOD = "assets: json(sort_keys, compact, ascii) -> zlib level 9 -> base64; rehearsal bytes: zlib level 9 -> base64"


class PackError(ValueError):
    pass


def pack_assets(assets):
    if not isinstance(assets, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in assets.items()):
        raise PackError("PACK_ASSETS_TYPE")
    data = json.dumps(assets, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return base64.b64encode(zlib.compress(data, ZLIB_LEVEL)).decode("ascii")


def unpack_assets(blob):
    return json.loads(zlib.decompress(base64.b64decode(blob, validate=True)).decode("ascii"))


def assets_bootstrap(assets):
    """Notebook lines that bind ``assets`` from the compressed blob (blob SHA-256 checked first)."""
    blob = pack_assets(assets)
    if unpack_assets(blob) != assets:
        raise PackError("PACK_ROUNDTRIP")
    return ("import zlib\n"
            "assets_z=" + repr(blob) + "\n"
            "assets_z_sha256=" + repr(hashlib.sha256(blob.encode("ascii")).hexdigest()) + "\n"
            "if hashlib.sha256(assets_z.encode('ascii')).hexdigest()!=assets_z_sha256: raise ValueError('BOOTSTRAP_PACK_CORRUPT')\n"
            "assets=json.loads(zlib.decompress(base64.b64decode(assets_z,validate=True)).decode('ascii'))\n")


def rehearsal_bootstrap_z(raw):
    """Like runtime_contract.rehearsal_bootstrap, but the embedded bytes are zlib-compressed; pins checked after decompression."""
    from runtime_contract import PINNED_BYTES, PINNED_SHA256, REHEARSAL_WORKING_PATH
    if hashlib.sha256(raw).hexdigest() != PINNED_SHA256 or len(raw) != PINNED_BYTES:
        raise ValueError("REPLAY_ARTIFACT_HASH")
    encoded = base64.b64encode(zlib.compress(raw, ZLIB_LEVEL)).decode("ascii")
    if zlib.decompress(base64.b64decode(encoded)) != raw:
        raise PackError("PACK_ROUNDTRIP_REHEARSAL")
    return (
        "embedded_rehearsal_b64=" + repr(encoded) + "\n"
        "replay_bytes=zlib.decompress(base64.b64decode(embedded_rehearsal_b64))\n"
        "if hashlib.sha256(replay_bytes).hexdigest()!=" + repr(PINNED_SHA256) + ": raise ValueError('REPLAY_ARTIFACT_HASH')\n"
        "if len(replay_bytes)!=" + str(PINNED_BYTES) + ": raise ValueError('REPLAY_ARTIFACT_BYTES')\n"
        "Path(" + repr(REHEARSAL_WORKING_PATH) + ").write_bytes(replay_bytes)\n"
    )


def notebook_bytes(built):
    """Exact bytes the builders write for a notebook dict."""
    return (json.dumps(built, indent=1, sort_keys=True) + "\n").encode("utf-8")


def check_size(size, limit=KAGGLE_SOURCE_LIMIT):
    if not isinstance(size, int) or size < 0:
        raise PackError("NOTEBOOK_SIZE_INVALID")
    if size >= limit:
        raise PackError("NOTEBOOK_TOO_LARGE %d >= %d" % (size, limit))
    return size


def check_notebook_file(path, limit=KAGGLE_SOURCE_LIMIT):
    path = Path(path)
    if not path.is_file():
        raise PackError("NOTEBOOK_MISSING")
    return check_size(path.stat().st_size, limit)


def embedded_assets(program):
    """Read the asset texts back out of a built notebook program (packed form)."""
    for line in program.splitlines():
        if line.startswith("assets_z="):
            import ast
            return unpack_assets(ast.literal_eval(line[len("assets_z="):]))
    raise PackError("PACK_LINE_MISSING")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "check":
        sys.exit("usage: notebook_pack.py check NOTEBOOK.ipynb")
    try:
        n = check_notebook_file(sys.argv[2])
    except PackError as exc:
        print(str(exc)); sys.exit(1)
    print("NOTEBOOK_SIZE_OK %d < %d" % (n, KAGGLE_SOURCE_LIMIT))
