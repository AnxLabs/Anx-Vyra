"""Per-file contract for what the notebook may read at runtime.

Observed on the failed Kaggle job (cee1dbc): papermill executed the notebook,
assets were written under /kaggle/working, and
Path('/submission/rehearsal_Y72_40.pt') raised FileNotFoundError. Kaggle did
not provide /submission. That path existed only in the CPU sandbox.

The 88749-byte rehearsal file is embedded in the notebook and checked before
use. It is not a new dataset mount. No upload is performed here. The X and Y
dataset paths below are the paths already named in the package; this failed
job did not reach them, so they are documented, not re-observed.
"""
import base64
import hashlib

from rehearsal_artifact import PINNED_BYTES, PINNED_SHA256

REHEARSAL_WORKING_PATH = "/kaggle/working/rehearsal_Y72_40.pt"
DOCUMENTED_DATASET_MOUNTS = (
    "/kaggle/input/datasets/anjorosimoaj/vyra-x-bundle",
    "/kaggle/input/datasets/anjorosimoaj/vyra-y-bundle",
)
# Harness-only. The notebook must not read these. /runtime is an import
# allowlist prefix inside the trainer text, so it is not forbidden as a
# substring; the sandbox command is what must not present /submission.
FORBIDDEN_NOTEBOOK_INPUTS = ("/submission",)


def rehearsal_bootstrap(raw: bytes) -> str:
    """Return notebook lines that materialize the rehearsal file from source bytes."""
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PINNED_SHA256 or len(raw) != PINNED_BYTES:
        raise ValueError("REPLAY_ARTIFACT_HASH")
    encoded = base64.b64encode(raw).decode("ascii")
    return (
        "import base64\n"
        "embedded_rehearsal_b64=" + repr(encoded) + "\n"
        "replay_bytes=base64.b64decode(embedded_rehearsal_b64)\n"
        "if hashlib.sha256(replay_bytes).hexdigest()!=" + repr(PINNED_SHA256) + ": raise ValueError('REPLAY_ARTIFACT_HASH')\n"
        "if len(replay_bytes)!=" + str(PINNED_BYTES) + ": raise ValueError('REPLAY_ARTIFACT_BYTES')\n"
        "Path(" + repr(REHEARSAL_WORKING_PATH) + ").write_bytes(replay_bytes)\n"
    )


def assert_notebook_contract(program: str) -> None:
    """Refuse a notebook that reads a sandbox-only path or skips the rehearsal pin."""
    for forbidden in FORBIDDEN_NOTEBOOK_INPUTS:
        if forbidden in program:
            raise ValueError("RUNTIME_CONTRACT_FORBIDDEN " + forbidden)
    if ".glob(" in program:
        raise ValueError("RUNTIME_CONTRACT_GLOB")
    if "embedded_rehearsal_b64=" not in program or "base64.b64decode(embedded_rehearsal_b64)" not in program:
        raise ValueError("RUNTIME_CONTRACT_REHEARSAL_NOT_EMBEDDED")
    if repr(PINNED_SHA256) not in program:
        raise ValueError("RUNTIME_CONTRACT_REHEARSAL_HASH")
    if repr(REHEARSAL_WORKING_PATH) not in program:
        raise ValueError("RUNTIME_CONTRACT_REHEARSAL_PATH")
    for mount in DOCUMENTED_DATASET_MOUNTS:
        if mount not in program:
            raise ValueError("RUNTIME_CONTRACT_DATASET " + mount)


def materialize_embedded(encoded: str) -> bytes:
    """Decode and pin-check rehearsal bytes. Does not touch the filesystem."""
    raw = base64.b64decode(encoded)
    if hashlib.sha256(raw).hexdigest() != PINNED_SHA256 or len(raw) != PINNED_BYTES:
        raise ValueError("REPLAY_ARTIFACT_HASH")
    return raw
