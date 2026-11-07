"""Receipt check for one already built package directory. This file never submits."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    receipt_path = root / "validation_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("Receipt required file missing: validation_receipt.json")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "VERIFIED_LOCAL_PACKAGE" or receipt.get("optimizer_steps") != 0:
        raise ValueError("Invalid validation receipt")
    if receipt.get("model_forward_calls") != 0:
        raise ValueError("Invalid validation receipt")
    for relative, expected in receipt["locked_files"].items():
        path = root / relative
        if not path.is_file():
            raise ValueError("Receipt required file missing: " + relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Changed since isolated preflight: " + relative)
    return {
        "status": "VALIDATED_ARTIFACT_UNCHANGED",
        "submission_authorization": "NOT PROVIDED by this tool; explicit user approval still required",
        "gpu_training_tested": False,
    }


def write_receipt(root, validation_report, locked_files):
    """Write a receipt only for an isolated preflight that did not train.

    locked_files maps paths relative to root to sha256 hex digests. The caller
    has to have produced those files. A report that is not the isolated
    preflight pass is refused and nothing is written.
    """
    root = Path(root)
    required = (
        validation_report.get("status") == "PASS"
        and validation_report.get("pretraining_status") == "PREFLIGHT_INITIALIZED_NO_TRAINING"
        and validation_report.get("optimizer_steps") == 0
        and validation_report.get("model_forward_calls") == 0
    )
    if not required:
        raise ValueError("REFUSING_RECEIPT")
    for relative in locked_files:
        path = root / relative
        if not path.is_file():
            raise ValueError("REFUSING_RECEIPT missing " + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != locked_files[relative]:
            raise ValueError("REFUSING_RECEIPT hash " + relative)
    receipt = {
        "status": "VERIFIED_LOCAL_PACKAGE",
        "optimizer_steps": 0,
        "model_forward_calls": 0,
        "pretraining_status": "PREFLIGHT_INITIALIZED_NO_TRAINING",
        "gpu_training_tested": False,
        "submission_authorization": "NOT PROVIDED",
        "locked_files": dict(locked_files),
        "notebook_sha256": validation_report.get("notebook_sha256"),
        "runtime": validation_report.get("runtime"),
    }
    (root / "validation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
