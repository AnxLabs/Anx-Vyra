"""Local private-artifact target: validate-experiment-package.

Builds the notebook twice from the supplied archives, runs the isolated CPU
preflight, runs the three negative cases, and writes a receipt bound to those
bytes. Missing archives, a missing runtime, or a missing isolation runner stop
the target. This file does not create stand-in datasets and it does not submit.
"""
import argparse
import filecmp
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from build import build
from negative_tests import run as run_negative
from validate_package import validate
from verify_receipt import verify, write_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(args):
    required = {
        "source-notebook": args.source_notebook,
        "x-archive": args.x_archive,
        "y-archive": args.y_archive,
        "builder-source": args.builder_source,
        "archive-audit": args.archive_audit,
        "inputs": args.inputs,
        "runtime": args.runtime,
        "replay-artifact": args.replay_artifact,
    }
    missing = [name + "=" + value for name, value in required.items() if not Path(value).exists()]
    if missing:
        return {
            "status": "REAL_INPUTS_UNAVAILABLE",
            "submission_ready": False,
            "gpu_training_tested": False,
            "missing": missing,
        }
    output = Path(args.output).resolve()
    if output == REPO_ROOT or REPO_ROOT in output.parents:
        return {
            "status": "REFUSING_REPOSITORY_OUTPUT",
            "submission_ready": False,
            "gpu_training_tested": False,
            "output": str(output),
        }
    incomplete = []
    for relative in ("vyra-x-bundle", "vyra-y-bundle"):
        path = Path(args.inputs) / relative
        if not path.is_dir():
            incomplete.append(str(path))
    runtime_python = Path(args.runtime) / "bin" / "python"
    if not runtime_python.is_file():
        incomplete.append(str(runtime_python))
    if incomplete:
        return {
            "status": "REAL_INPUTS_UNAVAILABLE",
            "submission_ready": False,
            "gpu_training_tested": False,
            "missing": incomplete,
        }
    return None


def _same(left, right):
    return filecmp.cmp(left, right, shallow=False)


def run_validated(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("REFUSING_EXISTING_OUTPUT " + str(output))
    output.mkdir(parents=True)
    first = output / "package"
    second = output / "rebuild"
    common = [
        "--source-notebook", args.source_notebook,
        "--x-archive", args.x_archive,
        "--y-archive", args.y_archive,
        "--builder-source", args.builder_source,
        "--archive-audit", args.archive_audit,
        "--replay-artifact", args.replay_artifact,
    ]
    build(argparse.Namespace(**_namespace(common, first)))
    build(argparse.Namespace(**_namespace(common, second)))
    for relative in (
        "kernel/replay.ipynb",
        "kernel/kernel-metadata.json",
        "dependency_lock.json",
    ):
        if not _same(first / relative, second / relative):
            raise ValueError("DETERMINISM_FAILED " + relative)
    work = output / "green_work"
    log = output / "logs" / "green_exact_payload.log"
    report = validate(first, args.inputs, work, log, args.runtime)
    run_negative(first, args.inputs, output / "negative", output / "logs", args.runtime)
    archives = output / "archives"
    archives.mkdir()
    shutil.copy2(args.x_archive, archives / "x-archive.zip")
    shutil.copy2(args.y_archive, archives / "y-archive.zip")
    locked_tools = output / "locked_tools"
    locked_tools.mkdir()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path, locked_tools / path.name)
    locked = {}
    for relative in (
        "package/kernel/replay.ipynb",
        "package/kernel/kernel-metadata.json",
        "package/dependency_lock.json",
        "archives/x-archive.zip",
        "archives/y-archive.zip",
    ):
        locked[relative] = _sha(output / relative)
    for path in sorted(locked_tools.glob("*.py")):
        locked["locked_tools/" + path.name] = _sha(path)
    write_receipt(output, report, locked)
    verified = verify(output)
    with tempfile.TemporaryDirectory(prefix="vyra-receipt-tamper-") as tmp:
        tampered = Path(tmp) / "bundle"
        shutil.copytree(output, tampered)
        notebook = tampered / "package" / "kernel" / "replay.ipynb"
        notebook.write_bytes(notebook.read_bytes() + b"\n")
        try:
            verify(tampered)
        except ValueError as exc:
            if "Changed since isolated preflight" not in str(exc):
                raise
        else:
            raise ValueError("TAMPERED_NOTEBOOK_WAS_ACCEPTED")
    return {
        "status": "ISOLATED_CPU_PREFLIGHT_PASS",
        "submission_ready": False,
        "gpu_training_tested": False,
        "submission_authorization": verified["submission_authorization"],
        "notebook_sha256": report["notebook_sha256"],
        "receipt": "validation_receipt.json",
        "determinism": "byte-identical rebuild",
    }


def _namespace(common, output):
    parser = argparse.ArgumentParser()
    for name in (
        "source-notebook",
        "x-archive",
        "y-archive",
        "builder-source",
        "archive-audit",
        "replay-artifact",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    return vars(parser.parse_args([*common, "--output", str(output)]))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="validate-experiment-package")
    parser.add_argument("--source-notebook", required=True)
    parser.add_argument("--x-archive", required=True)
    parser.add_argument("--y-archive", required=True)
    parser.add_argument("--builder-source", required=True)
    parser.add_argument("--archive-audit", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--replay-artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    blocked = assess(args)
    if blocked:
        print(json.dumps(blocked, indent=2, sort_keys=True))
        return 2
    try:
        payload = run_validated(args)
    except Exception as exc:
        print(json.dumps({
            "status": "VALIDATION_FAILED",
            "submission_ready": False,
            "gpu_training_tested": False,
            "error": str(exc),
        }, indent=2, sort_keys=True))
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
