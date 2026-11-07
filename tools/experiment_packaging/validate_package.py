"""Isolated CPU preflight of one exact generated notebook.

The sandbox is bubblewrap with a cleared environment, no network namespace,
no GPU devices, and no host workspace mount. The runtime interpreter is an
explicit directory. Missing isolation, missing runtime, or missing dataset
directories fail closed. This module does not invent those inputs and it
does not submit a job.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def _require_mounts(inputs):
    inputs = Path(inputs).resolve()
    x_root = inputs / "vyra-x-bundle"
    y_root = inputs / "vyra-y-bundle"
    if not x_root.is_dir() or not y_root.is_dir():
        raise ValueError("MISSING_DATASET_MOUNT")
    return x_root, y_root


def command(package, inputs, work, runtime):
    package = Path(package).resolve()
    work = Path(work).resolve()
    tools = Path(__file__).parent.resolve()
    runtime = Path(runtime).resolve()
    python = runtime / "bin" / "python"
    if not python.is_file():
        raise ValueError("CPU runtime python missing: " + str(python))
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise ValueError("ISOLATION_RUNNER_MISSING bwrap")
    x_root, y_root = _require_mounts(inputs)
    for host in ("/usr", "/lib"):
        if not Path(host).is_dir():
            raise ValueError("ISOLATION_HOST_PATH_MISSING " + host)
    cmd = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv", "PATH", "/runtime/bin:/usr/bin",
        "--setenv", "HOME", "/tmp",
        "--setenv", "CUDA_VISIBLE_DEVICES", "",
        "--setenv", "VYRA_PREFLIGHT_ONLY", "1",
        "--setenv", "OMP_NUM_THREADS", "2",
        "--setenv", "MKL_NUM_THREADS", "2",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
    ]
    if Path("/lib64").is_dir():
        cmd += ["--ro-bind", "/lib64", "/lib64"]
    notebook = package / "kernel" / "replay.ipynb"
    if not notebook.is_file():
        raise ValueError("NOTEBOOK_MISSING")
    # /tools and /runtime are the test harness, not Kaggle inputs. The notebook
    # file is mounted only there. /submission is not created. Dataset roots are
    # the two attachments already named by the package.
    cmd += [
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--tmpfs", "/tools",
        "--ro-bind", str(runtime), "/runtime",
        "--ro-bind", str(tools / "execute_notebook.py"), "/tools/execute_notebook.py",
        "--ro-bind", str(notebook), "/tools/replay.ipynb",
        "--ro-bind", str(x_root), "/kaggle/input/datasets/anjorosimoaj/vyra-x-bundle",
        "--ro-bind", str(y_root), "/kaggle/input/datasets/anjorosimoaj/vyra-y-bundle",
        "--bind", str(work), "/kaggle/working",
        "--chdir", "/kaggle/working",
        "/runtime/bin/python", "-I", "/tools/execute_notebook.py", "/tools/replay.ipynb",
    ]
    if "/submission" in cmd:
        raise ValueError("RUNTIME_CONTRACT_FORBIDDEN /submission")
    return cmd


def validate(package, inputs, work, log, runtime):
    package = Path(package).resolve()
    inputs = Path(inputs).resolve()
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    meta = json.loads((package / "kernel" / "kernel-metadata.json").read_text())
    if meta["is_private"] is not True or meta["enable_internet"] is not False:
        raise ValueError("Unsafe kernel privacy/network config")
    if meta["dataset_sources"] != ["anjorosimoaj/vyra-x-bundle", "anjorosimoaj/vyra-y-bundle"]:
        raise ValueError("Dataset config changed")
    cmd = command(package, inputs, work, runtime)
    start = time.monotonic()
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, timeout=90)
    if proc.returncode:
        raise RuntimeError("ISOLATED_NOTEBOOK_FAILED " + str(proc.returncode) + " see " + str(log))
    result = json.loads((work / "replay_out" / "result.json").read_text())
    timeout = json.loads((work / "timeout_status.json").read_text())
    if (
        result["status"] != "PREFLIGHT_INITIALIZED_NO_TRAINING"
        or result["initialization"]["optimizer_steps"] != 0
        or result["initialization"]["model_forward_calls"] != 0
    ):
        raise ValueError("Pretraining boundary failed")
    if timeout["timed_out"] or timeout["returncode"] != 0:
        raise ValueError("Child failed")
    environment = json.loads((work / "sandbox_probe.json").read_text())
    if not environment["network_blocked"] or environment["outside_workspace_visible"]:
        raise ValueError("Isolation failed")
    if environment.get("submission_exists") is not False:
        raise ValueError("SANDBOX_PRIVILEGED_MOUNT /submission")
    report = {
        "status": "PASS",
        "elapsed_s": time.monotonic() - start,
        "command": cmd,
        "notebook_sha256": hashlib.sha256((package / "kernel" / "replay.ipynb").read_bytes()).hexdigest(),
        "pretraining_status": result["status"],
        "model_forward_calls": 0,
        "optimizer_steps": 0,
        "dependency_checked_files": json.loads((work / "dependency_validation.json").read_text())["checked_files"],
        "sandbox": environment,
        "runtime": result["initialization"],
        "trainer_timeout_status": timeout,
        "gpu_training_tested": False,
    }
    (work / "ISOLATED_VALIDATION.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--runtime", required=True)
    args = parser.parse_args()
    result = validate(args.package, args.inputs, args.work, args.log, args.runtime)
    print(json.dumps({k: v for k, v in result.items() if k not in ("runtime", "command")}, indent=2))
