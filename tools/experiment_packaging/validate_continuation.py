"""Isolated validation of the built continuation notebook (CPU preflight + 3 notebook-level rejections).

Sandbox: bubblewrap, cleared environment, no network, no GPU. The checkpoint is
mounted read-only at its exact planned path. Preflight exits before any
forward, backward, optimizer step or checkpoint torch.load; it proves the
bootstrap, the asset/dataset lock (including the checkpoint SHA-256) and the
input pins. Negatives: checkpoint dataset absent, one flipped checkpoint byte,
right dataset but wrong file name. Nothing is uploaded or submitted.
"""
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

package, inputs, runtime, ckpt, work_root, logs = map(Path, sys.argv[1:7])
tools = Path(__file__).parent.resolve()
MOUNT = "/kaggle/input/datasets/anjorosimoaj/vyra-expanded40-200u-checkpoint"
logs.mkdir(parents=True, exist_ok=True)


def command(work, ckpt_mount):
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
           "--setenv", "PATH", "/runtime/bin:/usr/bin", "--setenv", "HOME", "/tmp", "--setenv", "CUDA_VISIBLE_DEVICES", "",
           "--setenv", "VYRA_PREFLIGHT_ONLY", "1", "--setenv", "OMP_NUM_THREADS", "2", "--setenv", "MKL_NUM_THREADS", "2",
           "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib"]
    if Path("/lib64").is_dir():
        cmd += ["--ro-bind", "/lib64", "/lib64"]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/tools",
            "--ro-bind", str(runtime), "/runtime",
            "--ro-bind", str(tools / "execute_notebook.py"), "/tools/execute_notebook.py",
            "--ro-bind", str(package / "kernel" / "replay.ipynb"), "/tools/replay.ipynb",
            "--ro-bind", str(inputs / "vyra-x-bundle"), "/kaggle/input/datasets/anjorosimoaj/vyra-x-bundle",
            "--ro-bind", str(inputs / "vyra-y-bundle"), "/kaggle/input/datasets/anjorosimoaj/vyra-y-bundle"]
    if ckpt_mount is not None:
        cmd += ["--ro-bind", str(ckpt_mount), MOUNT]
    cmd += ["--bind", str(work), "/kaggle/working", "--chdir", "/kaggle/working",
            "/runtime/bin/python", "-I", "/tools/execute_notebook.py", "/tools/replay.ipynb"]
    return cmd


def run(name, work, ckpt_mount, expect):
    work = work_root / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    log = logs / f"validate_{name}.log"
    t = time.monotonic()
    with log.open("w") as h:
        p = subprocess.run(command(work, ckpt_mount), stdout=h, stderr=subprocess.STDOUT, timeout=300)
    text = log.read_text()
    ok = (p.returncode == 0) if expect is None else (p.returncode != 0 and expect in text)
    out = {"case": name, "returncode": p.returncode, "expected": expect or "PASS", "ok": ok, "elapsed_s": round(time.monotonic() - t, 2)}
    if expect is None and ok:
        r = json.loads((work / "replay_out" / "result.json").read_text())
        timeout = json.loads((work / "timeout_status.json").read_text())
        dep = json.loads((work / "dependency_validation.json").read_text())
        out.update({"status": r["status"], "optimizer_steps": r["initialization"]["optimizer_steps"],
                    "model_forward_calls": r["initialization"]["model_forward_calls"],
                    "locked_files_checked": dep["checked_files"], "checkpoint_in_lock": any(k.endswith("checkpoint_200.pt") for k in dep["hashes"]),
                    "trainer_timeout": timeout, "plan_extension": r.get("plan_extension")})
        ok = ok and r["status"] == "PREFLIGHT_INITIALIZED_NO_TRAINING" and out["optimizer_steps"] == 0 and out["model_forward_calls"] == 0 and out["checkpoint_in_lock"]
        out["ok"] = ok
    elif not ok:
        out["tail"] = text[-600:]
    return out


results = []
results.append(run("preflight_pass", None, ckpt.parent, None))
bad = work_root / "stage_corrupt"; bad.mkdir(parents=True, exist_ok=True)
raw = bytearray(ckpt.read_bytes()); raw[len(raw) // 3] ^= 1
(bad / "checkpoint_200.pt").write_bytes(bytes(raw))
renamed = work_root / "stage_wrongname"; renamed.mkdir(parents=True, exist_ok=True)
shutil.copy2(ckpt, renamed / "checkpoint_200_copy.pt")
results.append(run("negative_corrupt_checkpoint", None, bad, "CORRUPT_DATASET_FILE"))
results.append(run("negative_missing_checkpoint_dataset", None, None, "MISSING_DATASET_ROOT C"))
results.append(run("negative_wrong_checkpoint_filename", None, renamed, "MISSING_DATASET_FILE"))
for d in (bad,):
    shutil.rmtree(d)
(logs / "validate_continuation_summary.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
sys.exit(0 if all(r["ok"] for r in results) else 1)
