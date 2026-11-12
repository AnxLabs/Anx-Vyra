"""Isolated validation of the built replay-weight (resume 225) notebook (CPU preflight + 3 notebook-level rejections).

Sandbox: bubblewrap, cleared environment, no network, no GPU. The checkpoint is
mounted read-only at its exact planned path. Preflight exits before any
forward, backward, optimizer step or checkpoint torch.load; it proves the
bootstrap, the asset/dataset lock (including the checkpoint SHA-256) and the
input pins. Negatives: checkpoint dataset absent, one flipped checkpoint byte,
right dataset but wrong file name. Nothing is uploaded or submitted.
"""
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import kaggle_config as K

package, inputs, runtime, ckpt, work_root, logs = map(Path, sys.argv[1:7])
tools = Path(__file__).parent.resolve()
MOUNT = K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_CKPT225@@")
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
            "--ro-bind", str(inputs / K.expand("@@SLUG_X_BUNDLE@@")), K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_X_BUNDLE@@"),
            "--ro-bind", str(inputs / K.expand("@@SLUG_Y_BUNDLE@@")), K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_Y_BUNDLE@@")]
    if ckpt_mount is not None:
        cmd += ["--ro-bind", str(ckpt_mount), MOUNT]
    cmd += ["--bind", str(work), K.expand("@@KROOT@@/working"), "--chdir", K.expand("@@KROOT@@/working"),
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
                    "locked_files_checked": dep["checked_files"], "checkpoint_in_lock": any(k.endswith("checkpoint_225.pt") for k in dep["hashes"]),
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
(bad / "checkpoint_225.pt").write_bytes(bytes(raw))
renamed = work_root / "stage_wrongname"; renamed.mkdir(parents=True, exist_ok=True)
shutil.copy2(ckpt, renamed / "checkpoint_225_copy.pt")
results.append(run("negative_corrupt_checkpoint", None, bad, "CORRUPT_DATASET_FILE"))
results.append(run("negative_missing_checkpoint_dataset", None, None, "MISSING_DATASET_ROOT C"))
results.append(run("negative_wrong_checkpoint_filename", None, renamed, "MISSING_DATASET_FILE"))
if len(sys.argv) > 7:                       # the real checkpoint_200 presented under the name checkpoint_225.pt
    other = work_root / "stage_other_checkpoint"; other.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sys.argv[7], other / "checkpoint_225.pt")
    results.append(run("negative_checkpoint_200_as_225", None, other, "CORRUPT_DATASET_FILE"))
    shutil.rmtree(other)
for d in (bad,):
    shutil.rmtree(d)
(logs / "validate_y72fix_summary.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
sys.exit(0 if all(r["ok"] for r in results) else 1)
