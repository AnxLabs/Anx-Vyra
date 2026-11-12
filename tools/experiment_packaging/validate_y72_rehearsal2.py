"""Isolated validation of the built Y72 rehearsal2 notebook (size gate + CPU preflight + rejections; prereg/y72_rehearsal2.json).

First, fail closed: the built notebook must be under Kaggle's 1 MB source limit (notebook_pack.check_notebook_file), and an
oversized copy of it must be refused by the same gate. CKPT is the checkpoint_300.pt FILE (not a folder).

Sandbox: bubblewrap, cleared environment, no network, no GPU. The checkpoint is
mounted read-only at its exact planned path. Preflight exits before any
forward, backward, optimizer step or checkpoint torch.load; it proves the
bootstrap, the asset/dataset lock (including the checkpoint SHA-256) and the
input pins and the 400-entry plan pins. Negatives: checkpoint dataset absent, one flipped checkpoint byte,
right dataset but wrong file name, the real checkpoint_225 presented as checkpoint_300.pt, one flipped character in the embedded
pool asset with the compressed blob's hash left stale (BOOTSTRAP_PACK_CORRUPT), the pool asset changed inside a consistently
re-packed blob with stale per-asset hashes (BOOTSTRAP_CORRUPT), and a different pool embedded with consistently re-computed asset hashes and lock (the pool module's
registered hashes refuse it). The pass case also checks the pool record and the training-presentation disjointness audit.
Nothing is uploaded or submitted.
"""
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import kaggle_config as K
import notebook_pack as NP

package, inputs, runtime, ckpt, work_root, logs = map(Path, sys.argv[1:7])
tools = Path(__file__).parent.resolve()
MOUNT = K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_CKPT300@@")
logs.mkdir(parents=True, exist_ok=True)
if not ckpt.is_file():
    sys.exit("CHECKPOINT_ARG_MUST_BE_FILE " + ckpt.name)
_nb = package / "kernel" / "replay.ipynb"
try:
    _size = NP.check_notebook_file(_nb)
except NP.PackError as exc:                               # fail closed before any sandbox run
    (logs / "validate_y72_rehearsal2_summary.json").write_text(json.dumps([{"case": "size_gate_package", "ok": False, "error": str(exc)}], indent=1))
    sys.exit("SIZE_GATE " + str(exc))


def command(work, ckpt_mount, notebook=None):
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
           "--setenv", "PATH", "/runtime/bin:/usr/bin", "--setenv", "HOME", "/tmp", "--setenv", "CUDA_VISIBLE_DEVICES", "",
           "--setenv", "VYRA_PREFLIGHT_ONLY", "1", "--setenv", "OMP_NUM_THREADS", "2", "--setenv", "MKL_NUM_THREADS", "2",
           "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib"]
    if Path("/lib64").is_dir():
        cmd += ["--ro-bind", "/lib64", "/lib64"]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/tools",
            "--ro-bind", str(runtime), "/runtime",
            "--ro-bind", str(tools / "execute_notebook.py"), "/tools/execute_notebook.py",
            "--ro-bind", str(notebook or (package / "kernel" / "replay.ipynb")), "/tools/replay.ipynb",
            "--ro-bind", str(inputs / K.expand("@@SLUG_X_BUNDLE@@")), K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_X_BUNDLE@@"),
            "--ro-bind", str(inputs / K.expand("@@SLUG_Y_BUNDLE@@")), K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_Y_BUNDLE@@")]
    if ckpt_mount is not None:
        cmd += ["--ro-bind", str(ckpt_mount), MOUNT]
    cmd += ["--bind", str(work), K.expand("@@KROOT@@/working"), "--chdir", K.expand("@@KROOT@@/working"),
            "/runtime/bin/python", "-I", "/tools/execute_notebook.py", "/tools/replay.ipynb"]
    return cmd


def run(name, work, ckpt_mount, expect, notebook=None):
    work = work_root / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    log = logs / f"validate_{name}.log"
    t = time.monotonic()
    with log.open("w") as h:
        p = subprocess.run(command(work, ckpt_mount, notebook), stdout=h, stderr=subprocess.STDOUT, timeout=300)
    text = log.read_text()
    ok = (p.returncode == 0) if expect is None else (p.returncode != 0 and expect in text)
    out = {"case": name, "returncode": p.returncode, "expected": expect or "PASS", "ok": ok, "elapsed_s": round(time.monotonic() - t, 2)}
    if expect is None and ok:
        r = json.loads((work / "replay_out" / "result.json").read_text())
        timeout = json.loads((work / "timeout_status.json").read_text())
        dep = json.loads((work / "dependency_validation.json").read_text())
        out.update({"status": r["status"], "optimizer_steps": r["initialization"]["optimizer_steps"],
                    "model_forward_calls": r["initialization"]["model_forward_calls"],
                    "locked_files_checked": dep["checked_files"], "checkpoint_in_lock": any(k.endswith("checkpoint_300.pt") for k in dep["hashes"]),
                    "trainer_timeout": timeout, "plan_extension": r.get("plan_extension")})
        pex = r.get("plan_extension") or {}
        out["plan_pins_ok"] = (pex.get("policy") == "vyra-x-plan-extension-v1-continued-400" and pex.get("length") == 400
                               and pex.get("prefix300_sha256") == "2c59794e261a88216e5d77141b3373073c6e518545dce8a2578f279e95de86aa"
                               and pex.get("full_sha256") == "d9f132342c569d400a737bc7739ff9b87b427858984006437db3fda7f654d943")
        pool = r.get("rehearsal2_pool") or {}
        out["rehearsal2_pool"] = pool
        out["pool_pins_ok"] = (pool.get("canonical_sha256") == "d4315c338f44855e1162099c4a42792f0368c60a854f6d047930fa865ef976d8"
                               and pool.get("text_sha256") == "604bba8a47c81bd3edc0f347487d908bc54769f48981fb0dac1f9252c96c3b45"
                               and pool.get("rows") == 200 and pool.get("rule") == "vyra-y72-rehearsal2-fresh-v1")
        dis = r.get("disjointness") or {}
        out["disjointness"] = dis
        out["disjoint_ok"] = bool(dis) and all(v["row_overlap"] == 0 and v["payload_overlap"] == 0 for v in dis["heldouts"].values()) and "Y72v2_600" in dis["heldouts"]
        ok = ok and r["status"] == "PREFLIGHT_INITIALIZED_NO_TRAINING" and out["optimizer_steps"] == 0 and out["model_forward_calls"] == 0 and out["checkpoint_in_lock"] and out["plan_pins_ok"] and out["pool_pins_ok"] and out["disjoint_ok"]
        out["ok"] = ok
    elif not ok:
        out["tail"] = text[-600:]
    return out


results = []
results.append(run("preflight_pass", None, ckpt.parent, None))
bad = work_root / "stage_corrupt"; bad.mkdir(parents=True, exist_ok=True)
raw = bytearray(ckpt.read_bytes()); raw[len(raw) // 3] ^= 1
(bad / "checkpoint_300.pt").write_bytes(bytes(raw))
renamed = work_root / "stage_wrongname"; renamed.mkdir(parents=True, exist_ok=True)
shutil.copy2(ckpt, renamed / "checkpoint_300_copy.pt")
results.append(run("negative_corrupt_checkpoint", None, bad, "CORRUPT_DATASET_FILE"))
results.append(run("negative_missing_checkpoint_dataset", None, None, "MISSING_DATASET_ROOT C"))
results.append(run("negative_wrong_checkpoint_filename", None, renamed, "MISSING_DATASET_FILE"))
if len(sys.argv) > 7:                       # the real checkpoint_225 presented under the name checkpoint_300.pt
    other = work_root / "stage_other_checkpoint"; other.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sys.argv[7], other / "checkpoint_300.pt")
    results.append(run("negative_checkpoint_225_as_300", None, other, "CORRUPT_DATASET_FILE"))
    shutil.rmtree(other)
for d in (bad,):
    shutil.rmtree(d)
import ast as _ast
nb_path = package / "kernel" / "replay.ipynb"
nb = json.loads(nb_path.read_text())
prog = "".join(nb["cells"][0]["source"])


def size_case(name, path, expect_ok):
    try:
        n = NP.check_notebook_file(path); msg = "NOTEBOOK_SIZE_OK %d" % n; passed = True
    except NP.PackError as exc:
        msg = str(exc); passed = False
    (logs / f"validate_{name}.log").write_text(msg + "\n")
    return {"case": name, "expected": "PASS" if expect_ok else "NOTEBOOK_TOO_LARGE", "ok": passed == expect_ok and (expect_ok or msg.startswith("NOTEBOOK_TOO_LARGE")),
            "bytes": path.stat().st_size, "limit": NP.KAGGLE_SOURCE_LIMIT, "message": msg}


results.insert(0, size_case("size_gate_package", nb_path, True))
# the same notebook padded with one comment line to just over the limit must be refused by the preflight size gate
big = work_root / "nb_oversized" / "replay.ipynb"; big.parent.mkdir(parents=True, exist_ok=True)
pad = "#" + "p" * (NP.KAGGLE_SOURCE_LIMIT - nb_path.stat().st_size + 16) + "\n"
big.write_bytes(NP.notebook_bytes(dict(nb, cells=[dict(nb["cells"][0], source=(prog + pad).splitlines(keepends=True))])))
results.append(size_case("negative_oversized_notebook", big, False))
shutil.rmtree(big.parent)

lines = prog.splitlines(keepends=True)
iz = next(k for k, l in enumerate(lines) if l.startswith("assets_z=")); izs = next(k for k, l in enumerate(lines) if l.startswith("assets_z_sha256="))
ih = next(k for k, l in enumerate(lines) if l.startswith("asset_hashes="))
blob = _ast.literal_eval(lines[iz][len("assets_z="):])
assets = NP.unpack_assets(blob)


def write_nb(name, new_lines):
    path = work_root / name / "replay.ipynb"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(NP.notebook_bytes(dict(nb, cells=[dict(nb["cells"][0], source="".join(new_lines).splitlines(keepends=True))])))
    return path


# 1) one character of the compressed blob changed, its SHA-256 left as built
k = len(blob) // 2
L = list(lines); L[iz] = "assets_z=" + repr(blob[:k] + ("A" if blob[k] != "A" else "B") + blob[k + 1:]) + "\n"
results.append(run("negative_tampered_pack_blob", None, ckpt.parent, "BOOTSTRAP_PACK_CORRUPT", write_nb("nb_tampered_pack", L)))
# 2) one hex character of the pool asset changed, blob re-packed with a consistent blob SHA-256, per-asset hashes left as built
a2 = dict(assets); s = a2["y72_fresh_pool_F200_set.json"]; key = '"rows_int16le_hex": "'; i = s.index(key) + len(key) + 40
a2["y72_fresh_pool_F200_set.json"] = s[:i] + ("1" if s[i] != "1" else "2") + s[i + 1:]
b2 = NP.pack_assets(a2)
L = list(lines); L[iz] = "assets_z=" + repr(b2) + "\n"; L[izs] = "assets_z_sha256=" + repr(hashlib.sha256(b2.encode()).hexdigest()) + "\n"
results.append(run("negative_tampered_pool_asset", None, ckpt.parent, "BOOTSTRAP_CORRUPT y72_fresh_pool_F200_set.json", write_nb("nb_tampered_pool", L)))
# 3) a different pool (one scored byte of row 0 changed) embedded with consistently re-computed blob hash, asset hashes and lock
a3 = dict(assets)
pj = json.loads(a3["y72_fresh_pool_F200_set.json"]); h = pj["rows_int16le_hex"]; j = 4 * 300
pj["rows_int16le_hex"] = h[:j] + ("41" if h[j:j + 2] != "41" else "42") + h[j + 2:]
a3["y72_fresh_pool_F200_set.json"] = json.dumps(pj, indent=1)
lock = json.loads(a3["dependency_lock.json"])
lock["assets"]["y72_fresh_pool_F200_set.json"] = hashlib.sha256(a3["y72_fresh_pool_F200_set.json"].encode()).hexdigest()
a3["dependency_lock.json"] = json.dumps(lock, indent=2, sort_keys=True)
b3 = NP.pack_assets(a3)
L = list(lines); L[iz] = "assets_z=" + repr(b3) + "\n"; L[izs] = "assets_z_sha256=" + repr(hashlib.sha256(b3.encode()).hexdigest()) + "\n"
L[ih] = "asset_hashes=" + repr({n: hashlib.sha256(t.encode()).hexdigest() for n, t in a3.items()}) + "\n"
results.append(run("negative_swapped_pool_consistent_hashes", None, ckpt.parent, "REHEARSAL2_POOL", write_nb("nb_swapped_pool", L)))
(logs / "validate_y72_rehearsal2_summary.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
sys.exit(0 if all(r["ok"] for r in results) else 1)
