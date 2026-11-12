"""CPU smoke of the BUILT replay-weight experiment treatment on the real checkpoint_225 (test-only).

Runs the notebook's treatment.py in the bubblewrap sandbox (no network, no GPU),
restoring the real checkpoint_225.pt from the exact planned mount path and
running two real updates (226, 227) with replay weight 0.5 with the real extended plan, real eval,
real X64 dump, real sparse diagnostics, real v2 checkpoint writes.

Mechanical, count-asserted substitutions ONLY (all recorded in SUBSTITUTIONS.json):
cuda->cpu, cuda stubs, gpu_guard stub, loop end 300->202, EVAL/CHECKPOINT/DIAGNOSTIC
updates 201 and 202, final-eval key, time caps. The continuation gates run
unchanged except `cpu_smoke=True` (VYRA_CONT_CPU_SMOKE=1): CUDA RNG restore is
skipped and the eval-count comparison is recorded, not fatal. Nothing is uploaded.
"""
import ast, json, shutil, subprocess, sys
from pathlib import Path
import kaggle_config as K

nb, inputs, runtime, ckpt_dir, rehearsal, work = sys.argv[1:7]
end = int(sys.argv[7]) if len(sys.argv) > 7 else 227
nb, inputs, runtime, ckpt_dir, rehearsal, work = map(Path, (nb, inputs, runtime, ckpt_dir, rehearsal, work))
code = "".join(json.loads(nb.read_text())["cells"][0]["source"])
assets = ast.literal_eval(next(n for n in ast.parse(code).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "assets").value)
upd = tuple(range(226, end + 1))
SUBS = [
    ("treatment.py", "HARD=600; SOFT=598; STEPS=300; START=225; RB=2; EVAL_AT={250,275,300}",
     "HARD=3000; SOFT=2998; STEPS=300; START=225; RB=2; EVAL_AT={%s}" % ",".join(map(str, upd)), 1),
    ("treatment.py", "DEVICE='cpu' if PREFLIGHT_ONLY else 'cuda'", "DEVICE='cpu'", 1),
    ("treatment.py", ".to('cuda')", ".to('cpu')", None),
    ("treatment.py", "  import torch\n",
     "  import torch\n  for _n in ('manual_seed_all','reset_peak_memory_stats','synchronize'): setattr(torch.cuda,_n,lambda *a,**k: None)\n  torch.cuda.max_memory_allocated=lambda *a,**k: 0; torch.cuda.max_memory_reserved=lambda *a,**k: 0\n", 1),
    ("treatment.py", "for s in range(START, STEPS):", "for s in range(START, %d):" % end, 1),
    ("treatment.py", "A['final']=A['trajectory'].get(str(STEPS))", "A['final']=A['trajectory'].get('%d')" % end, 1),
    ("continuation.py", "CHECKPOINT_UPDATES = (225, 250, 275, 300)", "CHECKPOINT_UPDATES = %r" % (upd,), 1),
    ("y72fix_gates.py", "CHECKPOINTS = (250, 275, 300)", "CHECKPOINTS = %r" % (upd,), 1),
    ("preclip_diag.py", "DIAGNOSTIC_UPDATES = (250, 275, 300)", "DIAGNOSTIC_UPDATES = %r" % (upd,), 1),
]
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
for name, text in assets.items():
    (work / name).write_text(text)
rec = []
for name, old, new, n in SUBS:
    t = (work / name).read_text()
    c = t.count(old)
    assert c and (n is None or c == n), (name, old, c)
    (work / name).write_text(t.replace(old, new))
    rec.append({"asset": name, "old": old, "new": new, "occurrences": c})
assert "'cuda'" not in (work / "treatment.py").read_text()
(work / "gpu_guard.py").write_text("def require_t4(torch):\n    return {'stub':'CPU smoke, no GPU'}\n")
(work / "SUBSTITUTIONS.json").write_text(json.dumps(rec, indent=1))
shutil.copy2(rehearsal, work / "rehearsal_Y72_40.pt")
x, y = inputs / K.expand("@@SLUG_X_BUNDLE@@"), inputs / K.expand("@@SLUG_Y_BUNDLE@@")
cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
       "--setenv", "PATH", "/runtime/bin:/usr/bin", "--setenv", "HOME", "/tmp", "--setenv", "CUDA_VISIBLE_DEVICES", "",
       "--setenv", "VYRA_PREFLIGHT_ONLY", "0", "--setenv", "VYRA_CONT_CPU_SMOKE", "1", "--setenv", "OMP_NUM_THREADS", "2",
       "--setenv", "XROOT", K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_X_BUNDLE@@"),
       "--setenv", "YROOT", K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_Y_BUNDLE@@"),
       "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib"] + (["--ro-bind", "/lib64", "/lib64"] if Path("/lib64").is_dir() else []) + [
       "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind", str(runtime), "/runtime",
       "--ro-bind", str(x), K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_X_BUNDLE@@"),
       "--ro-bind", str(y), K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_Y_BUNDLE@@"),
       "--ro-bind", str(ckpt_dir), K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_CKPT225@@"),
       "--bind", str(work), K.expand("@@KROOT@@/working"), "--chdir", K.expand("@@KROOT@@/working"),
       "/runtime/bin/python", "-u", K.expand("@@KROOT@@/working/treatment.py")]
rc = subprocess.run(cmd, timeout=3300).returncode
r = json.loads((work / "replay_out" / "result.json").read_text())
print("RC", rc, "STATUS", r.get("status"))
sys.exit(rc)
