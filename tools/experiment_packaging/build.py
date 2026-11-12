"""Deterministic notebook builder. Never regenerates data and never submits a job.

The treatment caller embedded in the source notebook is rewritten in two
places: the existing CPU-preflight stop, and the per-update expert-selection
union. Archive bytes have to match the audit. A wrong builder provenance hash
refuses the build before any notebook is written.
"""
import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path

from expert_union import apply_expert_union, apply_replay_telemetry
from expanded_rehearsal import apply_expanded_rehearsal
from update200 import apply_update200
from name_order import used_before_binding
from rehearsal_artifact import PINNED_SHA256, validate_rehearsal_file
from runtime_contract import assert_notebook_contract, rehearsal_bootstrap
import kaggle_config as K

K.load()  # fail closed at import if the private kaggle config is missing

BUILDER_SHA = "a34984d3aa60100cf282c3e98d8a55409ae6ed0fc0f543d93d9d0ba463e0b5cd"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_builder(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Builder provenance file missing")
    digest = sha(path)
    if digest != BUILDER_SHA:
        raise ValueError("Builder provenance SHA mismatch")
    return path.read_text()


def check_archives(x_archive, y_archive, audit):
    datasets = {}
    for role, path, slug in (
        ("X", x_archive, K.expand("@@SLUG_X_BUNDLE@@")),
        ("Y", y_archive, K.expand("@@SLUG_Y_BUNDLE@@")),
    ):
        path = Path(path)
        if not path.is_file():
            raise ValueError("ARCHIVE_MISSING " + role)
        if sha(path) != audit[role]["archive_sha256"]:
            raise ValueError("ARCHIVE_CHANGED " + role)
        with zipfile.ZipFile(path) as zipped:
            actual = {
                name: hashlib.sha256(zipped.read(name)).hexdigest()
                for name in zipped.namelist()
                if not name.endswith("/")
            }
        if actual != audit[role]["files"]:
            raise ValueError("ARCHIVE_CONTENT_CHANGED " + role)
        datasets[role] = {
            "id": K.expand("@@KUSER@@/") + slug,
            "mount": K.expand("@@KROOT@@/input/datasets/@@KUSER@@/") + slug,
            "archive_sha256": sha(path),
            "files": actual,
        }
    return datasets


def prepare_treatment(treatment, builder_text):
    if "BX/'make_corpus_X.py':" not in treatment:
        raise ValueError("TREATMENT_BUILDER_PIN_MISSING")
    treatment = treatment.replace(
        "BX/'make_corpus_X.py':",
        K.expand("Path('@@KROOT@@/working/make_corpus_X.py'):"),
    )
    gpu_block = (
        "if not torch.cuda.is_available(): stop('GPU unavailable; no CPU fallback')\n"
        "  from gpu_guard import require_t4\n"
        "  R['gpu']=require_t4(torch); save()"
    )
    preflight_block = (
        "PREFLIGHT_ONLY=os.environ.get('VYRA_PREFLIGHT_ONLY')=='1'\n"
        "  DEVICE='cpu' if PREFLIGHT_ONLY else 'cuda'\n"
        "  if not PREFLIGHT_ONLY:\n"
        "      from gpu_guard import require_t4\n"
        "      R['gpu']=require_t4(torch)\n"
        "  else: R['preflight_only']=True\n"
        "  save()"
    )
    if gpu_block not in treatment:
        raise ValueError("TREATMENT_GPU_BLOCK_MISSING")
    treatment = treatment.replace(gpu_block, preflight_block)
    if "PREFLIGHT_ONLY=" not in treatment:
        raise ValueError("TREATMENT_PREFLIGHT_MISSING")
    if "return m.to('cuda')" not in treatment:
        raise ValueError("TREATMENT_DEVICE_PIN_MISSING")
    treatment = treatment.replace("return m.to('cuda')", "return m.to(DEVICE)")
    before = "  m=make(); t=time.time(); R['baseline']=eval_block(m);"
    if before not in treatment:
        raise ValueError("TREATMENT_BASELINE_ANCHOR_MISSING")
    init = K.expand("""  if PREFLIGHT_ONLY:
      rv.LR=3e-4
      model=make(); opt=rv.make_optimizer(model)
      if any(p.device.type!='cpu' for p in model.parameters()): stop('CPU init escaped device restriction')
      if len(model.state_dict())!=90: stop('wrong model schema')
      if any(st['step']!=0 for st in opt.state.values()): stop('optimizer not fresh')
      for d in (32,64,109):
          assert tuple(XD[d]['unseen'].shape)==(40,512)
          assert tuple(XD[d]['unseen_mask'].shape)==(40,510)
      bt,bm=batch_at(0); rt,rm,_=replay_at(0)
      assert tuple(bt.shape)==(6,512) and tuple(rt.shape)==(2,512)
      assert tuple(bm.shape)==(6,510) and tuple(rm.shape)==(2,510)
      imports={k:str(v.__file__) for k,v in sys.modules.items() if getattr(v,'__file__',None)}
      allowed=('/runtime/','/usr/','@@KROOT@@/')
      unexpected={k:v for k,v in imports.items() if v.startswith('/') and not v.startswith(allowed)}
      if unexpected: stop('unexpected import origins '+repr(unexpected))
      R['initialization']={'model_tensors':len(model.state_dict()),'params':sum(p.numel() for p in model.parameters()),'optimizer':type(opt).__name__,'optimizer_states':len(opt.state),'optimizer_steps':0,'model_forward_calls':0,'device':'cpu','batch_shape':list(bt.shape),'replay_shape':list(rt.shape),'imports':imports,'python':sys.version,'torch':torch.__version__}
      R['status']='PREFLIGHT_INITIALIZED_NO_TRAINING'
      R['elapsed_s']=time.time()-T0;save();print('PREFLIGHT_INITIALIZED_NO_TRAINING',R['elapsed_s'],flush=True);sys.exit(0)
""")
    treatment = treatment.replace(before, init + before)
    treatment = apply_expert_union(treatment)
    treatment = apply_replay_telemetry(treatment)
    treatment = apply_expanded_rehearsal(treatment)
    treatment = apply_update200(treatment)
    return treatment, builder_text


def build(args):
    K.require_real()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_notebook)
    notebook = json.loads(source.read_text())
    code = "".join(notebook["cells"][0]["source"])
    node = next(
        n for n in ast.parse(code).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "assets" for t in n.targets)
    )
    assets = ast.literal_eval(node.value)
    for name in ("treatment.py", "input_preflight.py", "PROTOCOL.json", "entry.py"):
        if name not in assets:
            raise ValueError("NOTEBOOK_ASSET_MISSING " + name)
    builder_text = require_builder(args.builder_source)
    assets["make_corpus_X.py"] = builder_text
    preflight = assets["input_preflight.py"]
    preflight = preflight.replace("'X/make_corpus_X.py':", "'WORK/make_corpus_X.py':")
    preflight = preflight.replace(
        "roots=resolve(mapping);hashes={}",
        K.expand("roots=resolve(mapping);roots['WORK']=Path('@@KROOT@@/working');hashes={}"),
    )
    assets["input_preflight.py"] = preflight
    treatment, _builder = prepare_treatment(assets["treatment.py"], builder_text)
    assets["treatment.py"] = treatment
    assets["dependency_gate.py"] = K.read_asset(Path(__file__).with_name("dependency_gate.py"))
    assets["replay_telemetry.py"] = K.read_asset(Path(__file__).with_name("replay_telemetry.py"))
    assets["preclip_diag.py"] = K.read_asset(Path(__file__).with_name("preclip_diag.py"))
    assets["x64_checkpoint.py"] = K.read_asset(Path(__file__).with_name("x64_checkpoint.py"))
    assets["rehearsal_artifact.py"] = K.read_asset(Path(__file__).with_name("rehearsal_artifact.py"))
    assets["rehearsal_Y72_40_manifest.json"] = K.read_asset(Path(__file__).with_name("rehearsal_Y72_40_manifest.json"))
    audit = json.loads(Path(args.archive_audit).read_text())
    datasets = check_archives(args.x_archive, args.y_archive, audit)
    lock = {
        "schema": 1,
        "datasets": datasets,
        "assets": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in assets.items()},
        "rules": (
            "No fallback discovery, no regeneration, no submission, "
            "CPU preflight cannot forward/backward/step."
        ),
    }
    assets["dependency_lock.json"] = json.dumps(lock, indent=2, sort_keys=True)
    replay_path = Path(args.replay_artifact)
    validate_rehearsal_file(replay_path)
    program = """import os,sys,json,time,hashlib,subprocess,base64
from pathlib import Path
notebook_start=time.monotonic()
"""
    program += "assets=" + repr(assets) + "\n"
    program += "asset_hashes=" + repr({name: hashlib.sha256(text.encode()).hexdigest() for name, text in assets.items()}) + "\n"
    program += K.expand("""if set(assets)!=set(asset_hashes): raise ValueError('BOOTSTRAP_ASSET_SET_MISMATCH')
for name,text in assets.items():
    if hashlib.sha256(text.encode()).hexdigest()!=asset_hashes[name]:raise ValueError('BOOTSTRAP_CORRUPT '+name)
    Path('@@KROOT@@/working',name).write_text(text)
sys.path.insert(0,'@@KROOT@@/working')
""") + rehearsal_bootstrap(replay_path.read_bytes()) + K.expand("""from dependency_gate import verify_lock
verified=verify_lock()
Path('@@KROOT@@/working/dependency_validation.json').write_text(json.dumps(verified,indent=2))
from input_preflight import validate, XID, YID
mapping={XID:'@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_X_BUNDLE@@',YID:'@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_Y_BUNDLE@@'}
validation,roots=validate(mapping)
Path('@@KROOT@@/working/input_validation.json').write_text(json.dumps(validation,indent=2))
os.environ['XROOT']=str(roots['X']);os.environ['YROOT']=str(roots['Y'])
if os.environ.get('VYRA_PREFLIGHT_ONLY')!='1':
    print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True))
from entry import launch_gpu
result=launch_gpu('@@KROOT@@/working/timeout_status.json',cap=600,grace=2)
print('HARNESS',result)
Path('@@KROOT@@/working/notebook_timing.json').write_text(json.dumps({'notebook_cell_wall_s':time.monotonic()-notebook_start,'platform_timeout_requires_CLI_flag':900,'preflight_only':os.environ.get('VYRA_PREFLIGHT_ONLY')=='1','coverage':'queue/provisioning not bounded'}))
if result['timed_out'] or result['returncode']!=0:raise RuntimeError('Subprocess failed/timed out; no retry')
""")
    K.assert_clean(program, "notebook program")
    for _name, _text in assets.items():
        K.assert_clean(_text, "asset " + _name)
    assert_notebook_contract(program)
    ast.parse(program)
    for name, text in assets.items():
        if name.endswith(".py"):
            ast.parse(text)
            # treatment.py binds `build` by exec() of the pinned corpus builder.
            early = used_before_binding(text, predefined={"build"} if name == "treatment.py" else ())
            if early:
                raise ValueError(f"USE_BEFORE_BINDING {name} {early}")
    built = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [{
            "id": "closed-package-pretraining",
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": program.splitlines(keepends=True),
        }],
    }
    kernel = out / "kernel"
    kernel.mkdir(exist_ok=True)
    (kernel / "replay.ipynb").write_text(json.dumps(built, indent=1, sort_keys=True) + "\n")
    meta = {
        "id": K.expand("@@KUSER@@/@@SLUG_KERNEL_REPLAY175@@"),
        "title": K.expand("@@SLUG_KERNEL_REPLAY175@@"),
        "code_file": "replay.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [datasets[key]["id"] for key in ("X", "Y")],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kernel / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    (out / "dependency_lock.json").write_text(assets["dependency_lock.json"])
    report = {
        "status": "BUILT_NOT_VALIDATED",
        "source_notebook_sha256": sha(source),
        "builder_source_sha256": BUILDER_SHA,
        "notebook_sha256": sha(kernel / "replay.ipynb"),
        "assets": lock["assets"],
        "immutable_protocol_sha256": lock["assets"]["PROTOCOL.json"],
        "expert_union_applied": True,
        "replay_telemetry_embedded": True,
        "replay_artifact_sha256": PINNED_SHA256,
        "replay_transport": "embedded_in_notebook_base64",
        "submission_path_used": False,
        "replay_rows": 40,
        "updates": 200,
        "checkpoint_updates": [175, 200],
        "training_gates_unchanged": True,
        "no_scientific_gate_or_input_change": False,
        "no_submission": True,
        "real_input_validation": False,
    }
    (out / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2))
    print("BUILD_PASS", sha(kernel / "replay.ipynb"))
    return report


if __name__ == "__main__":
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
    build(parser.parse_args())
