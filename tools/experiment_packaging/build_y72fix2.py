"""Deterministic builder for the Y72 retention fix 2 notebook (resume checkpoint_225, per-task clip 0.5, lambda 0.25, stop at 300; prereg/y72fix2.json).

Never regenerates data, never trains, never uploads, never submits. Built from
the same source notebook, builder pin and archive audit as the 200-update
build (build.py), so the 200-update treatment is reproduced first and only then
rewritten by continue300. The checkpoint is bound by SHA-256 into the
dependency lock; the builder refuses a checkpoint file that does not match.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from build import BUILDER_SHA, check_archives, prepare_treatment, require_builder, sha
from continue225_y72fix2 import apply_y72fix2, patch_preclip_diag
from resume225_y72fix2 import (CKPT225_BYTES, CKPT225_DATASET_ID, CKPT225_MOUNT, CKPT225_SHA256, MODEL_STATE_SHA256_225)
from name_order import used_before_binding
from plan_extension import PINNED_FULL_PLAN_SHA256, PINNED_PREFIX_PLAN_SHA256
from rehearsal_artifact import PINNED_SHA256, validate_rehearsal_file
from runtime_contract import assert_notebook_contract, rehearsal_bootstrap

HERE = Path(__file__).parent
EXTRA_ASSETS = ("continuation.py", "plan_extension.py", "resume225.py", "resume225_y72fix2.py", "y72fix_gates.py", "y72fix2_gates.py",
                "y72_heldout_v2.py", "y72fix2_runtime.py", "clip_scheme.py")
PREREG_SHA256 = "8eaace70888b8ec6c02537354a6ec98118327505f1a482deea01221f8475a89a"
SET_FILE = Path(__file__).resolve().parents[2] / "prereg" / "y72_heldout_v2" / "y72_heldout_v2_set.json"
PREREG_FILE = Path(__file__).resolve().parents[2] / "prereg" / "y72fix2.json"
SOURCE_NOTEBOOK_SHA256 = "d12f782e75a9d2595a0b95df7f702ded1af46098d9aaa786edc523e86f17aa4a"


def check_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("CHECKPOINT_MISSING")
    raw = path.read_bytes()
    if len(raw) != CKPT225_BYTES or hashlib.sha256(raw).hexdigest() != CKPT225_SHA256:
        raise ValueError("CHECKPOINT_CHANGED")
    return {
        "id": CKPT225_DATASET_ID,
        "mount": CKPT225_MOUNT,
        "files": {"checkpoint_225.pt": CKPT225_SHA256},
        "bytes": CKPT225_BYTES,
    }


def build(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_notebook)
    if sha(source) != SOURCE_NOTEBOOK_SHA256:
        raise ValueError("SOURCE_NOTEBOOK_CHANGED")
    notebook = json.loads(source.read_text())
    code = "".join(notebook["cells"][0]["source"])
    node = next(
        n for n in ast.parse(code).body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "assets" for t in n.targets)
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
        "roots=resolve(mapping);roots['WORK']=Path('/kaggle/working');hashes={}",
    )
    assets["input_preflight.py"] = preflight
    treatment, _ = prepare_treatment(assets["treatment.py"], builder_text)
    assets["treatment.py"] = apply_y72fix2(treatment)
    for name in ("dependency_gate.py", "replay_telemetry.py", "x64_checkpoint.py", "rehearsal_artifact.py",
                 "rehearsal_Y72_40_manifest.json") + EXTRA_ASSETS:
        assets[name] = (HERE / name).read_text()
    if hashlib.sha256(PREREG_FILE.read_bytes()).hexdigest() != PREREG_SHA256:
        raise ValueError("PREREG_CHANGED")
    import y72_heldout_v2 as _heldout
    _heldout.verify_text_set(SET_FILE)                       # registered 600-segment set, fail closed (canonical + payload-list sha256)
    assets["y72_heldout_v2_set.json"] = SET_FILE.read_text()
    assets["prereg_y72fix2.json"] = PREREG_FILE.read_text()
    assets["preclip_diag.py"] = patch_preclip_diag((HERE / "preclip_diag.py").read_text())
    audit = json.loads(Path(args.archive_audit).read_text())
    datasets = check_archives(args.x_archive, args.y_archive, audit)
    datasets["C"] = check_checkpoint(args.checkpoint)
    lock = {
        "schema": 1,
        "datasets": datasets,
        "assets": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in assets.items()},
        "rules": (
            "No fallback discovery, no regeneration, no submission, no retry. "
            "checkpoint_225 SHA-256 is verified before any torch.load of it. "
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
    program += """if set(assets)!=set(asset_hashes): raise ValueError('BOOTSTRAP_ASSET_SET_MISMATCH')
for name,text in assets.items():
    if hashlib.sha256(text.encode()).hexdigest()!=asset_hashes[name]:raise ValueError('BOOTSTRAP_CORRUPT '+name)
    Path('/kaggle/working',name).write_text(text)
sys.path.insert(0,'/kaggle/working')
""" + rehearsal_bootstrap(replay_path.read_bytes()) + """from dependency_gate import verify_lock
verified=verify_lock()
Path('/kaggle/working/dependency_validation.json').write_text(json.dumps(verified,indent=2))
from input_preflight import validate, XID, YID
mapping={XID:'/kaggle/input/datasets/anjorosimoaj/vyra-x-bundle',YID:'/kaggle/input/datasets/anjorosimoaj/vyra-y-bundle'}
validation,roots=validate(mapping)
Path('/kaggle/working/input_validation.json').write_text(json.dumps(validation,indent=2))
os.environ['XROOT']=str(roots['X']);os.environ['YROOT']=str(roots['Y'])
if os.environ.get('VYRA_PREFLIGHT_ONLY')!='1':
    print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True))
from entry import launch_gpu
result=launch_gpu('/kaggle/working/timeout_status.json',cap=600,grace=2)
print('HARNESS',result)
Path('/kaggle/working/notebook_timing.json').write_text(json.dumps({'notebook_cell_wall_s':time.monotonic()-notebook_start,'platform_timeout_requires_CLI_flag':900,'preflight_only':os.environ.get('VYRA_PREFLIGHT_ONLY')=='1','coverage':'queue/provisioning not bounded'}))
if result['timed_out'] or result['returncode']!=0:raise RuntimeError('Subprocess failed/timed out; no retry')
"""
    assert_notebook_contract(program)
    ast.parse(program)
    for name, text in assets.items():
        if name.endswith(".py"):
            ast.parse(text)
            early = used_before_binding(text, predefined={"build"} if name == "treatment.py" else ())
            if early:
                raise ValueError(f"USE_BEFORE_BINDING {name} {early}")
    built = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [{
            "id": "closed-package-y72fix2-pertask",
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
        "id": "anjorosimoaj/vyra-y72fix2-pertask-packaged",
        "title": "vyra-y72fix2-pertask-packaged",
        "code_file": "replay.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [datasets[key]["id"] for key in ("X", "Y", "C")],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kernel / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    (out / "dependency_lock.json").write_text(assets["dependency_lock.json"])
    report = {
        "status": "BUILT_NOT_VALIDATED_NOT_SUBMITTED",
        "source_notebook_sha256": sha(source),
        "builder_source_sha256": BUILDER_SHA,
        "notebook_sha256": sha(kernel / "replay.ipynb"),
        "assets": lock["assets"],
        "checkpoint_dataset": datasets["C"],
        "plan_prefix_sha256": PINNED_PREFIX_PLAN_SHA256,
        "plan_full_sha256": PINNED_FULL_PLAN_SHA256,
        "replay_artifact_sha256": PINNED_SHA256,
        "updates": {"from": 225, "to": 300},
        "checkpoint_updates": [250, 275, 300],
        "only_variable": {"name": "gradient clip scheme", "from": "global clip 1.0", "to": "per-task clip 0.5 (lambda 0.25 unchanged)"},
        "prereg_sha256": PREREG_SHA256,
        "model_state_sha256_225": MODEL_STATE_SHA256_225,
        "limits": {"trainer_s": 600, "notebook_s": 900},
        "no_submission": True,
        "no_upload": True,
    }
    (out / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print("BUILD_PASS", sha(kernel / "replay.ipynb"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source-notebook", "x-archive", "y-archive", "builder-source", "archive-audit",
                 "replay-artifact", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    build(parser.parse_args())
