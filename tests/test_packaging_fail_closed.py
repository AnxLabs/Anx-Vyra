"""Fail-closed checks for the replay package tools.

These tests use temporary files that are not the private archives. They do not
write a validation receipt and they do not stand in for the isolated preflight.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
sys.path.insert(0, str(PACKAGING))

from build import check_archives, prepare_treatment, require_builder  # noqa: E402
from expert_union import apply_expert_union  # noqa: E402
from launch_gate import main as launch_main  # noqa: E402
from validate_package import command  # noqa: E402
from verify_receipt import verify, write_receipt  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
ORCH = PACKAGING / "validate_experiment_package.py"

MINIMAL_TREATMENT = """
if not torch.cuda.is_available(): stop('GPU unavailable; no CPU fallback')
  from gpu_guard import require_t4
  R['gpu']=require_t4(torch); save()
def make():
    return m.to('cuda')
BX/'make_corpus_X.py':'abc'
if tuple(Y72['rows'].shape)!=(8,512) or tuple(Y72['unseen'].shape)!=(40,512): stop('Y72 shape')
def replay_at(s):
      idx=[(s*RB+j)%8 for j in range(RB)]
      return Y72['rows'][idx].long(),Y72['mask'][idx].long(),idx
if any(v['row_overlap'] or v['payload_overlap'] for v in dis.values()): stop('BLOCKED overlap '+json.dumps(dis))
opt=rv.make_optimizer(m); torch.cuda.reset_peak_memory_stats()
for arm,lam in (('treatment',0.25),):
    pass
A={'lambda':lam,'steps':0,'loss':[],'trajectory':{},'eval_s':{}}
for s in range(STEPS):
    tok,msk=batch_at(s); tok=tok.to('cuda'); msk=msk.to('cuda'); lx=ce(m(tok),tok,msk); loss=lx; ly=None
    if lam: yt,ym,_=replay_at(s); yt=yt.to('cuda'); ym=ym.to('cuda'); ly=ce(m(yt),yt,ym); loss=lx+lam*ly
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.grad is not None],1.0)
    opt.step(rv.expert_selection(m)); m.zero_grad(set_to_none=True)
    A['trajectory'][str(step)]=eval_block(m)
R['baseline']=eval_block(m); R['baseline_eval_s']=time.time()-t; save()
  m=make(); t=time.time(); R['baseline']=eval_block(m);
A['replay_extra_note']='treatment adds 2-row replay CE+backward each update; equal updates != equal compute'
"""


def test_expert_union_matches_the_confirmed_caller_edit():
    patched = apply_expert_union(MINIMAL_TREATMENT)
    assert "selected=rv.expert_selection(m)\n" in patched
    assert "opt.step(selected)" in patched
    assert "opt.step(rv.expert_selection(" not in patched
    assert "set(selected[key]) | set(value)" in patched
    with pytest.raises(ValueError, match="EXPERT_UNION_ALREADY_APPLIED"):
        apply_expert_union(patched)


def test_expert_union_refuses_an_unknown_caller():
    with pytest.raises(ValueError, match="EXPERT_UNION_PATTERN_MISSING"):
        apply_expert_union("opt.step(rv.expert_selection(m))\n")


def test_prepare_treatment_keeps_preflight_and_applies_union_once():
    patched, _builder = prepare_treatment(MINIMAL_TREATMENT, "builder-text")
    assert "PREFLIGHT_ONLY=" in patched
    assert "return m.to(DEVICE)" in patched
    assert "model_forward_calls':0" in patched
    assert "opt.step(selected)" in patched
    assert "opt.step(rv.expert_selection(" not in patched
    assert "Path('/kaggle/working/make_corpus_X.py'):" in patched
    assert "tel.begin_update(m, opt)" in patched
    assert "tel.note_grads(m)" in patched
    assert "tel.finish_update(m, opt, tel_first, tel_second, selected)" in patched
    assert "A['expert_telemetry']=tel.summary()" in patched
    assert "replay_telemetry.py" in patched
    assert "lgx=m(tok)" in patched
    assert "%40 for j in range(RB)" in patched
    assert "%8 for j in range(RB)" not in patched
    assert "preclip_diag.begin_separate" in patched
    assert "rehearsal_Y72_40.pt" in patched
    with pytest.raises(ValueError, match="TREATMENT_BUILDER_PIN_MISSING"):
        prepare_treatment(patched, "builder-text")


def test_builder_provenance_mismatch_writes_no_report(tmp_path):
    builder = tmp_path / "not_the_corpus_builder.py"
    builder.write_text("print('synthetic provenance stand-in')\n")
    with pytest.raises(ValueError, match="Builder provenance SHA mismatch"):
        require_builder(builder)
    assert list(tmp_path.glob("BUILD_REPORT.json")) == []


def test_archive_hash_mismatch_is_refused(tmp_path):
    first = tmp_path / "one.bin"
    second = tmp_path / "two.bin"
    first.write_bytes(b"synthetic-bytes")
    second.write_bytes(b"other-synthetic-bytes")
    audit = {
        "X": {"archive_sha256": "0" * 64, "files": {}},
        "Y": {"archive_sha256": "0" * 64, "files": {}},
    }
    with pytest.raises(ValueError, match="ARCHIVE_CHANGED"):
        check_archives(first, second, audit)
    with pytest.raises(ValueError, match="ARCHIVE_MISSING"):
        check_archives(tmp_path / "absent.bin", second, audit)


def test_receipt_writer_refuses_anything_except_the_preflight_report(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("synthetic fixture, not a dataset\n")
    digest = hashlib.sha256(note.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="REFUSING_RECEIPT"):
        write_receipt(tmp_path, {"status": "PASS", "optimizer_steps": 1, "model_forward_calls": 0,
                                 "pretraining_status": "PREFLIGHT_INITIALIZED_NO_TRAINING"},
                      {"note.txt": digest})
    assert not (tmp_path / "validation_receipt.json").exists()


def test_receipt_checker_rejects_a_changed_or_incomplete_file(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("synthetic fixture, not a dataset\n")
    digest = hashlib.sha256(note.read_bytes()).hexdigest()
    (tmp_path / "validation_receipt.json").write_text(json.dumps({
        "status": "VERIFIED_LOCAL_PACKAGE",
        "optimizer_steps": 0,
        "model_forward_calls": 0,
        "locked_files": {"note.txt": digest},
    }))
    note.write_text("synthetic fixture, not a dataset\nchanged\n")
    with pytest.raises(ValueError, match="Changed since isolated preflight"):
        verify(tmp_path)
    with pytest.raises(ValueError, match="Receipt required file missing"):
        verify(tmp_path / "does-not-exist")


def test_launch_gate_refuses_push_without_a_network_call(tmp_path):
    assert launch_main(["--push", str(tmp_path)]) == 2


def test_missing_real_inputs_do_not_validate(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ORCH),
            "--source-notebook", str(tmp_path / "missing.ipynb"),
            "--x-archive", str(tmp_path / "missing-x.bin"),
            "--y-archive", str(tmp_path / "missing-y.bin"),
            "--builder-source", str(tmp_path / "missing-builder.py"),
            "--archive-audit", str(tmp_path / "missing-audit.json"),
            "--replay-artifact", str(tmp_path / "missing-replay.pt"),
            "--inputs", str(tmp_path / "missing-inputs"),
            "--runtime", str(tmp_path / "missing-runtime"),
            "--output", str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["status"] == "REAL_INPUTS_UNAVAILABLE"
    assert payload["submission_ready"] is False
    assert not (tmp_path / "out" / "validation_receipt.json").exists()


def test_package_output_inside_the_repository_is_refused(tmp_path):
    for name in ("notebook.ipynb", "x.bin", "y.bin", "builder.py", "audit.json", "replay.pt"):
        (tmp_path / name).write_text("present\n")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    runtime = tmp_path / "runtime" / "bin"
    runtime.mkdir(parents=True)
    (runtime / "python").write_text("#!/bin/sh\n")
    proc = subprocess.run(
        [
            sys.executable,
            str(ORCH),
            "--source-notebook", str(tmp_path / "notebook.ipynb"),
            "--x-archive", str(tmp_path / "x.bin"),
            "--y-archive", str(tmp_path / "y.bin"),
            "--builder-source", str(tmp_path / "builder.py"),
            "--archive-audit", str(tmp_path / "audit.json"),
            "--replay-artifact", str(tmp_path / "replay.pt"),
            "--inputs", str(inputs),
            "--runtime", str(tmp_path / "runtime"),
            "--output", str(REPO / "packaging-work"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["status"] == "REFUSING_REPOSITORY_OUTPUT"
    assert payload["submission_ready"] is False
    assert not (REPO / "packaging-work").exists()


def test_sandbox_command_does_not_mount_the_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr("validate_package.shutil.which", lambda _name: "/usr/bin/bwrap")
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin" / "python").write_text("#!/bin/sh\n")
    inputs = tmp_path / "inputs"
    (inputs / "vyra-x-bundle").mkdir(parents=True)
    (inputs / "vyra-y-bundle").mkdir(parents=True)
    package = tmp_path / "package"
    (package / "kernel").mkdir(parents=True)
    argv = command(package, inputs, tmp_path / "work", runtime)
    binds = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg in ("--ro-bind", "--bind")]
    assert str(REPO) not in binds
    assert str(PACKAGING) in binds
    assert str(inputs / "vyra-x-bundle") in binds
    assert "/workspace" not in argv
    assert "VYRA_PREFLIGHT_ONLY" in argv
