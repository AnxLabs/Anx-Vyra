"""Fail-closed resume from checkpoint_225 for the Y72 retention fix 2 (per-task clip, replay weight UNCHANGED at 0.25).

Same checks as resume225.load_checkpoint225 (SHA-256/size before torch.load, schema v2, update 225, optimizer config, plan hashes,
two-T4 device gate, complete v2 restore), except that the replay weight must stay exactly equal to the saved 0.25 (resume225 pins a change to
0.5 for the earlier experiment). The integer restoration check is resume225's, unchanged (model-state sha + exact 200-set / X64 counts).
Registered in prereg/y72fix2.json.
"""
import resume225 as base
import continuation as cont
from continuation import ContinuationGateError, _fail
from resume225 import (CKPT225_BYTES, CKPT225_DATASET_ID, CKPT225_MOUNT, CKPT225_PATH, CKPT225_SHA256, COUNTS_225,  # noqa: F401
                       MODEL_STATE_SHA256_225, PINNED_FULL_PLAN_SHA256, PINNED_PREFIX_PLAN_SHA256, RESUME_UPDATE, SAVED_CONFIG_EXCEPT_LAMBDA,
                       SAVED_LAMBDA, X64_225, check_resume_restoration)

NEW_LAMBDA = 0.25          # unchanged by design: the only variable is the gradient clip scheme


def validate_v2_resume(ck, *, new_lambda=NEW_LAMBDA, expected_update=RESUME_UPDATE, cuda_required=True):
    if ck.get("schema") != cont.V2_SCHEMA:
        _fail("CKPT_SCHEMA", f"schema {ck.get('schema')!r}")
    missing = [k for k in cont.V2_REQUIRED if k not in ck]
    if missing:
        _fail("CKPT_SCHEMA", f"missing {missing}")
    if ck["update"] != expected_update:
        _fail("CKPT_UPDATE", f"{ck['update']} != {expected_update}")
    cfg = dict(ck["optimizer_config"])
    if cfg.pop("replay_lambda", None) != SAVED_LAMBDA:
        _fail("OPTIMIZER_CONFIG_MISMATCH", f"saved replay_lambda != {SAVED_LAMBDA}")
    if cfg != SAVED_CONFIG_EXCEPT_LAMBDA:
        _fail("OPTIMIZER_CONFIG_MISMATCH", f"{cfg} != {SAVED_CONFIG_EXCEPT_LAMBDA}")
    if new_lambda != SAVED_LAMBDA:
        _fail("LAMBDA_CHANGED", f"{new_lambda} != saved {SAVED_LAMBDA}: the only variable of this experiment is the clip scheme")
    if cuda_required and (ck["cuda_rng"] is None or len(ck["cuda_rng"]) != cont.REQUIRED_CUDA_DEVICES):
        _fail("CKPT_SCHEMA", "cuda_rng must hold the two saved device states")
    if ck["sampler"].get("plan_prefix200_sha256") != PINNED_PREFIX_PLAN_SHA256 or ck["sampler"].get("plan_full_sha256") != PINNED_FULL_PLAN_SHA256:
        _fail("PLAN_PREFIX_MISMATCH", "saved sampler plan hashes differ from the pinned plan")
    if ck["sampler"].get("next_update_index") != expected_update:
        _fail("SAMPLER_MISMATCH", "next_update_index")


def load_checkpoint225(path, *, model, opt, plan, plan_module, lib, device, replay_batch=2, cpu_smoke=False,
                       expected_sha=CKPT225_SHA256, expected_bytes=CKPT225_BYTES, torch_mod=base.torch):
    raw = cont.read_verified_bytes(path, expected_sha, expected_bytes)
    ck = cont.load_full_checkpoint(raw)
    validate_v2_resume(ck, cuda_required=not cpu_smoke)
    cont.check_plan_prefix(plan, plan_module=plan_module)
    if plan_module.plan_sha256(plan) != PINNED_FULL_PLAN_SHA256:
        _fail("PLAN_PREFIX_MISMATCH", "live full plan hash differs from the pinned full plan")
    dev = cont.check_device(model, ck, device=device, cpu_smoke=cpu_smoke, torch_mod=torch_mod)
    rec = cont.restore_full(model, opt, ck, lib=lib, plan=plan, plan_module=plan_module, replay_batch=replay_batch, cuda=dev["cuda_rng_restored"])
    return {"path": str(path), "sha256": expected_sha, "bytes": expected_bytes, "update": ck["update"], "device": dev,
            "restored_components": rec["components"], "saved_lambda": SAVED_LAMBDA, "new_lambda": NEW_LAMBDA,
            "saved_config_except_lambda": SAVED_CONFIG_EXCEPT_LAMBDA, "provenance_of_checkpoint": ck["provenance"]}
